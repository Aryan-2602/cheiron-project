"""Stage 1: query understanding. The only place an LLM is used.

The model's job is deliberately narrow: read the question, classify what kind of
question it is, copy out the entities it mentions, and pick a visualization
shape. It never counts anything, and its output type
(:class:`~app.models.schemas.QueryUnderstanding`) has no field a count could
live in.

Two deterministic passes run over whatever the model returns:

* :func:`ground_entities` drops any entity string that does not actually appear
  in the user's question. A model that hallucinates "Keytruda" for a question
  about nivolumab gets that entity removed rather than issuing a search for it.
* :func:`reconcile_viz_type` overrides the model's chart choice when it is
  incompatible with the query type, so the rest of the pipeline can rely on the
  pair being coherent.

Structured outputs (``response_format=QueryUnderstanding``) guarantee a
schema-valid object, so there is no JSON repair or retry-on-parse-failure code.
"""

from __future__ import annotations

import logging
import re
import time
from difflib import SequenceMatcher

from openai import OpenAI, OpenAIError

from app.core.config import settings
from app.models.schemas import (
    ExtractedEntities,
    QueryPlan,
    QueryRequest,
    QueryUnderstanding,
    YearRange,
)

logger = logging.getLogger(__name__)


class UnderstandingError(RuntimeError):
    """LLM stage failed; the pipeline surfaces this as LLM_ERROR."""


SYSTEM_PROMPT = """\
You interpret natural-language questions about clinical trials so that a \
separate program can query the ClinicalTrials.gov API and build a chart.

You never estimate, invent, or output counts, statistics, totals, percentages, \
or any data value. You do not know how many trials exist. Your entire job is to \
classify the question and extract the entities that literally appear in it. \
All numbers in the final answer are computed by other code from real API \
responses.

Classify `query_type`:
- distribution: how trials break down across a category ("across phases", \
"by sponsor", "most common intervention types")
- time_trend: how a count changes over time ("per year", "since 2015", "over time")
- comparison: two or more named things set against each other ("A vs B", \
"compare X and Y")
- relationship: how entities connect to each other ("network of", "co-occur", \
"which drugs are combined with")
- geographic: breakdown by country or location ("which countries", "where")
- unsupported: not answerable from clinical-trial registry data

Pick `viz_type` to match:
- distribution -> bar_chart (or histogram when the axis is enrollment size)
- time_trend -> time_series
- comparison -> grouped_bar_chart
- relationship -> network_graph
- geographic -> geo_bar_chart

Set `group_by` to the axis the trials should be grouped along, for every \
query_type except relationship:
- phase, start_year, status, sponsor, sponsor_class, intervention_type, \
country, enrollment_bucket
Use start_year for time_trend. For comparison, group_by is the axis *within* \
each series (usually phase), and compare_entities holds the things being \
compared.

Set `network_kind` only for relationship queries:
- drug_drug: which drugs are studied together / co-occur in combination trials
- sponsor_drug: which sponsors work on which drugs

Entity extraction rules:
- Copy entity strings verbatim from the question. Do not expand abbreviations, \
substitute brand names for generic names, or add entities the user did not write.
- `phases` are bare integers 1-4 only, and only when the user names a phase.
- `statuses` use plain words like "recruiting" or "completed".
- Leave a list empty rather than guessing.

Use `assumptions` to record any interpretation you had to make, in one short \
sentence each, so the user can see how their question was read."""


def _client() -> OpenAI:
    api_key = settings.openai_api_key
    if not api_key:
        raise UnderstandingError(
            "No OpenAI API key configured. Set OPEN_AI_API_KEY (or OPENAI_API_KEY) "
            "in the environment or .env file."
        )
    return OpenAI(api_key=api_key)


def call_llm(query: str, *, model: str | None = None) -> QueryUnderstanding:
    """Ask the model to classify one question. Raises :class:`UnderstandingError`."""
    model = model or settings.LLM_MODEL
    started = time.perf_counter()
    # Prompt and completion are deliberately never logged -- only a summary of
    # what the model decided, which is what a reader actually needs.
    logger.info("llm call started", extra={"model": model, "query_chars": len(query)})

    def _elapsed_ms() -> float:
        return round((time.perf_counter() - started) * 1000, 1)

    try:
        completion = _client().chat.completions.parse(
            model=model,
            response_format=QueryUnderstanding,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": query},
            ],
        )
    except OpenAIError as exc:
        # The OpenAI SDK retries internally and does not expose per-attempt
        # callbacks, so this is the only point at which a failure is visible.
        logger.warning(
            "llm call failed",
            extra={
                "model": model,
                "error_type": type(exc).__name__,
                "duration_ms": _elapsed_ms(),
            },
        )
        raise UnderstandingError(f"LLM request failed: {exc}") from exc

    # An absent or empty `choices` is a documented possibility (content
    # filtering, upstream anomaly). Indexing it blindly raised IndexError /
    # TypeError, which OpenAIError above does not catch, so it escaped as an
    # unhandled 500 instead of the 502 LLM_ERROR the contract promises.
    if not completion.choices:
        logger.warning(
            "llm returned no choices",
            extra={"model": model, "duration_ms": _elapsed_ms()},
        )
        raise UnderstandingError("Model returned no completion choices.")

    message = completion.choices[0].message
    if message.refusal:
        logger.warning(
            "llm refused request",
            extra={"model": model, "duration_ms": _elapsed_ms()},
        )
        raise UnderstandingError(f"Model declined the request: {message.refusal}")
    if message.parsed is None:
        logger.warning(
            "llm returned no structured output",
            extra={"model": model, "duration_ms": _elapsed_ms()},
        )
        raise UnderstandingError("Model returned no parsable structured output.")

    logger.info(
        "llm call completed",
        extra={
            "model": model,
            "query_type": message.parsed.query_type,
            "viz_type": message.parsed.viz_type,
            "group_by": message.parsed.group_by,
            "network_kind": message.parsed.network_kind,
            # Counts only -- the extracted values themselves are user text.
            "entity_counts": {
                "drugs": len(message.parsed.entities.drugs),
                "conditions": len(message.parsed.entities.conditions),
                "sponsors": len(message.parsed.entities.sponsors),
                "countries": len(message.parsed.entities.countries),
                "phases": len(message.parsed.entities.phases),
                "statuses": len(message.parsed.entities.statuses),
            },
            "duration_ms": _elapsed_ms(),
        },
    )
    return message.parsed


# --------------------------------------------------------------------------
# Deterministic post-processing
# --------------------------------------------------------------------------

_WORD_RE = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


#: Below this length the fuzzy fallback is skipped and an exact, word-bounded
#: match is required. For a four-character acronym a single character is a
#: quarter of the string, so "SCLC" scores 0.889 against "NSCLC" -- above the
#: threshold, and a different disease. Acronyms need no inflection tolerance
#: anyway; the exact path already handles them.
#:
#: 6, not 8: a floor of 8 would stop "sarcoma" (7) matching "sarcomas", which
#: is exactly the inflection the fallback exists for.
_MIN_FUZZY_LENGTH = 6

#: Similarity required of the fuzzy fallback. Verified against similar drug
#: names (nivolumab/pembrolizumab, olaparib/niraparib, imatinib/dasatinib,
#: trastuzumab/pertuzumab and others) -- none come close to matching.
_FUZZY_THRESHOLD = 0.87


def _appears_in(candidate: str, haystack: str) -> bool:
    """True when ``candidate`` is plausibly present in ``haystack``.

    An exact, **word-bounded** match is the common case. Boundaries matter:
    a plain substring test let an acronym match inside a longer word, so the
    leukemia "ALL" matched the ordinary word "sm-ALL-cell", and "SCLC" matched
    inside "NSCLC" -- non-small-cell and small-cell lung cancer are different
    diseases with different treatments.

    The fuzzy fallback exists only to tolerate inflection and spacing ("lung
    cancers" for "lung cancer"), not to admit entities the user never wrote,
    and is skipped entirely for short strings (see ``_MIN_FUZZY_LENGTH``).
    """
    candidate = candidate.strip().lower()
    if not candidate:
        return False
    haystack = haystack.lower()

    # (?<!\w) / (?!\w) rather than \b: the candidate may begin or end with a
    # non-word character ("5-fu"), where \b would assert the wrong thing.
    if re.search(rf"(?<!\w){re.escape(candidate)}(?!\w)", haystack):
        return True

    candidate_tokens = _tokens(candidate)
    haystack_tokens = _tokens(haystack)
    if not candidate_tokens:
        return False

    joined = " ".join(candidate_tokens)
    window = len(candidate_tokens)
    for i in range(len(haystack_tokens) - window + 1):
        chunk = " ".join(haystack_tokens[i : i + window])
        if min(len(joined), len(chunk)) < _MIN_FUZZY_LENGTH:
            continue
        if SequenceMatcher(None, joined, chunk).ratio() >= _FUZZY_THRESHOLD:
            return True
    return False


# --------------------------------------------------------------------------
# Deterministic extraction for the closed-vocabulary fields
#
# Phases and statuses are small, closed sets with unambiguous surface forms, so
# they can be read straight out of the query text. That is a strictly stronger
# guarantee than checking the model's answer: the value provably comes from the
# user's words. The model's own extraction is kept only as a cross-check, and
# anything it proposed that the text does not support is dropped and reported.
# --------------------------------------------------------------------------

#: A phase mention plus the contiguous list that follows it. Anchoring on the
#: word "phase" and capturing only the run of connected tokens is what stops
#: "phase 2 study of 3 drugs" from yielding phase 3.
_PHASE_LIST = re.compile(
    r"\bphases?\b[\s:]*((?:(?:[1-4]|iv|iii|ii|i)\b[\s]*(?:[/,&+-]|or|and|to)?[\s]*)+)",
    re.IGNORECASE,
)
_PHASE_TOKEN = re.compile(r"\b([1-4]|iv|iii|ii|i)\b", re.IGNORECASE)
_ROMAN_PHASES = {"i": 1, "ii": 2, "iii": 3, "iv": 4}

#: Four-digit years plausible for a trial start date. Bounded so that a dose
#: ("200 mg") or an NCT id can never be mistaken for a year.
_YEAR = re.compile(r"\b(19[89]\d|20[0-3]\d)\b")

#: ClinicalTrials.gov ``overallStatus`` values, verified live against the API,
#: mapped from the words a person actually types. Ordered longest-phrase-first
#: within each entry so "not yet recruiting" is consumed before "recruiting".
STATUS_VOCABULARY: dict[str, tuple[str, ...]] = {
    "NOT_YET_RECRUITING": ("not yet recruiting", "not-yet-recruiting", "upcoming"),
    "ACTIVE_NOT_RECRUITING": ("active not recruiting", "active, not recruiting"),
    "ENROLLING_BY_INVITATION": ("enrolling by invitation",),
    "NO_LONGER_AVAILABLE": ("no longer available",),
    "RECRUITING": ("recruiting", "enrolling", "actively enrolling"),
    "COMPLETED": ("completed", "complete", "finished", "concluded"),
    "TERMINATED": ("terminated", "halted", "stopped early", "stopped"),
    "SUSPENDED": ("suspended", "paused", "on hold"),
    "WITHDRAWN": ("withdrawn",),
    "AVAILABLE": ("available",),
    "UNKNOWN": ("unknown status",),
}


def extract_phases_from_query(query: str) -> list[int]:
    """Phase numbers the user actually wrote. Handles "phase 3", "phase 1/2",
    "phases 2 and 3", and roman numerals."""
    found: set[int] = set()
    for match in _PHASE_LIST.finditer(query):
        for token in _PHASE_TOKEN.findall(match.group(1)):
            token = token.lower()
            found.add(int(token) if token.isdigit() else _ROMAN_PHASES[token])
    return sorted(found)


def extract_years_from_query(query: str) -> set[int]:
    """Every four-digit year literally present in the query."""
    return {int(y) for y in _YEAR.findall(query)}


def extract_statuses_from_query(query: str) -> list[str]:
    """Trial statuses the user actually named, as API enum values.

    Longer phrases are matched and masked out first, so "not yet recruiting"
    cannot also register as "recruiting".
    """
    haystack = query.lower()
    found: list[str] = []
    phrases = sorted(
        ((phrase, status) for status, ps in STATUS_VOCABULARY.items() for phrase in ps),
        key=lambda pair: -len(pair[0]),
    )
    for phrase, status in phrases:
        index = haystack.find(phrase)
        if index >= 0:
            if status not in found:
                found.append(status)
            haystack = haystack[:index] + " " * len(phrase) + haystack[index + len(phrase) :]
    return found


def ground_entities(
    understanding: QueryUnderstanding, query: str
) -> tuple[ExtractedEntities, list[str]]:
    """Constrain every extracted field to what the question actually supports.

    This is the guard against the one thing the LLM could still get wrong in a
    way that reaches real data: searching for something nobody asked about.
    *Every* field that reaches the query builder or the client-side filters is
    grounded here, because each of them changes which trials are retrieved:

    * free-text entities -- must appear in the question (fuzzy, for inflection)
    * phases and statuses -- read from the question directly, against closed
      vocabularies; the model's answer is only cross-checked against that
    * year bounds -- the year must appear literally in the question

    Returns the constrained entities and a warning per dropped item, so a
    divergence between what the model proposed and what the question supports
    is visible in the response rather than silently applied.
    """
    entities = understanding.entities
    warnings: list[str] = []

    def keep(values: list[str], label: str) -> list[str]:
        kept = []
        for value in values:
            if _appears_in(value, query):
                kept.append(value.strip())
            else:
                warnings.append(
                    f"Ignored {label} {value!r}: it does not appear in the question."
                )
        return kept

    # Phases: the text is authoritative.
    grounded_phases = extract_phases_from_query(query)
    for phase in entities.phases:
        if phase in (1, 2, 3, 4) and phase not in grounded_phases:
            warnings.append(
                f"Ignored phase {phase}: the question does not mention it."
            )

    # Statuses: likewise, and this also keeps an unrecognised status string out
    # of the client-side filter, where it would have matched no trial at all.
    grounded_statuses = extract_statuses_from_query(query)
    for status in entities.statuses:
        normalized = status.strip().upper().replace(" ", "_")
        if normalized and normalized not in grounded_statuses:
            warnings.append(
                f"Ignored status {status!r}: the question does not support it."
            )

    # Year bounds: the value must be written in the question. Note this grounds
    # *which* years may be used, not which end of the range each belongs to --
    # that reading stays with the model and is disclosed in meta.warnings when
    # the filter is applied.
    years_in_query = extract_years_from_query(query)
    grounded_range = None
    if entities.year_range is not None:
        start, end = entities.year_range.start, entities.year_range.end
        if start is not None and start not in years_in_query:
            warnings.append(
                f"Ignored start year {start}: it does not appear in the question."
            )
            start = None
        if end is not None and end not in years_in_query:
            warnings.append(
                f"Ignored end year {end}: it does not appear in the question."
            )
            end = None
        if start is not None or end is not None:
            grounded_range = YearRange(start=start, end=end)

    grounded = ExtractedEntities(
        drugs=keep(entities.drugs, "drug"),
        conditions=keep(entities.conditions, "condition"),
        sponsors=keep(entities.sponsors, "sponsor"),
        phases=grounded_phases,
        statuses=grounded_statuses,
        countries=keep(entities.countries, "country"),
        year_range=grounded_range,
    )
    return grounded, warnings


def ground_compare_entities(
    compare_entities: list[str], query: str
) -> tuple[list[str], list[str]]:
    """Comparison series must be named in the question too.

    These drive one upstream search each, so an invented entity would not just
    mislabel a series -- it would fetch and chart trials nobody asked about.
    """
    kept, warnings = [], []
    for entity in compare_entities:
        if _appears_in(entity, query):
            kept.append(entity.strip())
        else:
            warnings.append(
                f"Ignored comparison entity {entity!r}: it does not appear in "
                f"the question."
            )
    return kept, warnings


#: The chart shape each query type must produce. The LLM proposes; this decides.
VIZ_FOR_QUERY_TYPE = {
    "distribution": "bar_chart",
    "time_trend": "time_series",
    "comparison": "grouped_bar_chart",
    "relationship": "network_graph",
    "geographic": "geo_bar_chart",
}

#: Alternatives that are legitimate for a query type and left alone.
ALLOWED_ALTERNATIVES = {
    "distribution": {"histogram", "bar_chart"},
    "geographic": {"geo_bar_chart", "bar_chart"},
}

DEFAULT_GROUP_BY = {
    "distribution": "phase",
    "time_trend": "start_year",
    "comparison": "phase",
    "geographic": "country",
}


def reconcile_viz_type(understanding: QueryUnderstanding) -> tuple[str, list[str]]:
    """Force the visualization type to be coherent with the query type."""
    query_type = understanding.query_type
    proposed = understanding.viz_type
    if query_type == "unsupported":
        return proposed, []

    allowed = ALLOWED_ALTERNATIVES.get(query_type, {VIZ_FOR_QUERY_TYPE[query_type]})
    if proposed in allowed:
        return proposed, []

    corrected = VIZ_FOR_QUERY_TYPE[query_type]
    return corrected, [
        (
            f"Rendered as {corrected} rather than {proposed}, to match a "
            f"{query_type.replace('_', ' ')} question."
        )
    ]


def build_plan(request: QueryRequest, understanding: QueryUnderstanding) -> QueryPlan:
    """Ground, reconcile, and merge structured overrides into a final plan.

    Structured request fields win over LLM extraction: a caller who explicitly
    passes ``drug_name`` has stated the entity outright, so there is nothing to
    infer.
    """
    entities, warnings = ground_entities(understanding, request.query)
    viz_type, viz_warnings = reconcile_viz_type(understanding)
    assumptions = list(understanding.assumptions) + viz_warnings

    # Structured overrides.
    if request.drug_name:
        entities.drugs = [request.drug_name]
    if request.condition:
        entities.conditions = [request.condition]
    if request.sponsor:
        entities.sponsors = [request.sponsor]
    if request.country:
        entities.countries = [request.country]
    if request.phase is not None:
        entities.phases = [request.phase]
    if request.start_year is not None or request.end_year is not None:
        entities.year_range = YearRange(
            start=request.start_year, end=request.end_year
        )

    group_by = understanding.group_by
    if understanding.query_type != "relationship":
        group_by = group_by or DEFAULT_GROUP_BY.get(understanding.query_type, "phase")
        if understanding.query_type == "time_trend":
            group_by = "start_year"

    network_kind = understanding.network_kind
    if understanding.query_type == "relationship" and network_kind is None:
        network_kind = "drug_drug"
        assumptions.append(
            "Interpreted as a drug co-occurrence network; no relationship type "
            "was specified."
        )

    compare_entities, compare_warnings = ground_compare_entities(
        understanding.compare_entities, request.query
    )
    warnings.extend(compare_warnings)
    compare_kind = understanding.compare_entity_kind
    if understanding.query_type == "comparison" and len(compare_entities) < 2:
        # Nothing concrete to compare; fall back to a plain distribution so the
        # user still gets a grounded answer instead of an error.
        viz_type = "bar_chart"
        compare_entities, compare_kind = [], None
        assumptions.append(
            "Fewer than two comparable entities were named, so a single "
            "distribution is shown instead of a comparison."
        )

    return QueryPlan(
        query=request.query,
        query_type=understanding.query_type,
        entities=entities,
        group_by=group_by,
        compare_entities=compare_entities,
        compare_entity_kind=compare_kind,
        network_kind=network_kind,
        viz_type=viz_type,
        assumptions=assumptions,
        warnings=warnings,
        max_citations_per_datum=request.max_citations_per_datum,
        max_studies=request.max_studies,
    )


def understand(request: QueryRequest, *, model: str | None = None) -> QueryPlan:
    """Full stage 1: LLM call plus deterministic grounding and reconciliation."""
    understanding = call_llm(request.query, model=model)
    return build_plan(request, understanding)
