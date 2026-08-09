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

from openai import AsyncOpenAI, OpenAIError

from app.core.config import settings
from app.models.schemas import (
    ExtractedEntities,
    QueryPlan,
    QueryRequest,
    QueryUnderstanding,
    YearRange,
)
from app.services.ctgov import series_key

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


def _client() -> AsyncOpenAI:
    api_key = settings.openai_api_key
    if not api_key:
        raise UnderstandingError(
            "No OpenAI API key configured. Set OPEN_AI_API_KEY (or OPENAI_API_KEY) "
            "in the environment or .env file."
        )
    return AsyncOpenAI(api_key=api_key)


async def call_llm(query: str, *, model: str | None = None) -> QueryUnderstanding:
    """Ask the model to classify one question. Raises :class:`UnderstandingError`.

    Async because this is the one outbound call on the request path that is not
    already awaited. The synchronous client blocked the event-loop thread for
    the whole round trip, so a single slow completion stalled every other
    request the worker was serving -- a correctness-neutral but real
    availability problem. Structured-output parsing is unchanged; AsyncOpenAI
    exposes the same ``chat.completions.parse``.
    """
    model = model or settings.LLM_MODEL
    started = time.perf_counter()
    # Prompt and completion are deliberately never logged -- only a summary of
    # what the model decided, which is what a reader actually needs.
    logger.info("llm call started", extra={"model": model, "query_chars": len(query)})

    def _elapsed_ms() -> float:
        return round((time.perf_counter() - started) * 1000, 1)

    try:
        completion = await _client().chat.completions.parse(
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
#: Every phrase here must mean a *registry status* and nothing else. Bare
#: "available", "complete", and "stopped" were dropped because they are
#: ordinary English first and statuses second, and matching them silently
#: changed the population: "which countries have available melanoma trials"
#: filtered to expanded-access records, "trials with complete response rates"
#: filtered to COMPLETED, and "trials that stopped recruiting" matched
#: TERMINATED *and* RECRUITING at once. The longer forms they appear in
#: ("no longer available", "completed", "stopped early") are unambiguous and
#: stay.
STATUS_VOCABULARY: dict[str, tuple[str, ...]] = {
    "NOT_YET_RECRUITING": ("not yet recruiting", "not-yet-recruiting", "upcoming"),
    "ACTIVE_NOT_RECRUITING": (
        "active not recruiting",
        "active, not recruiting",
        # Unambiguous: a trial closed to enrolment but still running. Unlike
        # "stopped recruiting", these name the status rather than describing a
        # thing that several statuses have in common.
        "closed to enrollment",
        "closed to enrolment",
    ),
    "ENROLLING_BY_INVITATION": ("enrolling by invitation",),
    "NO_LONGER_AVAILABLE": ("no longer available",),
    "RECRUITING": ("recruiting", "enrolling", "actively enrolling"),
    "COMPLETED": ("completed", "finished", "concluded"),
    "TERMINATED": ("terminated", "halted", "stopped early"),
    "SUSPENDED": ("suspended", "paused", "on hold"),
    "WITHDRAWN": ("withdrawn",),
    "AVAILABLE": ("expanded access", "available for expanded access"),
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


#: A two-ended range, in the forms people actually write. Checked before the
#: one-ended cues, so "from 2015 to 2020" is not read as a bare "from".
_YEAR_SPAN = re.compile(
    r"\b(?:between\s+)?(19[89]\d|20[0-3]\d)\s*(?:-|--|–|—|to|and|through|until|till)\s*"
    r"(19[89]\d|20[0-3]\d)\b",
    re.IGNORECASE,
)

#: Cues that bind a single year to one end of the range, split by whether the
#: named year is itself included. At year granularity "after 2015" excludes
#: 2015 and "before 2020" excludes 2020, while "since"/"from"/"through"/"up to"
#: include theirs -- treating them all as inclusive quietly widened the range
#: by a year on either side. Structured start_year/end_year stay inclusive,
#: which is the documented contract.
_YEAR_START_INCLUSIVE = re.compile(
    r"\b(?:since|from|starting(?:\s+in)?|beginning(?:\s+in)?|onwards?\s+from)\s+"
    r"(19[89]\d|20[0-3]\d)\b",
    re.IGNORECASE,
)
_YEAR_START_EXCLUSIVE = re.compile(
    r"\b(?:after|later\s+than)\s+(19[89]\d|20[0-3]\d)\b", re.IGNORECASE
)
_YEAR_END_INCLUSIVE = re.compile(
    r"\b(?:until|till|through|up\s+to|by)\s+(19[89]\d|20[0-3]\d)\b", re.IGNORECASE
)
_YEAR_END_EXCLUSIVE = re.compile(
    r"\b(?:before|prior\s+to|earlier\s+than)\s+(19[89]\d|20[0-3]\d)\b", re.IGNORECASE
)


def extract_year_range_from_query(query: str) -> YearRange | None:
    """Read the *direction* of a year expression, not only the years.

    Grounding already required each year to appear literally in the question,
    but which end of the range each belonged to was left to the model -- and
    ``YearRange(start=2024, end=2020)`` validates happily while guaranteeing
    zero results. Reading the direction here removes that failure mode.

    Returns ``None`` when the question names no year expression this
    understands, in which case the model's own range is used as before.
    """
    span = _YEAR_SPAN.search(query)
    if span:
        # A bare span states no direction, so ordering it is reading it, not
        # rewriting it: "2020-2015" can only have meant 2015 to 2020.
        low, high = sorted((int(span.group(1)), int(span.group(2))))
        return YearRange(start=low, end=high)

    start = end = None
    if match := _YEAR_START_INCLUSIVE.search(query):
        start = int(match.group(1))
    elif match := _YEAR_START_EXCLUSIVE.search(query):
        start = int(match.group(1)) + 1
    if match := _YEAR_END_INCLUSIVE.search(query):
        end = int(match.group(1))
    elif match := _YEAR_END_EXCLUSIVE.search(query):
        end = int(match.group(1)) - 1

    if start is None and end is None:
        return None
    # A contradictory pair ("after 2020 and before 2015") is left as written.
    # Swapping it would rewrite the question into one that returns data, which
    # is a different question; the honest answer is that nothing matches.
    return YearRange(start=start, end=end)


#: Words that flip a status mention into an exclusion. Checked only against the
#: text immediately before a match, and only after longer phrases have been
#: masked -- so the "not" belonging to "not yet recruiting" or "active, not
#: recruiting" is already consumed and can never reach this check.
_NEGATION_CUES = (
    "not",
    "non",
    "n't",
    "no longer",
    "never",
    "except",
    "excluding",
    "other than",
    "without",
    # "stopped recruiting" reads as an exclusion rather than a status: a
    # completed, terminated, or active-not-recruiting trial has all stopped
    # recruiting, so claiming any one of them would over-read the question.
    # "stopped early" is a TERMINATED phrase and is matched and masked before
    # this check ever sees it.
    "stopped",
    "ceased",
    "halted",
    # A trial that paused, suspended, or withdrew its recruitment is not
    # recruiting. Read positively these produced the near-opposite of the
    # question: "paused recruiting trials" asked upstream for
    # RECRUITING *or* SUSPENDED.
    "paused",
    "suspended",
    "withdrawn",
)

#: How far back to look when masking a consumed cue. Comfortably covers the
#: word-based lookback without reaching into an earlier clause.
_CUE_MASK_WINDOW = 40

#: Scanning back stops here, so a negation in one clause cannot reach into the
#: next: in "not completed and recruiting", the "and" protects "recruiting".
_CLAUSE_BOUNDARIES = frozenset({"and", "or", "but", "plus", "also"})

#: How many words back a cue may sit, which covers an intervening adverb
#: ("not currently recruiting") without reaching across a whole clause.
_NEGATION_LOOKBACK = 3


def _negation_cue(text_before: str) -> str | None:
    """The cue negating a status match, or None.

    Returns the cue itself rather than a bool so the caller can mask it. A verb
    like "halted" or "suspended" is both a cue and a status phrase in its own
    right, so leaving it in the haystack made "halted recruiting" mean
    TERMINATED *and* not-RECRUITING at once -- while "stopped recruiting",
    whose verb is not a status phrase, meant only the exclusion. Masking the
    consumed cue makes every "<verb> recruiting" form behave the same way.

    Walks back a few words rather than requiring the cue to be adjacent, so
    "not currently recruiting" is caught, and stops at a clause boundary so
    "not completed and recruiting" leaves the recruiting request intact.
    """
    words = text_before.replace(",", " , ").replace(";", " ; ").split()
    tail: list[str] = []
    for word in reversed(words[-_NEGATION_LOOKBACK:]):
        if word in _CLAUSE_BOUNDARIES or word in {",", ";"}:
            break
        tail.insert(0, word)
    if not tail:
        return None
    # "non-recruiting" and "aren't recruiting" attach the cue to the match or
    # to the preceding word, so the joined tail is checked as well as each
    # token -- that also covers the two-word cues ("no longer", "other than").
    joined = " ".join(tail).rstrip("-")
    for cue in _NEGATION_CUES:
        if joined.endswith(cue):
            return cue
    for word in tail:
        if word.rstrip("-") in _NEGATION_CUES:
            return word.rstrip("-")
    return None


#: Vocabulary phrases as whole-word patterns, longest first. Every phrase both
#: starts and ends with a word character (including "active, not recruiting"
#: and "not-yet-recruiting"), so \b is well defined at both ends and the
#: embedded comma and hyphens are unaffected. Compiled once at import.
_STATUS_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = tuple(
    (re.compile(rf"\b{re.escape(phrase)}\b"), status)
    for phrase, status in sorted(
        ((phrase, status) for status, ps in STATUS_VOCABULARY.items() for phrase in ps),
        key=lambda pair: -len(pair[0]),
    )
)


def _match_statuses(query: str) -> tuple[list[str], list[str]]:
    """Statuses the question asks *for* and statuses it asks to *exclude*.

    Matching is on whole words: a raw substring search read "unavailable" as
    AVAILABLE and "incomplete" as COMPLETED, inventing a filter the question
    never asked for.

    Longer phrases are matched and masked out first, so "not yet recruiting"
    cannot also register as "recruiting". A match whose preceding text carries
    a negation cue is recorded as excluded rather than requested -- without
    that check, "trials that are not recruiting" matched the bare phrase and
    filtered to exactly the opposite set.
    """
    haystack = query.lower()
    found: list[str] = []
    negated: list[str] = []
    for pattern, status in _STATUS_PATTERNS:
        # Every occurrence, not just the first: in "recruiting and not
        # recruiting" the second mention is the one carrying the negation.
        match = pattern.search(haystack)
        while match:
            index = match.start()
            phrase = match.group()
            cue = _negation_cue(haystack[:index])
            bucket = negated if cue else found
            if status not in bucket:
                bucket.append(status)
            # Masked with spaces rather than deleted, so indices stay stable
            # for the negation lookback on later phrases.
            haystack = haystack[:index] + " " * len(phrase) + haystack[index + len(phrase) :]
            if cue:
                # Consume the cue too. Otherwise a verb that is also a status
                # phrase ("halted", "suspended") would be counted a second time
                # as a positive status it never meant.
                cue_at = haystack.rfind(cue, max(0, index - _CUE_MASK_WINDOW), index)
                if cue_at >= 0:
                    haystack = (
                        haystack[:cue_at] + " " * len(cue) + haystack[cue_at + len(cue) :]
                    )
            match = pattern.search(haystack, index + len(phrase))
    # A status asked for and excluded in the same question is contradictory;
    # excluding wins, because acting on the positive reading is the failure
    # mode this exists to prevent.
    found = [s for s in found if s not in negated]
    return found, negated


def extract_statuses_from_query(query: str) -> list[str]:
    """Trial statuses the user actually named, as API enum values."""
    return _match_statuses(query)[0]


def extract_excluded_statuses_from_query(query: str) -> list[str]:
    """Statuses the question asked to exclude ("trials that are not recruiting").

    Kept separate from :func:`ground_entities` so that function's signature --
    and the ``ExtractedEntities`` shape the LLM also fills -- stay unchanged.
    """
    return _match_statuses(query)[1]


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
    grounded_statuses, negated_statuses = _match_statuses(query)
    for status in entities.statuses:
        normalized = status.strip().upper().replace(" ", "_")
        if normalized in negated_statuses:
            # The question does name this status -- negatively. The exclusion
            # warning below explains it; "does not support it" would be wrong.
            continue
        if normalized and normalized not in grounded_statuses:
            warnings.append(
                f"Ignored status {status!r}: the question does not support it."
            )
    if negated_statuses:
        # There is no upstream parameter for "every status except X", so the
        # exclusion is enforced client-side as overallStatus not in {...}. A
        # negative predicate never enumerates, which is what makes it safe: a
        # status outside our vocabulary (APPROVED_FOR_MARKETING appears in live
        # data) is correctly kept rather than silently dropped.
        excluded = ", ".join(s.replace("_", " ").lower() for s in negated_statuses)
        warnings.append(
            f"Read the question as excluding {excluded} trials, applied after "
            f"fetching; ClinicalTrials.gov has no status-exclusion parameter."
        )

    # Year bounds: the value must be written in the question, and so must its
    # direction. A range the question states explicitly ("from 2015 to 2020",
    # "since 2015", "before 2020") is read here and wins outright -- the model
    # used to own that reading, and a reversed range validates happily while
    # guaranteeing zero results.
    years_in_query = extract_years_from_query(query)
    grounded_range = extract_year_range_from_query(query)
    if (
        grounded_range is not None
        and grounded_range.start is not None
        and grounded_range.end is not None
        and grounded_range.start > grounded_range.end
    ):
        # Kept as written. Swapping the bounds would turn an unanswerable
        # question into an answerable one and chart the result as if it were
        # what was asked.
        warnings.append(
            f"The question asks for trials starting after {grounded_range.start - 1} "
            f"and before {grounded_range.end + 1}, which no trial can satisfy. "
            f"The bounds were left as written rather than reordered, so no trials "
            f"match."
        )
    if grounded_range is not None and entities.year_range is not None:
        # The question stated the range outright, so it wins -- but say so when
        # the model proposed a different one, rather than dropping it silently.
        for label, proposed in (
            ("start year", entities.year_range.start),
            ("end year", entities.year_range.end),
        ):
            if proposed is not None and proposed not in years_in_query:
                warnings.append(
                    f"Ignored {label} {proposed}: it does not appear in the question."
                )
    if grounded_range is None and entities.year_range is not None:
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
        if start is not None and end is not None and start > end:
            warnings.append(
                f"Read the range as {end}-{start}: the bounds arrived reversed, "
                f"which would have matched no trials."
            )
            start, end = end, start
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
    """Comparison series must be named in the question too, and be distinct.

    These drive one upstream search each, so an invented entity would not just
    mislabel a series -- it would fetch and chart trials nobody asked about.

    Equivalent entities are collapsed here rather than in the fetch layer,
    which keys series membership and provenance by label: two entities that
    normalise to one label made the second search silently overwrite the
    first's ids *and* its filters, losing a series outright. Fixing it here
    means one series is one thing, instead of inventing a disambiguated label
    for what is really a duplicate.
    """
    kept, warnings = [], []
    seen: dict[str, str] = {}
    for entity in compare_entities:
        if not _appears_in(entity, query):
            warnings.append(
                f"Ignored comparison entity {entity!r}: it does not appear in "
                f"the question."
            )
            continue
        stripped = entity.strip()
        key = series_key(stripped)
        if key in seen:
            # First occurrence wins and keeps its original casing, so series
            # order and labels stay deterministic.
            warnings.append(
                f"Ignored duplicate comparison entity {stripped!r}: it names the "
                f"same series as {seen[key]!r}."
            )
            continue
        seen[key] = stripped
        kept.append(stripped)
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


#: Words that name an analytical axis outright. Used to tell "the question
#: asked for this axis" from "the model happened to fill the field" -- those
#: need opposite treatment when the model also guesses "histogram".
AXIS_EVIDENCE: dict[str, tuple[str, ...]] = {
    "phase": ("phase", "phases"),
    "sponsor": ("sponsor", "sponsors"),
    "country": ("country", "countries", "geography", "geographic"),
    "status": ("status", "statuses"),
    "intervention_type": ("intervention type", "intervention types"),
    "start_year": ("year", "yearly", "annually", "over time", "trend"),
    "enrollment_bucket": (
        "enrollment",
        "enrolment",
        "enrollment size",
        "sample size",
        "participants",
    ),
}

_AXIS_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = tuple(
    (re.compile(rf"\b{re.escape(phrase)}\b"), axis)
    for axis, phrases in AXIS_EVIDENCE.items()
    for phrase in phrases
)


def axis_named_in_query(query: str) -> str | None:
    """The analytical axis the question names, or None if it names none or several.

    Deliberately narrow: exact words on word boundaries, no inference. It exists
    only to answer "did the user actually say 'by phase'", which is a different
    question from "did the model populate group_by" -- and the two need opposite
    handling when the model also proposes a histogram.
    """
    haystack = query.lower()
    axes = {axis for pattern, axis in _AXIS_PATTERNS if pattern.search(haystack)}
    return axes.pop() if len(axes) == 1 else None


#: Combinations that are incoherent no matter what the model proposed, and the
#: axis each one must resolve to. A geographic chart of sponsors is not a
#: geographic chart; a histogram of phases is not a histogram. Deliberately a
#: small explicit table rather than a rules engine.
REQUIRED_GROUP_BY: dict[str, str] = {
    "geographic": "country",
    "time_trend": "start_year",
}

#: A histogram bins a continuous measure; enrollment size is the only one this
#: service has. Any other axis means the chart type was the mistake.
HISTOGRAM_GROUP_BY = "enrollment_bucket"

#: Phrases that settle which network the question asks for. Literal validation
#: proves the model picked a *valid* enum, not the *right* one -- the same gap
#: reconciliation already closes for query_type, group_by, and comparison kind.
#: Matched on word boundaries so "sponsor" cannot fire inside "sponsored".
NETWORK_KIND_EVIDENCE: dict[str, tuple[str, ...]] = {
    "sponsor_drug": (
        "sponsors and drugs",
        "sponsor and drug",
        "drugs and sponsors",
        "sponsor-to-drug",
        "sponsor to drug",
        "sponsor-drug",
        "sponsor drug",
        "sponsor network",
        "which sponsors",
        "between sponsors and drugs",
    ),
    "drug_drug": (
        "co-occur",
        "cooccur",
        "co-occurrence",
        "cooccurrence",
        "drug co-occurrence",
        "combination",
        "combination therapy",
        "combined with",
        "studied together",
        "used together",
        "drug-drug",
        "drug drug",
    ),
}

_NETWORK_KIND_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = tuple(
    (re.compile(rf"\b{re.escape(phrase)}"), kind)
    for kind, phrases in NETWORK_KIND_EVIDENCE.items()
    for phrase in phrases
)


def infer_network_kind(query: str) -> str | None:
    """The network the question's own words ask for, or None.

    Returns None when the evidence is absent *or* conflicting ("a network of
    sponsors and drugs studied together" says both), because a coin-flip
    between two valid readings is worse than keeping the model's proposal --
    which at least came from reading the whole sentence.
    """
    haystack = query.lower()
    kinds = {kind for pattern, kind in _NETWORK_KIND_PATTERNS if pattern.search(haystack)}
    return kinds.pop() if len(kinds) == 1 else None


#: Which entity list a comparison of each kind must have been drawn from.
COMPARE_KIND_SOURCES: dict[str, str] = {
    "drug": "drugs",
    "condition": "conditions",
    "sponsor": "sponsors",
}


def _grounds_every_entity(
    compare_entities: list[str], entities: ExtractedEntities, kind: str
) -> bool:
    """True when *every* compared name was grounded into ``kind``'s list.

    Full membership, not intersection. Any-overlap let one recognised name
    carry the rest: "Pembrolizumab vs melanoma" matched the drug list on
    Pembrolizumab alone and then searched ``query.intr=melanoma`` -- a
    head-to-head whose second series measured something nobody asked about.
    """
    field_name = COMPARE_KIND_SOURCES.get(kind)
    if not field_name or not compare_entities:
        return False
    grounded = {series_key(v) for v in getattr(entities, field_name, [])}
    return {series_key(v) for v in compare_entities} <= grounded


def _contradicts_kind(
    compare_entities: list[str], entities: ExtractedEntities, kind: str
) -> bool:
    """True when some compared name was grounded as a *different* kind.

    This is the real hazard the membership rule was reaching for: a name the
    model called a drug that the same extraction put in ``conditions`` would
    search ``query.intr`` for a disease. Absence from every list is not that --
    it is merely silence, and silence must not be read as contradiction.
    """
    if not compare_entities:
        return False
    claimed = COMPARE_KIND_SOURCES.get(kind)
    for entity in compare_entities:
        key = series_key(entity)
        if claimed and key in {series_key(v) for v in getattr(entities, claimed, [])}:
            continue
        for other_kind, field_name in COMPARE_KIND_SOURCES.items():
            if other_kind == kind:
                continue
            if key in {series_key(v) for v in getattr(entities, field_name, [])}:
                return True
    return False


def _infer_compare_kind(
    compare_entities: list[str], entities: ExtractedEntities
) -> str | None:
    """The one kind whose grounded list contains every compared name.

    Deliberately evidence-based rather than a guess: with no list containing
    them all there is nothing to infer from, and inventing a kind would send
    the names to the wrong ClinicalTrials.gov field.
    """
    return next(
        (
            kind
            for kind in COMPARE_KIND_SOURCES
            if _grounds_every_entity(compare_entities, entities, kind)
        ),
        None,
    )


def reconcile_plan_semantics(
    *,
    query_type: str,
    group_by: str | None,
    axis_named: str | None = None,
    axis_was_specified: bool = True,
    viz_type: str,
    network_kind: str | None,
    compare_entities: list[str],
    compare_entity_kind: str | None,
    entities: ExtractedEntities,
) -> tuple[str | None, str, str | None, str | None, list[str]]:
    """Force a plan's parts to describe the same question.

    ``reconcile_viz_type`` already normalises the chart type against the query
    type. This is the same idea one level down: the model may propose semantic
    combinations that cannot be rendered honestly, and deterministic code has
    to reject or normalise them before they reach a search or an axis.

    Returns the corrected ``(group_by, viz_type, network_kind,
    compare_entity_kind)`` plus one assumption per correction, so every change
    is visible to the reader rather than applied silently.
    """
    assumptions: list[str] = []

    required = REQUIRED_GROUP_BY.get(query_type)
    if required and group_by != required:
        if group_by is not None:
            assumptions.append(
                f"Grouped by {required.replace('_', ' ')} rather than "
                f"{group_by.replace('_', ' ')}, which a "
                f"{query_type.replace('_', ' ')} question cannot be plotted against."
            )
        group_by = required

    # A histogram is a statement about the axis, not just the rendering, so an
    # incompatible pair is resolved by trusting whichever side the question
    # actually supports: an explicit non-enrollment axis wins over the chart.
    if viz_type == "histogram" and group_by != HISTOGRAM_GROUP_BY:
        if axis_named == HISTOGRAM_GROUP_BY:
            # The question asked for enrolment and the model put something else
            # on the axis. What the user said wins.
            group_by = HISTOGRAM_GROUP_BY
            assumptions.append(
                "Grouped by enrollment size, which is the axis the question "
                "names and what a histogram measures."
            )
        elif axis_named and axis_named != HISTOGRAM_GROUP_BY:
            # The question named a categorical axis outright, so it wins over a
            # chart type the model guessed -- even when the model left group_by
            # empty. Testing only "did the model fill the field" turned
            # "distribution ... by phase" into an enrolment histogram.
            group_by = axis_named
            viz_type = "bar_chart"
            assumptions.append(
                f"Grouped by {axis_named.replace('_', ' ')} and rendered as a bar "
                f"chart: the question names that axis, and a histogram measures a "
                f"continuous quantity."
            )
        elif not axis_was_specified and query_type == "distribution":
            # No axis was named, so the chart type is the only signal there is.
            group_by = HISTOGRAM_GROUP_BY
            assumptions.append(
                "Read this as an enrollment-size distribution, which is what a "
                "histogram measures."
            )
        else:
            # An axis the question named explicitly outranks a chart type the
            # model guessed. Rewriting group_by here answered a different
            # question: "distributed by phase" became an enrollment histogram
            # purely because the model said "histogram".
            viz_type = "bar_chart"
            assumptions.append(
                f"Rendered as a bar chart rather than a histogram: "
                f"{(group_by or 'this axis').replace('_', ' ')} is categorical, "
                f"not a continuous measure."
            )

    # The compared entities came from grounding, so the list they appear in is
    # better evidence of their kind than the model's label. Querying
    # query.cond for drug names would return nothing and say nothing about it.
    #
    # A *missing* kind matters as much as a wrong one: build_searches requires
    # both, so two entities with kind=None built a single OR search while the
    # chart stayed a grouped bar -- a union presented as a comparison.
    if compare_entities and not _grounds_every_entity(
        compare_entities, entities, compare_entity_kind or ""
    ):
        # The claimed kind is not confirmed by entity-list membership. Three
        # different situations hide behind that, and they need different
        # answers -- treating all three as "ungrounded" is what wiped correctly
        # named comparisons and fell back to a whole-sentence query.term search.
        actual = _infer_compare_kind(compare_entities, entities)
        if actual:
            # Another list holds *all* of them: better evidence than the label.
            assumptions.append(
                f"Compared the entities as {actual}s"
                + (
                    f" rather than {compare_entity_kind}s"
                    if compare_entity_kind
                    else ""
                )
                + ", matching where they were found in the question."
            )
            compare_entity_kind = actual
        elif compare_entity_kind and not _contradicts_kind(
            compare_entities, entities, compare_entity_kind
        ):
            # No list contains them all, but no list contradicts the kind
            # either. They already passed ground_compare_entities, so every one
            # is named in the question -- which is the property that keeps an
            # invented entity out of an upstream field. Entity-list membership
            # was only ever a proxy for that, and demanding it as well threw
            # away comparisons the user plainly asked for.
            pass
        else:
            if compare_entity_kind:
                assumptions.append(
                    f"Did not compare these as {compare_entity_kind}s: the "
                    f"question grounds some compared value as something else, "
                    f"and searching the wrong field would answer a different "
                    f"question."
                )
            compare_entity_kind = None

    return group_by, viz_type, network_kind, compare_entity_kind, assumptions


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

    # Kept before defaulting, so reconciliation can tell "the question named no
    # axis" from "the question named phase" -- they need opposite histogram
    # treatment, and the default erases the difference.
    axis_was_specified = understanding.group_by is not None
    # What the *question* names, which is not the same as what the model filled
    # in -- the histogram rule needs to tell those apart.
    axis_named = axis_named_in_query(request.query)
    group_by = understanding.group_by
    if understanding.query_type != "relationship":
        group_by = group_by or DEFAULT_GROUP_BY.get(understanding.query_type, "phase")
        if understanding.query_type == "time_trend":
            group_by = "start_year"

    network_kind = understanding.network_kind
    if understanding.query_type == "relationship":
        # The question's own words outrank the model's pick, exactly as they do
        # for the chart axis and the comparison kind. A valid enum is not a
        # correct one: "which drugs co-occur" answered with sponsor_drug draws
        # a different graph than the one asked for.
        evidenced = infer_network_kind(request.query)
        if evidenced and network_kind != evidenced:
            if network_kind is not None:
                assumptions.append(
                    f"Built a {evidenced.replace('_', '-')} network rather than "
                    f"{network_kind.replace('_', '-')}, matching how the question "
                    f"describes the relationship."
                )
            network_kind = evidenced
        if network_kind is None:
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
        #
        # The survivor has to be promoted into the entity lists first. Clearing
        # compare state without it dropped the entity from the population
        # entirely -- "compare Pembrolizumab and <hallucinated>" searched the
        # condition alone, and pembrolizumab vanished from a question that
        # named it.
        for survivor in compare_entities:
            # Membership first, the claimed kind only as a fallback, and never
            # a default. Trusting the label put a sponsor into `drugs`, so the
            # search ANDed query.intr=Pfizer with query.spons=Pfizer and
            # returned nothing, silently.
            kind = _infer_compare_kind([survivor], entities) or compare_kind
            field_name = COMPARE_KIND_SOURCES.get(kind or "")
            if field_name is None:
                warnings.append(
                    f"Dropped {survivor!r} from the search: the question does not "
                    f"say whether it is a drug, condition, or sponsor, and "
                    f"guessing would search the wrong field."
                )
                continue
            current = list(getattr(entities, field_name))
            if not any(series_key(v) == series_key(survivor) for v in current):
                setattr(entities, field_name, [*current, survivor])
        viz_type = "bar_chart"
        compare_entities, compare_kind = [], None
        assumptions.append(
            "Fewer than two comparable entities were named, so a single "
            "distribution is shown instead of a comparison."
        )

    # Last, so it sees the grounded entities and every override already applied.
    group_by, viz_type, network_kind, compare_kind, semantic_notes = (
        reconcile_plan_semantics(
            query_type=understanding.query_type,
            group_by=group_by,
            axis_named=axis_named,
            axis_was_specified=axis_was_specified,
            viz_type=viz_type,
            network_kind=network_kind,
            compare_entities=compare_entities,
            compare_entity_kind=compare_kind,
            entities=entities,
        )
    )
    assumptions.extend(semantic_notes)

    # A grouped bar chart *is* a claim that the data has series, and series come
    # only from labelled per-entity searches. Without a resolvable kind there is
    # nothing to label them with, and the chart would ship one unlabelled series
    # built from a union -- a comparison in appearance only. Guessing the field
    # instead would send the names to the wrong ClinicalTrials.gov parameter.
    if viz_type == "grouped_bar_chart" and not (
        len(compare_entities) >= 2 and compare_kind
    ):
        viz_type = "bar_chart"
        compare_entities, compare_kind = [], None
        if not any("single distribution" in a for a in assumptions):
            assumptions.append(
                "Could not tell whether the compared values are drugs, "
                "conditions, or sponsors, so a single distribution is shown "
                "rather than a comparison of the wrong field."
            )

    return QueryPlan(
        query=request.query,
        # What actually shipped. A demoted comparison that still called itself
        # one made meta.query_interpretation describe a chart with series the
        # response does not contain.
        query_type=(
            "distribution"
            if understanding.query_type == "comparison" and not compare_entities
            else understanding.query_type
        ),
        entities=entities,
        excluded_statuses=extract_excluded_statuses_from_query(request.query),
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


async def understand(request: QueryRequest, *, model: str | None = None) -> QueryPlan:
    """Full stage 1: LLM call plus deterministic grounding and reconciliation.

    Only the LLM call awaits; grounding and reconciliation stay synchronous
    because they are pure CPU work over data already in hand.
    """
    understanding = await call_llm(request.query, model=model)
    return build_plan(request, understanding)
