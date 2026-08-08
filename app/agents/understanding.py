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
    try:
        completion = _client().chat.completions.parse(
            model=model or settings.LLM_MODEL,
            response_format=QueryUnderstanding,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": query},
            ],
        )
    except OpenAIError as exc:
        raise UnderstandingError(f"LLM request failed: {exc}") from exc

    message = completion.choices[0].message
    if message.refusal:
        raise UnderstandingError(f"Model declined the request: {message.refusal}")
    if message.parsed is None:
        raise UnderstandingError("Model returned no parsable structured output.")
    return message.parsed


# --------------------------------------------------------------------------
# Deterministic post-processing
# --------------------------------------------------------------------------

_WORD_RE = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


def _appears_in(candidate: str, haystack: str) -> bool:
    """True when ``candidate`` is plausibly present in ``haystack``.

    Exact substring is the common case. The fuzzy fallback exists only to
    tolerate inflection and spacing ("lung cancers" for "lung cancer"), not to
    admit entities the user never wrote -- the 0.87 threshold is tight enough
    that unrelated drug names do not pass.
    """
    candidate = candidate.strip().lower()
    if not candidate:
        return False
    haystack = haystack.lower()
    if candidate in haystack:
        return True

    candidate_tokens = _tokens(candidate)
    haystack_tokens = _tokens(haystack)
    if not candidate_tokens:
        return False

    window = len(candidate_tokens)
    for i in range(len(haystack_tokens) - window + 1):
        chunk = " ".join(haystack_tokens[i : i + window])
        if SequenceMatcher(None, " ".join(candidate_tokens), chunk).ratio() >= 0.87:
            return True
    return False


def ground_entities(
    understanding: QueryUnderstanding, query: str
) -> tuple[ExtractedEntities, list[str]]:
    """Drop extracted entities that do not appear in the user's question.

    This is the guard against the one thing the LLM could still get wrong in a
    way that reaches real data: searching for something nobody asked about.
    Returns the filtered entities and a warning per dropped item.
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

    grounded = ExtractedEntities(
        drugs=keep(entities.drugs, "drug"),
        conditions=keep(entities.conditions, "condition"),
        sponsors=keep(entities.sponsors, "sponsor"),
        # Phases are a closed set of integers, so range-checking is the whole
        # validation; there is nothing to hallucinate beyond an invalid number.
        phases=[p for p in entities.phases if p in (1, 2, 3, 4)],
        statuses=[s.strip() for s in entities.statuses if s.strip()],
        countries=keep(entities.countries, "country"),
        year_range=entities.year_range,
    )
    return grounded, warnings


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

    compare_entities = understanding.compare_entities
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
    logger.info(
        "understood query_type=%s viz_type=%s group_by=%s",
        understanding.query_type,
        understanding.viz_type,
        understanding.group_by,
    )
    return build_plan(request, understanding)
