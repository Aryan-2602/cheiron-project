"""The orchestrator: request in, validated visualization spec out.

Stage order is fixed and one-directional:

    understand -> build searches -> fetch -> aggregate -> format -> validate

Each arrow is a typed hand-off, and only the fetch and aggregate stages can
introduce a data value. That is the whole hallucination argument in one file:
the LLM's output reaches the network as *search parameters* and reaches the
chart as *labels*, never as numbers.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Literal

from app.agents.understanding import UnderstandingError, understand
from app.core.config import settings
from app.models.schemas import (
    AggregationResult,
    Meta,
    NetworkResult,
    QueryPlan,
    QueryRequest,
    QueryResponse,
    VisualizationSpec,
)
from app.services.aggregate import aggregate, zero_fill_years
from app.services.ctgov import (
    STATUS_FILTER_CODES,
    CTGovClient,
    CTGovError,
    CTGovSearch,
    build_searches,
    normalize_statuses,
)
from app.services.dimensions import extract_start_year, get_dimension
from app.services.drug_resolver import resolve_all
from app.services.network import (
    build_bipartite_network,
    build_cooccurrence_network,
    rank_candidate_names,
)
from app.services.store import StudyStore
from app.services.validate import ValidationFailure, validate_response
from app.services.viz import build_chart_spec, build_network_spec

logger = logging.getLogger(__name__)

#: Dimensions with unbounded cardinality get a top-N cap so the chart stays
#: readable; the cap is always disclosed in meta.warnings.
TOP_N_DIMENSIONS = {"sponsor": 15, "country": 20, "intervention_type": 15}


class EmptyResultError(RuntimeError):
    """There is nothing to chart. Carries the diagnostics explaining why.

    Two distinct outcomes, both correct answers rather than faults:
    ``NO_MATCHING_TRIALS`` (the search found nothing) and
    ``NO_CHARTABLE_DATA`` (trials matched, but the requested analysis produced
    no renderable rows -- no usable start dates, no surviving network edges).
    The second used to reach the validator, which rejected the empty spec and
    turned a legitimate analytical answer into an HTTP 500.
    """

    def __init__(
        self,
        message: str,
        meta: Meta,
        reason: Literal["NO_MATCHING_TRIALS", "NO_CHARTABLE_DATA"] = "NO_MATCHING_TRIALS",
    ) -> None:
        super().__init__(message)
        meta.empty_reason = reason
        self.meta = meta
        self.reason = reason


class UnsupportedQueryError(RuntimeError):
    """The question cannot be answered from trial registry data."""


@dataclass
class FetchResult:
    store: StudyStore = field(default_factory=StudyStore)
    series_membership: dict[str, set[str]] | None = None
    warnings: list[str] = field(default_factory=list)
    filters: dict[str, Any] = field(default_factory=dict)


async def fetch(
    plan: QueryPlan, searches: list[CTGovSearch], client: CTGovClient
) -> FetchResult:
    """Run every search, collecting records and per-series membership.

    Series membership is exactly "the trials this search returned", so a
    grouped comparison never has to infer which series a trial belongs to.
    """
    result = FetchResult()
    membership: dict[str, set[str]] = {}
    per_search_filters: dict[str, dict[str, Any]] = {}
    remaining_searches = len(searches)

    for search in searches:
        # Each search gets a fair share of the budget that is still unspent, so
        # one series cannot consume the whole allowance *and* the total can
        # never exceed max_studies. A previous fixed floor of 100 per search
        # broke the second guarantee: two searches under a cap of 100 fetched
        # 200. Unused allowance flows to later searches.
        budget = max(0, (plan.max_studies - len(result.store)) // remaining_searches)
        remaining_searches -= 1

        if budget == 0:
            # Disclose rather than quietly return an empty series.
            result.warnings.append(
                f"Reached the {plan.max_studies:,}-trial fetch cap before searching "
                f"{search.label or 'every filter'}; that part of the question is "
                f"not represented below. Raise max_studies to include it."
            )
            continue

        outcome = await client.run_search(search, result.store, max_studies=budget)
        result.warnings.extend(outcome.warnings)
        if search.label:
            membership[search.label] = outcome.nct_ids
            per_search_filters[search.label] = search.describe()
        else:
            result.filters.update(search.describe())

    if len(searches) > 1 and membership:
        result.series_membership = membership
        # Keyed by series label, mirroring series_membership. Flattening every
        # search into one dict let the last one overwrite the rest, so a
        # comparison response reported only its final entity's filters and
        # positively misattributed the others.
        result.filters = {**result.filters, **per_search_filters}
    elif per_search_filters:
        # A single labelled search keeps the flat shape callers already parse.
        result.filters.update(next(iter(per_search_filters.values())))

    return result


def apply_client_side_filters(store: StudyStore, plan: QueryPlan) -> list[str]:
    """Apply filters that cannot be trusted to the upstream API.

    What lands here is what the upstream API cannot express safely -- sending
    an unverified filter risks a silently-empty result rather than a loud
    failure:

    * **Statuses with no verified ``aggFilters`` code.** Nine statuses map to a
      live-verified code and are unioned upstream; AVAILABLE and
      NO_LONGER_AVAILABLE return HTTP 200 with zero results upstream, so they
      are matched here instead.
    * **Year ranges.** There is no verified date-range parameter, so "since
      2015" is enforced against the start date already present in each record.

    Phases are no longer among them: values within one ``aggFilters`` key are
    space-separated and union, so every requested phase goes upstream.

    Filtering locally is exact and keeps citations intact, because the records
    being filtered are the same ones the citations point at.
    """
    warnings: list[str] = []

    # Phases and verified statuses are filtered upstream in one aggFilters
    # clause each, so nothing is left to do here for them. Only statuses with
    # no live-verified code reach the client-side union below.
    wanted_status = normalize_statuses(plan.entities.statuses)
    if wanted_status <= set(STATUS_FILTER_CODES):
        wanted_status = set()
    if wanted_status:
        before = len(store.records)
        store.records = {
            nct_id: record
            for nct_id, record in store.records.items()
            if (
                record.get("protocolSection", {})
                .get("statusModule", {})
                .get("overallStatus")
            )
            in wanted_status
        }
        # sorted() so the disclosure text is stable rather than inheriting set
        # iteration order.
        listed = ", ".join(sorted(wanted_status))
        reason = (
            "only one status can be filtered upstream, so a multi-status "
            "request is narrowed after fetching"
            if len(wanted_status) > 1
            else "the upstream filter code for this status is not live-verified"
        )
        warnings.append(
            f"Kept trials whose status is any of {listed}, filtered client-side "
            f"({len(store.records):,} of {before:,} fetched trials); {reason}."
        )

    excluded = normalize_statuses(plan.excluded_statuses)
    if excluded:
        before = len(store.records)
        store.records = {
            nct_id: record
            for nct_id, record in store.records.items()
            # A negative predicate, never an enumeration of "everything else":
            # a status outside our vocabulary is kept, which is what the
            # question asked for.
            if (
                record.get("protocolSection", {})
                .get("statusModule", {})
                .get("overallStatus")
            )
            not in excluded
        }
        warnings.append(
            f"Excluded trials whose status is {', '.join(sorted(excluded))}, "
            f"filtered client-side ({len(store.records):,} of {before:,} fetched "
            f"trials); ClinicalTrials.gov has no status-exclusion parameter."
        )

    years = plan.entities.year_range
    if years and (years.start is not None or years.end is not None):
        low = years.start if years.start is not None else -9999
        high = years.end if years.end is not None else 9999
        before = len(store.records)
        kept: dict[str, Any] = {}
        undated = 0
        for nct_id, record in store.records.items():
            parsed = extract_start_year(record)
            if not parsed:
                undated += 1
                continue
            if low <= int(parsed[0]) <= high:
                kept[nct_id] = record
        store.records = kept
        bound = (
            f"{years.start}-{years.end}"
            if years.start is not None and years.end is not None
            else (f"from {years.start}" if years.start is not None else f"to {years.end}")
        )
        warnings.append(
            f"Restricted to trials starting {bound} client-side "
            f"({len(kept):,} of {before:,} fetched trials"
            + (f", {undated:,} excluded for having no start date" if undated else "")
            + "); ClinicalTrials.gov has no live-verified date-range parameter."
        )

    return warnings


def build_meta(
    plan: QueryPlan,
    fetched: FetchResult,
    *,
    data_as_of: str | None,
    aggregation: AggregationResult | None,
    network: NetworkResult | None,
    extra_warnings: list[str],
) -> Meta:
    warnings = list(plan.warnings) + list(fetched.warnings) + list(extra_warnings)

    # A question with predicates but no population scope has no upstream
    # parameter that narrows the registry -- there is no date filter, and a
    # phase or status clause still leaves hundreds of thousands of trials. The
    # fetch cap then takes the first max_studies in ClinicalTrials.gov's own
    # order, which is not a random sample. The cap itself is already disclosed;
    # what a reader cannot tell from that line alone is that the *shape* of the
    # chart inherits that ordering.
    entities = plan.entities
    has_scope = any(
        (entities.drugs, entities.conditions, entities.sponsors, entities.countries)
    )
    if not has_scope and fetched.store.truncated:
        warnings.append(
            "This question named no drug, condition, sponsor, or country, so the "
            "trials below are a capped slice of the whole registry in its default "
            "order rather than a random sample. Differences between the bars or "
            "years reflect that ordering as much as real activity -- add a "
            "condition or drug for a figure that can be relied on."
        )

    if aggregation is not None:
        if aggregation.multi_valued:
            warnings.append(
                "Trials can belong to more than one bucket on this axis (for "
                "example a combined Phase 1/2 trial), so bucket totals may sum "
                "to more than the number of trials."
            )
        if aggregation.unbucketed:
            warnings.append(
                f"{aggregation.unbucketed:,} of {aggregation.total_studies_matched:,} "
                f"trials had no value for this axis and are "
                + (
                    "shown as a separate bucket."
                    if aggregation.unbucketed_key_included
                    else "excluded from the chart."
                )
            )
        if aggregation.omitted_categories:
            # Without this, a 20-of-47 country chart reads as every country.
            axis = get_dimension(aggregation.dimension).axis_label.lower()
            warnings.append(
                f"Showing the top {aggregation.displayed_categories:,} of "
                f"{aggregation.total_categories:,} {axis} values by trial count; "
                f"{aggregation.omitted_categories:,} lower-frequency "
                f"{'value is' if aggregation.omitted_categories == 1 else 'values are'} "
                f"omitted."
            )
    if network is not None and network.truncated_to_top_n:
        # Say what the code actually ranks by. Pruning sorts on node size --
        # the number of trials a node appears in -- not on degree, and the
        # bipartite builder keeps the busiest sponsors first.
        how = (
            "retaining the busiest sponsors first, then the drugs those "
            "sponsors study"
            if network.kind == "sponsor_drug"
            else "keeping the nodes appearing in the most trials"
        )
        warnings.append(
            f"Graph truncated to {network.truncated_to_top_n} nodes for "
            f"readability, by {how}."
        )
    if network is not None and network.min_edge_weight > 1:
        warnings.append(
            f"Edges require at least {network.min_edge_weight} shared trials; "
            f"single-trial co-occurrences are omitted as incidental."
        )

    dimension_label = (
        get_dimension(aggregation.dimension).axis_label if aggregation else None
    )
    return Meta(
        filters=fetched.filters,
        data_as_of=data_as_of,
        total_studies_processed=len(fetched.store),
        api_urls=fetched.store.api_urls,
        query_interpretation=(
            f"Interpreted as a {plan.query_type.replace('_', ' ')} question"
            + (f", grouped by {dimension_label.lower()}" if dimension_label else "")
            + f", rendered as a {plan.viz_type.replace('_', ' ')}."
        ),
        assumptions=plan.assumptions,
        warnings=warnings,
        sorting=(
            get_dimension(aggregation.dimension).sorting
            if aggregation
            else "node size descending"
        ),
        time_granularity="year" if plan.viz_type == "time_series" else None,
    )


async def resolve_drug_names(
    plan: QueryPlan, store: StudyStore
) -> tuple[dict[str, Any], list[str]]:
    """Resolve the drug names that could plausibly reach the graph.

    Ordering matters and is deliberate: resolution happens **before** the graph
    is built, not after pruning. Merging changes node size and edge weight, and
    those are exactly what pruning ranks on -- resolving afterwards would let
    two fragments of one compound each miss the node cap that their merged form
    would clear, and would drop a low-weight edge before it could contribute to
    the merged edge it belongs to.

    What *is* deferred is the choice of which names to resolve: sizes are
    computed from string normalization alone (no API calls), the top candidates
    are taken from that ranking, and only those are sent to RxNorm.
    """
    if not settings.RXNORM_ENABLED:
        return {}, []
    candidates = rank_candidate_names(
        store.records, top_k=settings.RXNORM_CANDIDATE_POOL
    )
    if not candidates:
        return {}, []
    return await resolve_all(store.records, only=candidates)


async def analyze(
    plan: QueryPlan, fetched: FetchResult
) -> tuple[VisualizationSpec, AggregationResult | None, NetworkResult | None, list[str]]:
    """Stages 4 and 5: measure, then format. The only source of data values."""
    store = fetched.store

    if plan.query_type == "relationship":
        resolutions, warnings = await resolve_drug_names(plan, store)
        if plan.network_kind == "sponsor_drug":
            network = build_bipartite_network(store.records, resolutions=resolutions)
        else:
            network = build_cooccurrence_network(store.records, resolutions=resolutions)
        merged = sum(1 for n in network.nodes if n.merged_from)
        logger.info(
            "network built",
            extra={
                "kind": network.kind,
                "nodes": len(network.nodes),
                "edges": len(network.edges),
                "merged_nodes": merged,
                "truncated_to_top_n": network.truncated_to_top_n,
            },
        )
        if merged:
            warnings.append(
                f"Merged brand and generic names into {merged} shared compound "
                f"node(s) using RxNorm; each merged node lists its source names."
            )
        return build_network_spec(network, plan, store), None, network, warnings

    dimension = get_dimension(plan.group_by or "phase")
    is_time_series = plan.viz_type == "time_series"
    result = aggregate(
        store.records,
        dimension,
        series_membership=fetched.series_membership,
        top_n=TOP_N_DIMENSIONS.get(dimension.name),
        # A "no start date" category cannot be placed on a temporal axis, so
        # undated trials are reported in meta rather than charted. Every other
        # axis shows its unknown bucket, since hiding it would misstate the
        # distribution.
        include_unknown=not is_time_series,
    )
    if is_time_series:
        # The requested bounds, so "2020 through 2024" shows explicit zeros at
        # both ends rather than cropping the axis to whatever data exists.
        years = plan.entities.year_range
        result = zero_fill_years(
            result,
            start=years.start if years else None,
            end=years.end if years else None,
        )
    logger.info(
        "aggregation completed",
        extra={
            "dimension": result.dimension,
            "buckets": len(result.data),
            "total_studies_matched": result.total_studies_matched,
            "unbucketed": result.unbucketed,
            "series": result.series_dimension,
        },
    )
    # Chart queries never touch RxNorm -- resolution only affects graph identity.
    return build_chart_spec(result, plan, dimension, store), result, None, []


async def run_pipeline(
    request: QueryRequest,
    *,
    client: CTGovClient | None = None,
    plan: QueryPlan | None = None,
) -> QueryResponse:
    """Execute the full pipeline.

    Args:
        client: Injectable for tests; a fresh client is created when omitted.
        plan: Injectable to skip the LLM stage in tests.

    Raises:
        UnderstandingError: the LLM stage failed.
        UnsupportedQueryError: the question is not answerable from registry data.
        CTGovError: the upstream API failed.
        EmptyResultError: no trials matched.
        ValidationFailure: the response failed a grounding check.
    """
    plan = plan or understand(request)
    if plan.query_type == "unsupported":
        raise UnsupportedQueryError(
            "This question cannot be answered from ClinicalTrials.gov registry data."
        )

    searches, search_notes = build_searches(plan)
    # How multiple extracted values were combined is an interpretation the
    # reader should see, not a silent choice.
    plan.assumptions = [*plan.assumptions, *search_notes]
    owns_client = client is None
    client = client or CTGovClient()

    if owns_client:
        await client.__aenter__()
    try:
        data_as_of = await client.get_data_timestamp()
        fetched = await fetch(plan, searches, client)
    finally:
        if owns_client:
            await client.__aexit__()

    extra_warnings = apply_client_side_filters(fetched.store, plan)

    if not fetched.store.records:
        raise EmptyResultError(
            "No trials matched this query.",
            build_meta(
                plan,
                fetched,
                data_as_of=data_as_of,
                aggregation=None,
                network=None,
                extra_warnings=extra_warnings,
            ),
        )

    spec, aggregation, network, analysis_warnings = await analyze(plan, fetched)
    meta = build_meta(
        plan,
        fetched,
        data_as_of=data_as_of,
        aggregation=aggregation,
        network=network,
        extra_warnings=extra_warnings + analysis_warnings,
    )

    # Trials matched, but the requested analysis produced nothing renderable.
    # That is an answer, not a fault -- previously the empty spec reached the
    # validator and surfaced as an HTTP 500.
    unchartable = _describe_unchartable(spec, plan, fetched, aggregation, network)
    if unchartable:
        raise EmptyResultError(unchartable, meta, reason="NO_CHARTABLE_DATA")

    response = QueryResponse(visualization=spec, meta=meta)
    return validate_response(
        response,
        fetched.store,
        aggregation=aggregation,
        network=network,
        max_citations_per_datum=plan.max_citations_per_datum,
    )


def _describe_unchartable(
    spec: VisualizationSpec,
    plan: QueryPlan,
    fetched: FetchResult,
    aggregation: AggregationResult | None,
    network: NetworkResult | None,
) -> str | None:
    """Explain an empty chart built from a non-empty result set, or None.

    The count is stated because "no trials had a start date" and "no trials
    matched" are different answers, and the reader cannot tell them apart from
    an empty axis alone.
    """
    matched = len(fetched.store)
    if network is not None:
        if network.nodes and network.edges:
            return None
        # Isolated nodes are pruned, so nodes and edges empty together and the
        # result cannot say whether the entities were missing or merely never
        # paired. The wording is therefore true of both.
        if network.kind == "sponsor_drug":
            return (
                f"{matched:,} trials matched the filters, but none of them linked "
                f"a sponsor to a drug, so the graph has no edges to draw."
            )
        return (
            f"{matched:,} trials matched the filters, but no two drugs appeared "
            f"together in at least {network.min_edge_weight} of them, so the "
            f"graph has no edges to draw."
        )
    if spec.data:
        return None
    axis = get_dimension(aggregation.dimension).axis_label.lower() if aggregation else "axis"
    return (
        f"{matched:,} trials matched the filters, but none reported a usable "
        f"{axis} value, so there is nothing to chart."
    )


__all__ = [
    "CTGovError",
    "EmptyResultError",
    "UnderstandingError",
    "UnsupportedQueryError",
    "ValidationFailure",
    "run_pipeline",
]
