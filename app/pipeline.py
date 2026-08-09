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
from typing import Any

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
    CTGovClient,
    CTGovError,
    CTGovSearch,
    build_searches,
    normalize_statuses,
)
from app.services.dimensions import extract_phases, extract_start_year, get_dimension
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
    """No trials matched. Carries the diagnostics needed to explain why."""

    def __init__(self, message: str, meta: Meta) -> None:
        super().__init__(message)
        self.meta = meta


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

    Two cases end up here, both for the same reason -- the corresponding
    upstream parameter is either unverified or absent, and sending an unverified
    filter risks a silently-empty result rather than a loud failure:

    * **Status** other than ``recruiting``. Only ``status:rec`` was confirmed
      live; other abbreviations are guesses.
    * **Year ranges.** There is no verified date-range parameter, so "since
      2015" is enforced against the start date already present in each record.
    * **Multiple phases.** ``aggFilters`` carries only one phase and repeating
      the key does not OR them, so a request for phases 2 *and* 3 is fetched
      unfiltered and OR-ed here. Phases are multi-valued per trial, so a
      combined Phase 1/2 study legitimately matches a request for either.

    Filtering locally is exact and keeps citations intact, because the records
    being filtered are the same ones the citations point at.
    """
    warnings: list[str] = []

    wanted_phases = {p for p in plan.entities.phases if p in (1, 2, 3, 4)}
    if len(wanted_phases) > 1:
        wanted_enum = {f"PHASE{p}" for p in wanted_phases}
        before = len(store.records)
        store.records = {
            nct_id: record
            for nct_id, record in store.records.items()
            # Intersection, not equality: a Phase 1/2 trial matches either.
            if wanted_enum & set(extract_phases(record))
        }
        warnings.append(
            f"Filtered to phase {', '.join(str(p) for p in sorted(wanted_phases))} "
            f"client-side ({len(store.records):,} of {before:,} fetched trials); "
            f"ClinicalTrials.gov accepts only one phase per search, so a "
            f"multi-phase request is narrowed after fetching."
        )

    # A union, not an intersection: several requested statuses mean "status is
    # any of these", matching how multiple phases are already handled above.
    # RECRUITING is deliberately kept in the set -- stripping it assumed the
    # upstream filter had already applied, which is only true when it was the
    # sole request, and that assumption emptied every mixed-status result.
    wanted_status = normalize_statuses(plan.entities.statuses)
    if wanted_status == {"RECRUITING"}:
        # Already filtered upstream by aggFilters=status:rec; re-applying it
        # here would be a no-op, and the warning would be misleading.
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
    if network is not None and network.truncated_to_top_n:
        warnings.append(
            f"Graph truncated to the {network.truncated_to_top_n} highest-degree "
            f"nodes for readability."
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
        result = zero_fill_years(result)
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
    response = QueryResponse(visualization=spec, meta=meta)
    return validate_response(
        response,
        fetched.store,
        aggregation=aggregation,
        network=network,
        max_citations_per_datum=plan.max_citations_per_datum,
    )


__all__ = [
    "CTGovError",
    "EmptyResultError",
    "UnderstandingError",
    "UnsupportedQueryError",
    "ValidationFailure",
    "run_pipeline",
]
