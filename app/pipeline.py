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
from app.services.ctgov import CTGovClient, CTGovError, CTGovSearch, build_searches
from app.services.dimensions import extract_start_year, get_dimension
from app.services.network import build_bipartite_network, build_cooccurrence_network
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
    # Split the fetch budget across searches so a two-drug comparison cannot
    # spend the whole allowance on its first series.
    per_search = max(100, plan.max_studies // max(1, len(searches)))
    membership: dict[str, set[str]] = {}

    for search in searches:
        outcome = await client.run_search(search, result.store, max_studies=per_search)
        result.warnings.extend(outcome.warnings)
        if search.label:
            membership[search.label] = outcome.nct_ids
        result.filters.update(search.describe())

    if len(searches) > 1 and membership:
        result.series_membership = membership
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

    Filtering locally is exact and keeps citations intact, because the records
    being filtered are the same ones the citations point at.
    """
    warnings: list[str] = []

    wanted_status = {s.strip().upper().replace(" ", "_") for s in plan.entities.statuses}
    wanted_status = {s for s in wanted_status if s and not s.startswith("RECRUIT")}
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
        warnings.append(
            f"Filtered to status {', '.join(sorted(wanted_status))} client-side "
            f"({len(store.records):,} of {before:,} fetched trials); the upstream "
            f"filter code for this status is not live-verified."
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


def analyze(
    plan: QueryPlan, fetched: FetchResult
) -> tuple[VisualizationSpec, AggregationResult | None, NetworkResult | None]:
    """Stages 4 and 5: measure, then format. The only source of data values."""
    store = fetched.store

    if plan.query_type == "relationship":
        if plan.network_kind == "sponsor_drug":
            network = build_bipartite_network(store.records)
        else:
            network = build_cooccurrence_network(store.records)
        return build_network_spec(network, plan, store), None, network

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
    return build_chart_spec(result, plan, dimension, store), result, None


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

    searches = build_searches(plan)
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

    spec, aggregation, network = analyze(plan, fetched)
    meta = build_meta(
        plan,
        fetched,
        data_as_of=data_as_of,
        aggregation=aggregation,
        network=network,
        extra_warnings=extra_warnings,
    )
    response = QueryResponse(visualization=spec, meta=meta)
    return validate_response(
        response, fetched.store, aggregation=aggregation, network=network
    )


__all__ = [
    "CTGovError",
    "EmptyResultError",
    "UnderstandingError",
    "UnsupportedQueryError",
    "ValidationFailure",
    "run_pipeline",
]
