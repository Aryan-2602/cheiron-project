"""Stage 6: refuse to ship a response that cannot be backed by the fetched data.

This is the last line of the anti-hallucination argument. The earlier stages are
designed so a fabricated value has nowhere to come from; this stage assumes that
design could still be broken by a bug and checks the output against the raw
records anyway.

Every check fails loudly. A wrong chart that renders is worse than an error,
because a reader has no way to tell it is wrong.
"""

from __future__ import annotations

import logging

from app.models.schemas import (
    AggregationResult,
    NetworkResult,
    QueryResponse,
    VisualizationSpec,
)
from app.services.store import StudyStore
from app.services.viz import VALUE_FIELD

logger = logging.getLogger(__name__)


class ValidationFailure(RuntimeError):
    """A response failed a grounding check and must not be returned."""


def _check_schema(response: QueryResponse) -> None:
    """Round-trip through the model: what we return must re-parse as the
    documented contract, not merely resemble it."""
    try:
        QueryResponse.model_validate(response.model_dump())
    except Exception as exc:  # noqa: BLE001 - any parse failure must fail closed
        raise ValidationFailure(f"response does not conform to its own schema: {exc}")


def _check_encoding_matches_data(spec: VisualizationSpec) -> None:
    """Every channel the encoding names must exist in every data row.

    Without this, a frontend renders an empty axis and the failure looks like
    "no data" rather than a contract violation.
    """
    if spec.type == "network_graph":
        row = spec.data[0]
        for collection, key_map in (
            ("nodes", spec.encoding.nodes or {}),
            ("edges", spec.encoding.edges or {}),
        ):
            if collection not in row:
                raise ValidationFailure(f"network data is missing '{collection}'")
            for channel, field in key_map.items():
                for item in row[collection]:
                    if field not in item:
                        raise ValidationFailure(
                            f"encoding maps {collection}.{channel} to {field!r}, "
                            f"which is absent from a {collection[:-1]}"
                        )
        return

    required = [
        ref.field
        for ref in (spec.encoding.x, spec.encoding.y, spec.encoding.color)
        if ref is not None
    ]
    for row in spec.data:
        missing = [field for field in required if field not in row]
        if missing:
            raise ValidationFailure(
                f"encoding names field(s) {missing} absent from data row {row!r}"
            )


def _check_counts(
    spec: VisualizationSpec,
    aggregation: AggregationResult | None,
    network: NetworkResult | None,
) -> None:
    """Re-derive every published value from the pre-truncation id sets.

    The aggregator's own result is passed in alongside the spec precisely so
    the check has an independent source: if formatting dropped, duplicated, or
    rewrote a value, the two disagree here.
    """
    if aggregation is not None:
        for datum in aggregation.data:
            if datum.value != len(set(datum.nct_ids)):
                raise ValidationFailure(
                    f"bucket {datum.key!r} claims {datum.value} trials but carries "
                    f"{len(set(datum.nct_ids))} distinct NCT ids"
                )
        published = [
            (row.get(aggregation.dimension), row.get("series"), row.get(VALUE_FIELD))
            for row in spec.data
        ]
        computed = [(d.key, d.series, d.value) for d in aggregation.data]
        if published != computed:
            raise ValidationFailure(
                "published data rows do not match the aggregation they were built from"
            )

    if network is not None:
        for node in network.nodes:
            if node.size != len(set(node.nct_ids)):
                raise ValidationFailure(
                    f"node {node.id!r} claims size {node.size} but carries "
                    f"{len(set(node.nct_ids))} distinct NCT ids"
                )
        for edge in network.edges:
            if edge.weight != len(set(edge.nct_ids)):
                raise ValidationFailure(
                    f"edge {edge.source}->{edge.target} claims weight {edge.weight} "
                    f"but carries {len(set(edge.nct_ids))} distinct NCT ids"
                )
        node_ids = {n.id for n in network.nodes}
        for edge in network.edges:
            if edge.source not in node_ids or edge.target not in node_ids:
                raise ValidationFailure(
                    f"edge {edge.source}->{edge.target} references a node that is "
                    f"not in the graph"
                )
        for node in network.nodes:
            # A merge is a claim that several names are one compound. It is only
            # defensible if the resolved identity that justified it is present.
            if node.merged_from and not node.rxcui:
                raise ValidationFailure(
                    f"node {node.id!r} claims to merge {len(node.merged_from)} names "
                    f"but carries no RxNorm identity"
                )
            if node.merged_from and len(node.merged_from) < 2:
                raise ValidationFailure(
                    f"node {node.id!r} reports a merge of fewer than two names"
                )


def _iter_citation_blocks(spec: VisualizationSpec):
    if spec.type == "network_graph":
        row = spec.data[0]
        yield from list(row.get("nodes", [])) + list(row.get("edges", []))
    else:
        yield from spec.data


def _check_citations(spec: VisualizationSpec, store: StudyStore) -> None:
    """Every citation must point at a record we actually fetched.

    This is the check that makes "the LLM never invents data" verifiable rather
    than merely asserted: an id that is not in the store cannot have come from
    the aggregation, so it must have come from somewhere it shouldn't.
    """
    for item in _iter_citation_blocks(spec):
        citations = item.get("citations", [])
        total = item.get("total_supporting_trials")

        if total is None:
            raise ValidationFailure(f"datum {item!r} is missing total_supporting_trials")
        if len(citations) > total:
            raise ValidationFailure(
                f"datum carries {len(citations)} citations but claims only "
                f"{total} supporting trials"
            )

        measured = item.get(VALUE_FIELD, item.get("size", item.get("weight")))
        if measured and not citations:
            raise ValidationFailure(
                f"datum with value {measured} has no citations backing it"
            )

        for citation in citations:
            nct_id = citation.get("nct_id")
            if nct_id not in store:
                raise ValidationFailure(
                    f"citation references {nct_id!r}, which is not among the "
                    f"{len(store)} records fetched for this request"
                )
            if not (citation.get("excerpt") or "").strip():
                raise ValidationFailure(f"citation for {nct_id} has an empty excerpt")


def _check_non_empty(spec: VisualizationSpec) -> None:
    """A chart with no rows must never ship as if it were an answer."""
    if spec.type == "network_graph":
        if not spec.data or not spec.data[0].get("nodes"):
            raise ValidationFailure("network graph has no nodes")
        return
    if not spec.data:
        raise ValidationFailure("visualization has no data rows")


def _check_meta(response: QueryResponse, store: StudyStore) -> None:
    if response.meta.total_studies_processed != len(store):
        raise ValidationFailure(
            f"meta reports {response.meta.total_studies_processed} studies processed "
            f"but {len(store)} records were fetched"
        )
    if not response.meta.api_urls:
        raise ValidationFailure("meta must record the upstream API URLs actually called")


def validate_response(
    response: QueryResponse,
    store: StudyStore,
    *,
    aggregation: AggregationResult | None = None,
    network: NetworkResult | None = None,
) -> QueryResponse:
    """Run every grounding check. Raises :class:`ValidationFailure` on any breach."""
    _check_schema(response)
    _check_non_empty(response.visualization)
    _check_encoding_matches_data(response.visualization)
    _check_counts(response.visualization, aggregation, network)
    _check_citations(response.visualization, store)
    _check_meta(response, store)

    # Only the pass is logged here. A failure raises, and the route handler
    # emits a single ERROR carrying both the failing check and the request
    # context -- logging in both places would read as two separate faults.
    logger.info(
        "validation passed",
        extra={
            "viz_type": response.visualization.type,
            "rows": len(response.visualization.data),
            "citations": sum(
                len(item.get("citations", []))
                for item in _iter_citation_blocks(response.visualization)
            ),
        },
    )
    return response
