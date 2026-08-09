"""Deep citations: linking each visualized value back to the trial records
that produced it.

Citations are not assembled by searching for evidence after the fact. Bucket
membership *is* the evidence: the aggregator already knows exactly which trials
put a bar at the height it is, and this module renders a readable excerpt for
each of them. That ordering matters -- a citation can never disagree with the
number it supports, because both are projections of the same set of NCT ids.

Excerpts quote the exact field value from the API response that caused the
membership, plus the trial's brief title for human context. Nothing here calls
an LLM, and nothing here makes a second API request: every field quoted was
already fetched as part of the search.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from app.models.schemas import Citation
from app.services.dimensions import (
    extract_countries,
    extract_intervention_types,
    extract_phases,
    extract_sponsor,
    extract_sponsor_class,
    extract_status,
)
from app.services.network import DRUG_TYPES
from app.services.store import StudyStore

#: How many characters of a brief title to quote before eliding.
TITLE_CLIP = 160


def _title(record: dict[str, Any]) -> str:
    title = (
        record.get("protocolSection", {})
        .get("identificationModule", {})
        .get("briefTitle")
        or ""
    )
    if len(title) > TITLE_CLIP:
        return title[:TITLE_CLIP].rstrip() + "..."
    return title


def _raw_value(record: dict[str, Any], path: str) -> Any:
    """Read a dotted path out of ``protocolSection``, tolerating gaps."""
    node: Any = record.get("protocolSection", {})
    for part in path.split("."):
        if not isinstance(node, dict):
            return None
        node = node.get(part)
    return node


#: dimension name -> (response path quoted in the excerpt, value reader).
#: The path is included verbatim so a reader can verify the claim against the
#: raw API response without guessing which field was used.
EVIDENCE: dict[str, tuple[str, Callable[[dict[str, Any]], Any]]] = {
    "phase": ("designModule.phases", extract_phases),
    "start_year": (
        "statusModule.startDateStruct.date",
        lambda r: _raw_value(r, "statusModule.startDateStruct.date"),
    ),
    "status": ("statusModule.overallStatus", lambda r: extract_status(r)[:1] or None),
    "sponsor": (
        "sponsorCollaboratorsModule.leadSponsor.name",
        lambda r: (extract_sponsor(r) or [None])[0],
    ),
    "sponsor_class": (
        "sponsorCollaboratorsModule.leadSponsor.class",
        lambda r: (extract_sponsor_class(r) or [None])[0],
    ),
    "intervention_type": (
        "armsInterventionsModule.interventions[].type",
        extract_intervention_types,
    ),
    "country": (
        "contactsLocationsModule.locations[].country",
        extract_countries,
    ),
    "enrollment_bucket": (
        "designModule.enrollmentInfo.count",
        lambda r: _raw_value(r, "designModule.enrollmentInfo.count"),
    ),
    # Same type filter the graph builder uses, so a drug node's evidence quotes
    # the interventions that could have produced it. Listing every intervention
    # buried the drug among placebo and procedure entries the node was never
    # built from.
    "drug": (
        "armsInterventionsModule.interventions[].name",
        lambda r: [
            i.get("name")
            for i in (_raw_value(r, "armsInterventionsModule.interventions") or [])
            if isinstance(i, dict) and i.get("name") and i.get("type") in DRUG_TYPES
        ],
    ),
}


def _render_field(record: dict[str, Any], dimension: str) -> str:
    """One ``path: value`` fragment, exactly as it appears in the record."""
    path, reader = EVIDENCE[dimension]
    value = reader(record)
    if value in (None, [], ""):
        return f"{path}: not reported"
    if isinstance(value, list):
        return f"{path}: [{', '.join(str(v) for v in value)}]"
    return f'{path}: "{value}"' if isinstance(value, str) else f"{path}: {value}"


#: Dimensions whose evidence spans two fields, because the datum they support
#: has two endpoints. A sponsor-drug edge cited with intervention names alone
#: proves the drug end and leaves the sponsor end unevidenced -- the reader has
#: to take it on trust that this trial was Merck's.
COMPOSITE_EVIDENCE: dict[str, tuple[str, ...]] = {
    "sponsor_drug": ("sponsor", "drug"),
}


def build_excerpt(record: dict[str, Any], dimension: str) -> str:
    """Render the supporting excerpt for one trial on one dimension.

    Falls back to the brief title alone when the dimension has no registered
    evidence path or the record is missing the field -- an honest excerpt for a
    record that landed in an "unknown" bucket precisely because the field was
    absent.
    """
    title = _title(record)
    # One field, or several when the datum has several endpoints. The title is
    # carried once either way.
    parts = COMPOSITE_EVIDENCE.get(dimension, (dimension,))
    known = [part for part in parts if part in EVIDENCE]
    if not known:
        return f'"{title}"' if title else "(no title in record)"

    rendered = "; ".join(_render_field(record, part) for part in known)
    return f'"{title}" — {rendered}' if title else rendered


def _spread(nct_ids: list[str], limit: int) -> list[str]:
    """Pick ``limit`` ids spread evenly across ``nct_ids``, order preserved.

    The aggregator returns contributors sorted ascending, and NCT ids are
    assigned roughly chronologically, so taking the first ``limit`` -- as this
    once did -- cited every bucket's oldest members: a 373-trial bucket was
    evidenced by three 1990s studies. That is deterministic but arbitrary, and
    reads as cherry-picked from one end.

    A systematic sample keeps every property the design depends on: it is fully
    deterministic (same contributors always give the same citations), it needs
    no optionally-missing field such as enrolment, whose absence would make the
    choice non-deterministic, it is unbiased toward either end, and it is
    identical to the old behaviour when ``limit >= len(nct_ids)``.
    """
    n = len(nct_ids)
    if limit >= n:
        return list(nct_ids)
    if limit == 1:
        return [nct_ids[0]]
    # Anchors at both ends so the sample visibly spans the range.
    picked = {round(i * (n - 1) / (limit - 1)) for i in range(limit)}
    return [nct_ids[i] for i in sorted(picked)]


def build_citations(
    nct_ids: list[str],
    store: StudyStore,
    dimension: str,
    *,
    limit: int,
) -> tuple[list[Citation], int]:
    """Citations for one datum, plus the true number of contributing trials.

    Only ``limit`` citations are emitted to keep responses renderable, but the
    second return value always reports the full contributor count, so a
    truncated citation list never reads as the complete evidence set.
    """
    total = len(nct_ids)
    if limit <= 0:
        return [], total

    citations: list[Citation] = []
    for nct_id in _spread(nct_ids, limit):
        record = store.get(nct_id)
        if record is None:
            # Cannot happen for aggregator output (ids come from the store), but
            # emitting an unverifiable citation would be worse than skipping it.
            continue
        citations.append(
            Citation(
                nct_id=nct_id,
                excerpt=build_excerpt(record, dimension),
                url=store.study_url(nct_id),
            )
        )
    return citations, total
