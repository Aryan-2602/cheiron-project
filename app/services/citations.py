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
    "drug": (
        "armsInterventionsModule.interventions[].name",
        lambda r: [
            i.get("name")
            for i in (_raw_value(r, "armsInterventionsModule.interventions") or [])
            if isinstance(i, dict) and i.get("name")
        ],
    ),
}


def build_excerpt(record: dict[str, Any], dimension: str) -> str:
    """Render the supporting excerpt for one trial on one dimension.

    Falls back to the brief title alone when the dimension has no registered
    evidence path or the record is missing the field -- an honest excerpt for a
    record that landed in an "unknown" bucket precisely because the field was
    absent.
    """
    title = _title(record)
    entry = EVIDENCE.get(dimension)
    if entry is None:
        return f'"{title}"' if title else "(no title in record)"

    path, reader = entry
    value = reader(record)
    if value in (None, [], ""):
        return f'"{title}" — {path}: not reported' if title else f"{path}: not reported"
    if isinstance(value, list):
        rendered = ", ".join(str(v) for v in value)
        rendered = f"[{rendered}]"
    else:
        rendered = f'"{value}"' if isinstance(value, str) else str(value)
    return f'"{title}" — {path}: {rendered}' if title else f"{path}: {rendered}"


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
    for nct_id in nct_ids[:limit]:
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
