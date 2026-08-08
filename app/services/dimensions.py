"""The dimension registry: how to turn a raw study record into bucket keys.

This is the half of the "single coherent approach" that varies per question.
:func:`app.services.aggregate.aggregate` never changes; adding a new chart
axis means adding a :class:`Dimension` here and nothing else.

Every ``extract`` returns a *list* of keys, because several dimensions are
genuinely multi-valued in the source data: a combined Phase 1/2 trial has two
phases, a multinational trial has several countries. Returning ``[]`` means the
record has no value for this dimension, which the aggregator reports as an
explicit bucket rather than dropping.

All response paths below were confirmed against a live API response on
2026-08-08. They are *not* mechanical case conversions of the PascalCase names
sent in the ``fields`` request parameter (``Phase`` -> ``designModule.phases``,
``StartDate`` -> ``statusModule.startDateStruct.date``), so each is pinned by a
fixture test.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

ExtractFn = Callable[[dict[str, Any]], list[str]]

#: Canonical ordering for phase buckets; anything unknown sorts after these.
PHASE_ORDER = [
    "EARLY_PHASE1",
    "PHASE1",
    "PHASE2",
    "PHASE3",
    "PHASE4",
    "NA",
]

PHASE_LABELS = {
    "EARLY_PHASE1": "Early Phase 1",
    "PHASE1": "Phase 1",
    "PHASE2": "Phase 2",
    "PHASE3": "Phase 3",
    "PHASE4": "Phase 4",
    "NA": "Not Applicable",
}

ENROLLMENT_BINS: list[tuple[int, int | None, str]] = [
    (0, 10, "0-9"),
    (10, 50, "10-49"),
    (50, 100, "50-99"),
    (100, 500, "100-499"),
    (500, 1000, "500-999"),
    (1000, None, "1000+"),
]


def _protocol(record: dict[str, Any], module: str) -> dict[str, Any]:
    return record.get("protocolSection", {}).get(module, {}) or {}


# --------------------------------------------------------------------------
# Extractors
# --------------------------------------------------------------------------


def extract_phases(record: dict[str, Any]) -> list[str]:
    """``designModule.phases`` -- always an array upstream, possibly absent.

    Roughly a quarter of studies have no usable phase (observational studies
    report ``NA``, others omit the field). Both cases are preserved rather than
    silently dropped: ``NA`` is a real bucket, absence yields ``[]``.
    """
    phases = _protocol(record, "designModule").get("phases") or []
    return [p for p in phases if isinstance(p, str) and p]


_YEAR_RE = re.compile(r"^(\d{4})")


def extract_start_year(record: dict[str, Any]) -> list[str]:
    """``statusModule.startDateStruct.date`` -- granularity varies by record.

    Observed forms include ``2024-08-22``, ``2024-08`` and ``2024``. Only the
    year is used, so all three parse identically and nothing is discarded for
    lacking day precision.
    """
    date = _protocol(record, "statusModule").get("startDateStruct", {}).get("date")
    if not isinstance(date, str):
        return []
    match = _YEAR_RE.match(date.strip())
    return [match.group(1)] if match else []


def extract_status(record: dict[str, Any]) -> list[str]:
    status = _protocol(record, "statusModule").get("overallStatus")
    return [status] if isinstance(status, str) and status else []


def extract_sponsor(record: dict[str, Any]) -> list[str]:
    name = (
        _protocol(record, "sponsorCollaboratorsModule").get("leadSponsor", {}).get("name")
    )
    return [name] if isinstance(name, str) and name else []


def extract_sponsor_class(record: dict[str, Any]) -> list[str]:
    klass = (
        _protocol(record, "sponsorCollaboratorsModule").get("leadSponsor", {}).get("class")
    )
    return [klass] if isinstance(klass, str) and klass else []


def extract_intervention_types(record: dict[str, Any]) -> list[str]:
    """Distinct intervention types in a trial (a trial may combine several)."""
    interventions = _protocol(record, "armsInterventionsModule").get("interventions") or []
    types = {
        i.get("type")
        for i in interventions
        if isinstance(i, dict) and isinstance(i.get("type"), str) and i.get("type")
    }
    return sorted(types)


def extract_countries(record: dict[str, Any]) -> list[str]:
    """Distinct countries. A trial with 40 US sites counts once for the US."""
    locations = _protocol(record, "contactsLocationsModule").get("locations") or []
    countries = {
        loc.get("country")
        for loc in locations
        if isinstance(loc, dict)
        and isinstance(loc.get("country"), str)
        and loc.get("country")
    }
    return sorted(countries)


def extract_enrollment_bucket(record: dict[str, Any]) -> list[str]:
    count = _protocol(record, "designModule").get("enrollmentInfo", {}).get("count")
    if not isinstance(count, int) or count < 0:
        return []
    for low, high, label in ENROLLMENT_BINS:
        if count >= low and (high is None or count < high):
            return [label]
    return []


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Dimension:
    """Everything the generic aggregator needs to know about one axis."""

    name: str
    #: PascalCase names for the ``fields`` request parameter.
    api_fields: tuple[str, ...]
    extract: ExtractFn
    display: Callable[[str], str]
    #: Sort position for a bucket key; lower sorts first. Ties break by ``-value``
    #: then key, so output ordering is fully deterministic.
    order: Callable[[str], Any]
    #: Human-readable axis label for the chart.
    axis_label: str
    #: What to call records that have no value here.
    unknown_label: str
    #: True when one record can land in several buckets, so bucket values may
    #: legitimately sum to more than the number of trials.
    multi_valued: bool
    #: Frontend hint describing the emitted ordering.
    sorting: str
    field_type: str = "nominal"


def _by_value_desc(_key: str) -> Any:
    # Constant position -> the aggregator's (-value, key) tiebreak decides.
    return 0


DIMENSIONS: dict[str, Dimension] = {
    "phase": Dimension(
        name="phase",
        api_fields=("Phase",),
        extract=extract_phases,
        display=lambda k: PHASE_LABELS.get(k, k.replace("_", " ").title()),
        order=lambda k: PHASE_ORDER.index(k) if k in PHASE_ORDER else len(PHASE_ORDER),
        axis_label="Phase",
        unknown_label="No phase data",
        multi_valued=True,
        sorting="phase order (Early Phase 1 -> Phase 4)",
        field_type="ordinal",
    ),
    "start_year": Dimension(
        name="start_year",
        api_fields=("StartDate",),
        extract=extract_start_year,
        display=lambda k: k,
        order=lambda k: int(k) if k.isdigit() else 9999,
        axis_label="Start year",
        unknown_label="No start date",
        multi_valued=False,
        sorting="year ascending",
        field_type="temporal",
    ),
    "status": Dimension(
        name="status",
        api_fields=("OverallStatus",),
        extract=extract_status,
        display=lambda k: k.replace("_", " ").title(),
        order=_by_value_desc,
        axis_label="Overall status",
        unknown_label="No status",
        multi_valued=False,
        sorting="trial count descending",
    ),
    "sponsor": Dimension(
        name="sponsor",
        api_fields=("LeadSponsorName",),
        extract=extract_sponsor,
        display=lambda k: k,
        order=_by_value_desc,
        axis_label="Lead sponsor",
        unknown_label="No sponsor listed",
        multi_valued=False,
        sorting="trial count descending",
    ),
    "sponsor_class": Dimension(
        name="sponsor_class",
        api_fields=("LeadSponsorClass",),
        extract=extract_sponsor_class,
        display=lambda k: k.replace("_", " ").title(),
        order=_by_value_desc,
        axis_label="Sponsor category",
        unknown_label="No sponsor category",
        multi_valued=False,
        sorting="trial count descending",
    ),
    "intervention_type": Dimension(
        name="intervention_type",
        api_fields=("InterventionType",),
        extract=extract_intervention_types,
        display=lambda k: k.replace("_", " ").title(),
        order=_by_value_desc,
        axis_label="Intervention type",
        unknown_label="No interventions listed",
        multi_valued=True,
        sorting="trial count descending",
    ),
    "country": Dimension(
        name="country",
        api_fields=("LocationCountry",),
        extract=extract_countries,
        display=lambda k: k,
        order=_by_value_desc,
        axis_label="Country",
        unknown_label="No location listed",
        multi_valued=True,
        sorting="trial count descending",
    ),
    "enrollment_bucket": Dimension(
        name="enrollment_bucket",
        api_fields=("EnrollmentCount",),
        extract=extract_enrollment_bucket,
        display=lambda k: f"{k} participants",
        order=lambda k: [b[2] for b in ENROLLMENT_BINS].index(k)
        if k in [b[2] for b in ENROLLMENT_BINS]
        else len(ENROLLMENT_BINS),
        axis_label="Enrollment",
        unknown_label="No enrollment data",
        multi_valued=False,
        sorting="enrollment ascending",
        field_type="ordinal",
    ),
}


def get_dimension(name: str) -> Dimension:
    try:
        return DIMENSIONS[name]
    except KeyError:
        raise KeyError(
            f"unknown dimension {name!r}; known: {sorted(DIMENSIONS)}"
        ) from None
