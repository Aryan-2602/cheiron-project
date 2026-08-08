import json
from pathlib import Path
from typing import Any

import pytest

FIXTURE_DIR = Path(__file__).parent / "fixtures"


def make_record(
    nct_id: str = "NCT00000001",
    *,
    title: str = "A study of something",
    phases: list[str] | None = None,
    status: str | None = "RECRUITING",
    start_date: str | None = "2020-05-01",
    sponsor: str | None = "Acme Pharma",
    sponsor_class: str | None = "INDUSTRY",
    interventions: list[tuple[str, str]] | None = None,
    countries: list[str] | None = None,
    enrollment: int | None = 100,
) -> dict[str, Any]:
    """Build a study record shaped exactly like a real API response.

    Field paths mirror the live payload captured in
    ``fixtures/ctgov_pembrolizumab_page.json``; ``test_dimensions.py`` asserts
    the two stay in agreement.
    """
    design: dict[str, Any] = {}
    if phases is not None:
        design["phases"] = phases
    if enrollment is not None:
        design["enrollmentInfo"] = {"count": enrollment}

    status_module: dict[str, Any] = {}
    if status is not None:
        status_module["overallStatus"] = status
    if start_date is not None:
        status_module["startDateStruct"] = {"date": start_date}

    sponsor_module: dict[str, Any] = {}
    if sponsor is not None or sponsor_class is not None:
        lead: dict[str, Any] = {}
        if sponsor is not None:
            lead["name"] = sponsor
        if sponsor_class is not None:
            lead["class"] = sponsor_class
        sponsor_module["leadSponsor"] = lead

    arms: dict[str, Any] = {}
    if interventions is not None:
        arms["interventions"] = [{"type": t, "name": n} for t, n in interventions]

    contacts: dict[str, Any] = {}
    if countries is not None:
        contacts["locations"] = [{"country": c} for c in countries]

    return {
        "protocolSection": {
            "identificationModule": {"nctId": nct_id, "briefTitle": title},
            "statusModule": status_module,
            "sponsorCollaboratorsModule": sponsor_module,
            "designModule": design,
            "armsInterventionsModule": arms,
            "contactsLocationsModule": contacts,
        }
    }


@pytest.fixture
def live_page() -> dict[str, Any]:
    """A real, unmodified ClinicalTrials.gov page captured on 2026-08-08."""
    return json.loads((FIXTURE_DIR / "ctgov_pembrolizumab_page.json").read_text())


@pytest.fixture
def records() -> dict[str, dict[str, Any]]:
    """A small hand-built corpus exercising the awkward cases."""
    return {
        "NCT00000001": make_record(
            "NCT00000001",
            phases=["PHASE3"],
            start_date="2020-05-01",
            countries=["United States"],
            interventions=[("DRUG", "Pembrolizumab")],
        ),
        # Combined phase 1/2 trial: lands in two phase buckets.
        "NCT00000002": make_record(
            "NCT00000002",
            phases=["PHASE1", "PHASE2"],
            start_date="2021",
            countries=["United States", "Taiwan"],
            interventions=[("DRUG", "Pembrolizumab"), ("DRUG", "Lenvatinib")],
        ),
        # No phase field at all: must surface as "No phase data", not vanish.
        "NCT00000003": make_record(
            "NCT00000003",
            phases=None,
            start_date="2021-07",
            sponsor="Beta Institute",
            sponsor_class="OTHER",
            countries=["France"],
            interventions=[("BIOLOGICAL", "Nivolumab")],
        ),
        # Observational study: phase is the literal enum "NA".
        "NCT00000004": make_record(
            "NCT00000004",
            phases=["NA"],
            start_date=None,
            countries=None,
            interventions=None,
            enrollment=None,
        ),
    }
