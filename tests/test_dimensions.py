"""Dimension extractors, pinned against a real captured API response.

The response paths these extractors walk are *not* mechanical conversions of
the PascalCase names sent in the ``fields`` request parameter, so the live
fixture test below is the guard against drift: if ClinicalTrials.gov moves a
field, these fail rather than silently producing empty charts.
"""

from app.services.dimensions import (
    DIMENSIONS,
    extract_countries,
    extract_enrollment_bucket,
    extract_intervention_types,
    extract_phases,
    extract_sponsor,
    extract_sponsor_class,
    extract_start_year,
    extract_status,
)
from tests.conftest import make_record


class TestAgainstLiveFixture:
    """Every extractor must find real values in an unmodified API response."""

    def test_every_extractor_finds_values_in_real_data(self, live_page):
        studies = live_page["studies"]
        assert len(studies) == 25

        for name, dimension in DIMENSIONS.items():
            hits = sum(1 for s in studies if dimension.extract(s))
            assert hits > 0, (
                f"dimension {name!r} extracted nothing from 25 real records -- "
                f"the response path has probably changed"
            )

    def test_extractors_never_raise_on_real_data(self, live_page):
        for study in live_page["studies"]:
            for dimension in DIMENSIONS.values():
                assert isinstance(dimension.extract(study), list)

    def test_phase_values_are_the_enum_form(self, live_page):
        """Response phases are PHASE1/PHASE2/...; the *request* filter uses bare
        numbers. Conflating the two is the documented silent-failure trap."""
        seen = {p for s in live_page["studies"] for p in extract_phases(s)}
        assert seen, "no phases found in live data"
        assert all(p.startswith("PHASE") or p in {"NA", "EARLY_PHASE1"} for p in seen)


class TestPhases:
    def test_multi_phase_trial_yields_both(self):
        assert extract_phases(make_record(phases=["PHASE1", "PHASE2"])) == [
            "PHASE1",
            "PHASE2",
        ]

    def test_missing_phase_field_yields_empty(self):
        assert extract_phases(make_record(phases=None)) == []

    def test_na_is_a_real_bucket_not_a_missing_value(self):
        assert extract_phases(make_record(phases=["NA"])) == ["NA"]


class TestStartYear:
    def test_parses_all_three_observed_granularities(self):
        assert extract_start_year(make_record(start_date="2024-08-22")) == ["2024"]
        assert extract_start_year(make_record(start_date="2024-08")) == ["2024"]
        assert extract_start_year(make_record(start_date="2024")) == ["2024"]

    def test_missing_or_unparseable_date_yields_empty(self):
        assert extract_start_year(make_record(start_date=None)) == []
        assert extract_start_year(make_record(start_date="unknown")) == []


class TestOtherDimensions:
    def test_status_and_sponsor(self):
        record = make_record(status="COMPLETED", sponsor="Acme", sponsor_class="INDUSTRY")
        assert extract_status(record) == ["COMPLETED"]
        assert extract_sponsor(record) == ["Acme"]
        assert extract_sponsor_class(record) == ["INDUSTRY"]

    def test_countries_deduplicated(self):
        record = make_record(countries=["United States", "United States", "Taiwan"])
        assert extract_countries(record) == ["Taiwan", "United States"]

    def test_intervention_types_deduplicated(self):
        record = make_record(
            interventions=[("DRUG", "A"), ("DRUG", "B"), ("BIOLOGICAL", "C")]
        )
        assert extract_intervention_types(record) == ["BIOLOGICAL", "DRUG"]

    def test_enrollment_bins(self):
        assert extract_enrollment_bucket(make_record(enrollment=0)) == ["0-9"]
        assert extract_enrollment_bucket(make_record(enrollment=49)) == ["10-49"]
        assert extract_enrollment_bucket(make_record(enrollment=100)) == ["100-499"]
        assert extract_enrollment_bucket(make_record(enrollment=25000)) == ["1000+"]
        assert extract_enrollment_bucket(make_record(enrollment=None)) == []


class TestDisplayAndOrder:
    def test_phase_labels_are_human_readable(self):
        display = DIMENSIONS["phase"].display
        assert display("PHASE3") == "Phase 3"
        assert display("NA") == "Not Applicable"

    def test_phase_order_is_clinical_not_alphabetical(self):
        order = DIMENSIONS["phase"].order
        keys = ["PHASE4", "PHASE1", "EARLY_PHASE1", "PHASE10-unknown"]
        assert sorted(keys, key=order) == [
            "EARLY_PHASE1",
            "PHASE1",
            "PHASE4",
            "PHASE10-unknown",
        ]


def test_extractors_tolerate_a_totally_empty_record():
    """Sparse records are normal upstream; nothing here may raise."""
    for dimension in DIMENSIONS.values():
        assert dimension.extract({}) == []
        assert dimension.extract({"protocolSection": {}}) == []
