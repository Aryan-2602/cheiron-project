"""The single aggregation function, across every chart family it serves."""

import pytest

from app.services.aggregate import aggregate, zero_fill_years
from app.services.dimensions import DIMENSIONS
from tests.conftest import make_record


def by_key(result, series=None):
    return {d.key: d.value for d in result.data if d.series == series}


class TestCoreInvariant:
    """The property that makes the output trustworthy: every value is a set
    cardinality over real ids, never an asserted number."""

    def test_value_always_equals_distinct_contributor_count(self, records):
        for dimension in DIMENSIONS.values():
            result = aggregate(records, dimension)
            for datum in result.data:
                assert datum.value == len(set(datum.nct_ids)), (
                    f"{dimension.name}/{datum.key}: value {datum.value} != "
                    f"{len(set(datum.nct_ids))} distinct ids"
                )

    def test_every_cited_id_exists_in_the_source_records(self, records):
        result = aggregate(records, DIMENSIONS["phase"])
        for datum in result.data:
            for nct_id in datum.nct_ids:
                assert nct_id in records


class TestDistribution:
    def test_phase_distribution_counts_multi_phase_trials_in_both_buckets(
        self, records
    ):
        result = aggregate(records, DIMENSIONS["phase"])
        counts = by_key(result)
        # NCT...002 is a combined Phase 1/2 trial and belongs in both.
        assert counts["Phase 1"] == 1
        assert counts["Phase 2"] == 1
        assert counts["Phase 3"] == 1
        assert counts["Not Applicable"] == 1
        assert counts["No phase data"] == 1

    def test_bucket_sum_may_exceed_trial_count_and_that_is_flagged(self, records):
        result = aggregate(records, DIMENSIONS["phase"])
        assert sum(d.value for d in result.data) > result.total_studies_matched
        assert result.multi_valued is True

    def test_missing_data_is_surfaced_not_dropped(self, records):
        result = aggregate(records, DIMENSIONS["phase"])
        assert result.unbucketed == 1
        assert result.unbucketed_key_included is True
        assert "No phase data" in by_key(result)

    def test_unknown_bucket_can_be_suppressed(self, records):
        result = aggregate(records, DIMENSIONS["phase"], include_unknown=False)
        assert "No phase data" not in by_key(result)
        # Still reported in the metadata even when not charted.
        assert result.unbucketed == 1
        assert result.unbucketed_key_included is False

    def test_unknown_bucket_sorts_last(self, records):
        result = aggregate(records, DIMENSIONS["phase"])
        assert result.data[-1].key == "No phase data"

    def test_phase_buckets_are_in_clinical_order(self, records):
        result = aggregate(records, DIMENSIONS["phase"])
        labels = [d.key for d in result.data]
        assert labels.index("Phase 1") < labels.index("Phase 2") < labels.index("Phase 3")


class TestTimeSeries:
    def test_groups_by_year_across_mixed_date_granularity(self, records):
        result = aggregate(records, DIMENSIONS["start_year"])
        counts = by_key(result)
        assert counts["2020"] == 1
        # "2021" and "2021-07" both land in 2021.
        assert counts["2021"] == 2
        assert counts["No start date"] == 1

    def test_zero_fill_inserts_missing_years(self):
        records = {
            "NCT00000001": make_record("NCT00000001", start_date="2018-01-01"),
            "NCT00000002": make_record("NCT00000002", start_date="2021-01-01"),
        }
        result = zero_fill_years(
            aggregate(records, DIMENSIONS["start_year"], include_unknown=False)
        )
        counts = by_key(result)
        assert counts == {"2018": 1, "2019": 0, "2020": 0, "2021": 1}

    def test_zero_filled_years_carry_no_citations(self):
        records = {
            "NCT00000001": make_record("NCT00000001", start_date="2018-01-01"),
            "NCT00000002": make_record("NCT00000002", start_date="2020-01-01"),
        }
        result = zero_fill_years(
            aggregate(records, DIMENSIONS["start_year"], include_unknown=False)
        )
        gap = next(d for d in result.data if d.key == "2019")
        assert gap.value == 0
        assert gap.nct_ids == []

    def test_years_ascending(self):
        records = {
            f"NCT0000000{i}": make_record(f"NCT0000000{i}", start_date=f"{y}-01-01")
            for i, y in enumerate([2022, 2019, 2021], start=1)
        }
        result = aggregate(records, DIMENSIONS["start_year"], include_unknown=False)
        assert [d.key for d in result.data] == ["2019", "2021", "2022"]


class TestComparison:
    """Comparison is the same function with a series_membership map -- there is
    no separate grouped-bar code path."""

    def test_series_split(self, records):
        membership = {
            "Pembrolizumab": {"NCT00000001", "NCT00000002"},
            "Nivolumab": {"NCT00000003"},
        }
        result = aggregate(
            records, DIMENSIONS["phase"], series_membership=membership
        )
        assert by_key(result, "Pembrolizumab") == {"Phase 1": 1, "Phase 2": 1, "Phase 3": 1}
        assert by_key(result, "Nivolumab") == {"No phase data": 1}
        assert result.series_dimension == "series"

    def test_trial_in_both_series_counts_in_both(self, records):
        membership = {"A": {"NCT00000001"}, "B": {"NCT00000001"}}
        result = aggregate(records, DIMENSIONS["phase"], series_membership=membership)
        assert by_key(result, "A") == {"Phase 3": 1}
        assert by_key(result, "B") == {"Phase 3": 1}
        # ...but the trial is only counted once in the overall total.
        assert result.total_studies_matched == 1

    def test_ids_outside_the_store_are_ignored(self, records):
        membership = {"A": {"NCT00000001", "NCT99999999"}}
        result = aggregate(records, DIMENSIONS["phase"], series_membership=membership)
        assert result.total_studies_matched == 1


class TestGeographic:
    def test_multinational_trial_counts_once_per_country(self, records):
        result = aggregate(records, DIMENSIONS["country"], include_unknown=False)
        counts = by_key(result)
        assert counts["United States"] == 2
        assert counts["Taiwan"] == 1
        assert counts["France"] == 1

    def test_top_n_keeps_largest_buckets(self, records):
        result = aggregate(
            records, DIMENSIONS["country"], include_unknown=False, top_n=1
        )
        assert by_key(result) == {"United States": 2}


class TestDeterminism:
    def test_repeated_runs_are_byte_identical(self, records):
        first = aggregate(records, DIMENSIONS["country"])
        second = aggregate(records, DIMENSIONS["country"])
        assert first.model_dump() == second.model_dump()


class TestEdgeCases:
    def test_empty_input(self):
        result = aggregate({}, DIMENSIONS["phase"])
        assert result.data == []
        assert result.total_studies_matched == 0

    @pytest.mark.parametrize("dimension_name", list(DIMENSIONS))
    def test_all_dimensions_handle_a_record_with_no_data(self, dimension_name):
        records = {"NCT00000001": {"protocolSection": {"identificationModule": {}}}}
        result = aggregate(records, DIMENSIONS[dimension_name])
        assert result.unbucketed == 1
        assert result.total_studies_matched == 1


def test_aggregate_runs_over_real_api_data(live_page):
    """Sanity check on unmodified live data: real phases, plausible counts."""
    records = {
        s["protocolSection"]["identificationModule"]["nctId"]: s
        for s in live_page["studies"]
    }
    result = aggregate(records, DIMENSIONS["phase"])
    assert result.total_studies_matched == len(records) == 25
    assert sum(d.value for d in result.data) >= 25
    for datum in result.data:
        assert datum.value == len(set(datum.nct_ids))
