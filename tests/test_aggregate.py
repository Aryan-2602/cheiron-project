"""The single aggregation function, across every chart family it serves."""

import pytest

from app.models.schemas import AggregatedDatum, AggregationResult
from app.services.aggregate import aggregate, zero_fill_years
from app.services.dimensions import DIMENSIONS, get_dimension
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


class TestCategoryTruncationBookkeeping:
    """A top-N cap that is not disclosed presents a slice as the whole picture:
    20 of 47 countries reads as every country."""

    def records(self, n_sponsors, per=4):
        return {
            f"NCT{i:08d}": make_record(f"NCT{i:08d}", sponsor=f"Sponsor {i % n_sponsors}")
            for i in range(n_sponsors * per)
        }

    def test_no_cap_reports_no_omissions(self):
        result = aggregate(self.records(10), get_dimension("sponsor"), top_n=None)
        assert result.total_categories == 10
        assert result.displayed_categories == 10
        assert result.omitted_categories == 0
        assert result.category_limit is None

    def test_cap_exactly_at_the_category_count_omits_nothing(self):
        result = aggregate(self.records(10), get_dimension("sponsor"), top_n=10)
        assert (result.total_categories, result.displayed_categories) == (10, 10)
        assert result.omitted_categories == 0

    def test_cap_above_the_category_count_omits_nothing(self):
        result = aggregate(self.records(10), get_dimension("sponsor"), top_n=25)
        assert result.omitted_categories == 0

    def test_cap_below_the_category_count_reports_the_omitted_total(self):
        result = aggregate(self.records(30), get_dimension("sponsor"), top_n=15)
        assert result.total_categories == 30
        assert result.displayed_categories == 15
        assert result.omitted_categories == 15
        assert result.category_limit == 15

    def test_only_the_top_categories_survive(self):
        """The cap keeps the busiest categories, and the counts still describe
        exactly the rows that shipped."""
        records = {}
        for rank, sponsor in enumerate(["Big", "Mid", "Small"]):
            for i in range((3 - rank) * 5):
                nct = f"NCT{rank}{i:07d}"
                records[nct] = make_record(nct, sponsor=sponsor)
        result = aggregate(records, get_dimension("sponsor"), top_n=2)
        shown = {d.key for d in result.data}
        assert shown == {"Big", "Mid"}
        assert result.displayed_categories == len(shown)
        assert result.omitted_categories == 1


class TestZeroFillBounds:
    """Filling only the observed range cropped the axis to the data: "2020
    through 2024" with trials only in 2021-2023 hid the two empty years, which
    are exactly the answer to "how many started each year"."""

    def result(self, years):
        return AggregationResult(
            dimension="start_year",
            data=[
                AggregatedDatum(key=str(y), series=None, value=1, nct_ids=[f"NCT{y}"])
                for y in years
            ],
            total_studies_matched=len(years),
        )

    def keys(self, result):
        return [d.key for d in result.data]

    def test_explicit_bounds_fill_the_whole_requested_interval(self):
        filled = zero_fill_years(self.result([2021, 2022, 2023]), start=2020, end=2024)
        assert self.keys(filled) == ["2020", "2021", "2022", "2023", "2024"]
        assert [d.value for d in filled.data] == [0, 1, 1, 1, 0]

    def test_a_single_observed_year_still_fills_the_requested_interval(self):
        """Previously bailed out entirely when fewer than two years were seen."""
        filled = zero_fill_years(self.result([2022]), start=2020, end=2024)
        assert self.keys(filled) == ["2020", "2021", "2022", "2023", "2024"]

    def test_without_bounds_the_observed_range_is_used(self):
        filled = zero_fill_years(self.result([2020, 2023]))
        assert self.keys(filled) == ["2020", "2021", "2022", "2023"]

    def test_a_single_observed_year_without_bounds_is_left_alone(self):
        """Inventing an axis around one point would be inventing a bound."""
        assert self.keys(zero_fill_years(self.result([2022]))) == ["2022"]

    def test_internal_gaps_are_filled(self):
        filled = zero_fill_years(self.result([2018, 2021]))
        assert self.keys(filled) == ["2018", "2019", "2020", "2021"]

    def test_one_bound_extends_only_that_end(self):
        filled = zero_fill_years(self.result([2021, 2023]), start=2019)
        assert self.keys(filled) == ["2019", "2020", "2021", "2022", "2023"]

    def test_an_absurd_span_is_refused_rather_than_rendered(self):
        filled = zero_fill_years(self.result([2021]), start=1000, end=2024)
        assert self.keys(filled) == ["2021"]

    def test_reversed_bounds_are_refused(self):
        filled = zero_fill_years(self.result([2021]), start=2024, end=2020)
        assert self.keys(filled) == ["2021"]

    def test_no_data_stays_empty(self):
        """Fabricating a row of zeros would claim an answer; this case is
        reported as NO_CHARTABLE_DATA instead."""
        assert zero_fill_years(self.result([]), start=2020, end=2024).data == []

    def test_filled_years_carry_no_citations(self):
        filled = zero_fill_years(self.result([2021]), start=2020, end=2022)
        for datum in filled.data:
            if datum.value == 0:
                assert datum.nct_ids == []
