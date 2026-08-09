"""CTGov client: parameter construction, pagination, retry, and the
empty-filtered-result diagnostic."""

import httpx
import pytest
import respx

from app.models.schemas import ExtractedEntities, QueryPlan, YearRange
from app.services.ctgov import (
    AggFilter,
    CTGovClient,
    CTGovError,
    CTGovSearch,
    build_searches,
    normalize_statuses,
    series_key,
)
from app.services.store import StudyStore
from tests.conftest import make_record

BASE = "https://clinicaltrials.gov/api/v2"


def page(studies, *, total=None, next_token=None):
    body = {"studies": studies}
    if total is not None:
        body["totalCount"] = total
    if next_token:
        body["nextPageToken"] = next_token
    return httpx.Response(200, json=body)


async def run(search, *, max_studies=3000, page_size=1000, requested_max=None):
    store = StudyStore()
    async with CTGovClient(page_size=page_size, max_retries=1) as client:
        outcome = await client.run_search(
            search, store, max_studies=max_studies, requested_max=requested_max
        )
    return outcome, store


class TestAggFilter:
    """The documented trap: ``phase:PHASE3`` returns HTTP 200 with zero results.
    The type makes that form unrepresentable."""

    def test_phase_renders_as_a_bare_number(self):
        assert AggFilter.phase(3).render() == "phase:3"

    def test_phase_rejects_out_of_range(self):
        with pytest.raises(ValueError):
            AggFilter.phase(5)

    def test_only_verified_status_codes_accepted(self):
        assert AggFilter.status("rec").render() == "status:rec"
        # "avail" returns HTTP 200 with zero results -- the silent-failure trap.
        with pytest.raises(ValueError, match="not live-verified"):
            AggFilter.status("avail")

    def test_multiple_values_render_as_a_space_separated_union(self):
        """Verified live: rec 64,847 + com 326,301 = "status:rec com" 391,148."""
        assert AggFilter.status("rec", "com").render() == "status:com rec"
        # phase:2 89,652 + phase:3 49,614 -> "phase:2 3" 131,704 (overlap once).
        assert AggFilter.phase(2, 3).render() == "phase:2 3"

    def test_multi_value_rendering_is_order_independent(self):
        assert AggFilter.status("com", "rec") == AggFilter.status("rec", "com")
        assert AggFilter.phase(3, 2) == AggFilter.phase(2, 3)

    def test_empty_multi_value_is_rejected(self):
        with pytest.raises(ValueError):
            AggFilter.phase()
        with pytest.raises(ValueError):
            AggFilter.status()


class TestParams:
    def test_page_size_is_always_explicit(self):
        """Omitting pageSize silently gives 10 records, not everything."""
        params = CTGovSearch(intr="Pembrolizumab").to_params(page_size=1000)
        assert params["pageSize"] == "1000"

    def test_fields_are_comma_joined_pascal_case(self):
        params = CTGovSearch(intr="X", fields=("NCTId", "Phase")).to_params(page_size=10)
        assert params["fields"] == "NCTId,Phase"

    def test_search_terms_map_to_verified_query_params(self):
        params = CTGovSearch(
            intr="Pembrolizumab", cond="melanoma", spons="Merck", locn="France"
        ).to_params(page_size=10)
        assert params["query.intr"] == "Pembrolizumab"
        assert params["query.cond"] == "melanoma"
        assert params["query.spons"] == "Merck"
        assert params["query.locn"] == "France"

    def test_agg_filters_comma_joined(self):
        search = CTGovSearch(
            intr="X", agg_filters=(AggFilter.phase(3), AggFilter.status("rec"))
        )
        assert search.to_params(page_size=10)["aggFilters"] == "phase:3,status:rec"

    def test_no_filter_phase_parameter_is_ever_emitted(self):
        """filter.phase returns HTTP 400 -- it must never appear."""
        search = CTGovSearch(intr="X", agg_filters=(AggFilter.phase(3),))
        assert not any(k.startswith("filter.") for k in search.to_params(page_size=10))

    def test_count_total_and_page_token_are_opt_in(self):
        search = CTGovSearch(intr="X")
        assert "countTotal" not in search.to_params(page_size=10)
        params = search.to_params(page_size=10, count_total=True, page_token="abc")
        assert params["countTotal"] == "true"
        assert params["pageToken"] == "abc"


class TestPagination:
    @respx.mock
    async def test_follows_page_tokens_until_exhausted(self):
        route = respx.get(f"{BASE}/studies")
        route.side_effect = [
            page([make_record("NCT00000001")], total=3, next_token="t1"),
            page([make_record("NCT00000002")], next_token="t2"),
            page([make_record("NCT00000003")]),
        ]
        outcome, store = await run(CTGovSearch(intr="X"), page_size=1)
        assert len(store) == 3
        assert outcome.total_count == 3
        assert outcome.truncated is False
        assert route.call_count == 3

    @respx.mock
    async def test_second_request_carries_the_page_token(self):
        route = respx.get(f"{BASE}/studies")
        route.side_effect = [
            page([make_record("NCT00000001")], total=2, next_token="TOKEN123"),
            page([make_record("NCT00000002")]),
        ]
        await run(CTGovSearch(intr="X"), page_size=1)
        assert "pageToken=TOKEN123" in str(route.calls[1].request.url)
        # countTotal is requested once, on the first page only.
        assert "countTotal" in str(route.calls[0].request.url)
        assert "countTotal" not in str(route.calls[1].request.url)

    @respx.mock
    async def test_fetch_cap_stops_paging_and_is_disclosed(self):
        respx.get(f"{BASE}/studies").mock(
            return_value=page(
                [make_record(f"NCT0000{i:04d}") for i in range(100)],
                total=5000,
                next_token="more",
            )
        )
        outcome, store = await run(CTGovSearch(intr="X"), max_studies=100, page_size=100)
        assert len(store) == 100
        assert outcome.truncated is True
        assert store.truncated is True
        assert any("5,000" in w and "100" in w for w in outcome.warnings)

    @respx.mock
    async def test_a_complete_sample_is_never_called_capped(self):
        """The upstream hands back a nextPageToken after the last record.

        That tripped the budget check on the next iteration and produced
        "Fetched 2 of 2 matching trials (capped at max_studies=2)" -- a claim
        of incompleteness about a provably whole sample. Every disclosure here
        is load-bearing, so a false one costs as much as a missing one.
        """
        respx.get(f"{BASE}/studies").mock(
            return_value=page(
                [make_record("NCT00000001"), make_record("NCT00000002")],
                total=2,
                next_token="more",
            )
        )
        outcome, store = await run(CTGovSearch(intr="X"), max_studies=2, page_size=2)
        assert len(store) == 2
        assert outcome.truncated is False
        assert store.truncated is False
        assert outcome.warnings == []

    @respx.mock
    async def test_an_unreported_total_does_not_crash_the_warning(self):
        """totalCount is optional upstream. Formatting None with :, raised
        TypeError, which is not a CTGovError -- so it escaped the route's
        handlers as an undocumented 500 instead of the documented 502."""
        respx.get(f"{BASE}/studies").mock(
            return_value=page(
                [make_record(f"NCT0000{i:04d}") for i in range(5)], next_token="more"
            )
        )
        outcome, store = await run(CTGovSearch(intr="X"), max_studies=5, page_size=5)
        assert outcome.truncated is True
        assert len(store) == 5
        assert any("not report a total" in w for w in outcome.warnings)

    @respx.mock
    async def test_a_repeated_page_token_stops_instead_of_cycling(self):
        """A token pointing back at a page already read re-fetched the same
        records until the budget ran out, then reported a capped sample when
        the whole result set had been in hand after the first page."""
        route = respx.get(f"{BASE}/studies")
        route.mock(
            return_value=page([make_record("NCT00000001")], total=1, next_token="LOOP")
        )
        outcome, store = await run(CTGovSearch(intr="X"), max_studies=50, page_size=1)
        assert route.call_count == 2
        assert len(store) == 1
        assert outcome.truncated is False

    @respx.mock
    async def test_the_warning_names_the_requested_cap_not_the_series_share(self):
        """fetch() splits max_studies across a comparison's series and passes
        the share down. Quoting the share told the user they were capped at a
        number they never set, and then advised them to raise it."""
        respx.get(f"{BASE}/studies").mock(
            return_value=page(
                [make_record(f"NCT0000{i:04d}") for i in range(500)],
                total=2922,
                next_token="more",
            )
        )
        outcome, _store = await run(
            CTGovSearch(intr="Pembrolizumab", label="Pembrolizumab"),
            max_studies=500,
            page_size=500,
            requested_max=1000,
        )
        warning = outcome.warnings[0]
        assert "max_studies=1,000" in warning
        assert "max_studies=500" not in warning
        assert "500-trial share" in warning
        assert "for Pembrolizumab" in warning

    @respx.mock
    async def test_duplicate_records_across_pages_are_deduplicated(self):
        route = respx.get(f"{BASE}/studies")
        route.side_effect = [
            page([make_record("NCT00000001")], total=2, next_token="t1"),
            page([make_record("NCT00000001")]),
        ]
        _, store = await run(CTGovSearch(intr="X"), page_size=1)
        assert len(store) == 1


class TestErrorHandling:
    @respx.mock
    async def test_retries_on_429_then_succeeds(self):
        route = respx.get(f"{BASE}/studies")
        route.side_effect = [
            httpx.Response(429),
            page([make_record("NCT00000001")], total=1),
        ]
        _, store = await run(CTGovSearch(intr="X"))
        assert len(store) == 1
        assert route.call_count == 2

    @respx.mock
    async def test_gives_up_after_max_retries(self):
        respx.get(f"{BASE}/studies").mock(return_value=httpx.Response(503))
        with pytest.raises(CTGovError, match="503"):
            await run(CTGovSearch(intr="X"))

    @respx.mock
    async def test_400_is_not_retried_and_surfaces_the_api_diagnostic(self):
        """A 400 body carries the API's own message -- this is how the
        nonexistent filter.phase parameter was identified."""
        route = respx.get(f"{BASE}/studies").mock(
            return_value=httpx.Response(400, text='{"error": "unknown parameter"}')
        )
        with pytest.raises(CTGovError, match="unknown parameter"):
            await run(CTGovSearch(intr="X"))
        assert route.call_count == 1


class TestEmptyFilteredResult:
    """A 200 with ``studies: []`` cannot be distinguished from a filter that
    silently matched nothing, so an empty filtered result is explained rather
    than reported as a bare zero."""

    @respx.mock
    async def test_empty_filtered_result_reports_the_unfiltered_count(self):
        route = respx.get(f"{BASE}/studies")
        route.side_effect = [
            page([], total=0),  # filtered search: nothing
            page([], total=2922),  # same search without aggFilters
        ]
        outcome, _ = await run(
            CTGovSearch(intr="Pembrolizumab", agg_filters=(AggFilter.phase(4),))
        )
        assert any("2,922" in w and "phase:4" in w for w in outcome.warnings)
        assert "aggFilters" not in str(route.calls[1].request.url)

    @respx.mock
    async def test_no_probe_when_the_search_was_unfiltered(self):
        route = respx.get(f"{BASE}/studies").mock(return_value=page([], total=0))
        outcome, _ = await run(CTGovSearch(intr="Nonexistent"))
        assert route.call_count == 1
        assert outcome.warnings == []

    @respx.mock
    async def test_no_probe_when_results_were_found(self):
        route = respx.get(f"{BASE}/studies").mock(
            return_value=page([make_record("NCT00000001")], total=1)
        )
        outcome, _ = await run(
            CTGovSearch(intr="X", agg_filters=(AggFilter.phase(3),))
        )
        assert route.call_count == 1
        assert outcome.warnings == []


class TestVersionEndpoint:
    @respx.mock
    async def test_returns_data_timestamp(self):
        respx.get(f"{BASE}/version").mock(
            return_value=httpx.Response(
                200, json={"apiVersion": "2.0.5", "dataTimestamp": "2026-08-07T09:00:05"}
            )
        )
        async with CTGovClient() as client:
            assert await client.get_data_timestamp() == "2026-08-07T09:00:05"

    @respx.mock
    async def test_failure_degrades_to_none_rather_than_failing_the_request(self):
        respx.get(f"{BASE}/version").mock(return_value=httpx.Response(500))
        async with CTGovClient(max_retries=0) as client:
            assert await client.get_data_timestamp() is None


class TestBuildSearches:
    """The query builder is pure: no network, so URL shape is unit-testable."""

    def plan(self, **kwargs):
        entities = ExtractedEntities(
            drugs=kwargs.pop("drugs", []),
            conditions=kwargs.pop("conditions", []),
            sponsors=kwargs.pop("sponsors", []),
            phases=kwargs.pop("phases", []),
            statuses=kwargs.pop("statuses", []),
            countries=kwargs.pop("countries", []),
            year_range=kwargs.pop("year_range", None),
        )
        return QueryPlan(
            query=kwargs.pop("query", "a question"),
            query_type=kwargs.pop("query_type", "distribution"),
            entities=entities,
            group_by=kwargs.pop("group_by", "phase"),
            viz_type=kwargs.pop("viz_type", "bar_chart"),
            **kwargs,
        )

    def test_drug_becomes_query_intr(self):
        searches, _ = build_searches(self.plan(drugs=["Pembrolizumab"]))
        assert len(searches) == 1
        assert searches[0].intr == "Pembrolizumab"

    def test_phase_entity_becomes_a_bare_number_agg_filter(self):
        searches, _ = build_searches(self.plan(drugs=["X"], phases=[3]))
        assert searches[0].agg_filters[0].render() == "phase:3"

    def test_recruiting_status_becomes_the_one_verified_status_filter(self):
        plan = self.plan(conditions=["melanoma"], statuses=["recruiting"])
        searches, _ = build_searches(plan)
        assert "status:rec" in [f.render() for f in searches[0].agg_filters]

    def test_unverified_status_is_not_sent_upstream(self):
        """AVAILABLE has no live-verified code, so it is filtered client-side."""
        plan = self.plan(conditions=["melanoma"], statuses=["AVAILABLE"])
        searches, _ = build_searches(plan)
        assert searches[0].agg_filters == ()

    def test_comparison_produces_one_search_per_entity(self):
        searches, _ = build_searches(
            self.plan(
                query_type="comparison",
                viz_type="grouped_bar_chart",
                compare_entities=["Pembrolizumab", "Nivolumab"],
                compare_entity_kind="drug",
            )
        )
        assert [s.label for s in searches] == ["Pembrolizumab", "Nivolumab"]
        assert [s.intr for s in searches] == ["Pembrolizumab", "Nivolumab"]

    def test_comparison_keeps_shared_filters_on_every_series(self):
        searches, _ = build_searches(
            self.plan(
                query_type="comparison",
                viz_type="grouped_bar_chart",
                conditions=["melanoma"],
                compare_entities=["A", "B"],
                compare_entity_kind="drug",
            )
        )
        assert all(s.cond == "melanoma" for s in searches)

    def test_falls_back_to_free_text_when_no_entity_was_extracted(self):
        searches, _ = build_searches(self.plan(query="how many trials are there"))
        assert searches[0].term == "how many trials are there"


class TestMultipleExtractedValues:
    """Extra extracted values used to be dropped on the floor. ``OR`` is a
    verified set union in ``query.*`` (2,922 + 2,016 - 290 = 4,648), so they
    are unioned instead -- and the reading is disclosed."""

    plan = TestBuildSearches.plan

    def test_multiple_drugs_are_unioned_into_one_expression(self):
        searches, notes = build_searches(
            self.plan(drugs=["Pembrolizumab", "Nivolumab"])
        )
        assert len(searches) == 1
        assert searches[0].intr == "Pembrolizumab OR Nivolumab"
        assert any("either" in n or "any of them" in n for n in notes)

    def test_multiple_conditions_are_unioned(self):
        searches, _ = build_searches(
            self.plan(conditions=["melanoma", "lung cancer"])
        )
        assert searches[0].cond == "melanoma OR lung cancer"

    def test_a_single_value_is_unchanged_and_gains_no_note(self):
        searches, notes = build_searches(self.plan(drugs=["Pembrolizumab"]))
        assert searches[0].intr == "Pembrolizumab"
        assert notes == []

    def test_every_extracted_value_survives_into_the_query(self):
        """The regression itself: nothing extracted is silently discarded."""
        searches, _ = build_searches(
            self.plan(
                drugs=["Pembrolizumab", "Nivolumab"],
                conditions=["melanoma", "lung cancer"],
                sponsors=["Merck", "BMS"],
                countries=["France", "Japan"],
            )
        )
        params = searches[0].to_params(page_size=10)
        blob = " ".join(params.values())
        for value in ("Nivolumab", "lung cancer", "BMS", "Japan"):
            assert value in blob

    def test_comparison_still_gets_one_search_per_entity(self):
        """The union must not leak into the comparison path, where per-series
        membership is what makes the grouping exact."""
        searches, _ = build_searches(
            self.plan(
                query_type="comparison",
                viz_type="grouped_bar_chart",
                compare_entities=["Pembrolizumab", "Nivolumab"],
                compare_entity_kind="drug",
            )
        )
        assert len(searches) == 2
        assert all(" OR " not in (s.intr or "") for s in searches)

    def test_comparison_does_not_disclose_a_union_it_never_applied(self):
        """The compared field is replaced per series, so a note about how its
        values were combined would describe an interpretation never used."""
        searches, notes = build_searches(
            self.plan(
                query_type="comparison",
                viz_type="grouped_bar_chart",
                drugs=["Pembrolizumab", "Nivolumab"],
                compare_entities=["Pembrolizumab", "Nivolumab"],
                compare_entity_kind="drug",
            )
        )
        assert [s.intr for s in searches] == ["Pembrolizumab", "Nivolumab"]
        assert notes == []

    def test_comparison_still_discloses_unions_on_untouched_fields(self):
        """Only the compared field is exempt -- a shared filter that really was
        unioned applies to every series and must still be disclosed."""
        searches, notes = build_searches(
            self.plan(
                query_type="comparison",
                viz_type="grouped_bar_chart",
                conditions=["melanoma", "lung cancer"],
                compare_entities=["Pembrolizumab", "Nivolumab"],
                compare_entity_kind="drug",
            )
        )
        assert all(s.cond == "melanoma OR lung cancer" for s in searches)
        assert any("condition values" in n for n in notes)

    def test_over_cap_truncation_is_disclosed(self):
        drugs = [f"Drug{i}" for i in range(8)]
        searches, notes = build_searches(self.plan(drugs=drugs))
        assert searches[0].intr.count(" OR ") == 4  # 5 values kept
        assert "Drug5" not in searches[0].intr
        assert any("first 5 of 8" in n for n in notes)

    def test_a_value_that_is_already_an_expression_is_not_nested(self):
        searches, notes = build_searches(
            self.plan(drugs=["Pembrolizumab OR Nivolumab", "Atezolizumab"])
        )
        assert searches[0].intr == "Pembrolizumab OR Nivolumab"
        assert any("search operator" in n for n in notes)

    @pytest.mark.parametrize(
        "field,attr,values,expected",
        [
            ("conditions", "cond", ["Head and Neck Cancer", "melanoma"],
             "Head and Neck Cancer OR melanoma"),
            ("sponsors", "spons", ["Merck Sharp & Dohme, LLC", "Pfizer"],
             "Merck Sharp & Dohme, LLC OR Pfizer"),
            ("conditions", "cond", ["Ear, Nose and Throat Diseases", "melanoma"],
             "Ear, Nose and Throat Diseases OR melanoma"),
        ],
    )
    def test_a_connector_inside_a_name_does_not_suppress_the_join(
        self, field, attr, values, expected
    ):
        """The marker check was case-insensitive and counted commas, so an
        ordinary name read as an expression and every other value was dropped.

        Verified live that the join is exact in both directions:
        query.cond="melanoma OR Head and Neck Cancer" returns 11,953 =
        3,743 + 8,637 - 427, and query.spons="Merck Sharp & Dohme, LLC OR
        Pfizer" returns 10,260 = 4,276 + 6,061 - 77.
        """
        searches, notes = build_searches(self.plan(**{field: values}))
        assert getattr(searches[0], attr) == expected
        assert not any("search operator" in n for n in notes)


class TestStatusAggFilter:
    """Verified statuses are unioned in one aggFilters clause. An earlier
    version applied status:rec whenever *any* requested status was recruiting
    and then filtered client-side for the other, turning a union into an
    intersection: the fetch returned recruiting trials only, and the local pass
    kept just the other status."""

    plan = TestBuildSearches.plan

    def agg_filters(self, statuses):
        searches, _ = build_searches(self.plan(drugs=["X"], statuses=statuses))
        return searches[0].to_params(page_size=10).get("aggFilters")

    def test_sole_recruiting_filters_upstream(self):
        assert self.agg_filters(["RECRUITING"]) == "status:rec"

    @pytest.mark.parametrize(
        "statuses,expected",
        [
            (["RECRUITING", "COMPLETED"], "status:com rec"),
            (["RECRUITING", "TERMINATED"], "status:rec ter"),
            (["ACTIVE_NOT_RECRUITING", "RECRUITING"], "status:act rec"),
        ],
    )
    def test_mixed_statuses_union_upstream(self, statuses, expected):
        """Values inside one aggFilters key union, verified live, so a mixed
        request is expressed in a single clause instead of being fetched broad
        and narrowed afterwards."""
        assert self.agg_filters(statuses) == expected

    def test_any_verified_status_filters_upstream(self):
        assert self.agg_filters(["COMPLETED"]) == "status:com"

    def test_an_unverified_status_keeps_the_whole_request_client_side(self):
        """A partial upstream filter would silently drop AVAILABLE, since its
        code returns zero results rather than an error."""
        assert self.agg_filters(["COMPLETED", "AVAILABLE"]) is None

    @pytest.mark.parametrize(
        "statuses",
        [["recruiting"], ["  RECRUITING  "], ["Recruiting"], ["RECRUITING", "RECRUITING"]],
    )
    def test_normalisation_is_consistent(self, statuses):
        """A padded or oddly-cased value used to fail the builder's startswith
        test and then be discarded client-side too, so no filter applied
        anywhere. Duplicates collapse to the same sole-status case."""
        assert self.agg_filters(statuses) == "status:rec"

    def test_empty_status_list_filters_nothing(self):
        assert self.agg_filters([]) is None


class TestNormalizeStatuses:
    def test_trims_uppercases_and_collapses_duplicates(self):
        assert normalize_statuses(["  recruiting ", "RECRUITING"]) == {"RECRUITING"}

    def test_spaces_become_underscores(self):
        assert normalize_statuses(["active not recruiting"]) == {
            "ACTIVE_NOT_RECRUITING"
        }

    def test_blank_entries_are_dropped(self):
        assert normalize_statuses(["", "   ", "COMPLETED"]) == {"COMPLETED"}

    def test_empty_input_gives_an_empty_set(self):
        assert normalize_statuses([]) == set()


class TestComparisonLabelUniqueness:
    """Labels become dict keys in fetch(), so a duplicate would overwrite an
    earlier series. ground_compare_entities collapses equivalents first; this
    keeps the guarantee local to where the keys are minted, so it survives a
    caller that skips grounding."""

    plan = TestBuildSearches.plan

    def comparison(self, entities):
        return self.plan(
            query_type="comparison",
            viz_type="grouped_bar_chart",
            compare_entities=entities,
            compare_entity_kind="drug",
        )

    @pytest.mark.parametrize(
        "entities",
        [
            ["Aspirin", "Aspirin"],
            ["Aspirin", "aspirin"],
            ["  Aspirin ", "ASPIRIN"],
            ["Aspirin", "Aspirin", "aspirin"],
        ],
    )
    def test_duplicate_entities_never_produce_duplicate_labels(self, entities):
        searches, notes = build_searches(self.comparison(entities))
        labels = [s.label for s in searches]
        assert len(labels) == len(set(labels))
        assert len(searches) == 1
        # Never silent: the drop is disclosed through the existing notes channel.
        assert any("duplicate" in n for n in notes)

    def test_distinct_entities_all_survive(self):
        searches, notes = build_searches(
            self.comparison(["Pembrolizumab", "Nivolumab", "Atezolizumab"])
        )
        assert [s.label for s in searches] == [
            "Pembrolizumab",
            "Nivolumab",
            "Atezolizumab",
        ]
        assert not any("duplicate" in n for n in notes)

    def test_first_occurrence_wins_and_order_is_deterministic(self):
        entities = ["Nivolumab", "Pembrolizumab", "nivolumab"]
        for _ in range(5):
            searches, _ = build_searches(self.comparison(entities))
            assert [s.label for s in searches] == ["Nivolumab", "Pembrolizumab"]

    def test_a_duplicate_does_not_cost_an_upstream_search(self):
        """The collision used to be resolved after both searches had run."""
        searches, _ = build_searches(self.comparison(["Aspirin", "aspirin"]))
        assert len(searches) == 1


class TestSeriesKey:
    def test_collapses_case_and_surrounding_whitespace(self):
        assert series_key("  Aspirin  ") == series_key("aspirin") == "aspirin"

    def test_collapses_internal_whitespace(self):
        assert series_key("Drug  A") == series_key("Drug A")

    def test_distinct_names_keep_distinct_keys(self):
        assert series_key("Aspirin") != series_key("Ibuprofen")


class TestPopulationScopeVsPredicates:
    """A question has three separable parts and only two describe the
    population: scope (drug/condition/sponsor/country), predicates (phase,
    status, years), and the analytical instruction. Using the whole question as
    query.term whenever no scope was named ANDed the analytical wording into
    the retrieval predicate -- verified live, aggFilters=phase:3 alone returns
    49,614 trials and 0 with the question text attached."""

    plan = TestBuildSearches.plan

    def params(self, **kwargs):
        searches, _ = build_searches(self.plan(**kwargs))
        return {
            k: v
            for k, v in searches[0].to_params(page_size=10).items()
            if k not in ("fields", "pageSize")
        }

    def test_phase_only_query_sends_the_filter_and_no_free_text(self):
        params = self.params(
            query="How are Phase 3 trials distributed across sponsors?",
            phases=[3],
            group_by="sponsor",
        )
        assert params == {"aggFilters": "phase:3"}

    def test_multi_phase_only_query_sends_the_union_and_no_free_text(self):
        params = self.params(query="Phase 2 and 3 trials by sponsor", phases=[2, 3])
        assert params == {"aggFilters": "phase:2 3"}

    def test_status_only_query_sends_the_filter_and_no_free_text(self):
        params = self.params(
            query="Which countries have the most recruiting trials?",
            statuses=["RECRUITING"],
            group_by="country",
        )
        assert params == {"aggFilters": "status:rec"}

    def test_completed_only_query_sends_the_filter_and_no_free_text(self):
        params = self.params(
            query="How are completed trials distributed across phases?",
            statuses=["COMPLETED"],
        )
        assert params == {"aggFilters": "status:com"}

    def test_year_only_query_sends_no_free_text(self):
        """No verified date parameter exists, so the years are enforced
        client-side -- but the question must still not become a search term."""
        params = self.params(
            query="How many trials started each year from 2020 to 2024?",
            year_range=YearRange(start=2020, end=2024),
            query_type="time_trend",
            group_by="start_year",
        )
        assert params == {}

    def test_phase_and_status_query_sends_both_filters(self):
        params = self.params(
            query="Phase 3 recruiting trials by sponsor",
            phases=[3],
            statuses=["RECRUITING"],
            group_by="sponsor",
        )
        assert params["aggFilters"] == "phase:3,status:rec"
        assert "query.term" not in params

    def test_a_named_scope_is_still_used(self):
        params = self.params(query="pembrolizumab trials by phase", drugs=["Pembrolizumab"])
        assert params == {"query.intr": "Pembrolizumab"}

    def test_free_text_fallback_survives_for_a_query_with_nothing_structured(self):
        """The fallback is for questions carrying neither scope nor predicate --
        it must not be removed, only stopped from firing over a filter."""
        params = self.params(query="trials about long covid fatigue")
        assert params == {"query.term": "trials about long covid fatigue"}


class TestExclusionCountsAsAPredicate:
    """A status exclusion narrows the population exactly as a requested status
    does; it is simply enforced after fetching. Omitting it from the predicate
    test sent exclusion-only questions down the free-text fallback, which ANDs
    the analytical wording into retrieval."""

    plan = TestBuildSearches.plan

    def params(self, **kwargs):
        searches, _ = build_searches(self.plan(**kwargs))
        return {
            k: v
            for k, v in searches[0].to_params(page_size=10).items()
            if k not in ("fields", "pageSize")
        }

    @pytest.mark.parametrize(
        "query,excluded",
        [
            ("Show trials that are not recruiting", ["RECRUITING"]),
            ("Show trials that are not recruiting by phase", ["RECRUITING"]),
            ("Show trials that are not completed", ["COMPLETED"]),
        ],
    )
    def test_an_exclusion_only_query_sends_no_free_text(self, query, excluded):
        assert self.params(query=query, excluded_statuses=excluded) == {}

    def test_exclusion_plus_year_sends_no_free_text(self):
        assert self.params(
            query="trials not recruiting since 2020",
            excluded_statuses=["RECRUITING"],
            year_range=YearRange(start=2020, end=None),
        ) == {}

    def test_exclusion_plus_phase_keeps_the_upstream_filter(self):
        params = self.params(
            query="phase 3 trials not recruiting",
            phases=[3], excluded_statuses=["RECRUITING"],
        )
        assert params == {"aggFilters": "phase:3"}

    def test_a_positive_and_negative_status_together(self):
        params = self.params(
            query="completed trials that are not recruiting",
            statuses=["COMPLETED"], excluded_statuses=["RECRUITING"],
        )
        assert params == {"aggFilters": "status:com"}

    def test_analytical_words_never_become_the_search_topic(self):
        for query in (
            "Show trials that are not recruiting by phase",
            "How are trials that are not completed distributed across sponsors?",
        ):
            params = self.params(query=query, excluded_statuses=["RECRUITING"])
            assert "query.term" not in params

    def test_the_free_text_fallback_still_fires_with_nothing_structured(self):
        assert self.params(query="trials about long covid fatigue") == {
            "query.term": "trials about long covid fatigue"
        }
