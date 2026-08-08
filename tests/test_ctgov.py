"""CTGov client: parameter construction, pagination, retry, and the
empty-filtered-result diagnostic."""

import httpx
import pytest
import respx

from app.models.schemas import ExtractedEntities, QueryPlan
from app.services.ctgov import (
    AggFilter,
    CTGovClient,
    CTGovError,
    CTGovSearch,
    build_searches,
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


async def run(search, *, max_studies=3000, page_size=1000):
    store = StudyStore()
    async with CTGovClient(page_size=page_size, max_retries=1) as client:
        outcome = await client.run_search(search, store, max_studies=max_studies)
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
        with pytest.raises(ValueError, match="not live-verified"):
            AggFilter.status("com")


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
            year_range=None,
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
        searches = build_searches(self.plan(drugs=["Pembrolizumab"]))
        assert len(searches) == 1
        assert searches[0].intr == "Pembrolizumab"

    def test_phase_entity_becomes_a_bare_number_agg_filter(self):
        searches = build_searches(self.plan(drugs=["X"], phases=[3]))
        assert searches[0].agg_filters[0].render() == "phase:3"

    def test_recruiting_status_becomes_the_one_verified_status_filter(self):
        searches = build_searches(self.plan(conditions=["melanoma"], statuses=["recruiting"]))
        assert "status:rec" in [f.render() for f in searches[0].agg_filters]

    def test_unverified_status_is_not_sent_upstream(self):
        searches = build_searches(self.plan(conditions=["melanoma"], statuses=["completed"]))
        assert searches[0].agg_filters == ()

    def test_comparison_produces_one_search_per_entity(self):
        searches = build_searches(
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
        searches = build_searches(
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
        searches = build_searches(self.plan(query="how many trials are there"))
        assert searches[0].term == "how many trials are there"
