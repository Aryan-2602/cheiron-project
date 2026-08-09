"""End-to-end pipeline and HTTP behaviour, with the LLM and upstream API mocked.

These tests run the real pipeline -- fetch, aggregate, format, validate -- so
they cover the wiring between stages that unit tests cannot.
"""

from typing import ClassVar

import httpx
import pytest
import respx
from fastapi.testclient import TestClient

from app.core.config import settings
from app.main import app
from app.models.schemas import ExtractedEntities, Meta, QueryPlan, QueryRequest
from app.pipeline import (
    EmptyResultError,
    UnsupportedQueryError,
    apply_client_side_filters,
    fetch,
    run_pipeline,
)
from app.services.ctgov import CTGovClient, CTGovError, build_searches
from app.services.store import StudyStore
from tests.conftest import make_record

BASE = "https://clinicaltrials.gov/api/v2"
RXNORM = "https://rxnav.nlm.nih.gov/REST"


def plan(**kwargs):
    entities = ExtractedEntities(
        drugs=kwargs.pop("drugs", ["Pembrolizumab"]),
        conditions=kwargs.pop("conditions", []),
        sponsors=[],
        phases=kwargs.pop("phases", []),
        statuses=kwargs.pop("statuses", []),
        countries=[],
        year_range=kwargs.pop("year_range", None),
    )
    return QueryPlan(
        query=kwargs.pop("query", "How are trials distributed across phases?"),
        query_type=kwargs.pop("query_type", "distribution"),
        entities=entities,
        group_by=kwargs.pop("group_by", "phase"),
        viz_type=kwargs.pop("viz_type", "bar_chart"),
        **kwargs,
    )


def mock_studies(studies, *, total=None):
    respx.get(f"{BASE}/version").mock(
        return_value=httpx.Response(200, json={"dataTimestamp": "2026-08-07T09:00:05"})
    )
    respx.get(f"{BASE}/studies").mock(
        return_value=httpx.Response(
            200, json={"studies": studies, "totalCount": total or len(studies)}
        )
    )


SAMPLE = [
    make_record("NCT00000001", phases=["PHASE3"], start_date="2020-01-01",
                interventions=[("DRUG", "Pembrolizumab"), ("DRUG", "Carboplatin")]),
    make_record("NCT00000002", phases=["PHASE3"], start_date="2021-01-01",
                interventions=[("DRUG", "Pembrolizumab"), ("DRUG", "Carboplatin")]),
    make_record("NCT00000003", phases=["PHASE1", "PHASE2"], start_date="2022-01-01",
                interventions=[("DRUG", "Pembrolizumab")]),
    make_record("NCT00000004", phases=None, start_date=None, sponsor="Beta Institute"),
]


async def run(**kwargs):
    request = QueryRequest(query=kwargs.pop("query", "a question"), **kwargs.pop("request", {}))
    async with CTGovClient(max_retries=0) as client:
        return await run_pipeline(request, client=client, plan=plan(**kwargs))


class TestBarChart:
    @respx.mock
    async def test_end_to_end(self):
        mock_studies(SAMPLE)
        response = await run()
        viz = response.visualization
        assert viz.type == "bar_chart"
        assert viz.encoding.x.field == "phase"
        assert viz.encoding.y.field == "trial_count"
        counts = {r["phase"]: r["trial_count"] for r in viz.data}
        assert counts == {
            "Phase 1": 1, "Phase 2": 1, "Phase 3": 2, "No phase data": 1
        }

    @respx.mock
    async def test_every_row_carries_verifiable_citations(self):
        mock_studies(SAMPLE)
        response = await run()
        for row in response.visualization.data:
            assert row["citations"], f"row {row['phase']} has no citations"
            assert row["total_supporting_trials"] == row["trial_count"]
            for citation in row["citations"]:
                assert citation["nct_id"].startswith("NCT")
                assert citation["excerpt"].strip()
                assert citation["url"].endswith(citation["nct_id"])

    @respx.mock
    async def test_meta_records_provenance(self):
        mock_studies(SAMPLE, total=99)
        response = await run()
        meta = response.meta
        assert meta.source == "clinicaltrials.gov"
        assert meta.data_as_of == "2026-08-07T09:00:05"
        assert meta.total_studies_processed == 4
        assert meta.api_urls and all(u.startswith(BASE) for u in meta.api_urls)
        assert meta.filters["intervention"] == "Pembrolizumab"

    @respx.mock
    async def test_multi_valued_axis_is_disclosed(self):
        mock_studies(SAMPLE)
        response = await run()
        assert any("more than one bucket" in w for w in response.meta.warnings)

    @respx.mock
    async def test_missing_data_is_disclosed(self):
        mock_studies(SAMPLE)
        response = await run()
        assert any("had no value for this axis" in w for w in response.meta.warnings)


class TestTimeSeries:
    @respx.mock
    async def test_zero_fills_and_omits_undated_trials_from_a_temporal_axis(self):
        mock_studies(SAMPLE)
        response = await run(
            query_type="time_trend", group_by="start_year", viz_type="time_series"
        )
        keys = [r["start_year"] for r in response.visualization.data]
        assert keys == ["2020", "2021", "2022"]
        assert response.meta.time_granularity == "year"
        # The undated trial is not charted, but it is reported.
        assert any("no value for this axis" in w for w in response.meta.warnings)

    @respx.mock
    async def test_year_range_is_applied_client_side_and_disclosed(self):
        from app.models.schemas import YearRange

        mock_studies(SAMPLE)
        response = await run(
            query_type="time_trend",
            group_by="start_year",
            viz_type="time_series",
            year_range=YearRange(start=2021, end=None),
        )
        assert [r["start_year"] for r in response.visualization.data] == ["2021", "2022"]
        assert any("client-side" in w for w in response.meta.warnings)


class TestComparison:
    @respx.mock
    async def test_series_come_from_separate_searches(self):
        respx.get(f"{BASE}/version").mock(
            return_value=httpx.Response(200, json={"dataTimestamp": "2026-08-07T09:00:05"})
        )
        route = respx.get(f"{BASE}/studies")
        route.side_effect = [
            httpx.Response(200, json={"studies": SAMPLE[:2], "totalCount": 2}),
            httpx.Response(200, json={"studies": SAMPLE[2:], "totalCount": 2}),
        ]
        response = await run(
            query_type="comparison",
            viz_type="grouped_bar_chart",
            compare_entities=["Pembrolizumab", "Nivolumab"],
            compare_entity_kind="drug",
        )
        assert response.visualization.encoding.color.field == "series"
        by_series = {}
        for row in response.visualization.data:
            by_series.setdefault(row["series"], {})[row["phase"]] = row["trial_count"]
        assert by_series["Pembrolizumab"] == {"Phase 3": 2}
        assert by_series["Nivolumab"] == {"Phase 1": 1, "Phase 2": 1, "No phase data": 1}


def mock_rxnorm_unmatched():
    """RxNorm reachable but recognising nothing -- graphs behave as before."""
    respx.get(f"{RXNORM}/approximateTerm.json").mock(
        return_value=httpx.Response(200, json={"approximateGroup": {"inputTerm": "x"}})
    )


def mock_rxnorm_merging():
    """Term-aware RxNorm mock: brand and generic share an ingredient, and every
    other drug keeps its own identity.

    A blanket mock that maps everything to one ingredient would collapse the
    whole graph into a single node -- which says nothing about merging.
    """
    ingredients = {
        "keytruda": ("1547550", "1547545", "pembrolizumab"),
        "pembrolizumab": ("1547545", "1547545", "pembrolizumab"),
        "carboplatin": ("40048", "40048", "carboplatin"),
    }

    def approx(request):
        term = (request.url.params.get("term") or "").lower()
        entry = ingredients.get(term)
        if entry is None:
            return httpx.Response(200, json={"approximateGroup": {"inputTerm": term}})
        return httpx.Response(
            200,
            json={"approximateGroup": {"candidate": [{"rxcui": entry[0], "score": "14.2"}]}},
        )

    def related(request):
        concept = str(request.url).split("/rxcui/")[1].split("/")[0]
        entry = next((v for v in ingredients.values() if v[0] == concept), None)
        if entry is None:
            return httpx.Response(200, json={"relatedGroup": {"conceptGroup": []}})
        return httpx.Response(
            200,
            json={
                "relatedGroup": {
                    "conceptGroup": [
                        {
                            "tty": "IN",
                            "conceptProperties": [
                                {"rxcui": entry[1], "name": entry[2]}
                            ],
                        }
                    ]
                }
            },
        )

    respx.get(f"{RXNORM}/approximateTerm.json").mock(side_effect=approx)
    respx.get(url__regex=rf"{RXNORM}/rxcui/\d+/related\.json").mock(side_effect=related)


#: Two trials naming the same compound differently -- the case merging exists for.
BRAND_AND_GENERIC = [
    make_record("NCT00000001", interventions=[("DRUG", "Keytruda"), ("DRUG", "Carboplatin")]),
    make_record("NCT00000002", interventions=[("DRUG", "Pembrolizumab"), ("DRUG", "Carboplatin")]),
]


class TestNetwork:
    @respx.mock
    async def test_network_end_to_end(self):
        mock_studies(SAMPLE)
        mock_rxnorm_unmatched()
        response = await run(
            query_type="relationship", viz_type="network_graph",
            network_kind="drug_drug", group_by=None,
        )
        viz = response.visualization
        assert viz.type == "network_graph"
        assert viz.encoding.nodes and viz.encoding.edges
        nodes, edges = viz.data[0]["nodes"], viz.data[0]["edges"]
        labels = {n["label"] for n in nodes}
        assert labels == {"Pembrolizumab", "Carboplatin"}
        assert len(edges) == 1
        assert edges[0]["weight"] == 2
        # The edge's citations must show both endpoints.
        assert edges[0]["citations"]
        for citation in edges[0]["citations"]:
            assert "Pembrolizumab" in citation["excerpt"]
            assert "Carboplatin" in citation["excerpt"]

    @respx.mock
    async def test_rxnorm_outage_degrades_with_a_warning(self):
        """RxNorm is an enrichment: losing it must not fail the request."""
        mock_studies(SAMPLE)
        respx.get(f"{RXNORM}/approximateTerm.json").mock(
            return_value=httpx.Response(503)
        )
        response = await run(
            query_type="relationship", viz_type="network_graph",
            network_kind="drug_drug", group_by=None,
        )
        assert response.visualization.data[0]["nodes"]
        assert any(
            "Drug synonym resolution unavailable" in w
            for w in response.meta.warnings
        )

    @respx.mock
    async def test_merged_nodes_are_disclosed_and_carry_provenance(self):
        mock_studies(BRAND_AND_GENERIC)
        mock_rxnorm_merging()
        response = await run(
            query_type="relationship", viz_type="network_graph",
            network_kind="drug_drug", group_by=None,
        )
        nodes = response.visualization.data[0]["nodes"]
        merged = [n for n in nodes if n.get("merged_from")]
        assert merged, "Keytruda and Pembrolizumab should merge"
        node = merged[0]
        assert node["rxcui"] == "1547545"
        assert node["label"] == "pembrolizumab"
        assert sorted(node["merged_from"]) == ["Keytruda", "Pembrolizumab"]
        # Distinct drugs keep their own node.
        assert any(n["label"].lower() == "carboplatin" for n in nodes)
        assert any("RxNorm" in w for w in response.meta.warnings)

    @respx.mock
    async def test_merged_node_citations_span_both_source_names(self):
        """The union requirement: the brand trial and the generic trial must
        both be citable under the merged node."""
        mock_studies(BRAND_AND_GENERIC)
        mock_rxnorm_merging()
        response = await run(
            query_type="relationship", viz_type="network_graph",
            network_kind="drug_drug", group_by=None,
        )
        node = next(
            n for n in response.visualization.data[0]["nodes"] if n.get("merged_from")
        )
        assert node["total_supporting_trials"] == 2
        assert {c["nct_id"] for c in node["citations"]} == {
            "NCT00000001",
            "NCT00000002",
        }

    @respx.mock
    async def test_chart_queries_never_call_rxnorm(self):
        """Resolution only affects graph identity, so charts must not pay for it."""
        mock_studies(SAMPLE)
        route = respx.get(f"{RXNORM}/approximateTerm.json").mock(
            return_value=httpx.Response(200, json={"approximateGroup": {}})
        )
        await run()
        assert route.call_count == 0


class TestErrorPaths:
    @respx.mock
    async def test_no_matching_trials_raises_with_usable_meta(self):
        mock_studies([], total=0)
        with pytest.raises(EmptyResultError) as excinfo:
            await run()
        assert excinfo.value.meta.api_urls

    async def test_unsupported_query_is_refused_before_any_fetch(self):
        with pytest.raises(UnsupportedQueryError):
            await run(query_type="unsupported")

    @respx.mock
    async def test_upstream_failure_propagates(self):
        respx.get(f"{BASE}/version").mock(return_value=httpx.Response(200, json={}))
        respx.get(f"{BASE}/studies").mock(return_value=httpx.Response(500))
        with pytest.raises(CTGovError):
            await run()


class TestHttpApi:
    """Route-level behaviour, with the pipeline itself stubbed."""

    def client(self):
        return TestClient(app)

    def test_health(self):
        assert self.client().get("/health").json() == {"status": "ok"}

    def test_rejects_a_malformed_request_body(self):
        assert self.client().post("/api/v1/query", json={}).status_code == 422
        assert self.client().post("/api/v1/query", json={"query": "hi"}).status_code == 422

    def test_empty_result_is_a_200_with_an_explanation(self, monkeypatch):
        """'No trials match' is a correct answer to a well-formed question."""
        from app.models.schemas import Meta

        async def fake(*args, **kwargs):
            raise EmptyResultError("none", Meta(api_urls=["https://example.test"]))

        monkeypatch.setattr("app.api.routes.run_pipeline", fake)
        response = self.client().post("/api/v1/query", json={"query": "a question"})
        assert response.status_code == 200
        body = response.json()
        assert body["visualization"]["data"] == []
        assert any("No trials matched" in w for w in body["meta"]["warnings"])

    def test_unsupported_query_is_400_with_a_typed_code(self, monkeypatch):
        async def fake(*args, **kwargs):
            raise UnsupportedQueryError("not answerable from registry data")

        monkeypatch.setattr("app.api.routes.run_pipeline", fake)
        response = self.client().post("/api/v1/query", json={"query": "what is love"})
        # 400, not 422: FastAPI owns 422 for request-schema violations.
        assert response.status_code == 400
        assert response.json()["error"]["code"] == "UNSUPPORTED_QUERY"

    def test_upstream_failure_is_502(self, monkeypatch):
        async def fake(*args, **kwargs):
            raise CTGovError("boom")

        monkeypatch.setattr("app.api.routes.run_pipeline", fake)
        response = self.client().post("/api/v1/query", json={"query": "a question"})
        assert response.status_code == 502
        assert response.json()["error"]["code"] == "UPSTREAM_ERROR"

    def test_a_response_failing_validation_is_withheld(self, monkeypatch):
        """A chart that cannot be traced to source data must not render."""
        from app.services.validate import ValidationFailure

        async def fake(*args, **kwargs):
            raise ValidationFailure("citation references an unknown trial")

        monkeypatch.setattr("app.api.routes.run_pipeline", fake)
        response = self.client().post("/api/v1/query", json={"query": "a question"})
        assert response.status_code == 500
        assert response.json()["error"]["code"] == "VALIDATION_ERROR"


class TestCitationLimits:
    """max_citations_per_datum is documented as 0-20, and 0 is legitimate --
    a caller wanting counts without the citation payload."""

    @respx.mock
    async def test_zero_citations_is_a_valid_request(self):
        mock_studies(SAMPLE)
        response = await run(max_citations_per_datum=0)
        rows = response.visualization.data
        assert rows, "should still return data"
        for row in rows:
            assert row["citations"] == []
            # The evidence count is still reported -- only the payload is omitted.
            assert row["total_supporting_trials"] == row["trial_count"]

    @respx.mock
    async def test_zero_citations_on_a_network_query(self):
        mock_studies(SAMPLE)
        mock_rxnorm_unmatched()
        response = await run(
            query_type="relationship", viz_type="network_graph",
            network_kind="drug_drug", group_by=None,
            max_citations_per_datum=0,
        )
        row = response.visualization.data[0]
        assert row["nodes"]
        for item in row["nodes"] + row["edges"]:
            assert item["citations"] == []
            assert item["total_supporting_trials"] > 0

    @respx.mock
    async def test_a_nonzero_limit_still_requires_citations(self):
        """Suppressing citations must be opt-in, not a hole in the validator."""
        mock_studies(SAMPLE)
        response = await run(max_citations_per_datum=2)
        assert all(r["citations"] for r in response.visualization.data)


#: Phases 1-4 plus a combined Phase 1/2 trial, which must match a request for
#: either 1 or 2.
PHASE_MIX = [
    make_record("NCT00000001", phases=["PHASE1"]),
    make_record("NCT00000002", phases=["PHASE2"]),
    make_record("NCT00000003", phases=["PHASE3"]),
    make_record("NCT00000004", phases=["PHASE4"]),
    make_record("NCT00000005", phases=["PHASE1", "PHASE2"]),
]


class TestPhaseFiltering:
    """Values inside one aggFilters key are space-separated and union (verified
    live: phase:2 89,652 + phase:3 49,614 -> "phase:2 3" 131,704, the overlap
    counted once). So every requested phase goes upstream in one clause."""

    def upstream_filter(self, route):
        from urllib.parse import parse_qs, urlparse

        return parse_qs(urlparse(str(route.calls[0].request.url)).query).get(
            "aggFilters", [None]
        )[0]

    def mock_route(self, studies):
        respx.get(f"{BASE}/version").mock(
            return_value=httpx.Response(200, json={"dataTimestamp": "2026-08-07"})
        )
        return respx.get(f"{BASE}/studies").mock(
            return_value=httpx.Response(
                200, json={"studies": studies, "totalCount": len(studies)}
            )
        )

    @respx.mock
    async def test_single_phase_is_filtered_upstream(self):
        route = self.mock_route([PHASE_MIX[2]])
        await run(phases=[3])
        assert self.upstream_filter(route) == "phase:3"

    @respx.mock
    async def test_multiple_phases_union_in_one_upstream_clause(self):
        """Previously fetched unfiltered and narrowed afterwards, which spent
        the fetch budget on trials that were about to be discarded."""
        route = self.mock_route(PHASE_MIX)
        await run(phases=[2, 3])
        assert self.upstream_filter(route) == "phase:2 3"

    @respx.mock
    async def test_phases_are_not_truncated_to_the_first(self):
        route = self.mock_route(PHASE_MIX)
        await run(phases=[2, 3])
        assert self.upstream_filter(route) == "phase:2 3"

    @respx.mock
    async def test_no_client_side_phase_narrowing_warning_is_emitted(self):
        """Nothing is narrowed after fetching any more, so claiming it would be
        untrue."""
        self.mock_route(PHASE_MIX)
        response = await run(phases=[2, 3])
        assert not any("phase" in w and "client-side" in w for w in response.meta.warnings)

    @respx.mock
    async def test_duplicate_and_unordered_phases_normalise(self):
        route = self.mock_route(PHASE_MIX)
        await run(phases=[3, 2, 3])
        assert self.upstream_filter(route) == "phase:2 3"


class TestFetchCap:
    """max_studies is documented as an upper bound on trials fetched, so it has
    to hold across every search a query issues, not just each one."""

    def two_batches(self):
        respx.get(f"{BASE}/version").mock(
            return_value=httpx.Response(200, json={"dataTimestamp": "2026-08-07"})
        )
        first = [make_record(f"NCT{i:08d}", phases=["PHASE3"]) for i in range(100)]
        second = [make_record(f"NCT{i:08d}", phases=["PHASE2"]) for i in range(100, 200)]
        route = respx.get(f"{BASE}/studies")
        route.side_effect = [
            httpx.Response(200, json={"studies": first, "totalCount": 5000}),
            httpx.Response(200, json={"studies": second, "totalCount": 5000}),
        ]
        return route

    async def comparison(self, cap):
        return await run(
            query_type="comparison",
            viz_type="grouped_bar_chart",
            compare_entities=["Pembrolizumab", "Nivolumab"],
            compare_entity_kind="drug",
            max_studies=cap,
        )

    @respx.mock
    async def test_cap_holds_across_a_multi_search_comparison(self):
        """Two searches under a cap of 100 previously fetched 200."""
        self.two_batches()
        response = await self.comparison(100)
        assert response.meta.total_studies_processed <= 100

    @respx.mock
    async def test_budget_is_shared_so_the_second_series_is_not_starved(self):
        self.two_batches()
        response = await self.comparison(200)
        by_series = {row["series"] for row in response.visualization.data}
        assert by_series == {"Pembrolizumab", "Nivolumab"}

    @respx.mock
    async def test_a_single_search_still_gets_the_whole_budget(self):
        respx.get(f"{BASE}/version").mock(
            return_value=httpx.Response(200, json={"dataTimestamp": "2026-08-07"})
        )
        batch = [make_record(f"NCT{i:08d}", phases=["PHASE3"]) for i in range(150)]
        respx.get(f"{BASE}/studies").mock(
            return_value=httpx.Response(200, json={"studies": batch, "totalCount": 150})
        )
        response = await run(max_studies=150)
        assert response.meta.total_studies_processed == 150


class TestFilterProvenance:
    """meta.filters has to say which filters produced which series. Flattening
    them meant a comparison reported only its last entity."""

    @respx.mock
    async def test_comparison_keys_filters_by_series(self):
        respx.get(f"{BASE}/version").mock(
            return_value=httpx.Response(200, json={"dataTimestamp": "2026-08-07"})
        )
        route = respx.get(f"{BASE}/studies")
        route.side_effect = [
            httpx.Response(200, json={"studies": SAMPLE[:2], "totalCount": 2}),
            httpx.Response(200, json={"studies": SAMPLE[2:], "totalCount": 2}),
        ]
        response = await run(
            query_type="comparison",
            viz_type="grouped_bar_chart",
            compare_entities=["Pembrolizumab", "Nivolumab"],
            compare_entity_kind="drug",
        )
        filters = response.meta.filters
        # Both entities are present and correctly attributed -- previously the
        # second silently overwrote the first.
        assert set(filters) == {"Pembrolizumab", "Nivolumab"}
        assert filters["Pembrolizumab"]["intervention"] == "Pembrolizumab"
        assert filters["Nivolumab"]["intervention"] == "Nivolumab"

    @respx.mock
    async def test_single_search_keeps_the_flat_shape(self):
        """The common case is unchanged, so this is not a breaking change."""
        mock_studies(SAMPLE)
        response = await run()
        assert response.meta.filters == {"intervention": "Pembrolizumab"}


class TestMultiValueInterpretation:
    """Extra extracted values used to be dropped before the request was built.
    They now reach the query, and the union reading is disclosed."""

    @respx.mock
    async def test_both_drugs_reach_the_upstream_query(self):
        respx.get(f"{BASE}/version").mock(
            return_value=httpx.Response(200, json={"dataTimestamp": "2026-08-07"})
        )
        route = respx.get(f"{BASE}/studies").mock(
            return_value=httpx.Response(
                200, json={"studies": SAMPLE, "totalCount": len(SAMPLE)}
            )
        )
        response = await run(drugs=["Pembrolizumab", "Nivolumab"])
        url = str(route.calls[0].request.url)
        assert "Pembrolizumab+OR+Nivolumab" in url or "Pembrolizumab OR Nivolumab" in url
        assert any(
            "Pembrolizumab OR Nivolumab" in a for a in response.meta.assumptions
        ), response.meta.assumptions

    @respx.mock
    async def test_a_single_drug_adds_no_assumption(self):
        mock_studies(SAMPLE)
        response = await run()
        assert not any("OR" in a for a in response.meta.assumptions)


class TestComparisonSeriesAreNeverLost:
    """fetch() keys membership and provenance by search label. Two entities
    normalising to one label made the second search overwrite the first's ids
    *and* its filters -- a series vanished, and the survivor's label was
    attached to the other search's trials."""

    @respx.mock
    async def test_distinct_entities_each_keep_their_own_membership(self):
        respx.get(f"{BASE}/version").mock(
            return_value=httpx.Response(200, json={"dataTimestamp": "2026-08-07"})
        )
        route = respx.get(f"{BASE}/studies")
        route.side_effect = [
            httpx.Response(200, json={"studies": SAMPLE[:2], "totalCount": 2}),
            httpx.Response(200, json={"studies": SAMPLE[2:3], "totalCount": 1}),
        ]
        response = await run(
            query_type="comparison",
            viz_type="grouped_bar_chart",
            compare_entities=["Pembrolizumab", "Nivolumab"],
            compare_entity_kind="drug",
        )
        # One key per labelled search: nothing was overwritten.
        assert set(response.meta.filters) == {"Pembrolizumab", "Nivolumab"}

    def test_deduplicated_entities_cannot_collide_in_the_membership_dict(self):
        """Grounding removes the duplicate, so fetch() never sees two searches
        competing for one key."""
        from app.agents.understanding import ground_compare_entities
        from app.services.ctgov import build_searches

        query = "compare Aspirin and aspirin trials"
        kept, warnings = ground_compare_entities(["Aspirin", "aspirin"], query)
        plan_obj = plan(
            query=query,
            query_type="comparison",
            viz_type="grouped_bar_chart",
            compare_entities=kept,
            compare_entity_kind="drug",
        )
        searches, _ = build_searches(plan_obj)
        labels = [s.label for s in searches if s.label]
        assert len(labels) == len(set(labels))
        assert any("duplicate" in w for w in warnings)


class TestMixedStatusSemantics:
    """Several requested statuses mean "status is any of these". Nine statuses
    have a live-verified aggFilters code and union upstream in one clause; the
    rest fall back to the client-side union."""

    STATUSES = ("RECRUITING", "COMPLETED", "TERMINATED", "ACTIVE_NOT_RECRUITING")
    CODE_FOR: ClassVar[dict[str, str]] = {
        "rec": "RECRUITING",
        "com": "COMPLETED",
        "ter": "TERMINATED",
        "act": "ACTIVE_NOT_RECRUITING",
    }

    def registry(self):
        return {s: make_record(f"NCT_{s}", status=s) for s in self.STATUSES}

    async def run_with(self, statuses):
        """Drives the real fetch with an upstream that honours aggFilters the
        way the live API does, so a wrong filter would actually show up."""
        registry = self.registry()
        respx.get(f"{BASE}/version").mock(
            return_value=httpx.Response(200, json={"dataTimestamp": "2026-08-07"})
        )
        plan_obj = plan(statuses=statuses)
        searches, _ = build_searches(plan_obj)
        agg = searches[0].to_params(page_size=10).get("aggFilters")
        if agg and agg.startswith("status:"):
            wanted = {self.CODE_FOR[c] for c in agg.split("status:", 1)[1].split()}
            served = [r for s, r in registry.items() if s in wanted]
        else:
            served = list(registry.values())
        respx.get(f"{BASE}/studies").mock(
            return_value=httpx.Response(
                200, json={"studies": served, "totalCount": len(served)}
            )
        )
        async with CTGovClient() as client:
            fetched = await fetch(plan_obj, searches, client)
        apply_client_side_filters(fetched.store, plan_obj)
        kept = {n.replace("NCT_", "") for n in fetched.store.records}
        return agg, kept

    @respx.mock
    async def test_sole_recruiting_uses_the_upstream_filter(self):
        agg, kept = await self.run_with(["RECRUITING"])
        assert agg == "status:rec"
        assert kept == {"RECRUITING"}

    @respx.mock
    async def test_sole_other_status_also_filters_upstream(self):
        agg, kept = await self.run_with(["COMPLETED"])
        assert agg == "status:com"
        assert kept == {"COMPLETED"}

    @respx.mock
    async def test_recruiting_and_completed_is_a_union(self):
        agg, kept = await self.run_with(["RECRUITING", "COMPLETED"])
        assert agg == "status:com rec"
        assert kept == {"RECRUITING", "COMPLETED"}

    @respx.mock
    async def test_recruiting_and_terminated_is_a_union(self):
        agg, kept = await self.run_with(["RECRUITING", "TERMINATED"])
        assert agg == "status:rec ter"
        assert kept == {"RECRUITING", "TERMINATED"}

    @respx.mock
    async def test_active_not_recruiting_and_recruiting_is_a_union(self):
        """These two are genuinely different statuses, so both must survive."""
        agg, kept = await self.run_with(["ACTIVE_NOT_RECRUITING", "RECRUITING"])
        assert agg == "status:act rec"
        assert kept == {"ACTIVE_NOT_RECRUITING", "RECRUITING"}

    @respx.mock
    async def test_no_status_requested_keeps_everything(self):
        agg, kept = await self.run_with([])
        assert agg is None
        assert kept == set(self.STATUSES)

    @respx.mock
    async def test_an_unknown_status_matches_nothing_without_breaking_the_union(self):
        agg, kept = await self.run_with(["COMPLETED", "NOT_A_REAL_STATUS"])
        assert agg is None, "an unverified code would silently return nothing"
        assert kept == {"COMPLETED"}

    def test_a_client_side_union_is_disclosed_with_stable_wording(self):
        """AVAILABLE has no verified code, so this pair genuinely lands in the
        client-side union -- the only path that still needs a disclosure."""
        registry = {
            s: make_record(f"NCT_{s}", status=s)
            for s in ("AVAILABLE", "NO_LONGER_AVAILABLE", "COMPLETED")
        }
        store = StudyStore()
        store.add_records(registry.values())
        warnings = apply_client_side_filters(
            store, plan(statuses=["AVAILABLE", "NO_LONGER_AVAILABLE"])
        )
        text = " ".join(warnings)
        # sorted(), so the disclosure never inherits set iteration order.
        assert "AVAILABLE, NO_LONGER_AVAILABLE" in text
        assert "any of" in text
        assert set(store.records) == {"NCT_AVAILABLE", "NCT_NO_LONGER_AVAILABLE"}

    @respx.mock
    async def test_per_search_provenance_is_not_overwritten(self):
        """meta.filters must retain one entry per labelled search, correctly
        attributed -- a collision silently replaced one series' filters."""
        respx.get(f"{BASE}/version").mock(
            return_value=httpx.Response(200, json={"dataTimestamp": "2026-08-07"})
        )
        route = respx.get(f"{BASE}/studies")
        route.side_effect = [
            httpx.Response(200, json={"studies": SAMPLE[:2], "totalCount": 2}),
            httpx.Response(200, json={"studies": SAMPLE[2:3], "totalCount": 1}),
        ]
        response = await run(
            query_type="comparison",
            viz_type="grouped_bar_chart",
            compare_entities=["Pembrolizumab", "Nivolumab"],
            compare_entity_kind="drug",
        )
        filters = response.meta.filters
        assert set(filters) == {"Pembrolizumab", "Nivolumab"}
        assert filters["Pembrolizumab"]["intervention"] == "Pembrolizumab"
        assert filters["Nivolumab"]["intervention"] == "Nivolumab"

    @respx.mock
    async def test_an_ordinary_non_comparison_query_is_unaffected(self):
        """The dedup guard must not touch the single-search path."""
        mock_studies(SAMPLE)
        response = await run()
        assert response.meta.filters == {"intervention": "Pembrolizumab"}
        assert not any("duplicate" in w for w in response.meta.warnings)

    @respx.mock
    async def test_membership_key_count_always_equals_labelled_search_count(self):
        """The invariant stated directly rather than inferred: every labelled
        search must own exactly one membership entry, so none can be lost to a
        dict-key collision. Duplicates are collapsed before any search runs."""
        respx.get(f"{BASE}/version").mock(
            return_value=httpx.Response(200, json={"dataTimestamp": "2026-08-07"})
        )
        route = respx.get(f"{BASE}/studies")
        route.side_effect = [
            httpx.Response(
                200, json={"studies": [make_record(f"NCT0000000{i}")], "totalCount": 1}
            )
            for i in range(1, 5)
        ]
        plan_obj = plan(
            query_type="comparison",
            viz_type="grouped_bar_chart",
            compare_entities=["Pembrolizumab", "pembrolizumab", "Nivolumab"],
            compare_entity_kind="drug",
        )
        searches, notes = build_searches(plan_obj)
        async with CTGovClient() as client:
            fetched = await fetch(plan_obj, searches, client)
        labelled = [s for s in searches if s.label]
        assert len(fetched.series_membership) == len(labelled)
        assert len(fetched.filters) == len(labelled)
        # And the duplicate never reached the network.
        assert route.call_count == len(labelled)
        assert any("duplicate" in n for n in notes)

    @respx.mock
    async def test_a_single_compare_entity_does_not_produce_a_one_series_chart(self):
        """Comparison mode with fewer than two distinct entities falls back to
        a plain distribution rather than charting a degenerate comparison."""
        from app.agents.understanding import build_plan
        from app.models.schemas import QueryUnderstanding

        understanding = QueryUnderstanding(
            query_type="comparison",
            viz_type="grouped_bar_chart",
            group_by="phase",
            compare_entities=["Aspirin", "aspirin"],
            compare_entity_kind="drug",
            network_kind=None,
            assumptions=[],
            entities=ExtractedEntities(
                drugs=["Aspirin"], conditions=[], sponsors=[], phases=[],
                statuses=[], countries=[], year_range=None,
            ),
        )
        built = build_plan(
            QueryRequest(query="compare Aspirin and aspirin trials"), understanding
        )
        assert built.compare_entities == []
        assert built.viz_type == "bar_chart"


class TestCategoryTruncationDisclosure:
    """TOP_N_DIMENSIONS caps high-cardinality axes. The cap was applied but
    never disclosed, so a 20-of-47 country chart shipped as if complete."""

    def sponsors(self, n):
        return [
            make_record(f"NCT{i:08d}", sponsor=f"Sponsor {i % n}", phases=["PHASE3"])
            for i in range(n * 3)
        ]

    @respx.mock
    async def test_truncation_names_the_shown_and_omitted_counts(self):
        # TOP_N_DIMENSIONS caps sponsor at 15.
        mock_studies(self.sponsors(40))
        response = await run(group_by="sponsor")
        warning = next(
            (w for w in response.meta.warnings if "top" in w and "omitted" in w), None
        )
        assert warning, response.meta.warnings
        assert "top 15" in warning
        assert "of 40" in warning
        assert "25" in warning

    @respx.mock
    async def test_the_disclosure_matches_the_rows_actually_shipped(self):
        mock_studies(self.sponsors(40))
        response = await run(group_by="sponsor")
        shown = {row["sponsor"] for row in response.visualization.data}
        assert len(shown) == 15

    @respx.mock
    async def test_no_warning_when_nothing_was_truncated(self):
        mock_studies(self.sponsors(5))
        response = await run(group_by="sponsor")
        assert not any("omitted" in w for w in response.meta.warnings)

    @respx.mock
    async def test_no_warning_when_exactly_at_the_limit(self):
        mock_studies(self.sponsors(15))
        response = await run(group_by="sponsor")
        assert not any("omitted" in w for w in response.meta.warnings)

    @respx.mock
    async def test_an_uncapped_dimension_never_warns(self):
        """phase has no cap, so it can never trigger the disclosure."""
        mock_studies(SAMPLE)
        response = await run()
        assert not any("omitted" in w for w in response.meta.warnings)


class TestEmptyResultKinds:
    """"No trials matched" and "trials matched but nothing is chartable" are
    different answers, and neither is a server fault. The second used to reach
    the validator and surface as an HTTP 500."""

    @respx.mock
    async def test_zero_matching_trials_reports_no_matching_trials(self):
        mock_studies([])
        with pytest.raises(EmptyResultError) as exc:
            await run()
        assert exc.value.reason == "NO_MATCHING_TRIALS"
        assert exc.value.meta.empty_reason == "NO_MATCHING_TRIALS"

    @respx.mock
    async def test_matched_trials_with_no_usable_start_date(self):
        """A time series over trials that all lack a start date."""
        undated = [
            make_record("NCT00000001", start_date=None, phases=["PHASE3"]),
            make_record("NCT00000002", start_date=None, phases=["PHASE3"]),
        ]
        mock_studies(undated)
        with pytest.raises(EmptyResultError) as exc:
            await run(
                query_type="time_trend", group_by="start_year", viz_type="time_series"
            )
        assert exc.value.reason == "NO_CHARTABLE_DATA"
        # The count is stated, so the reader can tell this from "nothing matched".
        assert "2 trials matched" in str(exc.value)

    @respx.mock
    async def test_matched_trials_with_no_surviving_network_edge(self):
        """Single-drug trials: nodes exist, but no pair co-occurs."""
        singles = [
            make_record(f"NCT0000000{i}", interventions=[("DRUG", f"Drug{i}")])
            for i in range(1, 4)
        ]
        mock_rxnorm_unmatched()
        mock_studies(singles)
        with pytest.raises(EmptyResultError) as exc:
            await run(
                query_type="relationship", viz_type="network_graph",
                group_by=None, network_kind="drug_drug",
            )
        assert exc.value.reason == "NO_CHARTABLE_DATA"
        assert "no two drugs appeared together" in str(exc.value)

    @respx.mock
    async def test_the_route_returns_200_with_the_reason(self):
        """The whole point: a legitimate analytical emptiness is not a 500."""
        undated = [make_record("NCT00000001", start_date=None, phases=["PHASE3"])]
        mock_studies(undated)
        with respx.mock:
            mock_studies(undated)
            body = None
            try:
                await run(
                    query_type="time_trend", group_by="start_year",
                    viz_type="time_series",
                )
            except EmptyResultError as exc:
                from app.api.routes import _empty_response

                body = _empty_response(exc)
        assert body is not None
        assert body.visualization.data == []
        assert body.meta.empty_reason == "NO_CHARTABLE_DATA"
        assert any("usable" in w for w in body.meta.warnings)
        # And it must not offer the wrong advice: trials *were* found.
        assert not any("No trials matched" in w for w in body.meta.warnings)


class TestUnscopedSampleDisclosure:
    """A question with predicates but no population scope has no upstream
    parameter that narrows the registry, so the fetch cap takes the first N in
    ClinicalTrials.gov's own order. The cap was disclosed; that the *shape* of
    the chart inherits that ordering was not."""

    def studies(self, n):
        return [
            make_record(f"NCT{i:08d}", phases=["PHASE3"], start_date=f"202{i % 5}-01-01")
            for i in range(n)
        ]

    @respx.mock
    async def test_an_unscoped_truncated_query_says_the_sample_is_ordered(self):
        mock_studies(self.studies(30), total=597_691)
        response = await run(drugs=[], phases=[3], request={"max_studies": 100})
        assert any("capped slice of the whole registry" in w for w in response.meta.warnings)

    @respx.mock
    async def test_a_scoped_query_does_not_carry_the_warning(self):
        """A condition or drug narrows the population upstream, so the sample is
        of that population rather than of the registry."""
        mock_studies(self.studies(30), total=597_691)
        response = await run(drugs=["Pembrolizumab"], request={"max_studies": 100})
        assert not any("capped slice of the whole registry" in w for w in response.meta.warnings)

    @respx.mock
    async def test_an_unscoped_but_untruncated_query_does_not_carry_it(self):
        """Nothing was dropped, so there is no sampling to disclose."""
        mock_studies(self.studies(5), total=5)
        response = await run(drugs=[], phases=[3])
        assert not any("capped slice of the whole registry" in w for w in response.meta.warnings)


class TestEmptyPlaceholderMatchesTheRequestedShape:
    """A frontend routes on visualization.type, so a bar-chart placeholder for a
    failed network query sends an empty graph to the wrong renderer."""

    def placeholder(self, viz_type):
        from app.api.routes import _empty_response

        error = EmptyResultError(
            "nothing to chart", Meta(), reason="NO_CHARTABLE_DATA", viz_type=viz_type
        )
        return _empty_response(error).visualization

    def test_a_failed_network_query_returns_an_empty_network(self):
        viz = self.placeholder("network_graph")
        assert viz.type == "network_graph"
        assert viz.data == [{"nodes": [], "edges": []}]
        # The key maps a renderer needs are present, so it can parse the shape.
        assert viz.encoding.nodes and viz.encoding.edges

    def test_a_failed_chart_query_still_returns_a_bar_placeholder(self):
        viz = self.placeholder("bar_chart")
        assert viz.type == "bar_chart"
        assert viz.data == []
        assert viz.encoding.x and viz.encoding.y

    def test_the_time_series_case_keeps_the_chart_placeholder(self):
        assert self.placeholder("time_series").type == "bar_chart"

    @respx.mock
    async def test_the_requested_type_is_carried_from_the_plan(self):
        mock_rxnorm_unmatched()
        mock_studies(
            [make_record(f"NCT0000000{i}", interventions=[("DRUG", f"Drug{i}")])
             for i in range(1, 4)]
        )
        with pytest.raises(EmptyResultError) as exc:
            await run(
                query_type="relationship", viz_type="network_graph",
                group_by=None, network_kind="drug_drug",
            )
        assert exc.value.viz_type == "network_graph"


class TestClientSideStatusWarningIsAccurate:
    def test_it_no_longer_claims_only_one_status_can_go_upstream(self):
        """Verified statuses are unioned in one aggFilters clause; what reaches
        the client-side path does so because its code fails silently."""
        store = StudyStore()
        store.add_records(
            [make_record(f"NCT_{s}", status=s) for s in ("AVAILABLE", "COMPLETED")]
        )
        warnings = apply_client_side_filters(store, plan(statuses=["AVAILABLE"]))
        text = " ".join(warnings)
        assert "only one status can be filtered upstream" not in text
        assert "returns zero results rather than an error" in text


class TestPrunedNetworkDisclosure:
    @respx.mock
    async def test_truncation_discloses_that_node_sizes_are_global(self):
        """A hub keeps its true trial count while its edges are subgraph-only,
        so it can look larger than the drawing explains."""
        mock_rxnorm_unmatched()
        # Each pair appears twice, so it clears the default min_edge_weight,
        # and there are more partners than the node cap.
        mock_studies(
            [
                make_record(
                    f"NCT{i * 2 + rep:08d}",
                    interventions=[("DRUG", "HubDrug"), ("DRUG", f"Partner{i}")],
                )
                for i in range(40)
                for rep in (0, 1)
            ]
        )
        response = await run(
            query_type="relationship", viz_type="network_graph",
            group_by=None, network_kind="drug_drug",
        )
        assert any("Node sizes count every fetched trial" in w for w in response.meta.warnings)

    @respx.mock
    async def test_an_untruncated_graph_does_not_carry_the_note(self):
        mock_rxnorm_unmatched()
        mock_studies(SAMPLE)
        response = await run(
            query_type="relationship", viz_type="network_graph",
            group_by=None, network_kind="drug_drug",
        )
        assert not any("Node sizes count every fetched trial" in w for w in response.meta.warnings)


class TestEmptyComparisonSeries:
    """A series whose search returned nothing produced no rows, so it vanished
    from a chart whose title still named it."""

    def result(self, membership):
        records = {
            "NCT00000001": make_record("NCT00000001", phases=["PHASE3"]),
            "NCT00000002": make_record("NCT00000002", phases=["PHASE1"]),
        }
        from app.services.aggregate import aggregate
        from app.services.dimensions import get_dimension

        return aggregate(records, get_dimension("phase"), series_membership=membership)

    def test_an_empty_series_is_zero_filled_across_the_existing_keys(self):
        result = self.result(
            {"Pembrolizumab": {"NCT00000001", "NCT00000002"}, "Nivolumab": set()}
        )
        assert {d.series for d in result.data} == {"Pembrolizumab", "Nivolumab"}
        empty = [d for d in result.data if d.series == "Nivolumab"]
        assert {d.key for d in empty} == {"Phase 1", "Phase 3"}
        assert all(d.value == 0 and d.nct_ids == [] for d in empty)

    def test_a_populated_comparison_is_unchanged(self):
        result = self.result(
            {"Pembrolizumab": {"NCT00000001"}, "Nivolumab": {"NCT00000002"}}
        )
        assert len(result.data) == 2
        assert all(d.value == 1 for d in result.data)

    def test_nothing_is_fabricated_when_every_series_is_empty(self):
        """No keys exist, so there is nothing to zero-fill; this becomes
        NO_CHARTABLE_DATA instead."""
        assert self.result({"A": set(), "B": set()}).data == []


class TestHistogramEndToEnd:
    """The histogram path through the full pipeline, not just reconciliation."""

    @respx.mock
    async def test_enrollment_histogram_renders_and_validates(self):
        mock_studies(
            [
                make_record(f"NCT{i:08d}", enrollment=size)
                for i, size in enumerate([5, 40, 120, 900, 4000], start=1)
            ]
        )
        response = await run(
            group_by="enrollment_bucket", viz_type="histogram"
        )
        assert response.visualization.type == "histogram"
        assert response.visualization.encoding.x.field == "enrollment_bucket"
        assert sum(r["trial_count"] for r in response.visualization.data) == 5
        # Every bucket is still cited from the records that produced it.
        for row in response.visualization.data:
            assert row["total_supporting_trials"] == row["trial_count"]


class TestNetworkWithoutRxNorm:
    """RXNORM_ENABLED=false must still build a graph -- resolution is an
    enhancement, not a dependency."""

    @respx.mock
    async def test_graph_builds_with_resolution_disabled(self, monkeypatch):
        monkeypatch.setattr(settings, "RXNORM_ENABLED", False)
        mock_studies(
            [
                make_record(
                    f"NCT{i:08d}",
                    interventions=[("DRUG", "Pembrolizumab"), ("DRUG", "Carboplatin")],
                )
                for i in range(1, 4)
            ]
        )
        response = await run(
            query_type="relationship", viz_type="network_graph",
            group_by=None, network_kind="drug_drug",
        )
        graph = response.visualization.data[0]
        assert len(graph["nodes"]) == 2
        assert len(graph["edges"]) == 1
        # No RxNorm identity, but the graph is still fully cited.
        assert all(n["rxcui"] is None for n in graph["nodes"])
        assert graph["edges"][0]["citations"]
