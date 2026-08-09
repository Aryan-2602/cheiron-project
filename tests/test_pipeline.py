"""End-to-end pipeline and HTTP behaviour, with the LLM and upstream API mocked.

These tests run the real pipeline -- fetch, aggregate, format, validate -- so
they cover the wiring between stages that unit tests cannot.
"""

import httpx
import pytest
import respx
from fastapi.testclient import TestClient

from app.main import app
from app.models.schemas import ExtractedEntities, QueryPlan, QueryRequest
from app.pipeline import EmptyResultError, UnsupportedQueryError, run_pipeline
from app.services.ctgov import CTGovClient, CTGovError
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

    def test_unsupported_query_is_422_with_a_typed_code(self, monkeypatch):
        async def fake(*args, **kwargs):
            raise UnsupportedQueryError("not answerable from registry data")

        monkeypatch.setattr("app.api.routes.run_pipeline", fake)
        response = self.client().post("/api/v1/query", json={"query": "what is love"})
        assert response.status_code == 422
        assert response.json()["detail"]["error"]["code"] == "UNSUPPORTED_QUERY"

    def test_upstream_failure_is_502(self, monkeypatch):
        async def fake(*args, **kwargs):
            raise CTGovError("boom")

        monkeypatch.setattr("app.api.routes.run_pipeline", fake)
        response = self.client().post("/api/v1/query", json={"query": "a question"})
        assert response.status_code == 502
        assert response.json()["detail"]["error"]["code"] == "UPSTREAM_ERROR"

    def test_a_response_failing_validation_is_withheld(self, monkeypatch):
        """A chart that cannot be traced to source data must not render."""
        from app.services.validate import ValidationFailure

        async def fake(*args, **kwargs):
            raise ValidationFailure("citation references an unknown trial")

        monkeypatch.setattr("app.api.routes.run_pipeline", fake)
        response = self.client().post("/api/v1/query", json={"query": "a question"})
        assert response.status_code == 500
        assert response.json()["detail"]["error"]["code"] == "VALIDATION_ERROR"


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
    """aggFilters carries only one phase, so a multi-phase request has to be
    OR-ed client-side. Truncating to the first phase silently answered a
    different question."""

    @respx.mock
    async def test_single_phase_is_filtered_upstream(self):
        respx.get(f"{BASE}/version").mock(
            return_value=httpx.Response(200, json={"dataTimestamp": "2026-08-07"})
        )
        route = respx.get(f"{BASE}/studies").mock(
            return_value=httpx.Response(
                200, json={"studies": [PHASE_MIX[2]], "totalCount": 1}
            )
        )
        await run(phases=[3])
        assert "aggFilters=phase%3A3" in str(route.calls[0].request.url)

    @respx.mock
    async def test_multiple_phases_are_not_truncated_to_the_first(self):
        mock_studies(PHASE_MIX)
        response = await run(phases=[2, 3])
        cited = {
            c["nct_id"]
            for row in response.visualization.data
            for c in row["citations"]
        }
        # The union: phase 2, phase 3, and the combined 1/2 trial.
        assert cited == {"NCT00000002", "NCT00000003", "NCT00000005"}
        # Not just phase 2's results, and not phase 1 or 4 alone.
        assert "NCT00000001" not in cited
        assert "NCT00000004" not in cited

    @respx.mock
    async def test_multiple_phases_send_no_upstream_phase_filter(self):
        respx.get(f"{BASE}/version").mock(
            return_value=httpx.Response(200, json={"dataTimestamp": "2026-08-07"})
        )
        route = respx.get(f"{BASE}/studies").mock(
            return_value=httpx.Response(
                200, json={"studies": PHASE_MIX, "totalCount": 5}
            )
        )
        await run(phases=[2, 3])
        assert "aggFilters" not in str(route.calls[0].request.url)

    @respx.mock
    async def test_multi_phase_narrowing_is_disclosed(self):
        mock_studies(PHASE_MIX)
        response = await run(phases=[2, 3])
        assert any(
            "phase 2, 3" in w and "client-side" in w for w in response.meta.warnings
        )

    @respx.mock
    async def test_combined_phase_trial_matches_either_requested_phase(self):
        """A Phase 1/2 study is genuinely evidence for both."""
        mock_studies([PHASE_MIX[4]])
        response = await run(phases=[1, 4])
        cited = {
            c["nct_id"] for row in response.visualization.data for c in row["citations"]
        }
        assert cited == {"NCT00000005"}


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
