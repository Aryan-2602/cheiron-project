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


class TestNetwork:
    @respx.mock
    async def test_network_end_to_end(self):
        mock_studies(SAMPLE)
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
