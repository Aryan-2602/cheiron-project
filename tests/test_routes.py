from typing import ClassVar
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.main import app
from app.models.schemas import QueryRequest
from app.pipeline import CTGovError, UnsupportedQueryError

client = TestClient(app)


def test_health() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


class TestErrorResponseShape:
    """The runtime body must match what OpenAPI advertises. HTTPException
    nested the payload under "detail" while the schema claimed the bare model,
    and 422 carried two different shapes -- the domain error and FastAPI's own
    request-validation error."""

    def test_an_unanswerable_question_is_400_with_a_bare_error_body(self):
        with patch("app.api.routes.run_pipeline", side_effect=UnsupportedQueryError("nope")):
            response = client.post("/api/v1/query", json={"query": "capital of France?"})
        assert response.status_code == 400
        body = response.json()
        assert "detail" not in body
        assert body["error"]["code"] == "UNSUPPORTED_QUERY"
        assert body["error"]["message"]
        assert isinstance(body["error"]["details"], dict)

    def test_request_schema_violations_keep_fastapis_own_422(self):
        """Two shapes on one status code force callers to sniff; they now sit
        on different codes."""
        response = client.post("/api/v1/query", json={})
        assert response.status_code == 422
        assert "detail" in response.json()

    def test_the_openapi_schema_matches_the_runtime_body(self):
        responses = client.get("/openapi.json").json()["paths"]["/api/v1/query"]["post"][
            "responses"
        ]
        assert "ErrorResponse" in str(responses["400"])
        assert "ErrorResponse" in str(responses["502"])
        assert "ErrorResponse" in str(responses["500"])

    def test_an_upstream_failure_is_502_with_the_same_shape(self):
        with patch("app.api.routes.run_pipeline", side_effect=CTGovError("boom")):
            response = client.post("/api/v1/query", json={"query": "melanoma trials"})
        assert response.status_code == 502
        assert response.json()["error"]["code"] == "UPSTREAM_ERROR"


class TestRequestNormalization:
    def test_a_blank_query_is_rejected(self):
        response = client.post("/api/v1/query", json={"query": "   "})
        assert response.status_code == 422

    def test_query_whitespace_is_stripped(self):
        assert QueryRequest(query="  melanoma trials  ").query == "melanoma trials"

    def test_a_whitespace_only_override_becomes_none(self):
        """Otherwise it pins the entity to a value that matches nothing."""
        request = QueryRequest(query="melanoma trials", drug_name="   ")
        assert request.drug_name is None

    def test_override_whitespace_is_collapsed(self):
        request = QueryRequest(query="melanoma trials", condition="  lung   cancer ")
        assert request.condition == "lung cancer"

    def test_ordinary_values_are_untouched(self):
        request = QueryRequest(query="melanoma trials", drug_name="Pembrolizumab")
        assert request.drug_name == "Pembrolizumab"

    def test_a_too_short_query_is_still_rejected(self):
        with pytest.raises(ValidationError):
            QueryRequest(query="ab")


class TestStructuredFieldBounds:
    """Sized for real registry values, so an extreme string is rejected before
    it becomes an oversized upstream URL."""

    LIMITS: ClassVar[dict[str, int]] = {
        "drug_name": 300, "condition": 500, "sponsor": 500, "country": 200,
    }

    @pytest.mark.parametrize("field,limit", LIMITS.items())
    def test_a_value_at_the_limit_is_accepted(self, field, limit):
        request = QueryRequest(query="melanoma trials", **{field: "x" * limit})
        assert len(getattr(request, field)) == limit

    @pytest.mark.parametrize("field,limit", LIMITS.items())
    def test_one_character_over_is_rejected(self, field, limit):
        with pytest.raises(ValidationError):
            QueryRequest(query="melanoma trials", **{field: "x" * (limit + 1)})

    def test_normal_values_are_unaffected(self):
        request = QueryRequest(
            query="melanoma trials", drug_name="Pembrolizumab",
            condition="melanoma", sponsor="Merck Sharp & Dohme LLC",
            country="United States",
        )
        assert request.drug_name == "Pembrolizumab"
        assert request.sponsor == "Merck Sharp & Dohme LLC"

    def test_whitespace_normalisation_still_applies(self):
        request = QueryRequest(query="melanoma trials", condition="  lung   cancer ")
        assert request.condition == "lung cancer"

    def test_a_blank_override_still_becomes_none(self):
        assert QueryRequest(query="melanoma trials", sponsor="   ").sponsor is None

    def test_an_oversized_override_is_a_422_at_the_route(self):
        response = client.post(
            "/api/v1/query",
            json={"query": "melanoma trials", "drug_name": "x" * 5000},
        )
        assert response.status_code == 422
