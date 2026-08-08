"""Logging: formatter behaviour, stage coverage, and failure-path levels.

These tests assert on levels, message keys, and the presence of structured
fields -- not on exact prose -- so log wording can be improved without breaking
the suite. What they do pin down is the contract the logs are supposed to
provide: every stage emits a line, failures are loud, and nothing sensitive or
oversized is written out.
"""

import json
import logging
from typing import ClassVar

import httpx
import pytest
import respx
from fastapi.testclient import TestClient

from app.core.logging import (
    TRUNCATION_MARKER,
    JsonFormatter,
    request_id,
    setup_logging,
    truncate,
)
from app.main import app
from app.models.schemas import ExtractedEntities, QueryPlan, QueryRequest
from app.pipeline import run_pipeline
from app.services.ctgov import CTGovClient
from tests.conftest import make_record

BASE = "https://clinicaltrials.gov/api/v2"
RXNORM = "https://rxnav.nlm.nih.gov/REST"


def render(record: logging.LogRecord) -> dict:
    return json.loads(JsonFormatter().format(record))


def make_log_record(level=logging.INFO, msg="hello", **extra) -> logging.LogRecord:
    record = logging.LogRecord("demo", level, __file__, 1, msg, (), None)
    record.__dict__.update(extra)
    return record


SAMPLE = [
    make_record("NCT00000001", phases=["PHASE3"],
                interventions=[("DRUG", "Pembrolizumab"), ("DRUG", "Carboplatin")]),
    make_record("NCT00000002", phases=["PHASE1", "PHASE2"],
                interventions=[("DRUG", "Pembrolizumab")]),
    # A second co-occurrence, so the pembrolizumab-carboplatin edge clears the
    # default min_edge_weight=2 and the network tests have a graph to inspect.
    make_record("NCT00000003", phases=["PHASE2"],
                interventions=[("DRUG", "Pembrolizumab"), ("DRUG", "Carboplatin")]),
]


def plan(**kwargs):
    return QueryPlan(
        query=kwargs.pop("query", "How are trials distributed across phases?"),
        query_type=kwargs.pop("query_type", "distribution"),
        entities=ExtractedEntities(
            drugs=["Pembrolizumab"], conditions=[], sponsors=[], phases=[],
            statuses=[], countries=[], year_range=None,
        ),
        group_by=kwargs.pop("group_by", "phase"),
        viz_type=kwargs.pop("viz_type", "bar_chart"),
        **kwargs,
    )


def mock_studies(studies=SAMPLE):
    respx.get(f"{BASE}/version").mock(
        return_value=httpx.Response(200, json={"dataTimestamp": "2026-08-07T09:00:05"})
    )
    respx.get(f"{BASE}/studies").mock(
        return_value=httpx.Response(
            200, json={"studies": studies, "totalCount": len(studies)}
        )
    )


async def run(**kwargs):
    async with CTGovClient(max_retries=0) as client:
        return await run_pipeline(
            QueryRequest(query="a question"), client=client, plan=plan(**kwargs)
        )


class TestFormatter:
    def test_emits_the_four_required_fields_as_json(self):
        payload = render(make_log_record())
        assert set(payload) >= {"time", "level", "name", "message"}
        assert payload["level"] == "INFO"
        assert payload["name"] == "demo"
        assert payload["message"] == "hello"

    def test_extra_fields_become_top_level_keys(self):
        """Without this the logging is prose, not structured."""
        payload = render(make_log_record(records=42, truncated=False))
        assert payload["records"] == 42
        assert payload["truncated"] is False

    def test_request_id_included_only_when_set(self):
        assert "request_id" not in render(make_log_record())
        token = request_id.set("abc12345")
        try:
            assert render(make_log_record())["request_id"] == "abc12345"
        finally:
            request_id.reset(token)

    def test_non_serializable_values_degrade_instead_of_raising(self):
        payload = render(make_log_record(obj=object()))
        assert isinstance(payload["obj"], str)

    def test_exception_info_is_captured(self):
        try:
            raise ValueError("boom")
        except ValueError:
            import sys

            record = logging.LogRecord(
                "demo", logging.ERROR, __file__, 1, "failed", (), sys.exc_info()
            )
        assert "ValueError: boom" in render(record)["exc_info"]


class TestTruncate:
    def test_short_text_is_untouched(self):
        assert truncate("short") == "short"

    def test_long_text_is_clipped_and_marked(self):
        result = truncate("x" * 500, limit=100)
        assert result.endswith(TRUNCATION_MARKER)
        assert len(result) == 100 + len(TRUNCATION_MARKER)


class TestSetup:
    def test_installs_one_json_handler_at_the_configured_level(self):
        setup_logging()
        root = logging.getLogger()
        assert len(root.handlers) == 1
        assert isinstance(root.handlers[0].formatter, JsonFormatter)
        assert root.level == logging.INFO


class TestReservedKeyGuard:
    """An `extra` key colliding with a LogRecord attribute raises KeyError at
    call time -- a logging line that crashes the request it was meant to
    describe. Ruff's G101 catches it statically; this catches it in the payloads
    actually emitted."""

    RESERVED: ClassVar[frozenset[str]] = frozenset(
        {"name", "message", "args", "module", "levelname", "filename",
         "pathname", "lineno", "funcName", "created", "msg", "exc_info"}
    )

    @respx.mock
    async def test_no_stage_uses_a_reserved_extra_key(self, caplog):
        mock_studies()
        with caplog.at_level(logging.DEBUG):
            await run()
        ours = [r for r in caplog.records if r.name.startswith("app.")]
        assert ours, "expected our own log records"
        standard = set(logging.LogRecord("", 0, "", 0, "", (), None).__dict__)
        for record in ours:
            # `message`/`asctime` are written onto the record by Formatter.format
            # after the fact, so they are not evidence of a bad `extra`.
            custom = set(record.__dict__) - standard - {"message", "asctime"}
            assert not (custom & self.RESERVED), (
                f"{record.getMessage()!r} uses a reserved key"
            )


class TestStageCoverage:
    @respx.mock
    async def test_happy_path_emits_a_line_for_each_stage(self, caplog):
        mock_studies()
        with caplog.at_level(logging.INFO):
            await run()
        messages = [r.getMessage() for r in caplog.records]
        for expected in (
            "ctgov search started",
            "ctgov search completed",
            "aggregation completed",
            "validation passed",
        ):
            assert expected in messages, f"missing stage line: {expected}"

    @respx.mock
    async def test_stage_lines_carry_their_structured_fields(self, caplog):
        mock_studies()
        with caplog.at_level(logging.INFO):
            await run()
        by_message = {r.getMessage(): r for r in caplog.records}

        fetch = by_message["ctgov search completed"]
        assert fetch.records == len(SAMPLE)
        assert fetch.pages == 1
        assert isinstance(fetch.duration_ms, float)

        agg = by_message["aggregation completed"]
        assert agg.dimension == "phase"
        assert agg.buckets > 0
        assert agg.total_studies_matched == len(SAMPLE)

    @respx.mock
    async def test_logging_does_not_break_the_pipeline(self, caplog):
        """The point of the suite: instrumentation must be invisible to callers."""
        mock_studies()
        with caplog.at_level(logging.DEBUG):
            response = await run()
        assert response.visualization.data
        assert response.meta.total_studies_processed == len(SAMPLE)

    @respx.mock
    async def test_network_query_logs_resolution_and_graph_lines(self, caplog):
        mock_studies()
        respx.get(f"{RXNORM}/approximateTerm.json").mock(
            return_value=httpx.Response(200, json={"approximateGroup": {"inputTerm": "x"}})
        )
        with caplog.at_level(logging.INFO):
            await run(query_type="relationship", viz_type="network_graph",
                      network_kind="drug_drug", group_by=None)
        messages = [r.getMessage() for r in caplog.records]
        assert "drug resolution started" in messages
        assert "drug resolution completed" in messages
        assert "network built" in messages

        done = next(r for r in caplog.records
                    if r.getMessage() == "drug resolution completed")
        # Deltas for this request, not process-lifetime cache totals.
        assert done.live_lookups >= 0
        assert done.cache_hits >= 0


class TestFailurePaths:
    """The explicit ask: a failure must be visible at WARNING or ERROR."""

    @respx.mock
    async def test_ctgov_retry_emits_a_warning_with_the_attempt(self, caplog):
        respx.get(f"{BASE}/version").mock(
            return_value=httpx.Response(200, json={"dataTimestamp": "2026-08-07"})
        )
        route = respx.get(f"{BASE}/studies")
        route.side_effect = [
            httpx.Response(503),
            httpx.Response(200, json={"studies": SAMPLE, "totalCount": len(SAMPLE)}),
        ]
        with caplog.at_level(logging.WARNING):
            async with CTGovClient(max_retries=2) as client:
                await run_pipeline(
                    QueryRequest(query="a question"), client=client, plan=plan()
                )
        retries = [r for r in caplog.records if r.getMessage() == "ctgov request retrying"]
        assert retries, "a silent retry is exactly what this logging exists to surface"
        assert retries[0].levelno == logging.WARNING
        assert retries[0].attempt == 1
        assert "503" in retries[0].reason

    @respx.mock
    async def test_rxnorm_outage_warns_once_for_the_batch(self, caplog):
        """Resolution fans out over hundreds of names, so per-retry WARNINGs
        would drown the log -- a live outage emitted 116 of them. The detail
        stays at DEBUG and the batch warns once."""
        mock_studies()
        respx.get(f"{RXNORM}/approximateTerm.json").mock(
            return_value=httpx.Response(503)
        )
        with caplog.at_level(logging.DEBUG):
            await run(query_type="relationship", viz_type="network_graph",
                      network_kind="drug_drug", group_by=None)

        warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
        degraded = [r for r in warnings if r.getMessage() == "drug resolution degraded"]
        assert len(degraded) == 1
        assert degraded[0].failed > 0
        # Per-retry detail is still available, just not at WARNING.
        assert any(r.getMessage() == "rxnorm request retrying" for r in caplog.records)
        assert not any(
            r.getMessage() == "rxnorm request retrying" and r.levelno >= logging.WARNING
            for r in caplog.records
        )

    def test_validation_failure_logs_exactly_one_error(self, caplog, monkeypatch):
        """Logged in the route, not the validator, so one fault reads as one line."""
        from app.services.validate import ValidationFailure

        async def fake(*args, **kwargs):
            raise ValidationFailure("citation references an unknown trial")

        monkeypatch.setattr("app.api.routes.run_pipeline", fake)
        with caplog.at_level(logging.ERROR):
            response = TestClient(app).post("/api/v1/query", json={"query": "a question"})
        assert response.status_code == 500

        failures = [r for r in caplog.records if r.getMessage() == "request failed"]
        assert len(failures) == 1
        assert failures[0].code == "VALIDATION_ERROR"
        assert failures[0].stage == "validation"
        assert "unknown trial" in failures[0].detail

    def test_unsupported_query_logs_an_error_with_context(self, caplog, monkeypatch):
        from app.pipeline import UnsupportedQueryError

        async def fake(*args, **kwargs):
            raise UnsupportedQueryError("not answerable")

        monkeypatch.setattr("app.api.routes.run_pipeline", fake)
        with caplog.at_level(logging.ERROR):
            TestClient(app).post(
                "/api/v1/query",
                json={"query": "What is the capital of France?", "drug_name": "X"},
            )
        failure = next(r for r in caplog.records if r.getMessage() == "request failed")
        assert failure.code == "UNSUPPORTED_QUERY"
        assert "capital of France" in failure.query
        assert failure.filters == {"drug_name": "X"}


class TestMiddleware:
    def test_successful_request_logs_status_and_latency_at_info(self, caplog):
        with caplog.at_level(logging.INFO):
            TestClient(app).get("/health")
        completed = next(
            r for r in caplog.records if r.getMessage() == "request completed"
        )
        assert completed.levelno == logging.INFO
        assert completed.status == 200
        assert completed.path == "/health"
        assert completed.duration_ms >= 0

    def test_error_status_logs_at_error(self, caplog):
        """Covers FastAPI's own 422 for a malformed body, which never reaches
        the route handler and would otherwise leave no trace."""
        with caplog.at_level(logging.INFO):
            TestClient(app).post("/api/v1/query", json={})
        completed = next(
            r for r in caplog.records if r.getMessage() == "request completed"
        )
        assert completed.status == 422
        assert completed.levelno == logging.ERROR


class TestNoSecretsOrPayloads:
    SECRET = "sk-proj-THISISATESTKEYDONOTLOG"

    @respx.mock
    async def test_no_log_line_contains_a_credential(self, caplog, monkeypatch):
        from app.core import config

        monkeypatch.setattr(config.settings, "OPEN_AI_API_KEY", self.SECRET)
        mock_studies()
        with caplog.at_level(logging.DEBUG):
            await run()
        for record in caplog.records:
            assert self.SECRET not in json.dumps(record.__dict__, default=str)

    @respx.mock
    async def test_no_log_line_contains_a_raw_trial_record(self, caplog):
        """Only counts and ids -- a dumped record would swamp the output."""
        mock_studies()
        with caplog.at_level(logging.DEBUG):
            await run()
        for record in caplog.records:
            blob = json.dumps(record.__dict__, default=str)
            assert "protocolSection" not in blob
            assert "briefTitle" not in blob

    def test_long_queries_are_truncated_in_the_received_line(self, caplog, monkeypatch):
        from app.pipeline import UnsupportedQueryError

        async def fake(*args, **kwargs):
            raise UnsupportedQueryError("nope")

        monkeypatch.setattr("app.api.routes.run_pipeline", fake)
        with caplog.at_level(logging.INFO):
            TestClient(app).post("/api/v1/query", json={"query": "why " * 400})
        received = next(r for r in caplog.records if r.getMessage() == "query received")
        assert received.query.endswith(TRUNCATION_MARKER)


@pytest.mark.parametrize(
    "message",
    ["query received", "ctgov search started", "aggregation completed"],
)
def test_messages_are_constant_not_interpolated(message):
    """Messages stay greppable because variable data lives in `extra`."""
    assert "%" not in message and "{" not in message
