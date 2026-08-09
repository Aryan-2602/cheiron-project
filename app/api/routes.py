"""HTTP surface.

Error mapping is deliberate: an empty result is a 200, because "no trials match
your question" is a correct answer to a well-formed question, and the caller
gets the same ``meta`` (filters applied, URLs called, warnings) they would get
for a populated chart. Failures we cannot answer through -- an upstream outage,
a broken grounding check -- are non-2xx and say why.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app.agents.understanding import UnderstandingError
from app.core.logging import truncate
from app.models.schemas import (
    Encoding,
    ErrorBody,
    ErrorCode,
    ErrorResponse,
    FieldRef,
    QueryRequest,
    QueryResponse,
    VisualizationSpec,
)
from app.pipeline import (
    CTGovError,
    EmptyResultError,
    UnsupportedQueryError,
    ValidationFailure,
    run_pipeline,
)
from app.services.viz import EDGE_KEY_MAP, NODE_KEY_MAP

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["query"])


def _overrides(request: QueryRequest) -> dict[str, object]:
    """The structured override fields the caller actually supplied."""
    fields = ("drug_name", "condition", "sponsor", "phase", "country",
              "start_year", "end_year")
    return {f: getattr(request, f) for f in fields if getattr(request, f) is not None}


def _log_failure(
    request: QueryRequest, *, code: str, stage: str, exc: Exception
) -> None:
    """One ERROR line per failed request, with enough context to reproduce it.

    Deliberately an explicit allowlist of fields: config is never serialized, so
    no credential can reach the log by accident.
    """
    logger.error(
        "request failed",
        extra={
            "code": code,
            "stage": stage,
            # The one place the question text is kept: a failure that cannot be
            # reproduced cannot be fixed. Truncated, and only on the error path.
            "query": truncate(request.query),
            "filters": _overrides(request),
            "detail": truncate(str(exc), 300),
        },
    )


def _empty_response(error: EmptyResultError) -> QueryResponse:
    """An empty but fully-formed response, so a frontend renders 'no results'
    with the same code path it uses for data."""
    meta = error.meta
    if meta.empty_reason == "NO_CHARTABLE_DATA":
        # Trials *were* found, so "try broadening the filters" would be wrong
        # advice; the message from the pipeline names the actual cause.
        meta.warnings = [*meta.warnings, str(error)]
        title = "No chartable data"
    else:
        meta.warnings = [
            *meta.warnings,
            (
                "No trials matched this query. Try broadening the condition, drug, "
                "or phase filters."
            ),
        ]
        title = "No matching trials"
    # The placeholder has to be the shape that was asked for: a frontend routes
    # on `type`, so a bar-chart placeholder for a failed network query sends an
    # empty graph to the wrong renderer.
    if error.viz_type == "network_graph":
        visualization = VisualizationSpec(
            type="network_graph",
            title=title,
            encoding=Encoding(nodes=NODE_KEY_MAP, edges=EDGE_KEY_MAP),
            data=[{"nodes": [], "edges": []}],
        )
    else:
        visualization = VisualizationSpec(
            type="bar_chart",
            title=title,
            encoding=Encoding(
                x=FieldRef(field="key", label="Category", type="nominal"),
                y=FieldRef(field="trial_count", label="Number of trials", type="quantitative"),
            ),
            data=[],
        )
    return QueryResponse(
        visualization=visualization,
        meta=meta,
    )


def _error(status_code: int, code: ErrorCode, message: str, **details: Any) -> JSONResponse:
    """A response whose body *is* an ErrorResponse.

    HTTPException nests the payload under "detail", so the runtime body was
    {"detail": {"error": ...}} while OpenAPI advertised the bare model. Sending
    it directly makes the documented schema true.
    """
    body = ErrorResponse(error=ErrorBody(code=code, message=message, details=details))
    return JSONResponse(status_code=status_code, content=body.model_dump())


@router.post(
    "/query",
    response_model=QueryResponse,
    responses={
        # 400, not 422: FastAPI already uses 422 for request-schema violations
        # with its own body shape, and two different shapes on one status code
        # force callers to sniff. This one is a well-formed request asking an
        # unanswerable question.
        400: {"model": ErrorResponse, "description": "Question not answerable"},
        422: {"description": "Request body failed schema validation"},
        502: {"model": ErrorResponse, "description": "LLM or upstream API failure"},
        500: {"model": ErrorResponse, "description": "Response failed validation"},
    },
    summary="Turn a natural-language clinical-trials question into a visualization spec",
)
async def query(request: QueryRequest) -> QueryResponse:
    logger.info(
        "query received",
        extra={
            # Length, not text. A clinical-trials question is user-supplied
            # free text that could carry personal or clinical detail, and it is
            # not needed to follow a request through the logs -- the request id
            # correlates the lines, and the plan summary logged by the
            # understanding stage says how the question was read. See the
            # logging section of the README for the production posture.
            "query_chars": len(request.query),
            # Only the overrides actually supplied, so the line stays short and
            # a reader can see exactly what was pinned by the caller.
            "overrides": _overrides(request),
            "max_studies": request.max_studies,
            "max_citations_per_datum": request.max_citations_per_datum,
        },
    )
    try:
        return await run_pipeline(request)
    except EmptyResultError as exc:
        # A 200: an empty result is a correct answer, not a fault.
        logger.info(
            "no results for query",
            extra={"query_chars": len(request.query), "reason": exc.reason},
        )
        return _empty_response(exc)
    except UnsupportedQueryError as exc:
        _log_failure(request, code="UNSUPPORTED_QUERY", stage="understanding", exc=exc)
        return _error(400, "UNSUPPORTED_QUERY", str(exc))
    except UnderstandingError as exc:
        _log_failure(request, code="LLM_ERROR", stage="understanding", exc=exc)
        return _error(502, "LLM_ERROR", str(exc))
    except CTGovError as exc:
        _log_failure(request, code="UPSTREAM_ERROR", stage="fetch", exc=exc)
        return _error(
            502,
            "UPSTREAM_ERROR",
            "ClinicalTrials.gov could not be reached or returned an error.",
            upstream=str(exc),
        )
    except ValidationFailure as exc:
        # The response could not be grounded in the fetched records. Returning
        # an error is the correct outcome: a chart that renders but cannot be
        # traced to source data is worse than no chart.
        # One ERROR line, here rather than in the validator, so a failure reads
        # as a single fault carrying both the failing check and the query.
        _log_failure(request, code="VALIDATION_ERROR", stage="validation", exc=exc)
        return _error(
            500,
            "VALIDATION_ERROR",
            "The generated response failed its grounding checks and was withheld.",
            check=str(exc),
        )
