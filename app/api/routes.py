"""HTTP surface.

Error mapping is deliberate: an empty result is a 200, because "no trials match
your question" is a correct answer to a well-formed question, and the caller
gets the same ``meta`` (filters applied, URLs called, warnings) they would get
for a populated chart. Failures we cannot answer through -- an upstream outage,
a broken grounding check -- are non-2xx and say why.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from app.agents.understanding import UnderstandingError
from app.core.logging import truncate
from app.models.schemas import (
    Encoding,
    ErrorBody,
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
            "query": truncate(request.query),
            "filters": _overrides(request),
            "detail": truncate(str(exc), 300),
        },
    )


def _empty_response(error: EmptyResultError) -> QueryResponse:
    """An empty but fully-formed response, so a frontend renders 'no results'
    with the same code path it uses for data."""
    meta = error.meta
    meta.warnings = [
        *meta.warnings,
        (
            "No trials matched this query. Try broadening the condition, drug, "
            "or phase filters."
        ),
    ]
    return QueryResponse(
        visualization=VisualizationSpec(
            type="bar_chart",
            title="No matching trials",
            encoding=Encoding(
                x=FieldRef(field="key", label="Category", type="nominal"),
                y=FieldRef(field="trial_count", label="Number of trials", type="quantitative"),
            ),
            data=[],
        ),
        meta=meta,
    )


@router.post(
    "/query",
    response_model=QueryResponse,
    responses={
        422: {"model": ErrorResponse, "description": "Question not answerable"},
        502: {"model": ErrorResponse, "description": "LLM or upstream API failure"},
        500: {"model": ErrorResponse, "description": "Response failed validation"},
    },
    summary="Turn a natural-language clinical-trials question into a visualization spec",
)
async def query(request: QueryRequest) -> QueryResponse:
    logger.info(
        "query received",
        extra={
            "query": truncate(request.query),
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
        logger.info("no results for query", extra={"query": truncate(request.query)})
        return _empty_response(exc)
    except UnsupportedQueryError as exc:
        _log_failure(request, code="UNSUPPORTED_QUERY", stage="understanding", exc=exc)
        raise HTTPException(
            status_code=422,
            detail=ErrorResponse(
                error=ErrorBody(code="UNSUPPORTED_QUERY", message=str(exc))
            ).model_dump(),
        ) from exc
    except UnderstandingError as exc:
        _log_failure(request, code="LLM_ERROR", stage="understanding", exc=exc)
        raise HTTPException(
            status_code=502,
            detail=ErrorResponse(
                error=ErrorBody(code="LLM_ERROR", message=str(exc))
            ).model_dump(),
        ) from exc
    except CTGovError as exc:
        _log_failure(request, code="UPSTREAM_ERROR", stage="fetch", exc=exc)
        raise HTTPException(
            status_code=502,
            detail=ErrorResponse(
                error=ErrorBody(
                    code="UPSTREAM_ERROR",
                    message="ClinicalTrials.gov could not be reached or returned an error.",
                    details={"upstream": str(exc)},
                )
            ).model_dump(),
        ) from exc
    except ValidationFailure as exc:
        # The response could not be grounded in the fetched records. Returning
        # an error is the correct outcome: a chart that renders but cannot be
        # traced to source data is worse than no chart.
        # One ERROR line, here rather than in the validator, so a failure reads
        # as a single fault carrying both the failing check and the query.
        _log_failure(request, code="VALIDATION_ERROR", stage="validation", exc=exc)
        raise HTTPException(
            status_code=500,
            detail=ErrorResponse(
                error=ErrorBody(
                    code="VALIDATION_ERROR",
                    message="The generated response failed its grounding checks and "
                    "was withheld.",
                    details={"check": str(exc)},
                )
            ).model_dump(),
        ) from exc
