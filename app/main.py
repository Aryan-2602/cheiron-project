import logging
import time
import uuid
from collections.abc import Awaitable, Callable

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse

from app.api.routes import router
from app.core.config import settings
from app.core.logging import request_id, setup_logging
from app.models.schemas import ErrorBody, ErrorResponse

setup_logging()

logger = logging.getLogger(__name__)

app = FastAPI(
    title="ClinicalTrials.gov Query-to-Visualization Agent",
    version="1.0.0",
    description=(
        "Turns natural-language questions about clinical trials into structured "
        "visualization specifications backed by live ClinicalTrials.gov API v2 "
        "data.\n\n"
        "Every value in a response is computed by aggregation code over real "
        "trial records and carries citations to the trials that produced it. "
        "The language model is used only to interpret the question -- it never "
        "generates a data value."
    ),
)
app.include_router(router)

# Development-only CORS, so the demo page in frontend/ can call the API from a
# different origin (a static server on another port, or file:// which sends
# Origin: null). Wide-open is acceptable *only* because this is gated on ENV and
# the API uses no cookies or auth -- allow_credentials stays False, which is
# what makes a wildcard origin safe. A deployment with ENV=production gets no
# CORS at all. This is not a production CORS policy.
if settings.ENV == "development":
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )


@app.middleware("http")
async def log_requests(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    """Tag each request with an id and log how it finished.

    This is also the only place FastAPI's own 422s for malformed request bodies
    are visible -- those are rejected before the route handler runs, so without
    a middleware they would leave no trace at all.
    """
    token = request_id.set(uuid.uuid4().hex[:8])
    started = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        # An unhandled exception still produces a 500 downstream; log it here so
        # the stage that failed is not lost.
        logger.exception(
            "request failed",
            extra={
                "method": request.method,
                "path": request.url.path,
                "duration_ms": round((time.perf_counter() - started) * 1000, 1),
            },
        )
        request_id.reset(token)
        raise

    duration_ms = round((time.perf_counter() - started) * 1000, 1)
    logger.log(
        logging.ERROR if response.status_code >= 400 else logging.INFO,
        "request completed",
        extra={
            "method": request.method,
            "path": request.url.path,
            "status": response.status_code,
            "duration_ms": duration_ms,
        },
    )
    request_id.reset(token)
    return response


@app.exception_handler(Exception)
async def unhandled_exception(request: Request, exc: Exception) -> JSONResponse:
    """Make an unforeseen failure honour the documented error contract.

    The route advertises ``500: {"model": ErrorResponse}`` and the README says
    domain error bodies *are* an ErrorResponse. Without this, anything the
    specific handlers did not anticipate fell through to Starlette's default
    text/plain "Internal Server Error" -- a body no documented client can
    parse, on the one path where a caller most needs to know what happened.

    The message is fixed text. Nothing user-controlled and nothing from the
    exception goes into the body: an error string can carry a fragment of the
    query, an upstream URL, or a filesystem path, and this is the one response
    produced by a code path nobody reasoned about. The detail is logged
    instead, where the request id ties it back to this response.
    """
    logger.exception(
        "unhandled exception",
        extra={"method": request.method, "path": request.url.path},
    )
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error=ErrorBody(
                code="INTERNAL_ERROR",
                message=(
                    "An unexpected error occurred and the response was withheld."
                ),
                details={},
            )
        ).model_dump(),
    )


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse(url="/docs")


@app.get("/health", tags=["ops"])
def health() -> dict[str, str]:
    return {"status": "ok"}
