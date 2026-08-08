from fastapi import FastAPI
from fastapi.responses import RedirectResponse

from app.api.routes import router
from app.core.logging import setup_logging

setup_logging()

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


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse(url="/docs")


@app.get("/health", tags=["ops"])
def health() -> dict[str, str]:
    return {"status": "ok"}
