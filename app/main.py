from fastapi import FastAPI

from app.api.routes import router
from app.core.logging import setup_logging

setup_logging()

app = FastAPI(title="LLM Agent Service")
app.include_router(router)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
