# CLAUDE.md

## Project
Backend service for ClinicalTrials.gov query-to-visualization agent (Cheiron take-home).
Assignment spec: assignment/ (images)
Verified API findings: docs/api-notes.md  <- ground truth, beats general knowledge
Full plan: docs/PLAN.md

## Stack
Python 3.12, FastAPI, Pydantic v2, OpenAI SDK (structured outputs), httpx, pytest.
No langchain/langgraph — one LLM call with a fixed schema needs no framework.

## Conventions
- Every visualized data value must be computed from real API responses — never LLM-generated
- Structure: query-understanding, fetching, aggregation, citation, and validation are
  distinct testable modules (app/agents, app/services)
- Write tests alongside implementation, not after
- Commit after each working module with descriptive messages
- Aggregation stays behind the shared aggregate() interface, with per-axis variation in
  the Dimension registry — adding a chart axis means adding a Dimension and nothing else.
  Chart-*family* dispatch (network vs chart, temporal vs categorical) does exist in the
  pipeline and the empty-result handler, deliberately: an empty spec has to keep the
  family it was asked for, or a frontend routing on `type` renders it wrong.

## Current focus
Complete. All 6 pipeline stages implemented, plus RxNorm drug resolution, structured
logging, and a demo frontend in frontend/. Run `pytest -q` for the current test count --
do not hardcode it here, it only goes stale.
