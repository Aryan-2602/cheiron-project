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
- Aggregators stay behind the shared aggregate() interface — no per-chart-type logic
  in route handlers or the pipeline

## Current focus
Complete. All 6 pipeline stages implemented, 166 tests passing, 7 live examples in examples/.
