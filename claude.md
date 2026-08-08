# CLAUDE.md

## Project
Backend service for ClinicalTrials.gov query-to-visualization agent (Cheiron take-home).
Assignment spec: /assignments/ (images)
Verified API findings: /docs/api-notes.md
Full plan: /docs/PLAN.md (from Fable 5 planning session)

## Conventions
- Python, FastAPI, Pydantic v2
- Every visualized data value must be computed from real API responses — never LLM-generated
- Structure: separate query-understanding, fetching, aggregation, citation, and validation as distinct testable modules
- Write tests alongside implementation, not after
- Commit after each working module with descriptive messages
- Keep aggregators behind a shared interface — no per-chart-type one-off logic in route handlers

## Current focus
[update this as you move through phases — e.g. "building bar_chart + time_series aggregators"]