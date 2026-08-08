# [Project Name]

[One or two sentence summary of what this project does — fill in once the assignment is known.]

## Overview

[2-4 sentences: what problem this solves, how it's approached, and the core architecture at a glance —
e.g. "A FastAPI service exposing an agent endpoint that does X. The agent uses [framework] with Y tools
to accomplish Z."]

## Setup

### Prerequisites
- Python 3.11+
- [Any other requirements — Docker, API keys, etc.]

### Installation

```bash
# Clone and enter the project
git clone <repo-url>
cd project

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env and add your API keys
```

### Running the service

```bash
uvicorn app.main:app --reload
```

The service will be available at `http://localhost:8000`. Interactive API docs at `http://localhost:8000/docs`.

### Running tests

```bash
pytest
```

## API / Usage

[Document the main endpoints or CLI commands once built. Example format:]

### `POST /endpoint-name`

**Request:**
```json
{
  "field": "value"
}
```

**Response:**
```json
{
  "field": "value"
}
```

## Architecture

[Brief explanation of how the pieces fit together — the agent logic, how it's structured, any notable
design patterns used (e.g. state graphs, tool-calling, retrieval). A simple diagram or bullet list of
components is fine:]

- `app/main.py` — FastAPI entrypoint
- `app/api/` — route definitions
- `app/agents/` — agent logic
- `app/core/` — config and logging
- `app/models/` — request/response schemas

## Design Decisions

[This is the section that matters most for a take-home — explain the "why" behind non-obvious choices.
Examples of what to cover once you've built the thing:]

- Why this framework/approach was chosen over alternatives
- How errors, edge cases, or ambiguous requirements were handled
- Any assumptions made where the spec was unclear
- Trade-offs between simplicity and robustness given the time constraint

## Trade-offs & Limitations

[Be honest here — this signals engineering maturity more than pretending everything is perfect.]

- [What was deprioritized due to time]
- [Known limitations or edge cases not fully handled]
- [Anything that would need hardening before production use]

## What I'd Do With More Time

[3-5 bullets — shows you understand the gap between "take-home done" and "production-ready."
Examples: better test coverage, caching/performance work, more robust error handling,
observability/logging, handling additional edge cases, etc.]

## Notes

[Anything else worth flagging for the reviewer — how long it took, any blockers hit and how
they were resolved, or context that helps them evaluate the work fairly.]
