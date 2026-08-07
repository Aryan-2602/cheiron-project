# LLM Agent Service

FastAPI backend skeleton for hosting an LLM agent (LangChain / LangGraph).

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in API keys as needed.

## Run

```bash
uvicorn app.main:app --reload
```

## Test

```bash
pytest
```
