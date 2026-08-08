"""Run the documented example queries end to end and save their real outputs.

Every file under ``examples/`` is produced by this script against the live
ClinicalTrials.gov API -- none is hand-written or edited, so the README's
example outputs are exactly what the service returns.

Usage:
    ./venv/bin/python scripts/run_examples.py           # all examples
    ./venv/bin/python scripts/run_examples.py 01 04     # by number prefix
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
load_dotenv()

from app.models.schemas import QueryRequest  # noqa: E402
from app.pipeline import EmptyResultError, run_pipeline  # noqa: E402
from app.services.ctgov import CTGovClient  # noqa: E402

EXAMPLES: list[tuple[str, dict]] = [
    (
        "01_bar_phase",
        {"query": "How are lung cancer trials distributed across phases?", "max_studies": 1000},
    ),
    (
        "02_time_series",
        {
            "query": "How has the number of trials for Pembrolizumab changed per year since 2015?",
            "max_studies": 1000,
        },
    ),
    (
        "03_comparison",
        {
            "query": "Compare phases for trials involving Pembrolizumab vs Nivolumab.",
            "max_studies": 1000,
        },
    ),
    (
        "04_network_drug_drug",
        {
            "query": "Which drugs frequently co-occur in combination studies with pembrolizumab?",
            "max_studies": 600,
        },
    ),
    (
        "05_network_sponsor_drug",
        {
            "query": "Show a network of sponsors and drugs for melanoma trials.",
            "max_studies": 600,
        },
    ),
    (
        "06_geographic",
        {
            "query": "Which countries have the most recruiting trials for melanoma?",
            "max_studies": 600,
        },
    ),
    (
        "07_structured_override",
        {
            "query": "How are trials for this drug distributed across sponsors?",
            "drug_name": "Nivolumab",
            "phase": 3,
            "max_studies": 600,
        },
    ),
]

OUT_DIR = Path(__file__).resolve().parent.parent / "examples"


async def main() -> int:
    wanted = sys.argv[1:]
    OUT_DIR.mkdir(exist_ok=True)
    failures = 0

    async with CTGovClient() as client:
        for name, payload in EXAMPLES:
            if wanted and not any(name.startswith(w) for w in wanted):
                continue
            print(f"\n=== {name} ===\n{payload['query']}")
            try:
                response = await run_pipeline(QueryRequest(**payload), client=client)
                body = {"request": payload, "response": response.model_dump()}
                rows = response.visualization.data
                if response.visualization.type == "network_graph":
                    summary = (
                        f"{len(rows[0]['nodes'])} nodes, {len(rows[0]['edges'])} edges"
                    )
                else:
                    summary = f"{len(rows)} data rows"
                print(
                    f"  -> {response.visualization.type}: {summary}, "
                    f"{response.meta.total_studies_processed} trials processed"
                )
            except EmptyResultError as exc:
                body = {"request": payload, "empty": True, "meta": exc.meta.model_dump()}
                print("  -> no results")
            except Exception as exc:  # noqa: BLE001 - report and continue
                failures += 1
                print(f"  !! FAILED: {type(exc).__name__}: {exc}")
                continue

            path = OUT_DIR / f"{name}.json"
            path.write_text(json.dumps(body, indent=2))
            print(f"  -> wrote {path.relative_to(Path.cwd())}")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
