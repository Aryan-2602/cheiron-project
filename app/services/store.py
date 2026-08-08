"""Per-request store of the raw ClinicalTrials.gov records backing a response.

Everything downstream -- aggregation, network building, citations, validation --
reads from here. Keeping the raw records for the life of the request is what
makes deep citations possible without a second round of per-study API calls:
the excerpt for a datum is drawn from the same record that put the trial in
that bucket.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

CTGOV_STUDY_URL = "https://clinicaltrials.gov/study/{nct_id}"


def extract_nct_id(record: dict[str, Any]) -> str | None:
    """Pull the NCT id out of a raw study record, tolerating sparse responses."""
    return (
        record.get("protocolSection", {})
        .get("identificationModule", {})
        .get("nctId")
    )


@dataclass
class StudyStore:
    """Deduplicated collection of raw study records keyed by NCT id.

    Deduplication matters for comparison queries, where the same trial can be
    returned by more than one upstream search (a trial testing both compared
    drugs). The store holds one copy; series membership is tracked separately.
    """

    records: dict[str, dict[str, Any]] = field(default_factory=dict)
    api_urls: list[str] = field(default_factory=list)
    #: Reported ``totalCount`` per upstream search, before any fetch cap.
    total_counts: list[int] = field(default_factory=list)
    truncated: bool = False

    def add_records(self, records: Iterable[dict[str, Any]]) -> set[str]:
        """Add raw records; return the set of NCT ids seen in this batch."""
        seen: set[str] = set()
        for record in records:
            nct_id = extract_nct_id(record)
            if not nct_id:
                continue
            seen.add(nct_id)
            # First writer wins: identical records, and re-fetching cannot
            # improve on what we already have.
            self.records.setdefault(nct_id, record)
        return seen

    def add_url(self, url: str) -> None:
        self.api_urls.append(url)

    def get(self, nct_id: str) -> dict[str, Any] | None:
        return self.records.get(nct_id)

    def __contains__(self, nct_id: object) -> bool:
        return nct_id in self.records

    def __len__(self) -> int:
        return len(self.records)

    @property
    def nct_ids(self) -> set[str]:
        return set(self.records)

    @property
    def reported_total(self) -> int | None:
        """Largest upstream ``totalCount``, used to disclose fetch truncation."""
        return max(self.total_counts) if self.total_counts else None

    @staticmethod
    def study_url(nct_id: str) -> str:
        return CTGOV_STUDY_URL.format(nct_id=nct_id)
