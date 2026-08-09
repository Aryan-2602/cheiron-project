"""Per-request store of the raw ClinicalTrials.gov records backing a response.

Everything downstream -- aggregation, network building, citations, validation --
reads from here. Keeping the raw records for the life of the request is what
makes deep citations possible without a second round of per-study API calls:
the excerpt for a datum is drawn from the same record that put the trial in
that bucket.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from app.models.schemas import NCT_ID_PATTERN
from app.services.dimensions import protocol_module

logger = logging.getLogger(__name__)

CTGOV_STUDY_URL = "https://clinicaltrials.gov/study/{nct_id}"

#: Compiled from the schema's pattern rather than restated, so the store and
#: Citation can never disagree about what a citable id is.
_NCT_ID = re.compile(NCT_ID_PATTERN)


def extract_nct_id(record: dict[str, Any]) -> str | None:
    """Pull the NCT id out of a raw study record, tolerating sparse responses."""
    nct_id = protocol_module(record, "identificationModule").get("nctId")
    return nct_id if isinstance(nct_id, str) else None


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
        """Add raw records; return the set of NCT ids seen in this batch.

        Records whose id does not match :data:`NCT_ID_PATTERN` are refused at
        the door rather than at the citation site. ``Citation.nct_id`` enforces
        the same pattern, so a record with id "BADID" was admitted, reached a
        bucket, inflated its count, and then raised an uncaught ValidationError
        when the citation was built. A record we cannot cite cannot be
        evidence -- admitting it would let it support a figure that has no
        traceable backing, which is exactly what the validator exists to catch.
        """
        seen: set[str] = set()
        skipped = 0
        for record in records:
            nct_id = extract_nct_id(record)
            if not nct_id or not _NCT_ID.match(nct_id):
                # Counted, not logged per record: a malformed page would
                # otherwise emit one line per study.
                skipped += 1
                continue
            seen.add(nct_id)
            # First writer wins: identical records, and re-fetching cannot
            # improve on what we already have.
            self.records.setdefault(nct_id, record)
        if skipped:
            logger.debug(
                "records skipped for an unusable nct id", extra={"count": skipped}
            )
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
