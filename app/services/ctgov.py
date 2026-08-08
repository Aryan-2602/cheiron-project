"""ClinicalTrials.gov API v2 client and query builder.

Every parameter used here was verified against the live API (see
``docs/api-notes.md``). Two verified findings shape this module:

* ``filter.phase`` **does not exist** -- it returns HTTP 400. Phase filtering
  goes through ``aggFilters=phase:3``, with a **bare number**. The full enum
  form (``phase:PHASE3``) returns HTTP 200 with zero results, i.e. it fails
  silently. ``AggFilter`` makes the correct form the only representable one.
* Pagination is ``pageToken`` only; there is no offset. ``pageSize`` defaults to
  10 when omitted, so it is always set explicitly.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Iterable, Literal

import httpx

from app.core.config import settings
from app.services.store import StudyStore

logger = logging.getLogger(__name__)

# Response-path mappings for these were confirmed with a live call on
# 2026-08-08; the request names are PascalCase and do NOT convert mechanically
# to the camelCase response paths (see app/services/dimensions.py).
DEFAULT_FIELDS: tuple[str, ...] = (
    "NCTId",
    "BriefTitle",
    "OverallStatus",
    "Phase",
    "LeadSponsorName",
    "LeadSponsorClass",
    "StartDate",
    "EnrollmentCount",
    "InterventionName",
    "InterventionType",
    "LocationCountry",
)

#: Only ``status:rec`` was confirmed live. Other abbreviations are treated as
#: unverified and handled by client-side filtering instead (see the pipeline).
VERIFIED_STATUS_CODES = {"rec"}


class CTGovError(RuntimeError):
    """Upstream API failure that the pipeline surfaces as UPSTREAM_ERROR."""


@dataclass(frozen=True)
class AggFilter:
    """A single ``aggFilters`` clause.

    Constructing a phase filter through this type is the only supported path,
    which is what prevents the silently-empty ``phase:PHASE3`` form from ever
    being emitted.
    """

    key: Literal["phase", "status"]
    value: str

    @classmethod
    def phase(cls, phase: int) -> AggFilter:
        if phase not in (1, 2, 3, 4):
            raise ValueError(f"phase must be 1-4, got {phase!r}")
        # Bare number, verified live. "PHASE3" returns 200 with zero results.
        return cls(key="phase", value=str(phase))

    @classmethod
    def status(cls, code: str) -> AggFilter:
        code = code.lower()
        if code not in VERIFIED_STATUS_CODES:
            raise ValueError(
                f"status code {code!r} is not live-verified; filter client-side instead"
            )
        return cls(key="status", value=code)

    def render(self) -> str:
        return f"{self.key}:{self.value}"


@dataclass(frozen=True)
class CTGovSearch:
    """One upstream search. Pure data -- building it never touches the network,
    which is what makes the query builder unit-testable."""

    intr: str | None = None
    cond: str | None = None
    spons: str | None = None
    locn: str | None = None
    term: str | None = None
    agg_filters: tuple[AggFilter, ...] = ()
    fields: tuple[str, ...] = DEFAULT_FIELDS
    #: Series label for comparison queries ("Pembrolizumab" vs "Nivolumab").
    label: str | None = None

    def to_params(
        self,
        *,
        page_size: int,
        page_token: str | None = None,
        count_total: bool = False,
        include_agg_filters: bool = True,
    ) -> dict[str, str]:
        params: dict[str, str] = {
            "fields": ",".join(self.fields),
            "pageSize": str(page_size),
        }
        if self.intr:
            params["query.intr"] = self.intr
        if self.cond:
            params["query.cond"] = self.cond
        if self.spons:
            params["query.spons"] = self.spons
        if self.locn:
            params["query.locn"] = self.locn
        if self.term:
            params["query.term"] = self.term
        if include_agg_filters and self.agg_filters:
            params["aggFilters"] = ",".join(f.render() for f in self.agg_filters)
        if count_total:
            params["countTotal"] = "true"
        if page_token:
            params["pageToken"] = page_token
        return params

    def describe(self) -> dict[str, str]:
        """Human-readable filter summary for ``meta.filters``."""
        out: dict[str, str] = {}
        for name, value in (
            ("intervention", self.intr),
            ("condition", self.cond),
            ("sponsor", self.spons),
            ("location", self.locn),
            ("search_term", self.term),
        ):
            if value:
                out[name] = value
        for f in self.agg_filters:
            out[f.key] = f.value
        return out


@dataclass
class SearchOutcome:
    """Result of running one :class:`CTGovSearch` to completion."""

    search: CTGovSearch
    nct_ids: set[str] = field(default_factory=set)
    total_count: int | None = None
    truncated: bool = False
    warnings: list[str] = field(default_factory=list)


class CTGovClient:
    """Async client with pagination, retry/backoff, and a silent-filter check."""

    def __init__(
        self,
        client: httpx.AsyncClient | None = None,
        *,
        base_url: str | None = None,
        page_size: int | None = None,
        max_retries: int | None = None,
    ) -> None:
        self.base_url = (base_url or settings.CTGOV_BASE_URL).rstrip("/")
        self.page_size = page_size or settings.CTGOV_PAGE_SIZE
        self.max_retries = max_retries if max_retries is not None else settings.CTGOV_MAX_RETRIES
        self._client = client
        self._owns_client = client is None

    async def __aenter__(self) -> CTGovClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=settings.CTGOV_TIMEOUT_SECONDS)
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            raise RuntimeError("CTGovClient must be used as an async context manager")
        return self._client

    # ---------------------------------------------------------------- HTTP

    async def _get(self, path: str, params: dict[str, str]) -> tuple[dict[str, Any], str]:
        """GET with retry on 429 and 5xx. Returns (payload, request_url)."""
        url = f"{self.base_url}{path}"
        delay = 1.0
        last_error: str = ""
        for attempt in range(self.max_retries + 1):
            try:
                response = await self.client.get(url, params=params)
            except httpx.HTTPError as exc:  # network-level failure
                last_error = f"{type(exc).__name__}: {exc}"
                if attempt == self.max_retries:
                    raise CTGovError(f"request to {url} failed: {last_error}") from exc
            else:
                if response.status_code == 200:
                    return response.json(), str(response.request.url)
                if response.status_code == 429 or response.status_code >= 500:
                    last_error = f"HTTP {response.status_code}"
                    if attempt == self.max_retries:
                        raise CTGovError(
                            f"upstream returned {response.status_code} after "
                            f"{self.max_retries + 1} attempts: {response.text[:300]}"
                        )
                else:
                    # 4xx other than 429 will not succeed on retry. The response
                    # body carries the API's own diagnostic, which is worth
                    # surfacing verbatim (this is how filter.phase was caught).
                    raise CTGovError(
                        f"upstream returned {response.status_code} for {response.request.url}: "
                        f"{response.text[:300]}"
                    )
            await asyncio.sleep(delay)
            delay *= 2
        raise CTGovError(f"request to {url} failed: {last_error}")

    # -------------------------------------------------------------- public

    async def get_data_timestamp(self) -> str | None:
        """``dataTimestamp`` from ``GET /version``, for ``meta.data_as_of``."""
        try:
            payload, _ = await self._get("/version", {})
        except CTGovError:
            logger.warning("could not fetch CTGov version/dataTimestamp", exc_info=True)
            return None
        return payload.get("dataTimestamp")

    async def run_search(
        self, search: CTGovSearch, store: StudyStore, *, max_studies: int
    ) -> SearchOutcome:
        """Page through a search, adding records to ``store``.

        Stops at ``max_studies`` and records that it did so, so the response can
        state plainly that aggregates cover a capped sample rather than the
        whole result set.
        """
        outcome = SearchOutcome(search=search)
        page_token: str | None = None
        fetched = 0

        while True:
            remaining = max_studies - fetched
            if remaining <= 0:
                outcome.truncated = True
                break
            params = search.to_params(
                page_size=min(self.page_size, remaining),
                page_token=page_token,
                count_total=page_token is None,
            )
            payload, url = await self._get("/studies", params)
            store.add_url(url)

            studies = payload.get("studies") or []
            outcome.nct_ids |= store.add_records(studies)
            fetched += len(studies)

            if outcome.total_count is None and "totalCount" in payload:
                outcome.total_count = payload["totalCount"]
                store.total_counts.append(payload["totalCount"])

            page_token = payload.get("nextPageToken")
            if not page_token or not studies:
                break

        if outcome.truncated or (
            outcome.total_count is not None and fetched < outcome.total_count
        ):
            outcome.truncated = True
            store.truncated = True
            outcome.warnings.append(
                f"Fetched {fetched:,} of {outcome.total_count:,} matching trials "
                f"(capped at max_studies={max_studies:,}); figures below describe "
                f"that sample, not the full result set."
            )

        await self._check_empty_filtered_result(search, outcome)
        return outcome

    async def _check_empty_filtered_result(
        self, search: CTGovSearch, outcome: SearchOutcome
    ) -> None:
        """Explain an empty filtered result instead of reporting a bare zero.

        A 200 response with ``studies: []`` is indistinguishable from a filter
        that silently matched nothing (the documented ``phase:PHASE3`` trap).
        When a filtered search comes back empty we re-ask without the filter: if
        the unfiltered search has hits, the zero is real but worth explaining;
        if it does not, the query itself matched nothing.
        """
        if outcome.nct_ids or not search.agg_filters:
            return
        params = search.to_params(
            page_size=1, count_total=True, include_agg_filters=False
        )
        try:
            payload, _ = await self._get("/studies", params)
        except CTGovError:
            return
        unfiltered_total = payload.get("totalCount") or 0
        if unfiltered_total > 0:
            applied = ", ".join(f.render() for f in search.agg_filters)
            outcome.warnings.append(
                f"No trials matched the filter ({applied}), though {unfiltered_total:,} "
                f"trials matched the same search without it."
            )


# --------------------------------------------------------------------------
# Query builder (pure)
# --------------------------------------------------------------------------


def _first(values: Iterable[str]) -> str | None:
    for value in values:
        if value and value.strip():
            return value.strip()
    return None


def build_searches(plan: "Any") -> list[CTGovSearch]:
    """Translate a :class:`~app.models.schemas.QueryPlan` into upstream searches.

    Comparison queries produce one search per compared entity so that series
    membership is exactly "the trials this search returned" -- no client-side
    guessing about which trial belongs to which series.
    """
    entities = plan.entities
    base_intr = _first(entities.drugs)
    base_cond = _first(entities.conditions)
    base_spons = _first(entities.sponsors)
    base_locn = _first(entities.countries)

    agg: list[AggFilter] = []
    for phase in entities.phases:
        try:
            agg.append(AggFilter.phase(phase))
        except ValueError:
            continue
    # Only one phase filter is meaningful at a time upstream; extra phases are
    # applied client-side by the aggregator's own bucketing.
    agg = agg[:1]
    if any(s.lower().startswith("recruit") for s in entities.statuses):
        agg.append(AggFilter.status("rec"))

    if plan.compare_entities and plan.compare_entity_kind:
        kind = plan.compare_entity_kind
        searches: list[CTGovSearch] = []
        for entity in plan.compare_entities:
            searches.append(
                CTGovSearch(
                    intr=entity if kind == "drug" else base_intr,
                    cond=entity if kind == "condition" else base_cond,
                    spons=entity if kind == "sponsor" else base_spons,
                    locn=base_locn,
                    agg_filters=tuple(agg),
                    label=entity,
                )
            )
        return searches

    search = CTGovSearch(
        intr=base_intr,
        cond=base_cond,
        spons=base_spons,
        locn=base_locn,
        agg_filters=tuple(agg),
    )
    # Nothing structured to search on: fall back to free-text so the request
    # still hits real data rather than returning the entire registry.
    if not any((search.intr, search.cond, search.spons, search.locn)):
        search = CTGovSearch(term=plan.query, agg_filters=tuple(agg))
    return [search]
