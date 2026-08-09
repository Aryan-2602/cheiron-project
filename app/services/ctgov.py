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
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Literal, Self

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

#: ``overallStatus`` value -> ``aggFilters`` code, each confirmed live on
#: 2026-08-09 by checking that the filtered sample contains *only* that status:
#: rec 64,847 / com 326,301 / act 21,858 / not 29,086 / enr 5,236 /
#: ter 34,082 / sus 1,750 / wit 16,627 / unk 95,858.
#:
#: AVAILABLE and NO_LONGER_AVAILABLE are deliberately absent: ``status:avail``
#: and ``status:no_lon`` both return HTTP 200 with zero results -- the same
#: silent-failure trap as ``phase:PHASE3``. Anything not listed here is
#: filtered client-side instead.
STATUS_FILTER_CODES: dict[str, str] = {
    "RECRUITING": "rec",
    "COMPLETED": "com",
    "ACTIVE_NOT_RECRUITING": "act",
    "NOT_YET_RECRUITING": "not",
    "ENROLLING_BY_INVITATION": "enr",
    "TERMINATED": "ter",
    "SUSPENDED": "sus",
    "WITHDRAWN": "wit",
    "UNKNOWN": "unk",
}

VERIFIED_STATUS_CODES = frozenset(STATUS_FILTER_CODES.values())


def series_key(entity: str) -> str:
    """Identity of a comparison series: case- and whitespace-insensitive.

    ``fetch`` keys series membership and provenance by :attr:`CTGovSearch.label`,
    so two entities sharing a key would collide there. They also provably fetch
    the same trials -- verified live that ``query.intr`` is case-insensitive
    (Aspirin / aspirin / ASPIRIN all return 2,172). Deliberately no looser than
    that: stemming or fuzzy matching would merge genuinely distinct drugs.

    Lives here, next to the label it protects, so the query builder and the
    grounding stage cannot drift apart on what counts as the same series.
    """
    return " ".join(entity.split()).casefold()


def normalize_statuses(statuses: Iterable[str]) -> set[str]:
    """Requested statuses as canonical ``overallStatus`` values.

    Shared by the query builder and the client-side filter so both sides agree
    on what was asked for. They normalised differently before, and a padded
    value like ``"  RECRUITING  "`` slipped through both: the builder's
    ``startswith`` test failed on the leading space, and the filter then
    discarded it as already-handled-upstream, so no status filter was applied
    anywhere. Also collapses duplicates and casing differences.
    """
    return {s.strip().upper().replace(" ", "_") for s in statuses if s and s.strip()}


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
    def phase(cls, *phases: int) -> AggFilter:
        if not phases:
            raise ValueError("at least one phase is required")
        for phase in phases:
            if phase not in (1, 2, 3, 4):
                raise ValueError(f"phase must be 1-4, got {phase!r}")
        # Bare numbers, verified live. "PHASE3" returns 200 with zero results.
        # Space-separated values are a true union: phase:2 = 89,652,
        # phase:3 = 49,614, "phase:2 3" = 131,704 -- fewer than the sum,
        # because a combined Phase 2/3 trial is counted once.
        return cls(key="phase", value=" ".join(str(p) for p in sorted(set(phases))))

    @classmethod
    def status(cls, *codes: str) -> AggFilter:
        if not codes:
            raise ValueError("at least one status code is required")
        lowered = sorted({c.lower() for c in codes})
        for code in lowered:
            if code not in VERIFIED_STATUS_CODES:
                raise ValueError(
                    f"status code {code!r} is not live-verified; filter client-side instead"
                )
        # Verified union: rec = 64,847, com = 326,301, "status:rec com" =
        # 391,148 -- exactly the sum, statuses being mutually exclusive.
        return cls(key="status", value=" ".join(lowered))

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

    async def __aenter__(self) -> Self:
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
            # Reached only when the attempt failed retryably and retries remain.
            # Logged because a slow request caused by silent backoff is otherwise
            # indistinguishable from a slow upstream.
            logger.warning(
                "ctgov request retrying",
                extra={
                    "path": path,
                    "attempt": attempt + 1,
                    "max_retries": self.max_retries,
                    "reason": last_error,
                    "delay_s": delay,
                },
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
        self,
        search: CTGovSearch,
        store: StudyStore,
        *,
        max_studies: int,
        requested_max: int | None = None,
    ) -> SearchOutcome:
        """Page through a search, adding records to ``store``.

        Stops at ``max_studies`` and records that it did so, so the response can
        state plainly that aggregates cover a capped sample rather than the
        whole result set.

        ``max_studies`` is this search's budget; ``requested_max`` is the number
        the caller actually asked for, which differs whenever a comparison
        splits one budget across its series. The warning quotes the requested
        number, because quoting the share told the user they were capped at a
        figure they never set -- and then advised them to raise it.
        """
        outcome = SearchOutcome(search=search)
        page_token: str | None = None
        seen_tokens: set[str] = set()
        fetched = 0
        pages = 0
        started = time.perf_counter()

        logger.info(
            "ctgov search started",
            extra={
                "params": search.describe(),
                "max_studies": max_studies,
                "label": search.label,
            },
        )

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

            pages += 1
            page_token = payload.get("nextPageToken")
            logger.debug(
                "ctgov page fetched",
                extra={
                    "page": pages,
                    "records": len(studies),
                    "has_next": bool(page_token),
                },
            )
            if not page_token or not studies:
                break
            if page_token in seen_tokens:
                # A token that points back at a page already fetched is a cycle:
                # it re-reads the same records until the budget runs out and
                # then reports a capped sample, when in fact the whole result
                # set was in hand after the first pass.
                logger.warning(
                    "ctgov pagination token repeated; stopping",
                    extra={"pages": pages, "label": search.label},
                )
                break
            seen_tokens.add(page_token)

        reported_max = requested_max if requested_max is not None else max_studies
        series = f" for {search.label}" if search.label else ""
        # Naming the share as well, when it differs, keeps the advice coherent:
        # the reason this series stopped is its share, but the number to raise
        # is the one the caller set.
        share = (
            f", split into a {max_studies:,}-trial share for this series"
            if reported_max != max_studies
            else ""
        )

        if outcome.total_count is not None and fetched >= outcome.total_count:
            # Provably complete. The upstream hands back a nextPageToken even
            # after the last record, which tripped the budget check on the
            # following iteration and produced "Fetched 100 of 100 (capped)" --
            # a claim of incompleteness about a whole sample. Every disclosure
            # in this project is load-bearing, so a false one is as damaging as
            # the omission it imitates.
            outcome.truncated = False
        elif outcome.truncated or outcome.total_count is not None:
            outcome.truncated = True
            store.truncated = True
            if outcome.total_count is None:
                outcome.warnings.append(
                    f"Fetched {fetched:,} trials{series} and stopped at the "
                    f"max_studies={reported_max:,} cap{share}; the upstream did "
                    f"not report a total, so how much of the result set this "
                    f"covers is unknown."
                )
            else:
                outcome.warnings.append(
                    f"Fetched {fetched:,} of {outcome.total_count:,} matching "
                    f"trials{series} (capped at max_studies={reported_max:,}"
                    f"{share}); figures below describe that sample, not the "
                    f"full result set."
                )

        logger.info(
            "ctgov search completed",
            extra={
                "records": fetched,
                "pages": pages,
                "total_count": outcome.total_count,
                "truncated": outcome.truncated,
                "duration_ms": round((time.perf_counter() - started) * 1000, 1),
            },
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


#: Ceiling on how many extracted values are joined into one query expression,
#: so a pathological extraction cannot build an unbounded query string. Unlike
#: the previous behaviour, going over this is disclosed rather than silent.
MAX_JOINED_VALUES = 5

#: Operators that mean a value is already an expression; nesting one inside
#: another OR would change its meaning.
#:
#: Case-sensitive, and commas are absent, because the previous
#: case-insensitive form treated ordinary names as expressions and dropped
#: every other value: "Head and Neck Cancer" and "Merck Sharp & Dohme, LLC"
#: each suppressed the join. Verified live that neither needs the guard --
#: query.cond="melanoma OR Head and Neck Cancer" returns 11,953, exactly
#: 3,743 + 8,637 - 427, and query.spons="Merck Sharp & Dohme, LLC OR Pfizer"
#: returns 10,260 = 4,276 + 6,061 - 77. A lowercase connector is a stopword
#: inside a phrase and a comma is part of the name; only the uppercase
#: operators change what the expression means.
_EXPRESSION_MARKERS = (" OR ", " AND ", " NOT ", "(", ")")


def join_values(values: Iterable[str]) -> tuple[str | None, list[str]]:
    """Combine several extracted values into one query expression.

    ``OR`` performs a true set union in the ``query.*`` parameters -- verified
    live, and confirmed by inclusion-exclusion: ``query.intr=Pembrolizumab``
    returns 2,922, ``Nivolumab`` 2,016, their comma (intersection) form 290,
    and ``Pembrolizumab OR Nivolumab`` exactly 4,648 = 2922 + 2016 - 290.

    Taking only the first value -- as this once did -- answered a narrower
    question than the one asked and disclosed nothing. Union is also the safe
    reading of an ambiguous phrase like "melanoma and lung cancer trials",
    because it cannot silently exclude trials the user may have meant.

    Returns the expression and any warnings.
    """
    cleaned = [v.strip() for v in values if v and v.strip()]
    if not cleaned:
        return None, []

    warnings: list[str] = []
    if len(cleaned) > MAX_JOINED_VALUES:
        warnings.append(
            f"Used the first {MAX_JOINED_VALUES} of {len(cleaned)} values "
            f"({', '.join(cleaned[MAX_JOINED_VALUES:])} omitted) to keep the "
            f"upstream query bounded."
        )
        cleaned = cleaned[:MAX_JOINED_VALUES]

    if len(cleaned) == 1:
        return cleaned[0], warnings

    # A value that is already an expression is passed through untouched rather
    # than nested inside another OR, which would change what it means.
    if any(m in v for v in cleaned for m in _EXPRESSION_MARKERS):
        note = (
            f"Used {cleaned[0]!r} only: another extracted value already contains "
            f"a search operator, so combining them could change its meaning."
        )
        return cleaned[0], [*warnings, note]

    return " OR ".join(cleaned), warnings


def build_searches(plan: Any) -> tuple[list[CTGovSearch], list[str]]:
    """Translate a :class:`~app.models.schemas.QueryPlan` into upstream searches.

    Comparison queries produce one search per compared entity so that series
    membership is exactly "the trials this search returned" -- no client-side
    guessing about which trial belongs to which series.

    Returns the searches and any warnings about how values were combined.
    """
    entities = plan.entities
    # Notes are attributed to the field they describe. The comparison path
    # replaces one field per series, so a note about how that field's values
    # were combined would disclose an interpretation that was never applied.
    notes_by_field: dict[str, list[str]] = {}

    def combine(field: str, values: list[str], label: str) -> str | None:
        expression, warnings = join_values(values)
        notes = list(warnings)
        if expression and len(values) > 1 and " OR " in expression:
            notes.append(
                f"Interpreted the {len(values)} {label} values as trials matching "
                f"any of them ({expression})."
            )
        notes_by_field[field] = notes
        return expression

    def notes_except(field: str | None = None) -> list[str]:
        return [n for f, ns in notes_by_field.items() if f != field for n in ns]

    base_intr = combine("drug", entities.drugs, "drug")
    base_cond = combine("condition", entities.conditions, "condition")
    base_spons = combine("sponsor", entities.sponsors, "sponsor")
    base_locn = combine("country", entities.countries, "country")

    agg: list[AggFilter] = []
    # Values within one aggFilters key are space-separated and union, verified
    # live. So every requested phase goes upstream in one clause rather than
    # being fetched unfiltered and narrowed afterwards -- which spent the fetch
    # budget on trials that were about to be discarded.
    valid_phases = sorted({p for p in entities.phases if p in (1, 2, 3, 4)})
    if valid_phases:
        agg.append(AggFilter.phase(*valid_phases))
    # Likewise for status, but only when *every* requested status has a
    # live-verified code. A code that is not verified returns HTTP 200 with
    # zero results rather than an error, so a partial upstream filter would
    # silently drop the statuses it could not express; those fall back to the
    # client-side union in apply_client_side_filters.
    wanted_statuses = normalize_statuses(entities.statuses)
    if wanted_statuses and wanted_statuses <= set(STATUS_FILTER_CODES):
        agg.append(AggFilter.status(*(STATUS_FILTER_CODES[s] for s in wanted_statuses)))

    if plan.compare_entities and plan.compare_entity_kind:
        kind = plan.compare_entity_kind
        searches: list[CTGovSearch] = []
        # Labels become dict keys in fetch(), so a duplicate would overwrite an
        # earlier series' membership and provenance. ground_compare_entities
        # already collapses equivalents -- this keeps the guarantee local to the
        # place the keys are minted, rather than resting on a distant caller.
        seen_labels: set[str] = set()
        duplicate_notes: list[str] = []
        for entity in plan.compare_entities:
            key = series_key(entity)
            if key in seen_labels:
                duplicate_notes.append(
                    f"Ignored duplicate comparison entity {entity!r}: it would "
                    f"have replaced an earlier series with the same name."
                )
                continue
            seen_labels.add(key)
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
        # Each compared entity keeps its own search, which is what makes
        # per-series membership exact.
        return searches, [*notes_except(kind), *duplicate_notes]

    search = CTGovSearch(
        intr=base_intr,
        cond=base_cond,
        spons=base_spons,
        locn=base_locn,
        agg_filters=tuple(agg),
    )
    # A question has three separable parts, and only two of them describe the
    # population: the *scope* (drug, condition, sponsor, country), the
    # *predicates* (phase, status, year range), and the *analytical
    # instruction* ("distributed across sponsors", "which countries").
    #
    # Free-text is the last resort for a question that carries none of the
    # first two -- never a way to fill in for a missing scope when predicates
    # exist. Sending the whole question as query.term alongside a filter ANDs
    # the analytical wording into the retrieval predicate, which matches
    # nothing: verified live, aggFilters=phase:3 returns 49,614 trials on its
    # own and 0 with "How are Phase 3 trials distributed across sponsors?"
    # attached.
    has_scope = any((search.intr, search.cond, search.spons, search.locn))
    year_range = entities.year_range
    has_predicate = bool(
        agg
        or entities.phases
        or entities.statuses
        # A status *exclusion* narrows the population exactly as a requested one
        # does; it is simply enforced after fetching. Omitting it here sent
        # "Show trials that are not recruiting by phase" down the free-text
        # fallback, which ANDs the analytical wording into retrieval.
        or getattr(plan, "excluded_statuses", None)
        or (year_range and (year_range.start is not None or year_range.end is not None))
    )
    if not has_scope and not has_predicate:
        search = CTGovSearch(term=plan.query, agg_filters=tuple(agg))
    return [search], notes_except()
