"""Resolve free-text drug names to a canonical RxNorm ingredient.

``normalize_intervention`` in :mod:`app.services.network` collapses cosmetic
variation -- dosages, parentheticals, administration routes -- so
``Pembrolizumab 200 mg`` and ``Pembrolizumab (MK-3475)`` already share a node.
It cannot collapse ``Keytruda`` and ``pembrolizumab``, which share no
characters. That split understates every edge touching the compound, because
the trials naming the brand and the trials naming the generic are counted as
evidence for two different things.

This module closes that gap using RxNorm (NLM, no API key). It runs *after*
string normalization and only handles the brand-vs-generic case.

Why the ingredient walk
-----------------------
Brand and generic are **distinct RxNorm concepts** -- Keytruda is RxCUI
1547550, pembrolizumab is 1547545 -- so an ``approximateTerm`` match alone does
not merge them, and the concept's own preferred name for Keytruda is
"Keytruda". Resolution therefore takes two steps:

1. ``approximateTerm`` maps a messy string to a concept and a confidence score.
2. ``related.json?tty=IN`` walks that concept to its **ingredient**.

Step 2 is idempotent -- an ingredient relates to itself -- so brand and generic
input converge on one identity through a single code path, and the response
carries the canonical ingredient name, so no third call is needed.

Failure is always soft
----------------------
RxNorm is an enrichment, not the data. Every failure mode -- no match, a score
below threshold, a timeout, a 500, a combination product with no single
ingredient -- returns an *unresolved* :class:`DrugResolution` carrying the
cleaned name. Nothing in this module raises into the pipeline; the worst case
is that graphs behave exactly as they did before RxNorm existed.
"""

from __future__ import annotations

import asyncio
import logging
import math
import re
import time
from collections import OrderedDict
from collections.abc import Collection, Mapping
from typing import Any, Self

import httpx

from app.core.config import settings
from app.models.schemas import DrugResolution
from app.services.dimensions import protocol_module
from app.services.network import STOPWORD_DRUGS, normalize_intervention

logger = logging.getLogger(__name__)

#: Intervention types eligible for RxNorm lookup. ``COMBINATION_PRODUCT`` is
#: deliberately excluded: those names are compound descriptions that RxNorm
#: mis-matches to one arbitrary component (verified live -- "pembrolizumab and
#: lenvatinib" resolves to pembrolizumab alone, silently dropping the other
#: agent). They still become nodes, just via string normalization.
RESOLVABLE_TYPES = {"DRUG", "BIOLOGICAL"}

#: Strings that name more than one agent. RxNorm answers these confidently and
#: wrongly -- it matches one component and silently discards the rest -- so they
#: never reach the API.
_CONJUNCTION = re.compile(r"\s(?:and|or|with|plus|versus|vs)\s|[+/,]")

#: Control-arm and comparator phrasing. "Placebo for Pembrolizumab" is a
#: placebo arm, and resolving it to pembrolizumab would count control-arm
#: trials as evidence for the active drug.
_COMPARATOR = re.compile(r"\b(placebo|comparator|matching|vehicle|sham)\b")

#: Research/sponsor compound codes ("MK-3475", "ACE2016", "BMS 936558").
#: RxNorm either has no entry or matches something unrelated at low confidence.
_RESEARCH_CODE = re.compile(r"^[a-z]{1,4}[\s-]?\d{3,}[a-z]?$")

#: Formulation and sourcing words that decorate a single agent's name rather
#: than naming a second one. A multi-word name is only safe to resolve when
#: every token but one is drawn from this list -- that is what separates
#: "nab paclitaxel" (one drug, decorated) from "favezelimab pembrolizumab"
#: (two drugs, where resolution would steal the trial from favezelimab).
_QUALIFIER_TOKENS = frozenset(
    {
        "nab", "albumin", "bound", "pegylated", "peg", "liposomal", "liposome",
        "hydrochloride", "hcl", "sodium", "sulfate", "sulphate", "acetate",
        "citrate", "tartrate", "phosphate", "mesylate", "maleate", "succinate",
        "besylate", "fumarate", "dihydrate", "monohydrate", "calcium",
        "potassium", "magnesium", "injection", "injectable", "product",
        "concentrate", "conjugate", "recombinant", "human", "humanized",
        "biosimilar", "chemotherapy", "therapy", "monotherapy", "arm", "group",
        "drug", "agent", "acid", "free", "base", "alfa", "beta", "gamma",
        "vitamin", "solution", "suspension", "extended", "release",
    }
)


class RxNormError(RuntimeError):
    """Upstream RxNorm failure. Never escapes this module -- callers get an
    unresolved :class:`DrugResolution` instead."""


# --------------------------------------------------------------------------
# Cache
# --------------------------------------------------------------------------


class DrugCache:
    """Bounded name -> resolution cache.

    Scoped module-level rather than per-request on purpose: a name-to-ingredient
    mapping is globally stable (RxNorm ships monthly at most, ingredient
    identity effectively never changes) and carries no per-user variance, while
    the same drug names recur across nearly every oncology query. A per-request
    cache would discard the majority of the benefit.

    Negative results are cached too -- a name RxNorm does not know should cost
    one request, not one per trial that mentions it.
    """

    def __init__(self, max_entries: int = 5000) -> None:
        self.max_entries = max_entries
        self._entries: OrderedDict[str, DrugResolution] = OrderedDict()
        self.hits = 0
        self.misses = 0

    def get(self, name: str) -> DrugResolution | None:
        entry = self._entries.get(name)
        if entry is None:
            self.misses += 1
        else:
            self.hits += 1
        return entry

    def set(self, name: str, resolution: DrugResolution) -> None:
        if name in self._entries:
            self._entries[name] = resolution
            return
        if len(self._entries) >= self.max_entries:
            # FIFO eviction: bounded memory in a long-running process, and drug
            # popularity is stable enough that recency ranking is not worth the
            # extra bookkeeping.
            self._entries.popitem(last=False)
        self._entries[name] = resolution

    def clear(self) -> None:
        """Drop every entry. Used to isolate tests from the module singleton,
        and available operationally to force re-resolution after an RxNorm
        release without restarting the process."""
        self._entries.clear()
        self.hits = 0
        self.misses = 0

    def stats(self) -> dict[str, int]:
        return {"size": len(self._entries), "hits": self.hits, "misses": self.misses}

    def __len__(self) -> int:
        return len(self._entries)


#: Process-local singleton. Not shared across worker processes, which is fine
#: at this scale -- a cold worker simply re-resolves and warms up.
DRUG_CACHE = DrugCache()


# --------------------------------------------------------------------------
# Payload access
# --------------------------------------------------------------------------
#
# RxNorm answers 200 with a body it does not promise the shape of, and a
# malformed one used to surface as an AttributeError from deep inside a
# ``.get()`` chain -- an untyped crash rather than the soft degradation this
# module is built around. These three keep every access total: an unexpected
# shape reads as "no data here", which the callers already handle.


def _sub(payload: Any, key: str) -> Any:
    """``payload[key]`` when payload is a mapping, else ``None``."""
    return payload.get(key) if isinstance(payload, Mapping) else None


def _seq(value: Any) -> list[Any]:
    """``value`` as a list of items, or empty when it is not a real sequence.

    Strings are excluded deliberately: iterating one yields characters, which
    would silently turn a malformed field into a long run of junk entries.
    """
    return value if isinstance(value, list) else []


def _finite_float(value: Any) -> float | None:
    """A usable float, or ``None`` for anything unparseable or non-finite."""
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


# --------------------------------------------------------------------------
# Client
# --------------------------------------------------------------------------


class RxNormClient:
    """Async RxNorm client, mirroring the retry/backoff shape of
    :class:`app.services.ctgov.CTGovClient`."""

    def __init__(
        self,
        client: httpx.AsyncClient | None = None,
        *,
        base_url: str | None = None,
        max_retries: int | None = None,
        min_score: float | None = None,
    ) -> None:
        self.base_url = (base_url or settings.RXNORM_BASE_URL).rstrip("/")
        self.max_retries = (
            max_retries if max_retries is not None else settings.RXNORM_MAX_RETRIES
        )
        self.min_score = (
            min_score if min_score is not None else settings.RXNORM_MIN_SCORE
        )
        self._client = client
        self._owns_client = client is None

    async def __aenter__(self) -> Self:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=settings.RXNORM_TIMEOUT_SECONDS)
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            raise RuntimeError("RxNormClient must be used as an async context manager")
        return self._client

    async def _get(self, path: str, params: dict[str, str] | None = None) -> dict[str, Any]:
        """GET with retry on 429 and 5xx; other 4xx fail immediately."""
        url = f"{self.base_url}{path}"
        delay = 0.5
        last_error = ""
        for attempt in range(self.max_retries + 1):
            try:
                response = await self.client.get(url, params=params or {})
            except httpx.HTTPError as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                if attempt == self.max_retries:
                    raise RxNormError(f"request to {url} failed: {last_error}") from exc
            else:
                if response.status_code == 200:
                    try:
                        return response.json()
                    except ValueError as exc:
                        raise RxNormError(f"non-JSON response from {url}") from exc
                if response.status_code == 429 or response.status_code >= 500:
                    last_error = f"HTTP {response.status_code}"
                    if attempt == self.max_retries:
                        raise RxNormError(
                            f"RxNorm returned {response.status_code} after "
                            f"{self.max_retries + 1} attempts"
                        )
                else:
                    raise RxNormError(
                        f"RxNorm returned {response.status_code} for {url}"
                    )
            # DEBUG rather than WARNING: resolution fans out across hundreds
            # of names, so one outage would emit hundreds of near-identical
            # lines. The single aggregate WARNING from resolve_all carries the
            # signal without drowning the log. (CTGov retries stay at WARNING --
            # there are at most a handful per request.)
            logger.debug(
                "rxnorm request retrying",
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
        raise RxNormError(f"request to {url} failed: {last_error}")

    async def approximate_term(self, name: str) -> tuple[str, float] | None:
        """Best-matching concept for a messy name, or ``None`` below threshold.

        A no-match response omits the ``candidate`` key entirely rather than
        returning an empty list, so the lookup is written defensively.
        """
        payload = await self._get(
            "/approximateTerm.json", {"term": name, "maxEntries": "1"}
        )
        candidates = _sub(_sub(payload, "approximateGroup"), "candidate")
        if not isinstance(candidates, list) or not candidates:
            return None

        best = candidates[0]
        if not isinstance(best, Mapping):
            return None
        rxcui = best.get("rxcui")
        score = _finite_float(best.get("score"))
        # Stated positively on purpose. The gate used to be ``score <
        # min_score``, and every comparison with NaN is False -- so a score of
        # "NaN" sailed through the confidence check instead of tripping it and
        # merged an arbitrary RxCUI. An unusable score must fail closed.
        if not rxcui or score is None or not score >= self.min_score:
            return None
        return str(rxcui), score

    async def ingredient_for(self, rxcui: str) -> tuple[str, str] | None:
        """Walk a concept to its single ingredient: ``(rxcui, canonical_name)``.

        Returns ``None`` unless there is **exactly one** ingredient. A
        multi-ingredient brand (Trikafta) returns a concept group with no
        members at all, because it relates to ``MIN`` rather than ``IN``;
        refusing anything but a single ingredient is what stops combination
        products from being collapsed onto one arbitrary component.
        """
        payload = await self._get(f"/rxcui/{rxcui}/related.json", {"tty": "IN"})
        groups = _sub(_sub(payload, "relatedGroup"), "conceptGroup")
        if not isinstance(groups, list):
            return None
        concepts = [
            concept
            for group in groups
            for concept in _seq(_sub(group, "conceptProperties"))
            if isinstance(concept, Mapping)
            and concept.get("rxcui")
            and concept.get("name")
        ]
        if len(concepts) != 1:
            return None
        return str(concepts[0]["rxcui"]), str(concepts[0]["name"])


# --------------------------------------------------------------------------
# Resolution
# --------------------------------------------------------------------------


def is_resolvable(cleaned_name: str) -> bool:
    """Whether a cleaned name is worth asking RxNorm about.

    This is a correctness guard, not an optimization. RxNorm answers multi-drug
    strings *confidently and wrongly*: verified against live trial data,
    ``"ipilimumab pembrolizumab durvalumab idarubicin bevacizumab"`` resolves to
    durvalumab alone, and ``"favezelimab pembrolizumab"`` resolves to
    pembrolizumab -- which would move favezelimab's trials onto pembrolizumab's
    node. Because a wrong merge is invisible in the finished graph, anything
    ambiguous is refused rather than guessed at.

    A name passes only when it plausibly denotes exactly one agent:

    * no multi-agent connective (``and``, ``or``, ``with``, ``+``, ``/``, ...)
    * no comparator or control-arm wording (``placebo for pembrolizumab``)
    * not a bare research code (``MK-3475``)
    * at most one token that is not a known formulation qualifier, so
      ``nab paclitaxel`` and ``doxorubicin hydrochloride`` pass while
      ``gemcitabine nab-paclitaxel`` does not
    """
    name = cleaned_name.strip().lower()
    if len(name) < 3:
        return False
    if _CONJUNCTION.search(name) or _COMPARATOR.search(name):
        return False
    if _RESEARCH_CODE.match(name):
        return False

    # Hyphens join a qualifier to its agent ("nab-paclitaxel"), so split on them
    # too rather than treating the pair as one opaque token.
    tokens = [t for t in re.split(r"[\s-]+", name) if t]
    # Bare numbers are positional parts of a name ("5-FU", "vitamin B12"), not
    # separate agents, so they never count toward the one-agent budget.
    substantive = [
        t for t in tokens if t not in _QUALIFIER_TOKENS and not t.isdigit()
    ]
    return len(substantive) <= 1


def _unresolved(cleaned_name: str, originals: set[str] | None = None) -> DrugResolution:
    return DrugResolution(
        rxcui=None,
        canonical_name=cleaned_name,
        original_names=originals or {cleaned_name},
        resolved=False,
    )


async def resolve_drug(
    cleaned_name: str,
    *,
    client: RxNormClient,
    cache: DrugCache,
    originals: set[str] | None = None,
) -> DrugResolution:
    """Resolve one cleaned drug name.

    Cached either way, so an unknown name costs one request for the life of the
    process rather than one per trial that mentions it.

    Raises :class:`RxNormError` when the lookup itself fails, and does not cache
    that -- a transient outage must not poison the cache for the life of the
    process. Degrading to string normalization is the *batch* layer's decision,
    not this function's, because only the batch knows how many names failed.
    A malformed-but-successful response is not a failure: it resolves to
    unresolved, like any other name RxNorm has nothing useful to say about.
    """
    cached = cache.get(cleaned_name)
    if cached is not None:
        if originals:
            # Same identity reached from an additional surface form.
            cached = cached.model_copy(
                update={"original_names": cached.original_names | originals}
            )
        return cached

    if not is_resolvable(cleaned_name):
        resolution = _unresolved(cleaned_name, originals)
        cache.set(cleaned_name, resolution)
        return resolution

    try:
        match = await client.approximate_term(cleaned_name)
        if match is None:
            resolution = _unresolved(cleaned_name, originals)
        else:
            rxcui, score = match
            ingredient = await client.ingredient_for(rxcui)
            if ingredient is None:
                resolution = _unresolved(cleaned_name, originals)
            else:
                ingredient_rxcui, canonical = ingredient
                resolution = DrugResolution(
                    rxcui=ingredient_rxcui,
                    canonical_name=canonical,
                    original_names=originals or {cleaned_name},
                    resolved=True,
                    score=score,
                )
    except RxNormError as exc:
        # Soft failure: the graph degrades to string normalization rather than
        # the request failing. Not cached -- a transient outage should not
        # poison the cache for the life of the process.
        # DEBUG, not WARNING: during an outage this fires once per drug name,
        # and the retry warnings plus the batch summary already carry the signal.
        logger.debug(
            "rxnorm lookup failed",
            extra={"drug_name": cleaned_name, "reason": str(exc)},
        )
        raise

    cache.set(cleaned_name, resolution)
    return resolution


def collect_drug_names(
    records: Mapping[str, dict[str, Any]],
) -> dict[str, set[str]]:
    """Distinct cleaned drug names across all records -> their surface forms.

    Deduplicating here is what makes resolution affordable: a 600-trial
    pembrolizumab query mentions a few hundred distinct agents across thousands
    of intervention entries.
    """
    names: dict[str, set[str]] = {}
    for record in records.values():
        interventions = _seq(
            protocol_module(record, "armsInterventionsModule").get("interventions")
        )
        for intervention in interventions:
            if not isinstance(intervention, dict):
                continue
            if intervention.get("type") not in RESOLVABLE_TYPES:
                continue
            raw = intervention.get("name")
            if not isinstance(raw, str) or not raw.strip():
                continue
            cleaned = normalize_intervention(raw)
            if not cleaned or cleaned in STOPWORD_DRUGS:
                continue
            names.setdefault(cleaned, set()).add(raw.strip())
    return names


DEGRADED_WARNING = (
    "Drug synonym resolution unavailable; nodes reflect distinct name strings "
    "rather than merged compounds."
)


async def resolve_all(
    records: Mapping[str, dict[str, Any]],
    *,
    cache: DrugCache | None = None,
    client: RxNormClient | None = None,
    max_concurrency: int | None = None,
    only: Collection[str] | None = None,
) -> tuple[dict[str, DrugResolution], list[str]]:
    """Resolve every distinct drug name in ``records``.

    Runs as one bounded-concurrency batch before the graph is built, which
    keeps the network builders pure and synchronous.

    ``only`` restricts resolution to a set of cleaned names -- normally the
    candidate pool from :func:`app.services.network.rank_candidate_names`, so a
    query resolves the names that could reach the graph rather than every name
    in the corpus.

    Returns the resolution map plus warnings for ``meta.warnings``. If RxNorm is
    unreachable the map is empty and the caller falls back to string
    normalization -- a degraded graph, disclosed, rather than a failed request.
    """
    cache = cache if cache is not None else DRUG_CACHE
    names = collect_drug_names(records)
    if only is not None:
        allowed = set(only)
        names = {k: v for k, v in names.items() if k in allowed}
    if not names:
        return {}, []

    logger.info(
        "drug resolution started",
        extra={
            "distinct_names": len(names),
            "candidates": len(only) if only is not None else None,
        },
    )
    started = time.perf_counter()
    hits_before, misses_before = cache.hits, cache.misses

    limit = max_concurrency or settings.RXNORM_MAX_CONCURRENCY
    semaphore = asyncio.Semaphore(limit)
    owns_client = client is None
    client = client or RxNormClient()

    async def one(cleaned: str, originals: set[str]) -> tuple[str, DrugResolution | None]:
        async with semaphore:
            try:
                return cleaned, await resolve_drug(
                    cleaned, client=client, cache=cache, originals=originals
                )
            except RxNormError:
                return cleaned, None

    if owns_client:
        await client.__aenter__()
    try:
        # Per-request timeouts bound one lookup; nothing bounded the batch, so
        # an RxNorm that is slow without ever timing out could hold a response
        # open for as long as there were names to resolve. Falling back to
        # string normalization is this module's whole failure story, so the
        # ceiling costs nothing but a disclosed degradation.
        results = await asyncio.wait_for(
            asyncio.gather(
                *(one(cleaned, originals) for cleaned, originals in names.items())
            ),
            timeout=settings.RXNORM_BATCH_TIMEOUT_SECONDS,
        )
    except TimeoutError:
        logger.warning(
            "drug resolution timed out for the batch",
            extra={
                "names": len(names),
                "timeout_s": settings.RXNORM_BATCH_TIMEOUT_SECONDS,
            },
        )
        results = []
    finally:
        if owns_client:
            await client.__aexit__()

    resolutions = {cleaned: r for cleaned, r in results if r is not None}
    failed = len(names) - len(resolutions)

    warnings: list[str] = []
    if failed and not resolutions:
        warnings.append(DEGRADED_WARNING)
    elif failed:
        warnings.append(
            f"Drug synonym resolution failed for {failed:,} of {len(names):,} names; "
            f"those nodes reflect distinct name strings rather than merged compounds."
        )

    resolved = sum(1 for r in resolutions.values() if r.resolved)
    if failed:
        # One line for the whole batch: the per-name and per-retry detail is at
        # DEBUG, so an outage is loud once rather than hundreds of times.
        logger.warning(
            "drug resolution degraded",
            extra={"failed": failed, "of": len(names)},
        )
    logger.info(
        "drug resolution completed",
        extra={
            "resolved": resolved,
            "unresolved": len(resolutions) - resolved,
            # Deltas, so the figures describe this request rather than the
            # lifetime of the process-wide cache.
            "cache_hits": cache.hits - hits_before,
            "live_lookups": cache.misses - misses_before,
            "failed": failed,
            "duration_ms": round((time.perf_counter() - started) * 1000, 1),
        },
    )
    return resolutions, warnings
