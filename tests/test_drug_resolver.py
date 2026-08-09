"""RxNorm drug resolution: threshold, ingredient walk, caching, and the
failure modes that must degrade rather than raise.

RxNorm responses are mocked from fixtures captured against the live API, in the
same spirit as ``tests/test_dimensions.py`` -- the shapes asserted here are real
shapes, so a change upstream fails these tests rather than silently producing
unmerged graphs.
"""

import asyncio
import json

import httpx
import pytest
import respx

from app.models.schemas import DrugResolution
from app.services.drug_resolver import (
    DEGRADED_WARNING,
    DrugCache,
    RxNormClient,
    RxNormError,
    collect_drug_names,
    is_resolvable,
    resolve_all,
    resolve_drug,
)
from tests.conftest import FIXTURE_DIR, make_record

BASE = "https://rxnav.nlm.nih.gov/REST"

KEYTRUDA_RXCUI = "1547550"
PEMBRO_RXCUI = "1547545"


def fixture(name: str) -> dict:
    return json.loads((FIXTURE_DIR / f"rxnorm_{name}.json").read_text())


def approx_payload(rxcui: str, score: float) -> dict:
    return {
        "approximateGroup": {
            "inputTerm": None,
            "candidate": [{"rxcui": rxcui, "score": str(score), "rank": "1"}],
        }
    }


def ingredient_payload(rxcui: str, name: str) -> dict:
    return {
        "relatedGroup": {
            "conceptGroup": [
                {"tty": "IN", "conceptProperties": [{"rxcui": rxcui, "name": name}]}
            ]
        }
    }


def mock_rxnorm(approx: dict | None = None, ingredient: dict | None = None):
    """Wire the two-call chain with sensible pembrolizumab defaults."""
    respx.get(f"{BASE}/approximateTerm.json").mock(
        return_value=httpx.Response(
            200, json=approx if approx is not None else approx_payload(KEYTRUDA_RXCUI, 14.25)
        )
    )
    respx.get(url__regex=rf"{BASE}/rxcui/\d+/related\.json").mock(
        return_value=httpx.Response(
            200,
            json=ingredient
            if ingredient is not None
            else ingredient_payload(PEMBRO_RXCUI, "pembrolizumab"),
        )
    )


async def resolve(name, *, cache=None, **kwargs):
    async with RxNormClient(max_retries=0, **kwargs) as client:
        return await resolve_drug(name, client=client, cache=cache or DrugCache())


# --------------------------------------------------------------------------


class TestRealFixtures:
    """Pinned against responses captured from the live API.

    The headline assertion is the whole point of the feature: a brand name and
    its generic must land on the same ingredient RxCUI.
    """

    @respx.mock
    async def test_brand_and_generic_resolve_to_the_same_ingredient(self):
        cache = DrugCache()

        respx.get(f"{BASE}/approximateTerm.json").mock(
            return_value=httpx.Response(200, json=fixture("approx_keytruda"))
        )
        respx.get(url__regex=rf"{BASE}/rxcui/\d+/related\.json").mock(
            return_value=httpx.Response(200, json=fixture("ingredient_keytruda"))
        )
        brand = await resolve("keytruda", cache=cache)

        respx.get(f"{BASE}/approximateTerm.json").mock(
            return_value=httpx.Response(200, json=fixture("approx_pembrolizumab"))
        )
        respx.get(url__regex=rf"{BASE}/rxcui/\d+/related\.json").mock(
            return_value=httpx.Response(200, json=fixture("ingredient_pembrolizumab"))
        )
        generic = await resolve("pembrolizumab", cache=cache)

        assert brand.resolved and generic.resolved
        assert brand.rxcui == generic.rxcui == PEMBRO_RXCUI
        assert brand.canonical_name == generic.canonical_name == "pembrolizumab"

    @respx.mock
    async def test_no_match_response_omits_the_candidate_key_entirely(self):
        """The real miss payload has no 'candidate' key -- not an empty list."""
        payload = fixture("approx_nomatch")
        assert "candidate" not in payload["approximateGroup"]
        respx.get(f"{BASE}/approximateTerm.json").mock(
            return_value=httpx.Response(200, json=payload)
        )
        result = await resolve("zzzqqxnotadrug")
        assert result.resolved is False
        assert result.canonical_name == "zzzqqxnotadrug"

    @respx.mock
    async def test_multi_ingredient_brand_is_never_merged(self):
        """Trikafta relates to MIN, not IN, so the walk returns an empty group.

        Collapsing it onto one arbitrary component would silently misattribute
        trials, so anything but exactly one ingredient stays unresolved.
        """
        payload = fixture("ingredient_trikafta_empty")
        mock_rxnorm(approx=approx_payload("2263914", 14.0), ingredient=payload)
        result = await resolve("trikafta")
        assert result.resolved is False


class TestConfidenceThreshold:
    """Measured live: correct matches scored >= 11.49, the worst false positive
    6.38. The threshold sits in that gap."""

    @respx.mock
    @pytest.mark.parametrize(
        "score,expected",
        [
            (14.25, True),   # Keytruda
            (11.49, True),   # vitamin b12, the lowest verified-correct match
            (11.0, True),    # exactly at threshold
            (10.99, False),  # just below
            (6.38, False),   # "MK-3475" -> an unrelated concept
            (2.63, False),   # "study drug" -> a hand sanitizer gel
        ],
    )
    async def test_threshold_boundary(self, score, expected):
        mock_rxnorm(approx=approx_payload(KEYTRUDA_RXCUI, score))
        result = await resolve("some drug")
        assert result.resolved is expected

    @respx.mock
    async def test_score_is_recorded_for_auditability(self):
        mock_rxnorm(approx=approx_payload(KEYTRUDA_RXCUI, 14.25))
        assert (await resolve("keytruda")).score == pytest.approx(14.25)

    @respx.mock
    async def test_below_threshold_skips_the_ingredient_call(self):
        mock_rxnorm(approx=approx_payload(KEYTRUDA_RXCUI, 3.0))
        route = respx.routes[1]
        await resolve("study drug")
        assert route.call_count == 0


class TestMalformedSuccessPayloads:
    """RxNorm answers 200 without promising a shape.

    Two properties are asserted together, because either alone is a bug: an
    unusable body must not crash (it degrades to unresolved, like any name
    RxNorm cannot help with), and it must not resolve either -- a merge is a
    claim about identity that a malformed body cannot support.
    """

    @respx.mock
    @pytest.mark.parametrize(
        "score",
        ["NaN", "nan", "Infinity", "+Infinity", "-Infinity", "1e309", "abc", None],
    )
    async def test_a_non_finite_score_never_clears_the_threshold(self, score):
        """``score < min_score`` let "NaN" through: every comparison with NaN
        is False, so the guard that was meant to reject it did nothing and an
        arbitrary RxCUI was merged. The gate is stated positively now."""
        mock_rxnorm(
            approx={"approximateGroup": {"candidate": [{"rxcui": "1", "score": score}]}}
        )
        result = await resolve("some drug")
        assert result.resolved is False
        assert result.rxcui is None

    @respx.mock
    @pytest.mark.parametrize(
        "payload",
        [
            [1, 2],
            {"approximateGroup": None},
            {"approximateGroup": "x"},
            {"approximateGroup": {"candidate": "x"}},
            {"approximateGroup": {"candidate": [None]}},
            {"approximateGroup": {"candidate": [5]}},
            {"approximateGroup": {"candidate": [{"score": "50"}]}},
        ],
    )
    async def test_a_malformed_approximate_body_degrades_instead_of_raising(
        self, payload
    ):
        mock_rxnorm(approx=payload)
        result = await resolve("some drug")
        assert result.resolved is False

    @respx.mock
    @pytest.mark.parametrize(
        "payload",
        [
            [1, 2],
            {"relatedGroup": None},
            {"relatedGroup": {"conceptGroup": "x"}},
            {"relatedGroup": {"conceptGroup": [None]}},
            {"relatedGroup": {"conceptGroup": [{"conceptProperties": "x"}]}},
            {"relatedGroup": {"conceptGroup": [{"conceptProperties": [7]}]}},
        ],
    )
    async def test_a_malformed_ingredient_body_degrades_instead_of_raising(
        self, payload
    ):
        mock_rxnorm(approx=approx_payload(KEYTRUDA_RXCUI, 14.25), ingredient=payload)
        result = await resolve("keytruda")
        assert result.resolved is False


class TestPreFilter:
    """Names filtered before the network call. This is a correctness guard, not
    just an optimization -- RxNorm answers these confidently and wrongly."""

    @pytest.mark.parametrize(
        "name", ["pembrolizumab and lenvatinib", "carboplatin + paclitaxel",
                 "pembrolizumab with chemo", "cisplatin/etoposide",
                 "nivolumab or pembrolizumab", "carboplatin, paclitaxel",
                 "pembrolizumab versus placebo"]
    )
    def test_multi_agent_strings_rejected(self, name):
        assert is_resolvable(name) is False

    @pytest.mark.parametrize("name", ["mk-3475", "ace2016", "bms 936558", "ab-123"])
    def test_research_codes_rejected(self, name):
        assert is_resolvable(name) is False

    @pytest.mark.parametrize(
        "name", ["pembrolizumab", "keytruda", "folic acid", "vitamin b12", "5-fu"]
    )
    def test_real_drug_names_accepted(self, name):
        assert is_resolvable(name) is True

    def test_too_short_rejected(self):
        assert is_resolvable("ab") is False

    # ---- guards derived from real intervention names in live trial data ----

    @pytest.mark.parametrize(
        "name",
        [
            "placebo for pembrolizumab",
            "placebo to pembrolizumab",
            "matching placebo pembrolizumab",
            "pembrolizumab comparator",
        ],
    )
    def test_control_arm_names_rejected(self, name):
        """A placebo arm resolved to its active comparator would count control
        trials as evidence for the drug."""
        assert is_resolvable(name) is False

    @pytest.mark.parametrize(
        "name",
        [
            "favezelimab pembrolizumab",
            "pembrolizumab vibostolimab",
            "coformulation pembrolizumab quavonlimab",
            "gemcitabine nab-paclitaxel",
            "cisplatin carboplatin 5fu",
            "ipilimumab pembrolizumab durvalumab idarubicin bevacizumab",
            "her2 trastuzumab deruxtecan pertuzumab",
        ],
    )
    def test_space_separated_multi_drug_names_rejected(self, name):
        """No connective, but still several agents. RxNorm resolves these to one
        component and silently discards the rest, which would move the other
        drugs' trials onto the matched node."""
        assert is_resolvable(name) is False

    @pytest.mark.parametrize(
        "name",
        [
            "nab paclitaxel",
            "nab-paclitaxel",
            "albumin paclitaxel",
            "doxorubicin hydrochloride",
            "pegylated liposomal doxorubicin",
            "trastuzumab injectable product",
        ],
    )
    def test_qualifier_decorated_single_agents_accepted(self, name):
        """One agent plus formulation words is still one agent."""
        assert is_resolvable(name) is True

    @respx.mock
    async def test_filtered_names_never_reach_the_api(self):
        mock_rxnorm()
        result = await resolve("pembrolizumab and lenvatinib")
        assert result.resolved is False
        assert respx.routes[0].call_count == 0


class TestCaching:
    @respx.mock
    async def test_second_lookup_is_served_from_cache(self):
        mock_rxnorm()
        cache = DrugCache()
        async with RxNormClient(max_retries=0) as client:
            first = await resolve_drug("keytruda", client=client, cache=cache)
            second = await resolve_drug("keytruda", client=client, cache=cache)
        assert first.rxcui == second.rxcui
        assert respx.routes[0].call_count == 1
        assert cache.hits == 1

    @respx.mock
    async def test_negative_results_are_cached_too(self):
        """An unknown name should cost one request, not one per trial."""
        mock_rxnorm(approx={"approximateGroup": {"inputTerm": "x"}})
        cache = DrugCache()
        async with RxNormClient(max_retries=0) as client:
            for _ in range(3):
                await resolve_drug("unknownium", client=client, cache=cache)
        assert respx.routes[0].call_count == 1

    @respx.mock
    async def test_transient_failures_are_not_cached(self):
        """A momentary outage must not poison the cache for the process."""
        respx.get(f"{BASE}/approximateTerm.json").mock(return_value=httpx.Response(503))
        cache = DrugCache()
        async with RxNormClient(max_retries=0) as client:
            with pytest.raises(RxNormError):
                await resolve_drug("keytruda", client=client, cache=cache)
        assert len(cache) == 0

    async def test_cache_evicts_fifo_at_capacity(self):
        cache = DrugCache(max_entries=2)
        for name in ["a", "b", "c"]:
            cache.set(name, DrugResolution(rxcui=None, canonical_name=name, resolved=False))
        assert len(cache) == 2
        assert cache.get("a") is None
        assert cache.get("c") is not None

    async def test_repeated_set_updates_without_growing(self):
        cache = DrugCache(max_entries=2)
        for _ in range(5):
            cache.set("a", DrugResolution(rxcui=None, canonical_name="a", resolved=False))
        assert len(cache) == 1

    @respx.mock
    async def test_cache_hit_accumulates_additional_surface_forms(self):
        """Two spellings reaching one identity keep both for traceability."""
        mock_rxnorm()
        cache = DrugCache()
        async with RxNormClient(max_retries=0) as client:
            await resolve_drug(
                "keytruda", client=client, cache=cache, originals={"Keytruda"}
            )
            second = await resolve_drug(
                "keytruda", client=client, cache=cache, originals={"KEYTRUDA 200mg"}
            )
        assert second.original_names == {"Keytruda", "KEYTRUDA 200mg"}


class TestFailureHandling:
    @respx.mock
    async def test_retries_5xx_then_succeeds(self):
        route = respx.get(f"{BASE}/approximateTerm.json")
        route.side_effect = [
            httpx.Response(503),
            httpx.Response(200, json=approx_payload(KEYTRUDA_RXCUI, 14.25)),
        ]
        respx.get(url__regex=rf"{BASE}/rxcui/\d+/related\.json").mock(
            return_value=httpx.Response(200, json=ingredient_payload(PEMBRO_RXCUI, "pembrolizumab"))
        )
        async with RxNormClient(max_retries=2) as client:
            result = await resolve_drug("keytruda", client=client, cache=DrugCache())
        assert result.resolved is True
        assert route.call_count == 2

    @respx.mock
    async def test_4xx_is_not_retried(self):
        route = respx.get(f"{BASE}/approximateTerm.json").mock(
            return_value=httpx.Response(400)
        )
        async with RxNormClient(max_retries=2) as client:
            with pytest.raises(RxNormError):
                await resolve_drug("keytruda", client=client, cache=DrugCache())
        assert route.call_count == 1

    @respx.mock
    async def test_network_error_raises_rxnorm_error_not_httpx(self):
        respx.get(f"{BASE}/approximateTerm.json").mock(
            side_effect=httpx.ConnectError("boom")
        )
        async with RxNormClient(max_retries=0) as client:
            with pytest.raises(RxNormError):
                await resolve_drug("keytruda", client=client, cache=DrugCache())

    @respx.mock
    async def test_malformed_score_is_treated_as_no_match(self):
        mock_rxnorm(
            approx={"approximateGroup": {"candidate": [{"rxcui": "1", "score": "n/a"}]}}
        )
        assert (await resolve("keytruda")).resolved is False


class TestCollectDrugNames:
    def test_deduplicates_across_records_and_keeps_surface_forms(self):
        records = {
            "NCT00000001": make_record(
                "NCT00000001", interventions=[("DRUG", "Pembrolizumab 200 mg")]
            ),
            "NCT00000002": make_record(
                "NCT00000002", interventions=[("DRUG", "Pembrolizumab")]
            ),
        }
        names = collect_drug_names(records)
        assert set(names) == {"pembrolizumab"}
        assert names["pembrolizumab"] == {"Pembrolizumab 200 mg", "Pembrolizumab"}

    def test_skips_combination_products(self):
        """Scope guard: combination names are compound descriptions RxNorm
        mis-matches, so they never enter resolution."""
        records = {
            "NCT00000001": make_record(
                "NCT00000001",
                interventions=[
                    ("DRUG", "Pembrolizumab"),
                    ("COMBINATION_PRODUCT", "Pembrolizumab and Lenvatinib"),
                ],
            )
        }
        assert set(collect_drug_names(records)) == {"pembrolizumab"}

    def test_skips_non_drug_types_and_placebo(self):
        records = {
            "NCT00000001": make_record(
                "NCT00000001",
                interventions=[
                    ("DRUG", "Placebo"),
                    ("PROCEDURE", "Surgery"),
                    ("BIOLOGICAL", "Nivolumab"),
                ],
            )
        }
        assert set(collect_drug_names(records)) == {"nivolumab"}

    def test_tolerates_records_with_no_interventions(self):
        assert collect_drug_names({"NCT00000001": make_record(interventions=None)}) == {}


class TestResolveAll:
    @respx.mock
    async def test_resolves_every_distinct_name_once(self):
        mock_rxnorm()
        records = {
            f"NCT0000000{i}": make_record(
                f"NCT0000000{i}", interventions=[("DRUG", "Keytruda")]
            )
            for i in range(1, 6)
        }
        resolutions, warnings = await resolve_all(records, cache=DrugCache())
        assert set(resolutions) == {"keytruda"}
        assert respx.routes[0].call_count == 1
        assert warnings == []

    @respx.mock
    async def test_total_outage_degrades_with_a_warning(self):
        respx.get(f"{BASE}/approximateTerm.json").mock(return_value=httpx.Response(503))
        records = {
            "NCT00000001": make_record("NCT00000001", interventions=[("DRUG", "Keytruda")])
        }
        async with RxNormClient(max_retries=0) as client:
            resolutions, warnings = await resolve_all(
                records, cache=DrugCache(), client=client
            )
        assert resolutions == {}
        assert warnings == [DEGRADED_WARNING]

    @respx.mock
    async def test_partial_failure_reports_how_many(self):
        def handler(request):
            term = request.url.params.get("term")
            if term == "nivolumab":
                return httpx.Response(503)
            return httpx.Response(200, json=approx_payload(KEYTRUDA_RXCUI, 14.25))

        respx.get(f"{BASE}/approximateTerm.json").mock(side_effect=handler)
        respx.get(url__regex=rf"{BASE}/rxcui/\d+/related\.json").mock(
            return_value=httpx.Response(200, json=ingredient_payload(PEMBRO_RXCUI, "pembrolizumab"))
        )
        records = {
            "NCT00000001": make_record(
                "NCT00000001",
                interventions=[("DRUG", "Keytruda"), ("DRUG", "Nivolumab")],
            )
        }
        async with RxNormClient(max_retries=0) as client:
            resolutions, warnings = await resolve_all(
                records, cache=DrugCache(), client=client
            )
        assert set(resolutions) == {"keytruda"}
        assert warnings and "1 of 2" in warnings[0]

    @respx.mock
    async def test_no_drugs_means_no_requests(self):
        mock_rxnorm()
        records = {"NCT00000001": make_record("NCT00000001", interventions=None)}
        resolutions, warnings = await resolve_all(records, cache=DrugCache())
        assert (resolutions, warnings) == ({}, [])
        assert respx.routes[0].call_count == 0

    @respx.mock
    async def test_concurrency_stays_within_the_cap(self):
        in_flight = 0
        peak = 0

        async def handler(request):
            nonlocal in_flight, peak
            in_flight += 1
            peak = max(peak, in_flight)
            await asyncio.sleep(0.01)
            in_flight -= 1
            return httpx.Response(200, json={"approximateGroup": {"inputTerm": "x"}})

        respx.get(f"{BASE}/approximateTerm.json").mock(side_effect=handler)
        records = {
            f"NCT{i:08d}": make_record(f"NCT{i:08d}", interventions=[("DRUG", f"Drugium{i}ol")])
            for i in range(20)
        }
        async with RxNormClient(max_retries=0) as client:
            await resolve_all(
                records, cache=DrugCache(), client=client, max_concurrency=5
            )
        assert peak <= 5
        assert peak > 1, "should actually run concurrently"
