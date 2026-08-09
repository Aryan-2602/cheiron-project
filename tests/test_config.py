"""Settings bounds.

These are load-bearing rather than decorative. A misconfigured deployment
should fail at construction naming the offending setting, instead of at the
first request with a symptom several layers removed from its cause. The
motivating case: RXNORM_MAX_CONCURRENCY=0 builds an asyncio.Semaphore(0),
which does not raise -- it deadlocks, and the request simply never returns.
"""

import asyncio

import pytest
from pydantic import ValidationError

from app.core.config import Settings


class TestConcurrencyCannotDeadlock:
    def test_zero_concurrency_is_unconstructible(self):
        with pytest.raises(ValidationError, match="RXNORM_MAX_CONCURRENCY"):
            Settings(RXNORM_MAX_CONCURRENCY=0)

    def test_negative_concurrency_is_unconstructible(self):
        with pytest.raises(ValidationError, match="RXNORM_MAX_CONCURRENCY"):
            Settings(RXNORM_MAX_CONCURRENCY=-1)

    async def test_the_value_this_guards_against_really_does_hang(self):
        """Pins why the bound exists: Semaphore(0) is perfectly constructible
        and simply never lets anything through."""
        semaphore = asyncio.Semaphore(0)
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(semaphore.acquire(), timeout=0.05)

    def test_the_default_is_usable(self):
        assert Settings().RXNORM_MAX_CONCURRENCY > 0


class TestNumericBounds:
    @pytest.mark.parametrize(
        "field,bad",
        [
            ("CTGOV_PAGE_SIZE", 0),
            ("CTGOV_PAGE_SIZE", -1),
            # The live API caps pageSize at 1000; a larger value is silently
            # clamped upstream, so the request would not mean what it says.
            ("CTGOV_PAGE_SIZE", 5000),
            ("CTGOV_TIMEOUT_SECONDS", 0),
            ("CTGOV_TIMEOUT_SECONDS", -3.0),
            ("CTGOV_MAX_RETRIES", -1),
            ("RXNORM_TIMEOUT_SECONDS", 0),
            ("RXNORM_TIMEOUT_SECONDS", -1.0),
            ("RXNORM_BATCH_TIMEOUT_SECONDS", 0),
            ("RXNORM_MAX_RETRIES", -1),
            ("RXNORM_CANDIDATE_POOL", 0),
            ("RXNORM_MIN_SCORE", -1.0),
        ],
    )
    def test_an_out_of_range_value_is_rejected(self, field, bad):
        with pytest.raises(ValidationError, match=field):
            Settings(**{field: bad})

    @pytest.mark.parametrize(
        "field",
        [
            "CTGOV_TIMEOUT_SECONDS",
            "RXNORM_TIMEOUT_SECONDS",
            "RXNORM_BATCH_TIMEOUT_SECONDS",
            "RXNORM_MIN_SCORE",
        ],
    )
    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
    def test_a_non_finite_float_is_rejected(self, field, bad):
        """NaN is the dangerous one: every comparison with it is False, so it
        passes range checks by never failing them -- the same mechanism that
        let a NaN RxNorm score clear the confidence threshold."""
        with pytest.raises(ValidationError, match=field):
            Settings(**{field: bad})

    @pytest.mark.parametrize(
        "field,good",
        [
            ("CTGOV_PAGE_SIZE", 1000),
            ("CTGOV_MAX_RETRIES", 0),
            ("RXNORM_MAX_RETRIES", 0),
            ("RXNORM_MIN_SCORE", 0.0),
            ("RXNORM_MAX_CONCURRENCY", 1),
        ],
    )
    def test_a_boundary_value_is_accepted(self, field, good):
        assert getattr(Settings(**{field: good}), field) == good


class TestDefaultsAreValid:
    def test_the_shipped_defaults_construct(self):
        """The bounds must not exclude the values the project actually runs
        with -- otherwise the first thing they break is the service."""
        settings = Settings()
        assert settings.CTGOV_PAGE_SIZE == 1000
        assert settings.RXNORM_MIN_SCORE == 11.0
        assert settings.RXNORM_BATCH_TIMEOUT_SECONDS > settings.RXNORM_TIMEOUT_SECONDS
