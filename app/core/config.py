from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings, loaded from environment / .env.

    The OpenAI key is read from ``OPEN_AI_API_KEY`` (the name used in this
    project's .env) with ``OPENAI_API_KEY`` accepted as a fallback alias so the
    service also works in environments using the conventional variable name.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    OPEN_AI_API_KEY: str = ""
    OPENAI_API_KEY: str = ""

    # Model used for query understanding only (intent + entities, never data).
    LLM_MODEL: str = "gpt-5.4-mini"

    # ClinicalTrials.gov API v2.
    CTGOV_BASE_URL: str = "https://clinicaltrials.gov/api/v2"
    #: Bounds are load-bearing, not decorative. A misconfigured deployment
    #: should fail at import with the offending name, rather than at the first
    #: request with a symptom several layers removed from its cause --
    #: RXNORM_MAX_CONCURRENCY=0 built an asyncio.Semaphore(0), which does not
    #: raise: it deadlocks, and the request simply never returns. Every bound
    #: below is the range in which the value has a meaning.
    CTGOV_PAGE_SIZE: int = Field(default=1000, gt=0, le=1000)
    CTGOV_TIMEOUT_SECONDS: float = Field(default=30.0, gt=0, allow_inf_nan=False)
    CTGOV_MAX_RETRIES: int = Field(default=3, ge=0, le=10)

    # RxNorm (NLM), used only to resolve drug names to a canonical ingredient
    # for network graphs. No API key required.
    RXNORM_BASE_URL: str = "https://rxnav.nlm.nih.gov/REST"
    #: Minimum approximateTerm score to accept a match. The score scale is
    #: unbounded (~0-15), not 0-1. Measured against the live API: every correct
    #: match scored >= 11.49, while the worst false positive ("MK-3475" -> an
    #: unrelated concept) scored 6.38. 11.0 sits inside that empty band.
    #: Set strict deliberately: a wrong merge fuses two distinct compounds into
    #: one node and is invisible in the output, whereas a missed merge only
    #: leaves the pre-existing string-normalization behaviour.
    RXNORM_MIN_SCORE: float = Field(default=11.0, ge=0, allow_inf_nan=False)
    # Lower than CTGov's: this is an enrichment, not the data itself, so it
    # should give up quickly rather than hold up a response.
    RXNORM_TIMEOUT_SECONDS: float = Field(default=10.0, gt=0, allow_inf_nan=False)
    RXNORM_MAX_RETRIES: int = Field(default=2, ge=0, le=10)
    #: Becomes an asyncio.Semaphore. Zero is the dangerous value: it is
    #: constructible and simply never lets anything through.
    RXNORM_MAX_CONCURRENCY: int = Field(default=5, gt=0, le=50)
    #: Ceiling on the whole resolution batch, so a slow-but-not-timing-out
    #: RxNorm cannot hold a response open indefinitely. Comfortably above
    #: RXNORM_TIMEOUT_SECONDS, which bounds one request rather than all of them.
    RXNORM_BATCH_TIMEOUT_SECONDS: float = Field(
        default=45.0, gt=0, allow_inf_nan=False
    )
    #: How many of the most-mentioned drug names to resolve. Only a few
    #: dozen nodes survive pruning, so resolving every distinct name is
    #: mostly wasted work; the pool is kept several times larger than the
    #: node cap because merging only ever increases a node's size.
    RXNORM_CANDIDATE_POOL: int = Field(default=120, gt=0, le=5000)
    #: Kill switch. False -> graphs fall back to pure string normalization,
    #: which is byte-identical to the behaviour before RxNorm was introduced.
    RXNORM_ENABLED: bool = True

    ENV: str = "development"
    LOG_LEVEL: str = "INFO"

    @property
    def openai_api_key(self) -> str:
        return self.OPEN_AI_API_KEY or self.OPENAI_API_KEY


settings = Settings()
