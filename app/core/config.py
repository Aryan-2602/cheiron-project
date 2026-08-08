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
    CTGOV_PAGE_SIZE: int = 1000  # API maximum; default is only 10 if omitted.
    CTGOV_TIMEOUT_SECONDS: float = 30.0
    CTGOV_MAX_RETRIES: int = 3

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
    RXNORM_MIN_SCORE: float = 11.0
    # Lower than CTGov's: this is an enrichment, not the data itself, so it
    # should give up quickly rather than hold up a response.
    RXNORM_TIMEOUT_SECONDS: float = 10.0
    RXNORM_MAX_RETRIES: int = 2
    RXNORM_MAX_CONCURRENCY: int = 5
    #: Kill switch. False -> graphs fall back to pure string normalization,
    #: which is byte-identical to the behaviour before RxNorm was introduced.
    RXNORM_ENABLED: bool = True

    ENV: str = "development"
    LOG_LEVEL: str = "INFO"

    @property
    def openai_api_key(self) -> str:
        return self.OPEN_AI_API_KEY or self.OPENAI_API_KEY


settings = Settings()
