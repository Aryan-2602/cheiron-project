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

    ENV: str = "development"
    LOG_LEVEL: str = "INFO"

    @property
    def openai_api_key(self) -> str:
        return self.OPEN_AI_API_KEY or self.OPENAI_API_KEY


settings = Settings()
