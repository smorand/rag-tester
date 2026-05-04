"""Configuration management using pydantic-settings.

All configuration is loaded from environment variables with the RAG_TESTER_ prefix.
API keys for third-party embedding providers are also accepted under their
conventional names (GEMINI_API_KEY, OPENROUTER_API_KEY) for compatibility.
"""

from functools import lru_cache

from pydantic import AliasChoices, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables.

    All environment variables must be prefixed with RAG_TESTER_, except API
    keys which also accept their conventional unprefixed names.
    Example: RAG_TESTER_LOG_LEVEL=DEBUG
    """

    model_config = SettingsConfigDict(
        env_prefix="RAG_TESTER_",
        case_sensitive=False,
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Logging configuration
    log_level: str = "INFO"
    log_file: str = "logs/rag-tester.log"

    # Tracing configuration
    trace_file: str = "traces/rag-tester.jsonl"
    otel_endpoint: str | None = None

    # Retry configuration
    max_retry_attempts: int = 5
    retry_backoff_multiplier: float = 2.0
    retry_initial_delay: float = 1.0

    # API keys for third-party embedding providers (held as SecretStr so they
    # never appear in repr() or default logging output).
    gemini_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("RAG_TESTER_GEMINI_API_KEY", "GEMINI_API_KEY"),
    )
    openrouter_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("RAG_TESTER_OPENROUTER_API_KEY", "OPENROUTER_API_KEY"),
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached Settings instance.

    Tests that need to override env vars should call ``get_settings.cache_clear()``
    or instantiate ``Settings`` directly.
    """
    return Settings()
