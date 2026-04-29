"""Configuration management using pydantic-settings.

All configuration is loaded from environment variables with the RAG_TESTER_ prefix.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables.

    All environment variables must be prefixed with RAG_TESTER_.
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
