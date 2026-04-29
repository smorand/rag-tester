"""Tests for configuration management."""

import os
from unittest.mock import patch

from rag_tester.config import Settings


class TestSettings:
    """Tests for Settings class."""

    def test_default_values(self) -> None:
        """Test that Settings uses default values when no env vars are set.

        Corresponds to: E2E-INFRA-002
        """
        # Clear any RAG_TESTER_* env vars
        env_vars_to_clear = [key for key in os.environ if key.startswith("RAG_TESTER_")]
        with patch.dict(os.environ, dict.fromkeys(env_vars_to_clear, ""), clear=False):
            for key in env_vars_to_clear:
                os.environ.pop(key, None)

            settings = Settings()

            assert settings.log_level == "INFO"
            assert settings.log_file == "logs/rag-tester.log"
            assert settings.trace_file == "traces/rag-tester.jsonl"
            assert settings.otel_endpoint is None
            assert settings.max_retry_attempts == 5
            assert settings.retry_backoff_multiplier == 2.0
            assert settings.retry_initial_delay == 1.0

    def test_load_from_environment_variables(self) -> None:
        """Test that Settings loads values from environment variables.

        Corresponds to: E2E-INFRA-001
        """
        env_vars = {
            "RAG_TESTER_LOG_LEVEL": "DEBUG",
            "RAG_TESTER_LOG_FILE": "custom/logs/app.log",
            "RAG_TESTER_TRACE_FILE": "custom/trace.jsonl",
            "RAG_TESTER_OTEL_ENDPOINT": "http://localhost:4317",
            "RAG_TESTER_MAX_RETRY_ATTEMPTS": "3",
            "RAG_TESTER_RETRY_BACKOFF_MULTIPLIER": "1.5",
            "RAG_TESTER_RETRY_INITIAL_DELAY": "0.5",
        }

        with patch.dict(os.environ, env_vars, clear=False):
            settings = Settings()

            assert settings.log_level == "DEBUG"
            assert settings.log_file == "custom/logs/app.log"
            assert settings.trace_file == "custom/trace.jsonl"
            assert settings.otel_endpoint == "http://localhost:4317"
            assert settings.max_retry_attempts == 3
            assert settings.retry_backoff_multiplier == 1.5
            assert settings.retry_initial_delay == 0.5

    def test_partial_environment_variables(self) -> None:
        """Test that Settings uses defaults for unset env vars."""
        env_vars = {
            "RAG_TESTER_LOG_LEVEL": "DEBUG",
            "RAG_TESTER_TRACE_FILE": "custom/trace.jsonl",
        }

        with patch.dict(os.environ, env_vars, clear=False):
            settings = Settings()

            # Set values
            assert settings.log_level == "DEBUG"
            assert settings.trace_file == "custom/trace.jsonl"

            # Default values
            assert settings.log_file == "logs/rag-tester.log"
            assert settings.otel_endpoint is None
            assert settings.max_retry_attempts == 5

    def test_case_insensitive_env_vars(self) -> None:
        """Test that environment variables are case-insensitive."""
        env_vars = {
            "rag_tester_log_level": "WARNING",  # lowercase
            "RAG_TESTER_LOG_FILE": "test.log",  # uppercase
        }

        with patch.dict(os.environ, env_vars, clear=False):
            settings = Settings()

            assert settings.log_level == "WARNING"
            assert settings.log_file == "test.log"

    def test_type_conversion(self) -> None:
        """Test that environment variables are properly converted to correct types."""
        env_vars = {
            "RAG_TESTER_MAX_RETRY_ATTEMPTS": "10",
            "RAG_TESTER_RETRY_BACKOFF_MULTIPLIER": "3.5",
            "RAG_TESTER_RETRY_INITIAL_DELAY": "2.0",
        }

        with patch.dict(os.environ, env_vars, clear=False):
            settings = Settings()

            assert isinstance(settings.max_retry_attempts, int)
            assert settings.max_retry_attempts == 10

            assert isinstance(settings.retry_backoff_multiplier, float)
            assert settings.retry_backoff_multiplier == 3.5

            assert isinstance(settings.retry_initial_delay, float)
            assert settings.retry_initial_delay == 2.0

    def test_optional_otel_endpoint(self) -> None:
        """Test that otel_endpoint can be None or a string."""
        # Test None (default)
        with patch.dict(os.environ, {}, clear=False):
            settings = Settings()
            assert settings.otel_endpoint is None

        # Test with value
        with patch.dict(os.environ, {"RAG_TESTER_OTEL_ENDPOINT": "http://localhost:4317"}, clear=False):
            settings = Settings()
            assert settings.otel_endpoint == "http://localhost:4317"
