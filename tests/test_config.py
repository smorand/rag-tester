"""Tests for config module."""

import pytest

from rag_tester.config import Settings


def test_settings_defaults() -> None:
    """Test that Settings has correct defaults."""
    settings = Settings()
    assert settings.app_name == "rag-tester"
    assert settings.debug is False


def test_settings_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that Settings loads from environment variables."""
    monkeypatch.setenv("RAG_TESTER_APP_NAME", "test-app")
    monkeypatch.setenv("RAG_TESTER_DEBUG", "true")

    settings = Settings()
    assert settings.app_name == "test-app"
    assert settings.debug is True
