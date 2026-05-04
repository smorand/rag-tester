"""Tests for the OpenRouter chat-completions LLM provider."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from rag_tester.providers.llm.base import (
    LLMAuthenticationError,
    LLMError,
)
from rag_tester.providers.llm.openrouter import OpenRouterLLMProvider
from rag_tester.utils.retry import RetryError


@pytest.fixture(autouse=True)
def _set_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Provide a default API key so the provider can be constructed."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")


def _mock_response(status_code: int, json_payload: dict | None = None, headers: dict | None = None) -> MagicMock:
    response = MagicMock(spec=httpx.Response)
    response.status_code = status_code
    response.headers = headers or {}
    response.json.return_value = json_payload or {}
    response.raise_for_status = MagicMock()
    return response


class _FakeAsyncClient:
    """Minimal stand-in for ``httpx.AsyncClient`` as an async context manager."""

    def __init__(self, post_responses: list[Any]) -> None:
        self._responses = list(post_responses)
        self.post_calls = 0

    async def __aenter__(self) -> _FakeAsyncClient:
        return self

    async def __aexit__(self, *_exc: Any) -> None:
        return None

    async def post(self, *_args: Any, **_kwargs: Any) -> Any:
        self.post_calls += 1
        if not self._responses:
            raise AssertionError("no more queued responses")
        item = self._responses.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


def _patch_client(mocker: Any, responses: list[Any]) -> _FakeAsyncClient:
    """Patch httpx.AsyncClient to return a single shared _FakeAsyncClient."""
    fake = _FakeAsyncClient(responses)
    mocker.patch("httpx.AsyncClient", return_value=fake)
    return fake


class TestOpenRouterLLMProvider:
    def test_construct_without_key_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from rag_tester.config import get_settings

        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        get_settings.cache_clear()
        with pytest.raises(LLMError, match="Missing API key"):
            OpenRouterLLMProvider(model_name="openai/gpt-4o-mini")

    def test_get_model_name(self) -> None:
        provider = OpenRouterLLMProvider(model_name="openai/gpt-4o-mini")
        assert provider.get_model_name() == "openai/gpt-4o-mini"

    async def test_complete_success(self, mocker: Any) -> None:
        provider = OpenRouterLLMProvider(model_name="openai/gpt-4o-mini")
        _patch_client(
            mocker,
            [
                _mock_response(
                    200,
                    {
                        "choices": [{"message": {"content": "Hello there"}}],
                        "usage": {"total_tokens": 42},
                    },
                )
            ],
        )

        result = await provider.complete(system_prompt="sys", user_prompt="user")
        assert result.text == "Hello there"
        assert result.tokens == 42

    async def test_complete_authentication_error(self, mocker: Any) -> None:
        provider = OpenRouterLLMProvider(model_name="openai/gpt-4o-mini")
        _patch_client(mocker, [_mock_response(401)])
        with pytest.raises(LLMAuthenticationError):
            await provider.complete(system_prompt="sys", user_prompt="user")

    async def test_complete_rate_limit_then_success(self, mocker: Any) -> None:
        """Retry decorator must transparently recover from a 429."""
        provider = OpenRouterLLMProvider(model_name="openai/gpt-4o-mini", timeout=1.0)
        fake = _patch_client(
            mocker,
            [
                _mock_response(429, headers={"Retry-After": "0"}),
                _mock_response(
                    200,
                    {"choices": [{"message": {"content": "ok"}}], "usage": {"total_tokens": 1}},
                ),
            ],
        )
        mocker.patch("rag_tester.utils.retry.asyncio.sleep", new=AsyncMock())

        result = await provider.complete(system_prompt="sys", user_prompt="user")
        assert result.text == "ok"
        assert fake.post_calls == 2

    async def test_complete_raises_for_5xx_after_retries(self, mocker: Any) -> None:
        provider = OpenRouterLLMProvider(model_name="openai/gpt-4o-mini")
        # All attempts return 500; the retry wrapper exhausts and raises RetryError
        _patch_client(mocker, [_mock_response(500) for _ in range(10)])
        mocker.patch("rag_tester.utils.retry.asyncio.sleep", new=AsyncMock())
        with pytest.raises(RetryError):
            await provider.complete(system_prompt="sys", user_prompt="user")
