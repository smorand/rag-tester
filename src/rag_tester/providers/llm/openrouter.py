"""OpenRouter chat-completions provider for the answer command.

OpenRouter's chat-completions endpoint is OpenAI-compatible, so this module
also serves as a reference implementation for any future OpenAI-compatible
backend (e.g. self-hosted vLLM).
"""

from __future__ import annotations

import logging

import httpx
from opentelemetry import trace
from pydantic import SecretStr

from rag_tester.config import get_settings
from rag_tester.providers.llm.base import (
    LLMAuthenticationError,
    LLMError,
    LLMProvider,
    LLMRateLimitError,
    LLMResponse,
)
from rag_tester.utils.retry import retry_with_backoff

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)

OPENROUTER_CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"


class OpenRouterLLMProvider(LLMProvider):
    """Chat-completion provider backed by OpenRouter.

    Args:
        model_name: Provider-prefixed model id (e.g. ``openai/gpt-4o-mini``,
            ``anthropic/claude-sonnet-4-5``, ``google/gemini-2.5-flash``).
        api_key: Explicit key. When ``None``, falls back to
            ``Settings.openrouter_api_key`` (which accepts both
            ``RAG_TESTER_OPENROUTER_API_KEY`` and ``OPENROUTER_API_KEY``).
        timeout: HTTP timeout in seconds.
    """

    def __init__(
        self,
        model_name: str,
        api_key: str | SecretStr | None = None,
        timeout: float = 60.0,
    ) -> None:
        self._model_name = model_name
        self._timeout = timeout

        if api_key is None:
            settings_key = get_settings().openrouter_api_key
            resolved = settings_key.get_secret_value() if settings_key is not None else None
        elif isinstance(api_key, SecretStr):
            resolved = api_key.get_secret_value()
        else:
            resolved = api_key

        self._api_key = resolved
        if not self._api_key or not self._api_key.strip():
            msg = (
                "Missing API key: OPENROUTER_API_KEY. Set the environment variable to use the OpenRouter LLM provider."
            )
            logger.error(msg)
            raise LLMError(msg)

        logger.info("Using OpenRouter LLM with model: %s", self._model_name)

    def get_model_name(self) -> str:
        return self._model_name

    @retry_with_backoff(
        transient_errors=(
            httpx.TimeoutException,
            httpx.ConnectError,
            httpx.RemoteProtocolError,
            LLMRateLimitError,
        ),
    )
    async def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        with tracer.start_as_current_span("openrouter_chat_completion") as span:
            span.set_attribute("model.name", self._model_name)
            span.set_attribute("system_prompt.length", len(system_prompt))
            span.set_attribute("user_prompt.length", len(user_prompt))

            headers = {
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            }
            payload = {
                "model": self._model_name,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            }

            try:
                async with httpx.AsyncClient(timeout=self._timeout) as client:
                    response = await client.post(OPENROUTER_CHAT_URL, headers=headers, json=payload)

                    if response.status_code in (401, 403):
                        error_msg = "Authentication failed: invalid API key"
                        logger.error(error_msg)
                        span.set_attribute("error", error_msg)
                        raise LLMAuthenticationError(error_msg)

                    if response.status_code == 429:
                        retry_after = response.headers.get("Retry-After")
                        error_msg = f"Rate limit exceeded. Retry-After: {retry_after}"
                        logger.warning(error_msg)
                        span.set_attribute("rate_limit.retry_after", retry_after or "not provided")
                        raise LLMRateLimitError(error_msg)

                    if response.status_code >= 500:
                        error_msg = f"API server error: {response.status_code}"
                        logger.warning(error_msg)
                        span.set_attribute("error", error_msg)
                        raise httpx.RemoteProtocolError(error_msg)

                    response.raise_for_status()
                    data = response.json()

                    text = data["choices"][0]["message"]["content"]
                    tokens = int(data.get("usage", {}).get("total_tokens", 0))

                    span.set_attribute("tokens.count", tokens)
                    span.set_attribute("response.length", len(text))

                    return LLMResponse(text=text, tokens=tokens)

            except (LLMAuthenticationError, LLMRateLimitError):
                raise
            except httpx.HTTPStatusError as e:
                error_msg = f"OpenRouter API error: {e.response.status_code}"
                logger.error(error_msg)
                raise LLMError(error_msg) from e
            except Exception as e:
                if isinstance(e, httpx.RemoteProtocolError):
                    raise
                error_msg = f"OpenRouter request failed: {e}"
                logger.error(error_msg)
                raise LLMError(error_msg) from e
