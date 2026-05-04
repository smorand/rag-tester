"""Abstract base class for chat-completion LLM providers.

A minimal interface focused on what the ``answer`` command needs: take a
prompt and return a completion string. Streaming and tool-calling are out of
scope for v1.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class LLMResponse:
    """Wire-agnostic shape of a chat completion."""

    text: str
    """The generated assistant message content."""

    tokens: int
    """Total tokens reported by the provider (0 if unavailable)."""


class LLMProvider(ABC):
    """Abstract base for an LLM chat-completion provider."""

    @abstractmethod
    async def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        """Generate a single completion from a system + user prompt.

        Args:
            system_prompt: The system message defining behavior.
            user_prompt: The user message containing the query and context.

        Returns:
            The completion wrapped in :class:`LLMResponse`.

        Raises:
            LLMAuthenticationError: 401/403 from the API.
            LLMRateLimitError: 429 from the API.
            LLMError: Any other failure.
        """

    @abstractmethod
    def get_model_name(self) -> str:
        """Return the model identifier this provider was configured with."""


class LLMError(Exception):
    """Base exception for LLM provider failures."""


class LLMAuthenticationError(LLMError):
    """Raised when the LLM API rejects the credential (401/403)."""


class LLMRateLimitError(LLMError):
    """Raised when the LLM API returns a rate-limit response (429)."""
