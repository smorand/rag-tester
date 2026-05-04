"""LLM provider implementations for the answer command.

Exposes a factory ``get_llm_provider`` that maps a provider identifier to its
concrete class. New backends register themselves in ``_REGISTRY``.
"""

from __future__ import annotations

from rag_tester.providers.llm.base import (
    LLMAuthenticationError,
    LLMError,
    LLMProvider,
    LLMRateLimitError,
)
from rag_tester.providers.llm.openrouter import OpenRouterLLMProvider

_REGISTRY: dict[str, type[LLMProvider]] = {
    "openrouter": OpenRouterLLMProvider,
}


def get_llm_provider(name: str, model_name: str) -> LLMProvider:
    """Instantiate the LLM provider matching ``name``.

    Args:
        name: Provider identifier (currently ``openrouter``).
        model_name: Model identifier passed to the provider constructor
            (e.g. ``openai/gpt-4o-mini``).

    Returns:
        An initialised :class:`LLMProvider` instance.

    Raises:
        ValueError: If ``name`` is not a registered provider.
    """
    provider_cls = _REGISTRY.get(name)
    if provider_cls is None:
        msg = f"Unsupported LLM provider '{name}'. Known providers: {sorted(_REGISTRY.keys())}"
        raise ValueError(msg)
    return provider_cls(model_name=model_name)  # type: ignore[call-arg]


__all__ = [
    "LLMAuthenticationError",
    "LLMError",
    "LLMProvider",
    "LLMRateLimitError",
    "OpenRouterLLMProvider",
    "get_llm_provider",
]
