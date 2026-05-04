"""Embedding provider implementations.

Provides a factory ``get_embedding_provider`` that maps a provider identifier
(``local``, ``gemini``, ``openrouter``) to its concrete class.
"""

from __future__ import annotations

from rag_tester.providers.embeddings.base import (
    EmbeddingError,
    EmbeddingProvider,
    ModelLoadError,
)
from rag_tester.providers.embeddings.gemini import (
    AuthenticationError as GeminiAuthenticationError,
)
from rag_tester.providers.embeddings.gemini import (
    GeminiProvider,
)
from rag_tester.providers.embeddings.gemini import (
    MissingAPIKeyError as GeminiMissingAPIKeyError,
)
from rag_tester.providers.embeddings.gemini import (
    RateLimitError as GeminiRateLimitError,
)
from rag_tester.providers.embeddings.local import LocalEmbeddingProvider
from rag_tester.providers.embeddings.openrouter import (
    AuthenticationError as OpenRouterAuthenticationError,
)
from rag_tester.providers.embeddings.openrouter import (
    MissingAPIKeyError as OpenRouterMissingAPIKeyError,
)
from rag_tester.providers.embeddings.openrouter import (
    OpenRouterProvider,
)
from rag_tester.providers.embeddings.openrouter import (
    RateLimitError as OpenRouterRateLimitError,
)

_REGISTRY: dict[str, type[EmbeddingProvider]] = {
    "local": LocalEmbeddingProvider,
    "gemini": GeminiProvider,
    "openrouter": OpenRouterProvider,
}


def get_embedding_provider(name: str, model_name: str) -> EmbeddingProvider:
    """Instantiate the embedding provider matching ``name``.

    Args:
        name: Provider identifier (``local``, ``gemini``, ``openrouter``).
        model_name: Model identifier passed to the provider constructor.

    Returns:
        An initialised :class:`EmbeddingProvider` instance.

    Raises:
        ValueError: If ``name`` is not a registered provider.
    """
    provider_cls = _REGISTRY.get(name)
    if provider_cls is None:
        msg = f"Unsupported embedding provider '{name}'. Known providers: {sorted(_REGISTRY.keys())}"
        raise ValueError(msg)
    return provider_cls(model_name=model_name)  # type: ignore[call-arg]


__all__ = [
    "EmbeddingError",
    "EmbeddingProvider",
    "GeminiAuthenticationError",
    "GeminiMissingAPIKeyError",
    "GeminiProvider",
    "GeminiRateLimitError",
    "LocalEmbeddingProvider",
    "ModelLoadError",
    "OpenRouterAuthenticationError",
    "OpenRouterMissingAPIKeyError",
    "OpenRouterProvider",
    "OpenRouterRateLimitError",
    "get_embedding_provider",
]
