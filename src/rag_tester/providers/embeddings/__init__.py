"""Embedding provider implementations."""

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
]
