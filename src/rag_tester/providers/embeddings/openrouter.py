"""OpenRouter embedding provider using OpenRouter API."""

import logging
import os
from typing import Any

import httpx
from opentelemetry import trace

from rag_tester.providers.embeddings.base import EmbeddingError, EmbeddingProvider
from rag_tester.utils.retry import retry_with_backoff

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)

# OpenRouter API configuration
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/embeddings"
OPENROUTER_BATCH_LIMIT = 2048

# Model dimensions mapping
MODEL_DIMENSIONS = {
    "openai/text-embedding-3-small": 1536,
    "openai/text-embedding-3-large": 3072,
    "openai/text-embedding-ada-002": 1536,
}


class AuthenticationError(EmbeddingError):
    """Raised when API authentication fails."""

    pass


class RateLimitError(EmbeddingError):
    """Raised when API rate limit is exceeded."""

    pass


class MissingAPIKeyError(EmbeddingError):
    """Raised when API key is not set."""

    pass


class OpenRouterProvider(EmbeddingProvider):
    """OpenRouter embedding provider using OpenRouter API.

    This provider generates embeddings using OpenRouter's API, which supports
    multiple embedding models from various providers (OpenAI, Cohere, etc.).

    Args:
        model_name: Model identifier (e.g., "openai/text-embedding-3-small")
        api_key: OpenRouter API key (defaults to OPENROUTER_API_KEY env var)

    Raises:
        MissingAPIKeyError: If API key is not provided or set in environment
        AuthenticationError: If API authentication fails
    """

    def __init__(self, model_name: str, api_key: str | None = None) -> None:
        """Initialize the OpenRouter embedding provider.

        Args:
            model_name: Model identifier (e.g., "openai/text-embedding-3-small")
            api_key: OpenRouter API key (defaults to OPENROUTER_API_KEY env var)

        Raises:
            MissingAPIKeyError: If API key is not provided or set in environment
        """
        self._model_name = model_name
        self._api_key = api_key or os.getenv("OPENROUTER_API_KEY")

        if not self._api_key:
            msg = "Missing API key: OPENROUTER_API_KEY. Set the environment variable to use OpenRouter models."
            logger.error(msg)
            raise MissingAPIKeyError(msg)

        # Validate API key format (basic check)
        if not self._api_key.strip():
            msg = "Invalid API key: OPENROUTER_API_KEY is empty"
            logger.error(msg)
            raise MissingAPIKeyError(msg)

        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(60.0),
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
        )

        self._total_tokens = 0
        self._total_cost = 0.0

        logger.info(f"Initialized OpenRouter provider with model: {self._model_name}")
        logger.debug("API key validation: OK (key is set and non-empty)")

    async def __aenter__(self) -> "OpenRouterProvider":
        """Async context manager entry."""
        return self

    async def __aexit__(self, *args: Any) -> None:
        """Async context manager exit."""
        await self._client.aclose()

    async def close(self) -> None:
        """Close the HTTP client."""
        await self._client.aclose()

    @retry_with_backoff(
        transient_errors=(
            httpx.TimeoutException,
            httpx.NetworkError,
            httpx.RemoteProtocolError,
            RateLimitError,
        )
    )
    async def _make_api_request(self, texts: list[str]) -> dict[str, Any]:
        """Make API request to OpenRouter.

        Args:
            texts: List of texts to embed (max 2048)

        Returns:
            API response as dictionary

        Raises:
            AuthenticationError: If authentication fails (401/403)
            RateLimitError: If rate limit is exceeded (429)
            EmbeddingError: For other API errors
        """
        with tracer.start_as_current_span("openrouter_api_request") as span:
            span.set_attribute("model.name", self._model_name)
            span.set_attribute("texts.count", len(texts))

            try:
                response = await self._client.post(
                    OPENROUTER_API_URL,
                    json={
                        "model": self._model_name,
                        "input": texts,
                    },
                )

                span.set_attribute("http.status_code", response.status_code)

                # Handle authentication errors (permanent - don't retry)
                if response.status_code in (401, 403):
                    error_msg = "Authentication failed: invalid API key"
                    logger.error(error_msg)
                    span.set_attribute("error", error_msg)
                    raise AuthenticationError(error_msg)

                # Handle rate limiting (transient - retry)
                if response.status_code == 429:
                    retry_after = response.headers.get("Retry-After")
                    error_msg = f"Rate limit exceeded. Retry-After: {retry_after}"
                    logger.warning(error_msg)
                    span.set_attribute("error", error_msg)
                    span.set_attribute("retry_after", retry_after or "not provided")
                    raise RateLimitError(error_msg)

                # Handle server errors (transient - retry)
                if response.status_code >= 500:
                    error_msg = f"API server error: {response.status_code}"
                    logger.warning(error_msg)
                    span.set_attribute("error", error_msg)
                    # Use a custom exception that's in transient_errors
                    raise httpx.RemoteProtocolError(error_msg)

                # Handle other errors
                response.raise_for_status()

                data: dict[str, Any] = response.json()
                span.set_attribute("status", "success")

                # Extract token count
                tokens = data.get("usage", {}).get("total_tokens", 0)
                span.set_attribute("tokens", tokens)

                logger.info(f"API request successful: {len(texts)} texts, {tokens} tokens")

                return data

            except (AuthenticationError, RateLimitError, EmbeddingError):
                # Re-raise our custom exceptions without wrapping
                raise

            except (
                httpx.TimeoutException,
                httpx.NetworkError,
                httpx.RemoteProtocolError,
            ):
                # Re-raise transient errors without wrapping (for retry logic)
                raise

            except httpx.HTTPStatusError as e:
                error_msg = f"HTTP error: {e.response.status_code}"
                logger.error(f"{error_msg}: {e}")
                span.set_attribute("error", error_msg)
                raise EmbeddingError(error_msg) from e

            except Exception as e:
                error_msg = f"Unexpected error during API request: {e}"
                logger.error(error_msg)
                span.set_attribute("error", error_msg)
                raise EmbeddingError(error_msg) from e

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of texts.

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors (each vector is a list of floats)

        Raises:
            EmbeddingError: If embedding generation fails
        """
        if not texts:
            return []

        with tracer.start_as_current_span("embed_texts") as span:
            span.set_attribute("model.name", self._model_name)
            span.set_attribute("texts.count", len(texts))

            all_embeddings: list[list[float]] = []
            total_tokens = 0

            # Batch texts to respect API limits
            for i in range(0, len(texts), OPENROUTER_BATCH_LIMIT):
                batch = texts[i : i + OPENROUTER_BATCH_LIMIT]

                with tracer.start_as_current_span("batch_request") as batch_span:
                    batch_span.set_attribute("batch.index", i // OPENROUTER_BATCH_LIMIT)
                    batch_span.set_attribute("batch.size", len(batch))

                    logger.debug(f"Processing batch {i // OPENROUTER_BATCH_LIMIT + 1}: {len(batch)} texts")

                    # Make API request
                    response = await self._make_api_request(batch)

                    # Extract embeddings
                    embeddings_data = response.get("data", [])
                    batch_embeddings = [item["embedding"] for item in embeddings_data]
                    all_embeddings.extend(batch_embeddings)

                    # Track tokens
                    batch_tokens = response.get("usage", {}).get("total_tokens", 0)
                    total_tokens += batch_tokens
                    batch_span.set_attribute("tokens", batch_tokens)

            # Update totals
            self._total_tokens += total_tokens
            span.set_attribute("total_tokens", total_tokens)
            span.set_attribute("embeddings.count", len(all_embeddings))

            # Calculate cost
            from rag_tester.utils.cost import calculate_cost

            batch_cost = calculate_cost(self._model_name, total_tokens)
            self._total_cost += batch_cost
            span.set_attribute("cost", batch_cost)

            logger.info(f"Generated {len(all_embeddings)} embeddings: {total_tokens} tokens, ${batch_cost:.6f}")

            return all_embeddings

    def get_dimension(self) -> int:
        """Return the embedding dimension for this model.

        Returns:
            The number of dimensions in each embedding vector

        Raises:
            ValueError: If model dimension is unknown
        """
        if self._model_name not in MODEL_DIMENSIONS:
            msg = f"Unknown dimension for model: {self._model_name}"
            logger.error(msg)
            raise ValueError(msg)

        return MODEL_DIMENSIONS[self._model_name]

    def get_model_name(self) -> str:
        """Return the model identifier.

        Returns:
            The model identifier (e.g., "openai/text-embedding-3-small")
        """
        return self._model_name

    def get_total_tokens(self) -> int:
        """Return the total tokens consumed.

        Returns:
            Total tokens consumed across all API calls
        """
        return self._total_tokens

    def get_total_cost(self) -> float:
        """Return the total cost incurred.

        Returns:
            Total cost in USD across all API calls
        """
        return self._total_cost
