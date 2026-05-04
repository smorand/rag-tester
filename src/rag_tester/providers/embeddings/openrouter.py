"""OpenRouter embedding provider using OpenRouter API."""

import logging

import httpx
from opentelemetry import trace
from pydantic import SecretStr

from rag_tester.config import get_settings
from rag_tester.providers.embeddings.base import EmbeddingError, EmbeddingProvider
from rag_tester.utils.cost import calculate_cost
from rag_tester.utils.retry import retry_with_backoff

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)

# OpenRouter API configuration
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/embeddings"
OPENROUTER_MAX_BATCH_SIZE = 2048

# Model dimensions (from OpenRouter documentation)
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
        timeout: Request timeout in seconds (default: 30)

    Raises:
        MissingAPIKeyError: If API key is not provided or set in environment
        AuthenticationError: If API authentication fails
        RateLimitError: If rate limit is exceeded after retries
    """

    def __init__(
        self,
        model_name: str,
        api_key: str | SecretStr | None = None,
        timeout: float = 30.0,
    ) -> None:
        """Initialize the OpenRouter embedding provider.

        Args:
            model_name: Model identifier (e.g., "openai/text-embedding-3-small")
            api_key: OpenRouter API key. When ``None``, the value is read from
                ``Settings.openrouter_api_key`` which accepts both
                ``RAG_TESTER_OPENROUTER_API_KEY`` and ``OPENROUTER_API_KEY``.
            timeout: Request timeout in seconds
        """
        self._model_name = model_name
        self._timeout = timeout
        self._total_tokens = 0
        self._total_cost = 0.0

        # Resolve API key: explicit argument takes precedence over Settings.
        if api_key is None:
            settings_key = get_settings().openrouter_api_key
            resolved = settings_key.get_secret_value() if settings_key is not None else None
        elif isinstance(api_key, SecretStr):
            resolved = api_key.get_secret_value()
        else:
            resolved = api_key

        self._api_key = resolved
        if not self._api_key:
            msg = "Missing API key: OPENROUTER_API_KEY. Set the environment variable to use OpenRouter models."
            logger.error(msg)
            raise MissingAPIKeyError(msg)

        # Validate API key format (basic check)
        if not self._api_key.strip():
            msg = "Invalid API key: OPENROUTER_API_KEY is empty"
            logger.error(msg)
            raise MissingAPIKeyError(msg)

        logger.debug("API key validation passed (key not logged for security)")

        # Get model dimension
        if model_name not in MODEL_DIMENSIONS:
            msg = f"Unknown model dimension for '{model_name}'. Supported models: {list(MODEL_DIMENSIONS.keys())}"
            logger.error(msg)
            raise ValueError(msg)

        self._dimension = MODEL_DIMENSIONS[model_name]
        logger.info("Using OpenRouter API with model: %s (dimension: %s)", model_name, self._dimension)

    @retry_with_backoff(
        transient_errors=(
            httpx.TimeoutException,
            httpx.ConnectError,
            httpx.RemoteProtocolError,
            RateLimitError,
        ),
    )
    async def _make_api_request(
        self,
        texts: list[str],
    ) -> tuple[list[list[float]], int]:
        """Make API request to OpenRouter.

        Args:
            texts: List of texts to embed

        Returns:
            Tuple of (embeddings, token_count)

        Raises:
            AuthenticationError: If authentication fails (401/403)
            RateLimitError: If rate limit is exceeded (429)
            EmbeddingError: For other API errors
        """
        with tracer.start_as_current_span("openrouter_api_request") as span:
            span.set_attribute("model.name", self._model_name)
            span.set_attribute("texts.count", len(texts))

            # Prepare request
            headers = {
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            }

            payload = {
                "model": self._model_name,
                "input": texts,
            }

            try:
                async with httpx.AsyncClient(timeout=self._timeout) as client:
                    logger.debug("Making API request to OpenRouter for %s texts", len(texts))

                    response = await client.post(
                        OPENROUTER_API_URL,
                        headers=headers,
                        json=payload,
                    )

                    # Handle authentication errors (permanent - don't retry)
                    if response.status_code in (401, 403):
                        error_msg = "Authentication failed: invalid API key"
                        logger.error(error_msg)
                        span.set_attribute("error", error_msg)
                        raise AuthenticationError(error_msg)

                    # Handle rate limit errors (transient - retry)
                    if response.status_code == 429:
                        retry_after = response.headers.get("Retry-After")
                        error_msg = f"Rate limit exceeded. Retry-After: {retry_after}"
                        logger.warning(error_msg)
                        span.set_attribute("rate_limit.retry_after", retry_after or "not provided")
                        raise RateLimitError(error_msg)

                    # Handle other errors (including 5xx which will be retried)
                    if response.status_code >= 500:
                        error_msg = f"API server error: {response.status_code}"
                        logger.warning(error_msg)
                        span.set_attribute("error", error_msg)
                        raise httpx.RemoteProtocolError(error_msg)

                    # Handle other client errors
                    response.raise_for_status()

                    # Parse response
                    data = response.json()

                    # Extract embeddings
                    embeddings = [item["embedding"] for item in data["data"]]

                    # Extract token count
                    token_count = data.get("usage", {}).get("total_tokens", 0)

                    # Calculate cost
                    cost = calculate_cost(self._model_name, token_count)

                    # Update totals
                    self._total_tokens += token_count
                    self._total_cost += cost

                    # Add tracing attributes
                    span.set_attribute("tokens.count", token_count)
                    span.set_attribute("cost.usd", cost)
                    span.set_attribute("embeddings.count", len(embeddings))

                    logger.info(
                        "API request successful: %s embeddings, %s tokens, $%.6f",
                        len(embeddings),
                        token_count,
                        cost,
                    )

                    return embeddings, token_count

            except httpx.TimeoutException as e:
                error_msg = "API request timeout"
                logger.error(error_msg)
                span.set_attribute("error", error_msg)
                raise EmbeddingError(error_msg) from e

            except (AuthenticationError, RateLimitError, httpx.RemoteProtocolError):
                # Re-raise these without wrapping (retry decorator will handle RemoteProtocolError)
                raise

            except Exception as e:
                error_msg = f"API request failed: {e}"
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

            # Batch texts to respect API limits
            for i in range(0, len(texts), OPENROUTER_MAX_BATCH_SIZE):
                batch = texts[i : i + OPENROUTER_MAX_BATCH_SIZE]

                logger.debug("Processing batch %s: %s texts", i // OPENROUTER_MAX_BATCH_SIZE + 1, len(batch))

                embeddings, _token_count = await self._make_api_request(batch)
                all_embeddings.extend(embeddings)

            span.set_attribute("embeddings.count", len(all_embeddings))
            span.set_attribute("total_tokens", self._total_tokens)
            span.set_attribute("total_cost", self._total_cost)

            logger.info(
                "Generated %s embeddings (total: %s tokens, $%.6f)",
                len(all_embeddings),
                self._total_tokens,
                self._total_cost,
            )

            return all_embeddings

    def get_dimension(self) -> int:
        """Return the embedding dimension for this model.

        Returns:
            The number of dimensions in each embedding vector
        """
        return self._dimension

    def get_model_name(self) -> str:
        """Return the model identifier.

        Returns:
            The model identifier (e.g., "openai/text-embedding-3-small")
        """
        return self._model_name

    def get_total_tokens(self) -> int:
        """Return the total number of tokens consumed.

        Returns:
            Total token count across all API calls
        """
        return self._total_tokens

    def get_total_cost(self) -> float:
        """Return the total cost in USD.

        Returns:
            Total cost across all API calls
        """
        return self._total_cost
