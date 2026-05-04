"""Google Gemini embedding provider using Gemini API."""

import logging

import httpx
from opentelemetry import trace
from pydantic import SecretStr

from rag_tester.config import get_settings
from rag_tester.providers.embeddings.base import EmbeddingError, EmbeddingProvider
from rag_tester.utils.retry import retry_with_backoff

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)

# Gemini API configuration
GEMINI_API_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
GEMINI_MAX_BATCH_SIZE = 100

# Model dimensions (from Gemini documentation)
MODEL_DIMENSIONS = {
    "models/text-embedding-004": 768,
    "models/embedding-001": 768,
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


class GeminiProvider(EmbeddingProvider):
    """Google Gemini embedding provider using Gemini API.

    This provider generates embeddings using Google's Gemini API.
    Note: Gemini API does not provide token usage data, so tokens are estimated.

    Args:
        model_name: Model identifier (e.g., "models/text-embedding-004")
        api_key: Gemini API key (defaults to GEMINI_API_KEY env var)
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
        """Initialize the Gemini embedding provider.

        Args:
            model_name: Model identifier (e.g., "models/text-embedding-004")
            api_key: Gemini API key. When ``None``, the value is read from
                ``Settings.gemini_api_key`` which accepts both
                ``RAG_TESTER_GEMINI_API_KEY`` and ``GEMINI_API_KEY``.
            timeout: Request timeout in seconds
        """
        self._model_name = model_name
        self._timeout = timeout
        self._total_tokens = 0

        # Resolve API key: explicit argument takes precedence over Settings.
        if api_key is None:
            settings_key = get_settings().gemini_api_key
            resolved = settings_key.get_secret_value() if settings_key is not None else None
        elif isinstance(api_key, SecretStr):
            resolved = api_key.get_secret_value()
        else:
            resolved = api_key

        self._api_key = resolved
        if not self._api_key:
            msg = "Missing API key: GEMINI_API_KEY. Set the environment variable to use Gemini models."
            logger.error(msg)
            raise MissingAPIKeyError(msg)

        # Validate API key format (basic check)
        if not self._api_key.strip():
            msg = "Invalid API key: GEMINI_API_KEY is empty"
            logger.error(msg)
            raise MissingAPIKeyError(msg)

        logger.debug("API key validation passed (key not logged for security)")

        # Get model dimension
        if model_name not in MODEL_DIMENSIONS:
            msg = f"Unknown model dimension for '{model_name}'. Supported models: {list(MODEL_DIMENSIONS.keys())}"
            logger.error(msg)
            raise ValueError(msg)

        self._dimension = MODEL_DIMENSIONS[model_name]
        logger.info("Using Gemini API with model: %s (dimension: %s)", model_name, self._dimension)

    def _estimate_tokens(self, texts: list[str]) -> int:
        """Estimate token count from text length.

        Gemini API does not provide token usage data, so we estimate
        tokens as approximately 1 token per 4 characters.

        Args:
            texts: List of texts

        Returns:
            Estimated token count
        """
        total_chars = sum(len(text) for text in texts)
        estimated_tokens = total_chars // 4
        logger.debug("Estimated %s tokens from %s characters", estimated_tokens, total_chars)
        return estimated_tokens

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
        """Make API request to Gemini.

        Args:
            texts: List of texts to embed

        Returns:
            Tuple of (embeddings, estimated_token_count)

        Raises:
            AuthenticationError: If authentication fails (401/403)
            RateLimitError: If rate limit is exceeded (429)
            EmbeddingError: For other API errors
        """
        with tracer.start_as_current_span("gemini_api_request") as span:
            span.set_attribute("model.name", self._model_name)
            span.set_attribute("texts.count", len(texts))

            # Prepare request
            # Gemini uses API key in query parameter
            url = f"{GEMINI_API_BASE_URL}/{self._model_name}:batchEmbedContents"
            params = {"key": self._api_key}

            # Gemini expects content.parts format
            payload = {
                "requests": [{"model": self._model_name, "content": {"parts": [{"text": text}]}} for text in texts]
            }

            try:
                async with httpx.AsyncClient(timeout=self._timeout) as client:
                    logger.debug("Making API request to Gemini for %s texts", len(texts))

                    response = await client.post(
                        url,
                        params=params,
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

                    # Extract embeddings from Gemini response format
                    embeddings = [item["values"] for item in data["embeddings"]]

                    # Estimate token count (Gemini doesn't provide usage data)
                    estimated_tokens = self._estimate_tokens(texts)

                    # Update totals
                    self._total_tokens += estimated_tokens

                    # Add tracing attributes
                    span.set_attribute("tokens.estimated", estimated_tokens)
                    span.set_attribute("embeddings.count", len(embeddings))

                    logger.info(
                        "API request successful: %s embeddings, %s tokens (estimated)",
                        len(embeddings),
                        estimated_tokens,
                    )

                    return embeddings, estimated_tokens

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
            for i in range(0, len(texts), GEMINI_MAX_BATCH_SIZE):
                batch = texts[i : i + GEMINI_MAX_BATCH_SIZE]

                logger.debug("Processing batch %s: %s texts", i // GEMINI_MAX_BATCH_SIZE + 1, len(batch))

                embeddings, _estimated_tokens = await self._make_api_request(batch)
                all_embeddings.extend(embeddings)

            span.set_attribute("embeddings.count", len(all_embeddings))
            span.set_attribute("total_tokens.estimated", self._total_tokens)

            logger.info("Generated %s embeddings (total: %s tokens estimated)", len(all_embeddings), self._total_tokens)

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
            The model identifier (e.g., "models/text-embedding-004")
        """
        return self._model_name

    def get_total_tokens(self) -> int:
        """Return the total number of tokens consumed (estimated).

        Returns:
            Total estimated token count across all API calls
        """
        return self._total_tokens
