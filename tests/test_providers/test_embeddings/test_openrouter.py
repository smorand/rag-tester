"""Tests for OpenRouterProvider."""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from rag_tester.providers.embeddings.openrouter import (
    AuthenticationError,
    MissingAPIKeyError,
    OpenRouterProvider,
    RateLimitError,
)


class TestOpenRouterProvider:
    """Tests for OpenRouterProvider."""

    @pytest.fixture
    def mock_api_key(self, monkeypatch: pytest.MonkeyPatch) -> str:
        """Set up mock API key."""
        api_key = "test-api-key-12345"
        monkeypatch.setenv("OPENROUTER_API_KEY", api_key)
        return api_key

    @pytest.fixture
    def provider(self, mock_api_key: str) -> OpenRouterProvider:
        """Create an OpenRouterProvider instance."""
        return OpenRouterProvider("openai/text-embedding-3-small")

    @pytest.fixture
    def mock_response_data(self) -> dict:
        """Create mock API response data."""
        return {
            "data": [
                {"embedding": [0.1] * 1536},
                {"embedding": [0.2] * 1536},
                {"embedding": [0.3] * 1536},
            ],
            "usage": {"total_tokens": 150},
        }

    async def test_init_with_api_key(self, mock_api_key: str) -> None:
        """Test initialization with API key from environment."""
        provider = OpenRouterProvider("openai/text-embedding-3-small")
        assert provider.get_model_name() == "openai/text-embedding-3-small"
        assert provider.get_dimension() == 1536

    async def test_init_with_explicit_api_key(self) -> None:
        """Test initialization with explicit API key."""
        provider = OpenRouterProvider("openai/text-embedding-3-small", api_key="explicit-key")
        assert provider.get_model_name() == "openai/text-embedding-3-small"

    async def test_init_missing_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that missing API key raises MissingAPIKeyError."""
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        with pytest.raises(
            MissingAPIKeyError,
            match="Missing API key: OPENROUTER_API_KEY",
        ):
            OpenRouterProvider("openai/text-embedding-3-small")

    async def test_init_empty_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that empty API key raises MissingAPIKeyError."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "")
        with pytest.raises(MissingAPIKeyError, match="Missing API key"):
            OpenRouterProvider("openai/text-embedding-3-small")

    async def test_init_invalid_model(self, mock_api_key: str) -> None:
        """Test that invalid model raises ValueError."""
        with pytest.raises(ValueError, match="Unknown model dimension"):
            OpenRouterProvider("invalid/model")

    async def test_embed_texts_success(
        self,
        provider: OpenRouterProvider,
        mock_response_data: dict,
    ) -> None:
        """Test successful embedding generation."""
        texts = ["text1", "text2", "text3"]

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client

            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = mock_response_data
            mock_client.post.return_value = mock_response

            embeddings = await provider.embed_texts(texts)

            assert len(embeddings) == 3
            assert all(len(emb) == 1536 for emb in embeddings)
            assert embeddings[0] == [0.1] * 1536
            assert embeddings[1] == [0.2] * 1536
            assert embeddings[2] == [0.3] * 1536

            # Verify API call
            mock_client.post.assert_called_once()
            call_args = mock_client.post.call_args
            assert call_args[0][0] == "https://openrouter.ai/api/v1/embeddings"
            assert call_args[1]["json"]["model"] == "openai/text-embedding-3-small"
            assert call_args[1]["json"]["input"] == texts

    async def test_embed_texts_empty_list(self, provider: OpenRouterProvider) -> None:
        """Test embedding empty list returns empty list."""
        embeddings = await provider.embed_texts([])
        assert embeddings == []

    async def test_embed_texts_authentication_error(
        self,
        provider: OpenRouterProvider,
    ) -> None:
        """Test that 401 raises AuthenticationError."""
        texts = ["text1"]

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client

            mock_response = MagicMock()
            mock_response.status_code = 401
            mock_client.post.return_value = mock_response

            with pytest.raises(AuthenticationError, match="Authentication failed"):
                await provider.embed_texts(texts)

    async def test_embed_texts_rate_limit_error(
        self,
        provider: OpenRouterProvider,
    ) -> None:
        """Test that 429 is retried and eventually raises RetryError."""
        from rag_tester.utils.retry import RetryError

        texts = ["text1"]

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client

            mock_response = MagicMock()
            mock_response.status_code = 429
            mock_response.headers = {"Retry-After": "60"}
            mock_client.post.return_value = mock_response

            with pytest.raises(RetryError, match="Max retry attempts"):
                await provider.embed_texts(texts)

    async def test_embed_texts_server_error_retry(
        self,
        provider: OpenRouterProvider,
        mock_response_data: dict,
    ) -> None:
        """Test that 5xx errors are retried and eventually succeed."""
        texts = ["text1"]

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client

            # First two calls return 500, third call succeeds
            mock_response_error1 = MagicMock()
            mock_response_error1.status_code = 500

            mock_response_error2 = MagicMock()
            mock_response_error2.status_code = 500

            mock_response_success = MagicMock()
            mock_response_success.status_code = 200
            mock_response_success.json.return_value = {
                "data": [{"embedding": [0.1] * 1536}],
                "usage": {"total_tokens": 50},
            }

            mock_client.post.side_effect = [
                mock_response_error1,
                mock_response_error2,
                mock_response_success,
            ]

            embeddings = await provider.embed_texts(texts)

            assert len(embeddings) == 1
            assert mock_client.post.call_count == 3

    async def test_embed_texts_timeout(self, provider: OpenRouterProvider) -> None:
        """Test that timeout raises EmbeddingError."""
        texts = ["text1"]

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client
            mock_client.post.side_effect = httpx.TimeoutException("Timeout")

            with pytest.raises(Exception, match="API request timeout"):
                await provider.embed_texts(texts)

    async def test_embed_texts_batch_splitting(
        self,
        provider: OpenRouterProvider,
    ) -> None:
        """Test that large batches are split correctly."""
        # Create 3000 texts (exceeds max batch size of 2048)
        texts = [f"text{i}" for i in range(3000)]

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client

            # Mock responses for two batches
            mock_response1 = MagicMock()
            mock_response1.status_code = 200
            mock_response1.json.return_value = {
                "data": [{"embedding": [0.1] * 1536} for _ in range(2048)],
                "usage": {"total_tokens": 10000},
            }

            mock_response2 = MagicMock()
            mock_response2.status_code = 200
            mock_response2.json.return_value = {
                "data": [{"embedding": [0.2] * 1536} for _ in range(952)],
                "usage": {"total_tokens": 5000},
            }

            mock_client.post.side_effect = [mock_response1, mock_response2]

            embeddings = await provider.embed_texts(texts)

            assert len(embeddings) == 3000
            assert mock_client.post.call_count == 2

            # Verify batch sizes
            call1_batch = mock_client.post.call_args_list[0][1]["json"]["input"]
            call2_batch = mock_client.post.call_args_list[1][1]["json"]["input"]
            assert len(call1_batch) == 2048
            assert len(call2_batch) == 952

    async def test_token_counting(
        self,
        provider: OpenRouterProvider,
        mock_response_data: dict,
    ) -> None:
        """Test that tokens are counted correctly."""
        texts = ["text1", "text2", "text3"]

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client

            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = mock_response_data
            mock_client.post.return_value = mock_response

            await provider.embed_texts(texts)

            assert provider.get_total_tokens() == 150

    async def test_cost_calculation(
        self,
        provider: OpenRouterProvider,
        mock_response_data: dict,
    ) -> None:
        """Test that cost is calculated correctly."""
        texts = ["text1", "text2", "text3"]

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client

            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = mock_response_data
            mock_client.post.return_value = mock_response

            await provider.embed_texts(texts)

            # 150 tokens * $0.02 per 1M tokens = $0.000003
            expected_cost = (150 / 1_000_000) * 0.02
            assert provider.get_total_cost() == pytest.approx(expected_cost, abs=1e-6)

    async def test_cumulative_tokens_and_cost(
        self,
        provider: OpenRouterProvider,
    ) -> None:
        """Test that tokens and cost accumulate across multiple calls."""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client

            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "data": [{"embedding": [0.1] * 1536}],
                "usage": {"total_tokens": 100},
            }
            mock_client.post.return_value = mock_response

            # First call
            await provider.embed_texts(["text1"])
            assert provider.get_total_tokens() == 100

            # Second call
            await provider.embed_texts(["text2"])
            assert provider.get_total_tokens() == 200

            # Cost should also accumulate
            expected_cost = (200 / 1_000_000) * 0.02
            assert provider.get_total_cost() == pytest.approx(expected_cost, abs=1e-6)

    def test_get_dimension(self, provider: OpenRouterProvider) -> None:
        """Test getting embedding dimension."""
        assert provider.get_dimension() == 1536

    def test_get_model_name(self, provider: OpenRouterProvider) -> None:
        """Test getting model name."""
        assert provider.get_model_name() == "openai/text-embedding-3-small"

    async def test_large_model_dimension(self, mock_api_key: str) -> None:
        """Test that large model has correct dimension."""
        provider = OpenRouterProvider("openai/text-embedding-3-large")
        assert provider.get_dimension() == 3072
