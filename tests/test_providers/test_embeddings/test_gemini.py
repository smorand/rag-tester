"""Tests for GeminiProvider."""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from rag_tester.providers.embeddings.gemini import (
    AuthenticationError,
    GeminiProvider,
    MissingAPIKeyError,
    RateLimitError,
)


class TestGeminiProvider:
    """Tests for GeminiProvider."""

    @pytest.fixture
    def mock_api_key(self, monkeypatch: pytest.MonkeyPatch) -> str:
        """Set up mock API key."""
        api_key = "test-gemini-api-key-12345"
        monkeypatch.setenv("GEMINI_API_KEY", api_key)
        return api_key

    @pytest.fixture
    def provider(self, mock_api_key: str) -> GeminiProvider:
        """Create a GeminiProvider instance."""
        return GeminiProvider("models/text-embedding-004")

    @pytest.fixture
    def mock_response_data(self) -> dict:
        """Create mock API response data."""
        return {
            "embeddings": [
                {"values": [0.1] * 768},
                {"values": [0.2] * 768},
                {"values": [0.3] * 768},
            ],
        }

    async def test_init_with_api_key(self, mock_api_key: str) -> None:
        """Test initialization with API key from environment."""
        provider = GeminiProvider("models/text-embedding-004")
        assert provider.get_model_name() == "models/text-embedding-004"
        assert provider.get_dimension() == 768

    async def test_init_with_explicit_api_key(self) -> None:
        """Test initialization with explicit API key."""
        provider = GeminiProvider("models/text-embedding-004", api_key="explicit-key")
        assert provider.get_model_name() == "models/text-embedding-004"

    async def test_init_missing_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that missing API key raises MissingAPIKeyError."""
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        with pytest.raises(
            MissingAPIKeyError,
            match="Missing API key: GEMINI_API_KEY",
        ):
            GeminiProvider("models/text-embedding-004")

    async def test_init_empty_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that empty API key raises MissingAPIKeyError."""
        monkeypatch.setenv("GEMINI_API_KEY", "")
        with pytest.raises(MissingAPIKeyError, match="Missing API key"):
            GeminiProvider("models/text-embedding-004")

    async def test_init_invalid_model(self, mock_api_key: str) -> None:
        """Test that invalid model raises ValueError."""
        with pytest.raises(ValueError, match="Unknown model dimension"):
            GeminiProvider("invalid/model")

    async def test_embed_texts_success(
        self,
        provider: GeminiProvider,
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
            assert all(len(emb) == 768 for emb in embeddings)
            assert embeddings[0] == [0.1] * 768
            assert embeddings[1] == [0.2] * 768
            assert embeddings[2] == [0.3] * 768

            # Verify API call
            mock_client.post.assert_called_once()
            call_args = mock_client.post.call_args
            assert "generativelanguage.googleapis.com" in call_args[0][0]
            assert "batchEmbedContents" in call_args[0][0]
            assert call_args[1]["params"]["key"] == "test-gemini-api-key-12345"

    async def test_embed_texts_empty_list(self, provider: GeminiProvider) -> None:
        """Test embedding empty list returns empty list."""
        embeddings = await provider.embed_texts([])
        assert embeddings == []

    async def test_embed_texts_authentication_error(
        self,
        provider: GeminiProvider,
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
        provider: GeminiProvider,
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
        provider: GeminiProvider,
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
                "embeddings": [{"values": [0.1] * 768}],
            }

            mock_client.post.side_effect = [
                mock_response_error1,
                mock_response_error2,
                mock_response_success,
            ]

            embeddings = await provider.embed_texts(texts)

            assert len(embeddings) == 1
            assert mock_client.post.call_count == 3

    async def test_embed_texts_timeout(self, provider: GeminiProvider) -> None:
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
        provider: GeminiProvider,
    ) -> None:
        """Test that large batches are split correctly."""
        # Create 250 texts (exceeds max batch size of 100)
        texts = [f"text{i}" for i in range(250)]

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client

            # Mock responses for three batches
            mock_response1 = MagicMock()
            mock_response1.status_code = 200
            mock_response1.json.return_value = {
                "embeddings": [{"values": [0.1] * 768} for _ in range(100)],
            }

            mock_response2 = MagicMock()
            mock_response2.status_code = 200
            mock_response2.json.return_value = {
                "embeddings": [{"values": [0.2] * 768} for _ in range(100)],
            }

            mock_response3 = MagicMock()
            mock_response3.status_code = 200
            mock_response3.json.return_value = {
                "embeddings": [{"values": [0.3] * 768} for _ in range(50)],
            }

            mock_client.post.side_effect = [mock_response1, mock_response2, mock_response3]

            embeddings = await provider.embed_texts(texts)

            assert len(embeddings) == 250
            assert mock_client.post.call_count == 3

            # Verify batch sizes in request payloads
            call1_payload = mock_client.post.call_args_list[0][1]["json"]
            call2_payload = mock_client.post.call_args_list[1][1]["json"]
            call3_payload = mock_client.post.call_args_list[2][1]["json"]
            assert len(call1_payload["requests"]) == 100
            assert len(call2_payload["requests"]) == 100
            assert len(call3_payload["requests"]) == 50

    async def test_token_estimation(
        self,
        provider: GeminiProvider,
        mock_response_data: dict,
    ) -> None:
        """Test that tokens are estimated correctly."""
        # Each text has 20 characters, so estimated tokens = 20 * 3 / 4 = 15
        texts = ["a" * 20, "b" * 20, "c" * 20]

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client

            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = mock_response_data
            mock_client.post.return_value = mock_response

            await provider.embed_texts(texts)

            # Total chars = 60, estimated tokens = 60 / 4 = 15
            assert provider.get_total_tokens() == 15

    async def test_cumulative_tokens(
        self,
        provider: GeminiProvider,
    ) -> None:
        """Test that tokens accumulate across multiple calls."""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client

            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "embeddings": [{"values": [0.1] * 768}],
            }
            mock_client.post.return_value = mock_response

            # First call: 100 chars = 25 tokens
            await provider.embed_texts(["a" * 100])
            assert provider.get_total_tokens() == 25

            # Second call: 100 chars = 25 tokens
            await provider.embed_texts(["b" * 100])
            assert provider.get_total_tokens() == 50

    def test_get_dimension(self, provider: GeminiProvider) -> None:
        """Test getting embedding dimension."""
        assert provider.get_dimension() == 768

    def test_get_model_name(self, provider: GeminiProvider) -> None:
        """Test getting model name."""
        assert provider.get_model_name() == "models/text-embedding-004"

    async def test_alternative_model(self, mock_api_key: str) -> None:
        """Test that alternative model has correct dimension."""
        provider = GeminiProvider("models/embedding-001")
        assert provider.get_dimension() == 768
        assert provider.get_model_name() == "models/embedding-001"

    async def test_request_format(
        self,
        provider: GeminiProvider,
    ) -> None:
        """Test that request format matches Gemini API expectations."""
        texts = ["test text"]

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client

            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "embeddings": [{"values": [0.1] * 768}],
            }
            mock_client.post.return_value = mock_response

            await provider.embed_texts(texts)

            # Verify request format
            call_args = mock_client.post.call_args
            payload = call_args[1]["json"]

            assert "requests" in payload
            assert len(payload["requests"]) == 1
            assert payload["requests"][0]["model"] == "models/text-embedding-004"
            assert "content" in payload["requests"][0]
            assert "parts" in payload["requests"][0]["content"]
            assert payload["requests"][0]["content"]["parts"][0]["text"] == "test text"
