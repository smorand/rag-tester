"""Tests for GeminiProvider."""

from unittest.mock import AsyncMock, MagicMock

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
    def mock_response_success(self) -> dict:
        """Mock successful API response."""
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

    async def test_embed_texts_success(
        self,
        provider: GeminiProvider,
        mock_response_success: dict,
        mocker: pytest.MockerFixture,
    ) -> None:
        """Test successful embedding generation."""
        # Mock the HTTP client
        mock_post = mocker.patch.object(
            provider._client,
            "post",
            new_callable=AsyncMock,
        )
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_response_success
        mock_post.return_value = mock_response

        texts = ["Text 1", "Text 2", "Text 3"]
        embeddings = await provider.embed_texts(texts)

        assert len(embeddings) == 3
        assert all(len(emb) == 768 for emb in embeddings)
        assert embeddings[0] == [0.1] * 768
        assert embeddings[1] == [0.2] * 768
        assert embeddings[2] == [0.3] * 768

        # Verify API call
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert "models/text-embedding-004:batchEmbedContents" in call_args.args[0]
        assert call_args.kwargs["params"]["key"] == provider._api_key

    async def test_embed_texts_empty_list(self, provider: GeminiProvider) -> None:
        """Test embedding empty list returns empty list."""
        embeddings = await provider.embed_texts([])
        assert embeddings == []

    async def test_embed_texts_authentication_error(
        self,
        provider: GeminiProvider,
        mocker: pytest.MockerFixture,
    ) -> None:
        """Test that 401 raises AuthenticationError."""
        mock_post = mocker.patch.object(
            provider._client,
            "post",
            new_callable=AsyncMock,
        )
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_post.return_value = mock_response

        with pytest.raises(AuthenticationError, match="Authentication failed"):
            await provider.embed_texts(["Test"])

    async def test_embed_texts_rate_limit_error(
        self,
        provider: GeminiProvider,
        mocker: pytest.MockerFixture,
    ) -> None:
        """Test that 429 is retried and eventually raises RetryError."""
        from rag_tester.utils.retry import RetryError

        mock_post = mocker.patch.object(
            provider._client,
            "post",
            new_callable=AsyncMock,
        )
        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.headers = {"Retry-After": "60"}
        mock_post.return_value = mock_response

        with pytest.raises(RetryError, match="Max retry attempts"):
            await provider.embed_texts(["Test"])

    async def test_embed_texts_rate_limit_retry_success(
        self,
        provider: GeminiProvider,
        mock_response_success: dict,
        mocker: pytest.MockerFixture,
    ) -> None:
        """Test that rate limit is retried and succeeds."""
        mock_post = mocker.patch.object(
            provider._client,
            "post",
            new_callable=AsyncMock,
        )

        # First call returns 429, second call succeeds
        mock_response_429 = MagicMock()
        mock_response_429.status_code = 429
        mock_response_429.headers = {"Retry-After": "1"}

        mock_response_200 = MagicMock()
        mock_response_200.status_code = 200
        mock_response_200.json.return_value = {
            "embeddings": [{"values": [0.1] * 768}],
        }

        mock_post.side_effect = [mock_response_429, mock_response_200]

        embeddings = await provider.embed_texts(["Test"])

        assert len(embeddings) == 1
        assert len(embeddings[0]) == 768
        assert mock_post.call_count == 2

    async def test_embed_texts_server_error_retry(
        self,
        provider: GeminiProvider,
        mock_response_success: dict,
        mocker: pytest.MockerFixture,
    ) -> None:
        """Test that 5xx errors are retried."""
        mock_post = mocker.patch.object(
            provider._client,
            "post",
            new_callable=AsyncMock,
        )

        # First 2 calls return 500, third call succeeds
        mock_response_500 = MagicMock()
        mock_response_500.status_code = 500

        mock_response_200 = MagicMock()
        mock_response_200.status_code = 200
        mock_response_200.json.return_value = {
            "embeddings": [{"values": [0.1] * 768}],
        }

        mock_post.side_effect = [mock_response_500, mock_response_500, mock_response_200]

        embeddings = await provider.embed_texts(["Test"])

        assert len(embeddings) == 1
        assert mock_post.call_count == 3

    async def test_embed_texts_batch_handling(
        self,
        provider: GeminiProvider,
        mocker: pytest.MockerFixture,
    ) -> None:
        """Test that large batches are split correctly."""
        mock_post = mocker.patch.object(
            provider._client,
            "post",
            new_callable=AsyncMock,
        )

        # Create response for each batch
        def create_response(batch_size: int) -> MagicMock:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "embeddings": [{"values": [0.1] * 768} for _ in range(batch_size)],
            }
            return mock_response

        # Test with 250 texts (should be split into 100 + 100 + 50)
        texts = [f"Text {i}" for i in range(250)]
        mock_post.side_effect = [create_response(100), create_response(100), create_response(50)]

        embeddings = await provider.embed_texts(texts)

        assert len(embeddings) == 250
        assert mock_post.call_count == 3

        # Verify batch sizes
        first_call = mock_post.call_args_list[0]
        second_call = mock_post.call_args_list[1]
        third_call = mock_post.call_args_list[2]
        assert len(first_call.kwargs["json"]["requests"]) == 100
        assert len(second_call.kwargs["json"]["requests"]) == 100
        assert len(third_call.kwargs["json"]["requests"]) == 50

    async def test_token_estimation(
        self,
        provider: GeminiProvider,
        mocker: pytest.MockerFixture,
    ) -> None:
        """Test that tokens are estimated correctly."""
        mock_post = mocker.patch.object(
            provider._client,
            "post",
            new_callable=AsyncMock,
        )
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "embeddings": [{"values": [0.1] * 768}],
        }
        mock_post.return_value = mock_response

        # Text with 400 characters should estimate ~100 tokens (400 / 4)
        text = "a" * 400
        await provider.embed_texts([text])

        assert provider.get_total_tokens() == 100

        # Second call should accumulate
        await provider.embed_texts([text])
        assert provider.get_total_tokens() == 200

    async def test_estimate_tokens_method(self, provider: GeminiProvider) -> None:
        """Test the _estimate_tokens method directly."""
        texts = ["a" * 400, "b" * 600]  # 1000 chars total
        estimated = provider._estimate_tokens(texts)
        assert estimated == 250  # 1000 / 4

    def test_get_dimension(self, provider: GeminiProvider) -> None:
        """Test getting embedding dimension."""
        assert provider.get_dimension() == 768

    def test_get_dimension_unknown_model(self, mock_api_key: str) -> None:
        """Test that unknown model raises ValueError."""
        provider = GeminiProvider("models/unknown-model")
        with pytest.raises(ValueError, match="Unknown dimension for model"):
            provider.get_dimension()

    def test_get_model_name(self, provider: GeminiProvider) -> None:
        """Test getting model name."""
        assert provider.get_model_name() == "models/text-embedding-004"

    async def test_context_manager(self, mock_api_key: str) -> None:
        """Test async context manager."""
        async with GeminiProvider("models/text-embedding-004") as provider:
            assert provider.get_model_name() == "models/text-embedding-004"

    async def test_close(self, provider: GeminiProvider) -> None:
        """Test closing the provider."""
        await provider.close()
        # Verify client is closed (should not raise)

    async def test_request_body_format(
        self,
        provider: GeminiProvider,
        mocker: pytest.MockerFixture,
    ) -> None:
        """Test that request body is formatted correctly for Gemini API."""
        mock_post = mocker.patch.object(
            provider._client,
            "post",
            new_callable=AsyncMock,
        )
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "embeddings": [{"values": [0.1] * 768}, {"values": [0.2] * 768}],
        }
        mock_post.return_value = mock_response

        texts = ["Text 1", "Text 2"]
        await provider.embed_texts(texts)

        # Verify request body structure
        call_args = mock_post.call_args
        request_body = call_args.kwargs["json"]
        assert "requests" in request_body
        assert len(request_body["requests"]) == 2
        assert request_body["requests"][0] == {"content": {"parts": [{"text": "Text 1"}]}}
        assert request_body["requests"][1] == {"content": {"parts": [{"text": "Text 2"}]}}
