"""Tests for OpenRouterProvider."""

from unittest.mock import AsyncMock, MagicMock

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
    def mock_response_success(self) -> dict:
        """Mock successful API response."""
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

    async def test_embed_texts_success(
        self,
        provider: OpenRouterProvider,
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
        assert all(len(emb) == 1536 for emb in embeddings)
        assert embeddings[0] == [0.1] * 1536
        assert embeddings[1] == [0.2] * 1536
        assert embeddings[2] == [0.3] * 1536

        # Verify API call
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert call_args.kwargs["json"]["model"] == "openai/text-embedding-3-small"
        assert call_args.kwargs["json"]["input"] == texts

    async def test_embed_texts_empty_list(self, provider: OpenRouterProvider) -> None:
        """Test embedding empty list returns empty list."""
        embeddings = await provider.embed_texts([])
        assert embeddings == []

    async def test_embed_texts_authentication_error(
        self,
        provider: OpenRouterProvider,
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
        provider: OpenRouterProvider,
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
        provider: OpenRouterProvider,
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
            "data": [{"embedding": [0.1] * 1536}],
            "usage": {"total_tokens": 50},
        }

        mock_post.side_effect = [mock_response_429, mock_response_200]

        embeddings = await provider.embed_texts(["Test"])

        assert len(embeddings) == 1
        assert len(embeddings[0]) == 1536
        assert mock_post.call_count == 2

    async def test_embed_texts_server_error_retry(
        self,
        provider: OpenRouterProvider,
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
            "data": [{"embedding": [0.1] * 1536}],
            "usage": {"total_tokens": 50},
        }

        mock_post.side_effect = [mock_response_500, mock_response_500, mock_response_200]

        embeddings = await provider.embed_texts(["Test"])

        assert len(embeddings) == 1
        assert mock_post.call_count == 3

    async def test_embed_texts_batch_handling(
        self,
        provider: OpenRouterProvider,
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
                "data": [{"embedding": [0.1] * 1536} for _ in range(batch_size)],
                "usage": {"total_tokens": batch_size * 10},
            }
            return mock_response

        # Test with 3000 texts (should be split into 2048 + 952)
        texts = [f"Text {i}" for i in range(3000)]
        mock_post.side_effect = [create_response(2048), create_response(952)]

        embeddings = await provider.embed_texts(texts)

        assert len(embeddings) == 3000
        assert mock_post.call_count == 2

        # Verify batch sizes
        first_call = mock_post.call_args_list[0]
        second_call = mock_post.call_args_list[1]
        assert len(first_call.kwargs["json"]["input"]) == 2048
        assert len(second_call.kwargs["json"]["input"]) == 952

    async def test_token_counting(
        self,
        provider: OpenRouterProvider,
        mocker: pytest.MockerFixture,
    ) -> None:
        """Test that tokens are counted correctly."""
        mock_post = mocker.patch.object(
            provider._client,
            "post",
            new_callable=AsyncMock,
        )
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [{"embedding": [0.1] * 1536}],
            "usage": {"total_tokens": 150},
        }
        mock_post.return_value = mock_response

        await provider.embed_texts(["Test"])

        assert provider.get_total_tokens() == 150

        # Second call should accumulate
        await provider.embed_texts(["Test 2"])
        assert provider.get_total_tokens() == 300

    async def test_cost_calculation(
        self,
        provider: OpenRouterProvider,
        mocker: pytest.MockerFixture,
    ) -> None:
        """Test that cost is calculated correctly."""
        mock_post = mocker.patch.object(
            provider._client,
            "post",
            new_callable=AsyncMock,
        )
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [{"embedding": [0.1] * 1536}],
            "usage": {"total_tokens": 1_000_000},  # 1M tokens
        }
        mock_post.return_value = mock_response

        await provider.embed_texts(["Test"])

        # For openai/text-embedding-3-small: $0.02 per 1M tokens
        assert provider.get_total_cost() == 0.02

    def test_get_dimension(self, provider: OpenRouterProvider) -> None:
        """Test getting embedding dimension."""
        assert provider.get_dimension() == 1536

    def test_get_dimension_unknown_model(self, mock_api_key: str) -> None:
        """Test that unknown model raises ValueError."""
        provider = OpenRouterProvider("unknown/model")
        with pytest.raises(ValueError, match="Unknown dimension for model"):
            provider.get_dimension()

    def test_get_model_name(self, provider: OpenRouterProvider) -> None:
        """Test getting model name."""
        assert provider.get_model_name() == "openai/text-embedding-3-small"

    async def test_context_manager(self, mock_api_key: str) -> None:
        """Test async context manager."""
        async with OpenRouterProvider("openai/text-embedding-3-small") as provider:
            assert provider.get_model_name() == "openai/text-embedding-3-small"

    async def test_close(self, provider: OpenRouterProvider) -> None:
        """Test closing the provider."""
        await provider.close()
        # Verify client is closed (should not raise)
