"""E2E tests for US-008: API Embedding Providers (OpenRouter, Gemini).

These tests validate the complete integration of:
- OpenRouterProvider with API-based embeddings
- GeminiProvider with API-based embeddings
- Rate limiting and retry logic
- Token counting and cost tracking
- Error handling for authentication and API failures
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from rag_tester.providers.embeddings.gemini import (
    AuthenticationError as GeminiAuthenticationError,
)
from rag_tester.providers.embeddings.gemini import (
    GeminiProvider,
)
from rag_tester.providers.embeddings.gemini import (
    MissingAPIKeyError as GeminiMissingAPIKeyError,
)
from rag_tester.providers.embeddings.openrouter import (
    AuthenticationError as OpenRouterAuthenticationError,
)
from rag_tester.providers.embeddings.openrouter import (
    MissingAPIKeyError as OpenRouterMissingAPIKeyError,
)
from rag_tester.providers.embeddings.openrouter import (
    OpenRouterProvider,
    RateLimitError,
)


@pytest.mark.e2e
@pytest.mark.critical
class TestE2E008OpenRouterEmbedding:
    """E2E-008: OpenRouter Embedding."""

    @pytest.fixture
    def mock_api_key(self, monkeypatch: pytest.MonkeyPatch) -> str:
        """Set up mock API key."""
        api_key = "test-openrouter-key"
        monkeypatch.setenv("OPENROUTER_API_KEY", api_key)
        return api_key

    async def test_openrouter_load_workflow(
        self,
        mock_api_key: str,
        tmp_path,
        mocker: pytest.MockerFixture,
    ) -> None:
        """Test loading 100 documents with OpenRouter embeddings."""
        provider = OpenRouterProvider("openai/text-embedding-3-small")

        # Mock API responses
        mock_post = mocker.patch.object(
            provider._client,
            "post",
            new_callable=AsyncMock,
        )

        def create_batch_response(batch_size: int, tokens: int) -> MagicMock:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "data": [{"embedding": [0.1] * 1536} for _ in range(batch_size)],
                "usage": {"total_tokens": tokens},
            }
            return mock_response

        # For 100 texts, single batch
        mock_post.return_value = create_batch_response(100, 15000)

        try:
            # Generate embeddings
            texts = [f"Document {i} content" for i in range(100)]
            embeddings = await provider.embed_texts(texts)

            # Verify embeddings
            assert len(embeddings) == 100
            assert all(len(emb) == 1536 for emb in embeddings)

            # Verify token counting
            assert provider.get_total_tokens() == 15000

            # Verify cost calculation (15000 tokens * $0.02 / 1M = $0.0003)
            assert provider.get_total_cost() == 0.0003

        finally:
            await provider.close()


@pytest.mark.e2e
@pytest.mark.critical
class TestE2E009GeminiEmbedding:
    """E2E-009: Google Gemini Embedding."""

    @pytest.fixture
    def mock_api_key(self, monkeypatch: pytest.MonkeyPatch) -> str:
        """Set up mock API key."""
        api_key = "test-gemini-key"
        monkeypatch.setenv("GEMINI_API_KEY", api_key)
        return api_key

    async def test_gemini_load_workflow(
        self,
        mock_api_key: str,
        tmp_path,
        mocker: pytest.MockerFixture,
    ) -> None:
        """Test loading 100 documents with Gemini embeddings."""
        provider = GeminiProvider("models/text-embedding-004")

        # Mock API responses
        mock_post = mocker.patch.object(
            provider._client,
            "post",
            new_callable=AsyncMock,
        )

        def create_batch_response(batch_size: int) -> MagicMock:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "embeddings": [{"values": [0.1] * 768} for _ in range(batch_size)],
            }
            return mock_response

        # For 100 texts, single batch
        mock_post.return_value = create_batch_response(100)

        try:
            # Generate embeddings
            texts = [f"Document {i} content" for i in range(100)]
            embeddings = await provider.embed_texts(texts)

            # Verify embeddings
            assert len(embeddings) == 100
            assert all(len(emb) == 768 for emb in embeddings)

            # Verify token estimation (100 texts * ~20 chars * 1/4 = ~500 tokens)
            assert provider.get_total_tokens() > 0

        finally:
            await provider.close()


@pytest.mark.e2e
@pytest.mark.high
class TestE2E038RateLimitExceeded:
    """E2E-038: Rate Limit Exceeded."""

    @pytest.fixture
    def mock_api_key(self, monkeypatch: pytest.MonkeyPatch) -> str:
        """Set up mock API key."""
        api_key = "test-key"
        monkeypatch.setenv("OPENROUTER_API_KEY", api_key)
        return api_key

    async def test_rate_limit_retry_success(
        self,
        mock_api_key: str,
        mocker: pytest.MockerFixture,
    ) -> None:
        """Test that rate limit is retried and succeeds."""
        provider = OpenRouterProvider("openai/text-embedding-3-small")

        mock_post = mocker.patch.object(
            provider._client,
            "post",
            new_callable=AsyncMock,
        )

        # First 2 calls return 429, third succeeds
        mock_response_429 = MagicMock()
        mock_response_429.status_code = 429
        mock_response_429.headers = {"Retry-After": "1"}

        mock_response_success = MagicMock()
        mock_response_success.status_code = 200
        mock_response_success.json.return_value = {
            "data": [{"embedding": [0.1] * 1536}],
            "usage": {"total_tokens": 50},
        }

        mock_post.side_effect = [
            mock_response_429,
            mock_response_429,
            mock_response_success,
        ]

        try:
            # Should succeed after retries
            embeddings = await provider.embed_texts(["Test"])

            assert len(embeddings) == 1
            assert mock_post.call_count == 3

        finally:
            await provider.close()


@pytest.mark.e2e
@pytest.mark.critical
class TestE2E051APIKeyValidationOpenRouter:
    """E2E-051: API Key Validation (OpenRouter)."""

    async def test_invalid_api_key(
        self,
        monkeypatch: pytest.MonkeyPatch,
        mocker: pytest.MockerFixture,
    ) -> None:
        """Test that invalid API key raises AuthenticationError."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "invalid-key")
        provider = OpenRouterProvider("openai/text-embedding-3-small")

        mock_post = mocker.patch.object(
            provider._client,
            "post",
            new_callable=AsyncMock,
        )
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_post.return_value = mock_response

        try:
            with pytest.raises(
                OpenRouterAuthenticationError,
                match="Authentication failed",
            ):
                await provider.embed_texts(["Test"])
        finally:
            await provider.close()


@pytest.mark.e2e
@pytest.mark.critical
class TestE2E052APIKeyValidationGemini:
    """E2E-052: API Key Validation (Gemini)."""

    async def test_invalid_api_key(
        self,
        monkeypatch: pytest.MonkeyPatch,
        mocker: pytest.MockerFixture,
    ) -> None:
        """Test that invalid API key raises AuthenticationError."""
        monkeypatch.setenv("GEMINI_API_KEY", "invalid-key")
        provider = GeminiProvider("models/text-embedding-004")

        mock_post = mocker.patch.object(
            provider._client,
            "post",
            new_callable=AsyncMock,
        )
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_post.return_value = mock_response

        try:
            with pytest.raises(
                GeminiAuthenticationError,
                match="Authentication failed",
            ):
                await provider.embed_texts(["Test"])
        finally:
            await provider.close()


@pytest.mark.e2e
@pytest.mark.critical
class TestE2EAPI001MissingAPIKeyOpenRouter:
    """E2E-API-001: Missing API Key (OpenRouter)."""

    async def test_missing_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that missing API key raises MissingAPIKeyError."""
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

        with pytest.raises(
            OpenRouterMissingAPIKeyError,
            match="Missing API key: OPENROUTER_API_KEY",
        ):
            OpenRouterProvider("openai/text-embedding-3-small")


@pytest.mark.e2e
@pytest.mark.critical
class TestE2EAPI002MissingAPIKeyGemini:
    """E2E-API-002: Missing API Key (Gemini)."""

    async def test_missing_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that missing API key raises MissingAPIKeyError."""
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)

        with pytest.raises(
            GeminiMissingAPIKeyError,
            match="Missing API key: GEMINI_API_KEY",
        ):
            GeminiProvider("models/text-embedding-004")


@pytest.mark.e2e
@pytest.mark.high
class TestE2EAPI003RateLimitExhausted:
    """E2E-API-003: Rate Limit Exhausted."""

    @pytest.fixture
    def mock_api_key(self, monkeypatch: pytest.MonkeyPatch) -> str:
        """Set up mock API key."""
        api_key = "test-key"
        monkeypatch.setenv("OPENROUTER_API_KEY", api_key)
        return api_key

    async def test_rate_limit_exhausted(
        self,
        mock_api_key: str,
        mocker: pytest.MockerFixture,
    ) -> None:
        """Test that persistent rate limiting fails after max retries."""
        provider = OpenRouterProvider("openai/text-embedding-3-small")

        mock_post = mocker.patch.object(
            provider._client,
            "post",
            new_callable=AsyncMock,
        )

        # Always return 429
        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.headers = {"Retry-After": "1"}
        mock_post.return_value = mock_response

        try:
            with pytest.raises(RateLimitError, match="Rate limit exceeded"):
                await provider.embed_texts(["Test"])

            # Should have tried 5 times (max retries)
            assert mock_post.call_count == 5

        finally:
            await provider.close()


@pytest.mark.e2e
@pytest.mark.high
class TestE2EAPI005APIServerError:
    """E2E-API-005: API Server Error (5xx)."""

    @pytest.fixture
    def mock_api_key(self, monkeypatch: pytest.MonkeyPatch) -> str:
        """Set up mock API key."""
        api_key = "test-key"
        monkeypatch.setenv("OPENROUTER_API_KEY", api_key)
        return api_key

    async def test_server_error_retry_success(
        self,
        mock_api_key: str,
        mocker: pytest.MockerFixture,
    ) -> None:
        """Test that 5xx errors are retried and succeed."""
        provider = OpenRouterProvider("openai/text-embedding-3-small")

        mock_post = mocker.patch.object(
            provider._client,
            "post",
            new_callable=AsyncMock,
        )

        # First 2 calls return 500, third succeeds
        mock_response_500 = MagicMock()
        mock_response_500.status_code = 500

        mock_response_success = MagicMock()
        mock_response_success.status_code = 200
        mock_response_success.json.return_value = {
            "data": [{"embedding": [0.1] * 1536}],
            "usage": {"total_tokens": 50},
        }

        mock_post.side_effect = [
            mock_response_500,
            mock_response_500,
            mock_response_success,
        ]

        try:
            embeddings = await provider.embed_texts(["Test"])

            assert len(embeddings) == 1
            assert mock_post.call_count == 3

        finally:
            await provider.close()


@pytest.mark.e2e
@pytest.mark.medium
class TestE2EAPI006BatchSizeLimitOpenRouter:
    """E2E-API-006: Batch Size Limit (OpenRouter)."""

    @pytest.fixture
    def mock_api_key(self, monkeypatch: pytest.MonkeyPatch) -> str:
        """Set up mock API key."""
        api_key = "test-key"
        monkeypatch.setenv("OPENROUTER_API_KEY", api_key)
        return api_key

    async def test_large_batch_splitting(
        self,
        mock_api_key: str,
        mocker: pytest.MockerFixture,
    ) -> None:
        """Test that large batches are split correctly (max 2048 per call)."""
        provider = OpenRouterProvider("openai/text-embedding-3-small")

        mock_post = mocker.patch.object(
            provider._client,
            "post",
            new_callable=AsyncMock,
        )

        def create_response(batch_size: int) -> MagicMock:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "data": [{"embedding": [0.1] * 1536} for _ in range(batch_size)],
                "usage": {"total_tokens": batch_size * 10},
            }
            return mock_response

        # 3000 texts should be split into 2048 + 952
        mock_post.side_effect = [create_response(2048), create_response(952)]

        try:
            texts = [f"Text {i}" for i in range(3000)]
            embeddings = await provider.embed_texts(texts)

            assert len(embeddings) == 3000
            assert mock_post.call_count == 2

            # Verify batch sizes
            first_call = mock_post.call_args_list[0]
            second_call = mock_post.call_args_list[1]
            assert len(first_call.kwargs["json"]["input"]) == 2048
            assert len(second_call.kwargs["json"]["input"]) == 952

        finally:
            await provider.close()


@pytest.mark.e2e
@pytest.mark.medium
class TestE2EAPI007BatchSizeLimitGemini:
    """E2E-API-007: Batch Size Limit (Gemini)."""

    @pytest.fixture
    def mock_api_key(self, monkeypatch: pytest.MonkeyPatch) -> str:
        """Set up mock API key."""
        api_key = "test-key"
        monkeypatch.setenv("GEMINI_API_KEY", api_key)
        return api_key

    async def test_large_batch_splitting(
        self,
        mock_api_key: str,
        mocker: pytest.MockerFixture,
    ) -> None:
        """Test that large batches are split correctly (max 100 per call)."""
        provider = GeminiProvider("models/text-embedding-004")

        mock_post = mocker.patch.object(
            provider._client,
            "post",
            new_callable=AsyncMock,
        )

        def create_response(batch_size: int) -> MagicMock:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "embeddings": [{"values": [0.1] * 768} for _ in range(batch_size)],
            }
            return mock_response

        # 250 texts should be split into 100 + 100 + 50
        mock_post.side_effect = [
            create_response(100),
            create_response(100),
            create_response(50),
        ]

        try:
            texts = [f"Text {i}" for i in range(250)]
            embeddings = await provider.embed_texts(texts)

            assert len(embeddings) == 250
            assert mock_post.call_count == 3

            # Verify batch sizes
            calls = mock_post.call_args_list
            assert len(calls[0].kwargs["json"]["requests"]) == 100
            assert len(calls[1].kwargs["json"]["requests"]) == 100
            assert len(calls[2].kwargs["json"]["requests"]) == 50

        finally:
            await provider.close()


@pytest.mark.e2e
@pytest.mark.medium
class TestE2EAPI008CostCalculationAccuracy:
    """E2E-API-008: Cost Calculation Accuracy."""

    @pytest.fixture
    def mock_api_key(self, monkeypatch: pytest.MonkeyPatch) -> str:
        """Set up mock API key."""
        api_key = "test-key"
        monkeypatch.setenv("OPENROUTER_API_KEY", api_key)
        return api_key

    async def test_cost_calculation(
        self,
        mock_api_key: str,
        mocker: pytest.MockerFixture,
    ) -> None:
        """Test that cost is calculated accurately."""
        provider = OpenRouterProvider("openai/text-embedding-3-small")

        mock_post = mocker.patch.object(
            provider._client,
            "post",
            new_callable=AsyncMock,
        )

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [{"embedding": [0.1] * 1536}],
            "usage": {"total_tokens": 15000},
        }
        mock_post.return_value = mock_response

        try:
            await provider.embed_texts(["Test"])

            # For openai/text-embedding-3-small: $0.02 per 1M tokens
            # 15000 tokens = 15000 / 1,000,000 * 0.02 = $0.0003
            assert provider.get_total_cost() == 0.0003

        finally:
            await provider.close()


@pytest.mark.e2e
@pytest.mark.low
class TestE2EAPI009TokenEstimationGemini:
    """E2E-API-009: Token Estimation (Gemini)."""

    @pytest.fixture
    def mock_api_key(self, monkeypatch: pytest.MonkeyPatch) -> str:
        """Set up mock API key."""
        api_key = "test-key"
        monkeypatch.setenv("GEMINI_API_KEY", api_key)
        return api_key

    async def test_token_estimation(
        self,
        mock_api_key: str,
        mocker: pytest.MockerFixture,
    ) -> None:
        """Test that tokens are estimated correctly for Gemini."""
        provider = GeminiProvider("models/text-embedding-004")

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

        try:
            # Text with 1000 characters should estimate ~250 tokens (1000 / 4)
            text = "a" * 1000
            await provider.embed_texts([text])

            assert provider.get_total_tokens() == 250

        finally:
            await provider.close()


@pytest.mark.e2e
@pytest.mark.medium
class TestE2E065LoadLatencyAPIModel:
    """E2E-065: Load Latency (API Model)."""

    @pytest.fixture
    def mock_api_key(self, monkeypatch: pytest.MonkeyPatch) -> str:
        """Set up mock API key."""
        api_key = "test-key"
        monkeypatch.setenv("OPENROUTER_API_KEY", api_key)
        return api_key

    async def test_load_performance(
        self,
        mock_api_key: str,
        mocker: pytest.MockerFixture,
    ) -> None:
        """Test that loading 100 documents completes in reasonable time."""
        import time

        provider = OpenRouterProvider("openai/text-embedding-3-small")

        mock_post = mocker.patch.object(
            provider._client,
            "post",
            new_callable=AsyncMock,
        )

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [{"embedding": [0.1] * 1536} for _ in range(100)],
            "usage": {"total_tokens": 15000},
        }
        mock_post.return_value = mock_response

        try:
            texts = [f"Document {i}" for i in range(100)]

            start_time = time.time()
            embeddings = await provider.embed_texts(texts)
            elapsed_time = time.time() - start_time

            assert len(embeddings) == 100
            # With mocked API, should be very fast (< 1 second)
            assert elapsed_time < 1.0

        finally:
            await provider.close()


@pytest.mark.e2e
@pytest.mark.critical
@pytest.mark.skip(reason="Requires real API keys - run manually for integration testing")
class TestE2E075OpenRouterAPIIntegration:
    """E2E-075: OpenRouter API Integration (requires real API key)."""

    async def test_real_api_integration(self) -> None:
        """Test real OpenRouter API integration."""
        # This test requires OPENROUTER_API_KEY to be set
        provider = OpenRouterProvider("openai/text-embedding-3-small")

        try:
            texts = ["Machine learning", "Artificial intelligence"]
            embeddings = await provider.embed_texts(texts)

            assert len(embeddings) == 2
            assert all(len(emb) == 1536 for emb in embeddings)
            assert provider.get_total_tokens() > 0
            assert provider.get_total_cost() > 0

        finally:
            await provider.close()


@pytest.mark.e2e
@pytest.mark.critical
@pytest.mark.skip(reason="Requires real API keys - run manually for integration testing")
class TestE2E076GeminiAPIIntegration:
    """E2E-076: Google Gemini API Integration (requires real API key)."""

    async def test_real_api_integration(self) -> None:
        """Test real Gemini API integration."""
        # This test requires GEMINI_API_KEY to be set
        provider = GeminiProvider("models/text-embedding-004")

        try:
            texts = ["Machine learning", "Artificial intelligence"]
            embeddings = await provider.embed_texts(texts)

            assert len(embeddings) == 2
            assert all(len(emb) == 768 for emb in embeddings)
            assert provider.get_total_tokens() > 0

        finally:
            await provider.close()
