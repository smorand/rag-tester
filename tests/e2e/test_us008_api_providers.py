"""E2E tests for US-008: API Embedding Providers (OpenRouter, Gemini).

These tests validate the complete integration of:
- OpenRouterProvider with OpenRouter API
- GeminiProvider with Google Gemini API
- Rate limiting and retry logic
- Cost tracking and token counting
- Error handling (authentication, timeouts, server errors)
"""

import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
import yaml

from rag_tester.providers.databases.chromadb import ChromaDBProvider
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
)
from rag_tester.utils.file_io import read_yaml


@pytest.fixture
def test_data_100(tmp_path: Path) -> Path:
    """Create test data YAML file with 100 records."""
    file_path = tmp_path / "test_data_100.yaml"
    records = [{"id": f"doc{i}", "text": f"Document {i} content about various topics"} for i in range(1, 101)]
    data = {"records": records}
    with open(file_path, "w") as f:
        yaml.dump(data, f)
    return file_path


@pytest.fixture
def test_data_large(tmp_path: Path) -> Path:
    """Create test data YAML file with 3000 records for batch testing."""
    file_path = tmp_path / "test_data_large.yaml"
    records = [{"id": f"doc{i}", "text": f"Document {i} content"} for i in range(1, 3001)]
    data = {"records": records}
    with open(file_path, "w") as f:
        yaml.dump(data, f)
    return file_path


@pytest.fixture
def chroma_persistent_path(tmp_path: Path) -> Path:
    """Create temporary directory for ChromaDB persistent storage."""
    chroma_path = tmp_path / "chroma_data"
    chroma_path.mkdir(exist_ok=True)
    return chroma_path


@pytest.mark.e2e
@pytest.mark.critical
@pytest.mark.skipif(
    not os.getenv("OPENROUTER_API_KEY"),
    reason="OPENROUTER_API_KEY not set - skipping real API test",
)
class TestE2E008OpenRouterEmbedding:
    """E2E-008: OpenRouter Embedding."""

    async def test_openrouter_load_100_documents(
        self,
        test_data_100: Path,
        chroma_persistent_path: Path,
    ) -> None:
        """Load 100 documents with OpenRouter embeddings into ChromaDB."""
        # Setup providers
        embedding_provider = OpenRouterProvider("openai/text-embedding-3-small")
        connection_string = f"chromadb:///{chroma_persistent_path}/openrouter_test"
        db_provider = ChromaDBProvider(connection_string)

        try:
            # Read records
            records = []
            async for record in read_yaml(test_data_100):
                records.append(record)

            assert len(records) == 100

            # Generate embeddings
            texts = [record["text"] for record in records]
            embeddings = await embedding_provider.embed_texts(texts)

            assert len(embeddings) == 100
            assert all(len(emb) == 1536 for emb in embeddings)

            # Verify token counting
            total_tokens = embedding_provider.get_total_tokens()
            assert total_tokens > 0, "Should have consumed tokens"

            # Verify cost calculation
            total_cost = embedding_provider.get_total_cost()
            assert total_cost > 0, "Should have calculated cost"

            # Insert into ChromaDB
            db_records = [
                {"id": record["id"], "text": record["text"], "embedding": embedding}
                for record, embedding in zip(records, embeddings, strict=False)
            ]

            await db_provider.insert("openrouter_test", db_records)

            # Verify collection
            info = await db_provider.get_collection_info("openrouter_test")
            assert info["dimension"] == 1536
            assert info["count"] == 100

        finally:
            await db_provider.delete_collection("openrouter_test")
            await db_provider.close()


@pytest.mark.e2e
@pytest.mark.critical
@pytest.mark.skipif(
    not os.getenv("GEMINI_API_KEY"),
    reason="GEMINI_API_KEY not set - skipping real API test",
)
class TestE2E009GeminiEmbedding:
    """E2E-009: Google Gemini Embedding."""

    async def test_gemini_load_100_documents(
        self,
        test_data_100: Path,
        chroma_persistent_path: Path,
    ) -> None:
        """Load 100 documents with Gemini embeddings into ChromaDB."""
        # Setup providers
        embedding_provider = GeminiProvider("models/text-embedding-004")
        connection_string = f"chromadb:///{chroma_persistent_path}/gemini_test"
        db_provider = ChromaDBProvider(connection_string)

        try:
            # Read records
            records = []
            async for record in read_yaml(test_data_100):
                records.append(record)

            assert len(records) == 100

            # Generate embeddings
            texts = [record["text"] for record in records]
            embeddings = await embedding_provider.embed_texts(texts)

            assert len(embeddings) == 100
            assert all(len(emb) == 768 for emb in embeddings)

            # Verify token estimation
            total_tokens = embedding_provider.get_total_tokens()
            assert total_tokens > 0, "Should have estimated tokens"

            # Insert into ChromaDB
            db_records = [
                {"id": record["id"], "text": record["text"], "embedding": embedding}
                for record, embedding in zip(records, embeddings, strict=False)
            ]

            await db_provider.insert("gemini_test", db_records)

            # Verify collection
            info = await db_provider.get_collection_info("gemini_test")
            assert info["dimension"] == 768
            assert info["count"] == 100

        finally:
            await db_provider.delete_collection("gemini_test")
            await db_provider.close()


@pytest.mark.e2e
@pytest.mark.high
class TestE2E038RateLimitExceeded:
    """E2E-038: Rate Limit Exceeded."""

    async def test_rate_limit_retry_success(self) -> None:
        """Test that rate limit errors are retried and eventually succeed."""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client

            # First two calls return 429, third succeeds
            mock_response_429_1 = MagicMock()
            mock_response_429_1.status_code = 429
            mock_response_429_1.headers = {"Retry-After": "1"}

            mock_response_429_2 = MagicMock()
            mock_response_429_2.status_code = 429
            mock_response_429_2.headers = {"Retry-After": "1"}

            mock_response_success = MagicMock()
            mock_response_success.status_code = 200
            mock_response_success.json.return_value = {
                "data": [{"embedding": [0.1] * 1536}],
                "usage": {"total_tokens": 50},
            }

            mock_client.post.side_effect = [
                mock_response_429_1,
                mock_response_429_2,
                mock_response_success,
            ]

            # Create provider with mock API key
            provider = OpenRouterProvider("openai/text-embedding-3-small", api_key="test-key")

            # Should succeed after retries
            embeddings = await provider.embed_texts(["test text"])

            assert len(embeddings) == 1
            assert mock_client.post.call_count == 3


@pytest.mark.e2e
@pytest.mark.critical
class TestE2E051APIKeyValidationOpenRouter:
    """E2E-051: API Key Validation (OpenRouter)."""

    async def test_invalid_api_key_raises_authentication_error(self) -> None:
        """Test that invalid API key raises AuthenticationError."""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client

            mock_response = MagicMock()
            mock_response.status_code = 401
            mock_client.post.return_value = mock_response

            provider = OpenRouterProvider("openai/text-embedding-3-small", api_key="invalid-key")

            with pytest.raises(OpenRouterAuthenticationError, match="Authentication failed"):
                await provider.embed_texts(["test text"])


@pytest.mark.e2e
@pytest.mark.critical
class TestE2E052APIKeyValidationGemini:
    """E2E-052: API Key Validation (Gemini)."""

    async def test_invalid_api_key_raises_authentication_error(self) -> None:
        """Test that invalid API key raises AuthenticationError."""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client

            mock_response = MagicMock()
            mock_response.status_code = 401
            mock_client.post.return_value = mock_response

            provider = GeminiProvider("models/text-embedding-004", api_key="invalid-key")

            with pytest.raises(GeminiAuthenticationError, match="Authentication failed"):
                await provider.embed_texts(["test text"])


@pytest.mark.e2e
@pytest.mark.critical
class TestE2EAPI001MissingAPIKeyOpenRouter:
    """E2E-API-001: Missing API Key (OpenRouter)."""

    def test_missing_api_key_raises_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
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

    def test_missing_api_key_raises_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
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

    async def test_rate_limit_exhausted_raises_error(self) -> None:
        """Test that persistent rate limiting raises RateLimitError after max retries."""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client

            # Always return 429
            mock_response = MagicMock()
            mock_response.status_code = 429
            mock_response.headers = {"Retry-After": "1"}
            mock_client.post.return_value = mock_response

            provider = OpenRouterProvider("openai/text-embedding-3-small", api_key="test-key")

            with pytest.raises(Exception, match="Max retry attempts"):
                await provider.embed_texts(["test text"])


@pytest.mark.e2e
@pytest.mark.high
class TestE2EAPI004APITimeout:
    """E2E-API-004: API Timeout."""

    async def test_timeout_raises_error(self) -> None:
        """Test that API timeout raises EmbeddingError."""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client
            mock_client.post.side_effect = httpx.TimeoutException("Timeout")

            provider = OpenRouterProvider("openai/text-embedding-3-small", api_key="test-key")

            with pytest.raises(Exception, match="timeout"):
                await provider.embed_texts(["test text"])


@pytest.mark.e2e
@pytest.mark.high
class TestE2EAPI005APIServerError:
    """E2E-API-005: API Server Error (5xx)."""

    async def test_server_error_retry_success(self) -> None:
        """Test that 5xx errors are retried and eventually succeed."""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client

            # First two calls return 500, third succeeds
            mock_response_500_1 = MagicMock()
            mock_response_500_1.status_code = 500

            mock_response_500_2 = MagicMock()
            mock_response_500_2.status_code = 500

            mock_response_success = MagicMock()
            mock_response_success.status_code = 200
            mock_response_success.json.return_value = {
                "data": [{"embedding": [0.1] * 1536}],
                "usage": {"total_tokens": 50},
            }

            mock_client.post.side_effect = [
                mock_response_500_1,
                mock_response_500_2,
                mock_response_success,
            ]

            provider = OpenRouterProvider("openai/text-embedding-3-small", api_key="test-key")

            embeddings = await provider.embed_texts(["test text"])

            assert len(embeddings) == 1
            assert mock_client.post.call_count == 3


@pytest.mark.e2e
@pytest.mark.medium
class TestE2EAPI006BatchSizeLimitOpenRouter:
    """E2E-API-006: Batch Size Limit (OpenRouter)."""

    async def test_large_batch_splitting(self, test_data_large: Path) -> None:
        """Test that batches exceeding 2048 are split correctly."""
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

            provider = OpenRouterProvider("openai/text-embedding-3-small", api_key="test-key")

            # Read 3000 records
            records = []
            async for record in read_yaml(test_data_large):
                records.append(record)

            texts = [record["text"] for record in records]
            embeddings = await provider.embed_texts(texts)

            assert len(embeddings) == 3000
            assert mock_client.post.call_count == 2

            # Verify batch sizes
            call1_batch = mock_client.post.call_args_list[0][1]["json"]["input"]
            call2_batch = mock_client.post.call_args_list[1][1]["json"]["input"]
            assert len(call1_batch) == 2048
            assert len(call2_batch) == 952


@pytest.mark.e2e
@pytest.mark.medium
class TestE2EAPI007BatchSizeLimitGemini:
    """E2E-API-007: Batch Size Limit (Gemini)."""

    async def test_large_batch_splitting(self, tmp_path: Path) -> None:
        """Test that batches exceeding 100 are split correctly."""
        # Create 250 record file
        file_path = tmp_path / "test_data_250.yaml"
        records = [{"id": f"doc{i}", "text": f"Document {i}"} for i in range(1, 251)]
        data = {"records": records}
        with open(file_path, "w") as f:
            yaml.dump(data, f)

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

            provider = GeminiProvider("models/text-embedding-004", api_key="test-key")

            # Read 250 records
            records_list = []
            async for record in read_yaml(file_path):
                records_list.append(record)

            texts = [record["text"] for record in records_list]
            embeddings = await provider.embed_texts(texts)

            assert len(embeddings) == 250
            assert mock_client.post.call_count == 3

            # Verify batch sizes
            call1_payload = mock_client.post.call_args_list[0][1]["json"]
            call2_payload = mock_client.post.call_args_list[1][1]["json"]
            call3_payload = mock_client.post.call_args_list[2][1]["json"]
            assert len(call1_payload["requests"]) == 100
            assert len(call2_payload["requests"]) == 100
            assert len(call3_payload["requests"]) == 50


@pytest.mark.e2e
@pytest.mark.medium
class TestE2EAPI008CostCalculationAccuracy:
    """E2E-API-008: Cost Calculation Accuracy."""

    async def test_cost_calculation(self) -> None:
        """Test that cost is calculated accurately."""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client

            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "data": [{"embedding": [0.1] * 1536}],
                "usage": {"total_tokens": 15000},
            }
            mock_client.post.return_value = mock_response

            provider = OpenRouterProvider("openai/text-embedding-3-small", api_key="test-key")

            await provider.embed_texts(["test text"])

            # 15000 tokens * $0.02 per 1M tokens = $0.0003
            expected_cost = (15000 / 1_000_000) * 0.02
            actual_cost = provider.get_total_cost()

            assert actual_cost == pytest.approx(expected_cost, abs=1e-6)


@pytest.mark.e2e
@pytest.mark.low
class TestE2EAPI009TokenEstimationGemini:
    """E2E-API-009: Token Estimation (Gemini)."""

    async def test_token_estimation(self) -> None:
        """Test that Gemini token estimation is reasonable."""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client

            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "embeddings": [{"values": [0.1] * 768}],
            }
            mock_client.post.return_value = mock_response

            provider = GeminiProvider("models/text-embedding-004", api_key="test-key")

            # Text with 1000 characters
            text = "a" * 1000
            await provider.embed_texts([text])

            # Estimated tokens should be ~250 (1000 / 4)
            estimated_tokens = provider.get_total_tokens()
            assert 200 <= estimated_tokens <= 300, f"Expected ~250 tokens, got {estimated_tokens}"


@pytest.mark.e2e
@pytest.mark.medium
@pytest.mark.skipif(
    not os.getenv("OPENROUTER_API_KEY"),
    reason="OPENROUTER_API_KEY not set - skipping performance test",
)
class TestE2E065LoadLatencyAPIModel:
    """E2E-065: Load Latency (API Model)."""

    async def test_load_latency_under_60_seconds(
        self,
        test_data_100: Path,
        chroma_persistent_path: Path,
    ) -> None:
        """Test that loading 100 documents completes within 60 seconds."""
        import time

        start_time = time.time()

        # Setup providers
        embedding_provider = OpenRouterProvider("openai/text-embedding-3-small")
        connection_string = f"chromadb:///{chroma_persistent_path}/perf_test"
        db_provider = ChromaDBProvider(connection_string)

        try:
            # Read records
            records = []
            async for record in read_yaml(test_data_100):
                records.append(record)

            # Generate embeddings (sequential, parallel=1)
            texts = [record["text"] for record in records]
            embeddings = await embedding_provider.embed_texts(texts)

            # Insert into ChromaDB
            db_records = [
                {"id": record["id"], "text": record["text"], "embedding": embedding}
                for record, embedding in zip(records, embeddings, strict=False)
            ]

            await db_provider.insert("perf_test", db_records)

            elapsed_time = time.time() - start_time

            assert elapsed_time < 60, f"Load took {elapsed_time:.2f}s, expected < 60s"

        finally:
            await db_provider.delete_collection("perf_test")
            await db_provider.close()


@pytest.mark.e2e
@pytest.mark.critical
@pytest.mark.skipif(
    not os.getenv("OPENROUTER_API_KEY"),
    reason="OPENROUTER_API_KEY not set - skipping real API integration test",
)
class TestE2E075OpenRouterAPIIntegration:
    """E2E-075: OpenRouter API Integration."""

    async def test_real_api_integration(self) -> None:
        """Test real OpenRouter API integration."""
        provider = OpenRouterProvider("openai/text-embedding-3-small")

        texts = ["Machine learning is a subset of AI", "Python is a programming language"]

        embeddings = await provider.embed_texts(texts)

        # Verify embeddings
        assert len(embeddings) == 2
        assert all(len(emb) == 1536 for emb in embeddings)

        # Verify token counting
        tokens = provider.get_total_tokens()
        assert tokens > 0

        # Verify cost calculation
        cost = provider.get_total_cost()
        assert cost > 0


@pytest.mark.e2e
@pytest.mark.critical
@pytest.mark.skipif(
    not os.getenv("GEMINI_API_KEY"),
    reason="GEMINI_API_KEY not set - skipping real API integration test",
)
class TestE2E076GeminiAPIIntegration:
    """E2E-076: Google Gemini API Integration."""

    async def test_real_api_integration(self) -> None:
        """Test real Gemini API integration."""
        provider = GeminiProvider("models/text-embedding-004")

        texts = ["Machine learning is a subset of AI", "Python is a programming language"]

        embeddings = await provider.embed_texts(texts)

        # Verify embeddings
        assert len(embeddings) == 2
        assert all(len(emb) == 768 for emb in embeddings)

        # Verify token estimation
        tokens = provider.get_total_tokens()
        assert tokens > 0
