"""Unit tests for the Tester class."""

import json

import pytest

from rag_tester.core.tester import Tester, TestError, ValidationError
from rag_tester.providers.databases.base import DatabaseError
from rag_tester.providers.embeddings.base import EmbeddingError


@pytest.fixture
def mock_embedding_provider(mocker):
    """Create a mock embedding provider."""
    provider = mocker.AsyncMock()
    provider.embed_texts.return_value = [[0.1] * 384]  # 384-dim embedding
    provider.get_dimension.return_value = 384
    provider.get_model_name.return_value = "test-model"
    return provider


@pytest.fixture
def mock_database(mocker):
    """Create a mock database provider."""
    db = mocker.AsyncMock()
    db.collection_exists.return_value = True
    db.get_collection_info.return_value = {
        "name": "test_collection",
        "dimension": 384,
        "count": 100,
        "metadata": {},
    }
    db.query.return_value = [
        {"id": "doc1", "text": "First result", "score": 0.95},
        {"id": "doc2", "text": "Second result", "score": 0.87},
        {"id": "doc3", "text": "Third result", "score": 0.82},
    ]
    return db


@pytest.fixture
def tester(mock_database, mock_embedding_provider):
    """Create a Tester instance with mocked dependencies."""
    return Tester(
        database=mock_database,
        embedding_provider=mock_embedding_provider,
        collection_name="test_collection",
    )


class TestTesterValidation:
    """Test input validation methods."""

    @pytest.mark.asyncio
    async def test_validate_query_empty_string(self, tester):
        """Test that empty query string raises ValidationError."""
        with pytest.raises(ValidationError, match="Query cannot be empty"):
            await tester.test_query("", top_k=5)

    @pytest.mark.asyncio
    async def test_validate_query_whitespace_only(self, tester):
        """Test that whitespace-only query raises ValidationError."""
        with pytest.raises(ValidationError, match="Query cannot be empty"):
            await tester.test_query("   ", top_k=5)

    @pytest.mark.asyncio
    async def test_validate_top_k_zero(self, tester):
        """Test that top_k=0 raises ValidationError."""
        with pytest.raises(ValidationError, match="Top-K must be between 1 and 100"):
            await tester.test_query("test query", top_k=0)

    @pytest.mark.asyncio
    async def test_validate_top_k_negative(self, tester):
        """Test that negative top_k raises ValidationError."""
        with pytest.raises(ValidationError, match="Top-K must be between 1 and 100"):
            await tester.test_query("test query", top_k=-1)

    @pytest.mark.asyncio
    async def test_validate_top_k_too_large(self, tester):
        """Test that top_k > 100 raises ValidationError."""
        with pytest.raises(ValidationError, match="Top-K must be between 1 and 100"):
            await tester.test_query("test query", top_k=101)

    @pytest.mark.asyncio
    async def test_validate_format_invalid(self, tester):
        """Test that invalid format raises ValidationError."""
        with pytest.raises(ValidationError, match="Invalid format"):
            await tester.test_query("test query", top_k=5, output_format="xml")

    def test_validate_format_valid_table(self, tester):
        """Test that 'table' format is valid."""
        # Should not raise
        tester._validate_format("table")

    def test_validate_format_valid_json(self, tester):
        """Test that 'json' format is valid."""
        # Should not raise
        tester._validate_format("json")

    def test_validate_format_valid_text(self, tester):
        """Test that 'text' format is valid."""
        # Should not raise
        tester._validate_format("text")


class TestTesterQuery:
    """Test query execution."""

    @pytest.mark.asyncio
    async def test_query_success(self, tester, mock_database, mock_embedding_provider):
        """Test successful query execution."""
        result = await tester.test_query("test query", top_k=3)

        # Verify result structure
        assert result["query"] == "test query"
        assert len(result["results"]) == 3
        assert result["tokens"] == 0
        assert result["time"] > 0

        # Verify results are properly formatted
        assert result["results"][0]["rank"] == 1
        assert result["results"][0]["id"] == "doc1"
        assert result["results"][0]["text"] == "First result"
        assert result["results"][0]["score"] == 0.95

        # Verify provider calls
        mock_embedding_provider.embed_texts.assert_called_once_with(["test query"])
        mock_database.query.assert_called_once()

    @pytest.mark.asyncio
    async def test_query_collection_not_exists(self, tester, mock_database):
        """Test query when collection doesn't exist."""
        mock_database.collection_exists.return_value = False

        with pytest.raises(TestError, match="Collection 'test_collection' does not exist"):
            await tester.test_query("test query", top_k=5)

    @pytest.mark.asyncio
    async def test_query_empty_database(self, tester, mock_database):
        """Test query when database is empty."""
        mock_database.get_collection_info.return_value = {
            "name": "test_collection",
            "dimension": 384,
            "count": 0,
            "metadata": {},
        }

        with pytest.raises(TestError, match="Database is empty"):
            await tester.test_query("test query", top_k=5)

    @pytest.mark.asyncio
    async def test_query_dimension_mismatch(self, tester, mock_database, mock_embedding_provider):
        """Test query when embedding dimension doesn't match database."""
        mock_database.get_collection_info.return_value = {
            "name": "test_collection",
            "dimension": 768,  # Different from embedding (384)
            "count": 100,
            "metadata": {},
        }

        with pytest.raises(TestError, match="Dimension mismatch: model=384, database=768"):
            await tester.test_query("test query", top_k=5)

    @pytest.mark.asyncio
    async def test_query_database_error(self, tester, mock_database):
        """Test query when database operation fails."""
        mock_database.query.side_effect = DatabaseError("Connection failed")

        with pytest.raises(TestError, match="Database operation failed"):
            await tester.test_query("test query", top_k=5)

    @pytest.mark.asyncio
    async def test_query_embedding_error(self, tester, mock_embedding_provider):
        """Test query when embedding generation fails."""
        mock_embedding_provider.embed_texts.side_effect = EmbeddingError("Model failed")

        with pytest.raises(TestError, match="Embedding generation failed"):
            await tester.test_query("test query", top_k=5)

    @pytest.mark.asyncio
    async def test_query_top_k_exceeds_collection_size(self, tester, mock_database):
        """Test query when top_k exceeds collection size."""
        mock_database.get_collection_info.return_value = {
            "name": "test_collection",
            "dimension": 384,
            "count": 2,  # Only 2 documents
            "metadata": {},
        }
        mock_database.query.return_value = [
            {"id": "doc1", "text": "First result", "score": 0.95},
            {"id": "doc2", "text": "Second result", "score": 0.87},
        ]

        result = await tester.test_query("test query", top_k=10)

        # Should return all available documents
        assert len(result["results"]) == 2


class TestTesterFormatting:
    """Test result formatting."""

    @pytest.fixture
    def sample_result(self):
        """Create sample result data."""
        return {
            "query": "test query",
            "results": [
                {"rank": 1, "id": "doc1", "text": "First result", "score": 0.95},
                {"rank": 2, "id": "doc2", "text": "Second result", "score": 0.87},
            ],
            "tokens": 0,
            "time": 0.123,
        }

    def test_format_table(self, tester, sample_result):
        """Test table formatting."""
        output = tester.format_results(sample_result, "table")

        # Verify output contains expected elements
        assert "test query" in output
        assert "doc1" in output
        assert "doc2" in output
        assert "First result" in output
        assert "Second result" in output
        assert "0.9500" in output
        assert "0.8700" in output
        assert "Tokens: 0" in output
        assert "Time: 0.12s" in output

    def test_format_json(self, tester, sample_result):
        """Test JSON formatting."""
        output = tester.format_results(sample_result, "json")

        # Verify output is valid JSON
        parsed = json.loads(output)
        assert parsed["query"] == "test query"
        assert len(parsed["results"]) == 2
        assert parsed["results"][0]["id"] == "doc1"
        assert parsed["tokens"] == 0
        assert parsed["time"] == 0.123

    def test_format_text(self, tester, sample_result):
        """Test text formatting."""
        output = tester.format_results(sample_result, "text")

        # Verify output contains expected elements
        assert "Query: test query" in output
        assert "1. [doc1] (score: 0.95)" in output
        assert "First result" in output
        assert "2. [doc2] (score: 0.87)" in output
        assert "Second result" in output
        assert "Tokens: 0" in output
        assert "Time: 0.12s" in output

    def test_format_table_truncates_long_text(self, tester):
        """Test that table format truncates text longer than 80 chars."""
        long_text = "A" * 100
        result = {
            "query": "test",
            "results": [
                {"rank": 1, "id": "doc1", "text": long_text, "score": 0.95},
            ],
            "tokens": 0,
            "time": 0.1,
        }

        output = tester.format_results(result, "table")

        # Should contain truncated text with ellipsis
        assert "A" * 77 + "..." in output
        assert "A" * 100 not in output

    def test_format_invalid_format(self, tester, sample_result):
        """Test that invalid format raises ValidationError."""
        with pytest.raises(ValidationError, match="Invalid format"):
            tester.format_results(sample_result, "xml")
