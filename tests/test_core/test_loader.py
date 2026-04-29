"""Tests for core.loader module."""

import pytest
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from rag_tester.core.loader import (
    LoadStatistics,
    generate_embeddings_batch,
    load_records,
    stream_records,
)
from rag_tester.core.validator import ValidationError
from rag_tester.providers.databases.base import (
    DatabaseError,
    DimensionMismatchError,
    VectorDatabase,
)
from rag_tester.providers.embeddings.base import EmbeddingError, EmbeddingProvider


class MockEmbeddingProvider(EmbeddingProvider):
    """Mock embedding provider for testing."""

    def __init__(self, dimension: int = 384, model_name: str = "test-model") -> None:
        """Initialize mock provider."""
        self.dimension = dimension
        self.model_name = model_name
        self.embed_calls: list[list[str]] = []

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Generate mock embeddings."""
        self.embed_calls.append(texts)
        return [[0.1] * self.dimension for _ in texts]

    def get_dimension(self) -> int:
        """Return embedding dimension."""
        return self.dimension

    def get_model_name(self) -> str:
        """Return model name."""
        return self.model_name


class MockVectorDatabase(VectorDatabase):
    """Mock vector database for testing."""

    def __init__(self) -> None:
        """Initialize mock database."""
        self.collections: dict[str, dict[str, Any]] = {}
        self.records: dict[str, list[dict[str, Any]]] = {}

    async def create_collection(
        self, name: str, dimension: int, metadata: dict[str, Any] | None = None
    ) -> None:
        """Create mock collection."""
        self.collections[name] = {
            "name": name,
            "dimension": dimension,
            "count": 0,
            "metadata": metadata or {},
        }
        self.records[name] = []

    async def collection_exists(self, name: str) -> bool:
        """Check if collection exists."""
        return name in self.collections

    async def insert(self, collection: str, records: list[dict[str, Any]]) -> None:
        """Insert records into collection."""
        if collection not in self.collections:
            raise DatabaseError(f"Collection not found: {collection}")
        self.records[collection].extend(records)
        self.collections[collection]["count"] = len(self.records[collection])

    async def query(
        self,
        collection: str,
        query_embedding: list[float],
        top_k: int = 5,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Mock query implementation."""
        return []

    async def delete_collection(self, name: str) -> None:
        """Delete collection."""
        if name in self.collections:
            del self.collections[name]
            del self.records[name]

    async def get_collection_info(self, name: str) -> dict[str, Any]:
        """Get collection info."""
        if name not in self.collections:
            raise DatabaseError(f"Collection not found: {name}")
        return self.collections[name]


class TestLoadStatistics:
    """Tests for LoadStatistics class."""

    def test_initialization(self) -> None:
        """Test statistics initialization."""
        stats = LoadStatistics()
        assert stats.total_records == 0
        assert stats.loaded_records == 0
        assert stats.failed_records == 0
        assert stats.skipped_records == 0
        assert stats.total_tokens == 0
        assert stats.embedding_model == ""
        assert stats.database == ""

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        stats = LoadStatistics()
        stats.total_records = 100
        stats.loaded_records = 98
        stats.failed_records = 1
        stats.skipped_records = 1
        stats.embedding_model = "test-model"
        stats.database = "test-collection"

        result = stats.to_dict()
        assert result["total_records"] == 100
        assert result["loaded_records"] == 98
        assert result["failed_records"] == 1
        assert result["skipped_records"] == 1
        assert result["embedding_model"] == "test-model"
        assert result["database"] == "test-collection"


class TestStreamRecords:
    """Tests for stream_records function."""

    @pytest.mark.asyncio
    async def test_stream_yaml_file(self, tmp_path: Path) -> None:
        """Test streaming records from YAML file."""
        yaml_content = """
- id: doc1
  text: "First document"
- id: doc2
  text: "Second document"
"""
        file_path = tmp_path / "test.yaml"
        file_path.write_text(yaml_content)

        records = []
        async for record in stream_records(file_path):
            records.append(record)

        assert len(records) == 2
        assert records[0]["id"] == "doc1"
        assert records[0]["text"] == "First document"
        assert records[1]["id"] == "doc2"
        assert records[1]["text"] == "Second document"

    @pytest.mark.asyncio
    async def test_stream_json_file(self, tmp_path: Path) -> None:
        """Test streaming records from JSON file."""
        json_content = """[
  {"id": "doc1", "text": "First document"},
  {"id": "doc2", "text": "Second document"}
]"""
        file_path = tmp_path / "test.json"
        file_path.write_text(json_content)

        records = []
        async for record in stream_records(file_path):
            records.append(record)

        assert len(records) == 2
        assert records[0]["id"] == "doc1"
        assert records[1]["id"] == "doc2"

    @pytest.mark.asyncio
    async def test_empty_yaml_file(self, tmp_path: Path) -> None:
        """Test streaming from empty YAML file."""
        file_path = tmp_path / "empty.yaml"
        file_path.write_text("")

        with pytest.raises(ValidationError, match="Input file is empty"):
            async for _ in stream_records(file_path):
                pass

    @pytest.mark.asyncio
    async def test_invalid_yaml_format(self, tmp_path: Path) -> None:
        """Test streaming from invalid YAML file."""
        file_path = tmp_path / "invalid.yaml"
        file_path.write_text("invalid: yaml: content:")

        with pytest.raises(ValidationError, match="Invalid file format"):
            async for _ in stream_records(file_path):
                pass

    @pytest.mark.asyncio
    async def test_yaml_not_list(self, tmp_path: Path) -> None:
        """Test streaming from YAML file that's not a list."""
        file_path = tmp_path / "notlist.yaml"
        file_path.write_text("key: value")

        with pytest.raises(ValidationError, match="must contain a list"):
            async for _ in stream_records(file_path):
                pass


class TestGenerateEmbeddingsBatch:
    """Tests for generate_embeddings_batch function."""

    @pytest.mark.asyncio
    async def test_single_batch(self) -> None:
        """Test embedding generation with single batch."""
        provider = MockEmbeddingProvider(dimension=384)
        texts = ["text1", "text2", "text3"]

        embeddings = await generate_embeddings_batch(texts, provider, batch_size=10)

        assert len(embeddings) == 3
        assert all(len(emb) == 384 for emb in embeddings)
        assert len(provider.embed_calls) == 1
        assert provider.embed_calls[0] == texts

    @pytest.mark.asyncio
    async def test_multiple_batches(self) -> None:
        """Test embedding generation with multiple batches."""
        provider = MockEmbeddingProvider(dimension=384)
        texts = [f"text{i}" for i in range(5)]

        embeddings = await generate_embeddings_batch(texts, provider, batch_size=2)

        assert len(embeddings) == 5
        assert len(provider.embed_calls) == 3  # 2+2+1
        assert provider.embed_calls[0] == ["text0", "text1"]
        assert provider.embed_calls[1] == ["text2", "text3"]
        assert provider.embed_calls[2] == ["text4"]

    @pytest.mark.asyncio
    async def test_exact_batch_size(self) -> None:
        """Test embedding generation with exact batch size."""
        provider = MockEmbeddingProvider(dimension=384)
        texts = [f"text{i}" for i in range(4)]

        embeddings = await generate_embeddings_batch(texts, provider, batch_size=2)

        assert len(embeddings) == 4
        assert len(provider.embed_calls) == 2
        assert provider.embed_calls[0] == ["text0", "text1"]
        assert provider.embed_calls[1] == ["text2", "text3"]


class TestLoadRecords:
    """Tests for load_records function."""

    @pytest.mark.asyncio
    async def test_load_new_collection(self, tmp_path: Path) -> None:
        """Test loading records into new collection."""
        # Create test file
        yaml_content = """
- id: doc1
  text: "First document"
- id: doc2
  text: "Second document"
"""
        file_path = tmp_path / "test.yaml"
        file_path.write_text(yaml_content)

        # Setup mocks
        database = MockVectorDatabase()
        provider = MockEmbeddingProvider()

        # Load records
        stats = await load_records(
            file_path=file_path,
            database=database,
            embedding_provider=provider,
            collection_name="test_collection",
            batch_size=32,
            parallel=1,
        )

        # Verify statistics
        assert stats.total_records == 2
        assert stats.loaded_records == 2
        assert stats.failed_records == 0
        assert stats.skipped_records == 0

        # Verify collection created
        assert await database.collection_exists("test_collection")
        info = await database.get_collection_info("test_collection")
        assert info["dimension"] == 384
        assert info["count"] == 2

        # Verify records inserted
        records = database.records["test_collection"]
        assert len(records) == 2
        assert records[0]["id"] == "doc1"
        assert records[1]["id"] == "doc2"

    @pytest.mark.asyncio
    async def test_load_existing_collection(self, tmp_path: Path) -> None:
        """Test loading records into existing collection."""
        # Create test file
        yaml_content = """
- id: doc3
  text: "Third document"
"""
        file_path = tmp_path / "test.yaml"
        file_path.write_text(yaml_content)

        # Setup mocks with existing collection
        database = MockVectorDatabase()
        provider = MockEmbeddingProvider()
        await database.create_collection("test_collection", dimension=384)

        # Load records
        stats = await load_records(
            file_path=file_path,
            database=database,
            embedding_provider=provider,
            collection_name="test_collection",
            batch_size=32,
            parallel=1,
        )

        # Verify statistics
        assert stats.total_records == 1
        assert stats.loaded_records == 1

        # Verify records inserted
        info = await database.get_collection_info("test_collection")
        assert info["count"] == 1

    @pytest.mark.asyncio
    async def test_dimension_mismatch(self, tmp_path: Path) -> None:
        """Test loading with dimension mismatch."""
        # Create test file
        yaml_content = """
- id: doc1
  text: "First document"
"""
        file_path = tmp_path / "test.yaml"
        file_path.write_text(yaml_content)

        # Setup mocks with wrong dimension
        database = MockVectorDatabase()
        provider = MockEmbeddingProvider(dimension=384)
        await database.create_collection("test_collection", dimension=768)

        # Load should fail
        with pytest.raises(DimensionMismatchError, match="Dimension mismatch"):
            await load_records(
                file_path=file_path,
                database=database,
                embedding_provider=provider,
                collection_name="test_collection",
                batch_size=32,
                parallel=1,
            )

    @pytest.mark.asyncio
    async def test_duplicate_ids(self, tmp_path: Path) -> None:
        """Test loading with duplicate IDs."""
        # Create test file with duplicates
        yaml_content = """
- id: doc1
  text: "First document"
- id: doc2
  text: "Second document"
- id: doc1
  text: "Duplicate document"
"""
        file_path = tmp_path / "test.yaml"
        file_path.write_text(yaml_content)

        # Setup mocks
        database = MockVectorDatabase()
        provider = MockEmbeddingProvider()

        # Load records
        stats = await load_records(
            file_path=file_path,
            database=database,
            embedding_provider=provider,
            collection_name="test_collection",
            batch_size=32,
            parallel=1,
        )

        # Verify statistics
        assert stats.total_records == 3
        assert stats.loaded_records == 2
        assert stats.skipped_records == 1

        # Verify only first occurrence loaded
        records = database.records["test_collection"]
        assert len(records) == 2
        assert records[0]["id"] == "doc1"
        assert records[0]["text"] == "First document"
        assert records[1]["id"] == "doc2"

    @pytest.mark.asyncio
    async def test_invalid_records(self, tmp_path: Path) -> None:
        """Test loading with invalid records."""
        # Create test file with invalid record
        yaml_content = """
- id: doc1
  text: "Valid document"
- id: doc2
"""
        file_path = tmp_path / "test.yaml"
        file_path.write_text(yaml_content)

        # Setup mocks
        database = MockVectorDatabase()
        provider = MockEmbeddingProvider()

        # Load records
        stats = await load_records(
            file_path=file_path,
            database=database,
            embedding_provider=provider,
            collection_name="test_collection",
            batch_size=32,
            parallel=1,
        )

        # Verify statistics
        assert stats.total_records == 2
        assert stats.loaded_records == 1
        assert stats.failed_records == 1

    @pytest.mark.asyncio
    async def test_empty_file(self, tmp_path: Path) -> None:
        """Test loading from empty file."""
        file_path = tmp_path / "empty.yaml"
        file_path.write_text("")

        database = MockVectorDatabase()
        provider = MockEmbeddingProvider()

        with pytest.raises(ValidationError, match="Input file is empty"):
            await load_records(
                file_path=file_path,
                database=database,
                embedding_provider=provider,
                collection_name="test_collection",
                batch_size=32,
                parallel=1,
            )

    @pytest.mark.asyncio
    async def test_records_with_metadata(self, tmp_path: Path) -> None:
        """Test loading records with metadata."""
        yaml_content = """
- id: doc1
  text: "Document with metadata"
  metadata:
    category: "test"
    priority: 1
"""
        file_path = tmp_path / "test.yaml"
        file_path.write_text(yaml_content)

        database = MockVectorDatabase()
        provider = MockEmbeddingProvider()

        stats = await load_records(
            file_path=file_path,
            database=database,
            embedding_provider=provider,
            collection_name="test_collection",
            batch_size=32,
            parallel=1,
        )

        assert stats.loaded_records == 1
        records = database.records["test_collection"]
        assert "metadata" in records[0]
        assert records[0]["metadata"]["category"] == "test"
        assert records[0]["metadata"]["priority"] == 1
