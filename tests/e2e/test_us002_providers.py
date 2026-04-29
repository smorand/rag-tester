"""E2E tests for US-002: Local Embeddings + ChromaDB Foundation.

These tests validate the complete integration of:
- LocalEmbeddingProvider with sentence-transformers
- ChromaDBProvider in HTTP and persistent modes
- File I/O utilities for YAML and JSON
"""

import json
import shutil
from pathlib import Path

import pytest
import yaml

from rag_tester.providers.databases.chromadb import ChromaDBProvider
from rag_tester.providers.embeddings.local import LocalEmbeddingProvider
from rag_tester.utils.file_io import ValidationError, read_json, read_yaml


@pytest.fixture
def test_data_yaml(tmp_path: Path) -> Path:
    """Create test data YAML file with 100 records."""
    file_path = tmp_path / "test_data.yaml"
    records = [{"id": f"doc{i}", "text": f"Document {i} content about various topics"} for i in range(1, 101)]
    # Add special doc42 for query test
    records[41] = {
        "id": "doc42",
        "text": "Machine learning is a subset of artificial intelligence that enables computers to learn from data.",
    }
    data = {"records": records}
    with open(file_path, "w") as f:
        yaml.dump(data, f)
    return file_path


@pytest.fixture
def test_data_json(tmp_path: Path) -> Path:
    """Create test data JSON file with 50 records."""
    file_path = tmp_path / "test_data.json"
    records = [{"id": f"doc{i}", "text": f"Document {i} content about various topics"} for i in range(1, 51)]
    data = {"records": records}
    with open(file_path, "w") as f:
        json.dump(data, f)
    return file_path


@pytest.fixture
def embedding_provider() -> LocalEmbeddingProvider:
    """Create LocalEmbeddingProvider instance."""
    return LocalEmbeddingProvider("sentence-transformers/all-MiniLM-L6-v2", device="cpu")


@pytest.fixture
def chroma_persistent_path(tmp_path: Path) -> Path:
    """Create temporary directory for ChromaDB persistent storage."""
    chroma_path = tmp_path / "chroma_data"
    chroma_path.mkdir(exist_ok=True)
    return chroma_path


@pytest.mark.e2e
@pytest.mark.critical
class TestE2E001InitialDatasetLoad:
    """E2E-001: Initial Dataset Load with Local Embedding Model."""

    async def test_load_100_documents(
        self,
        test_data_yaml: Path,
        embedding_provider: LocalEmbeddingProvider,
        chroma_persistent_path: Path,
    ) -> None:
        """Load 100 documents with local embeddings into ChromaDB."""
        # Setup ChromaDB provider
        connection_string = f"chromadb:///{chroma_persistent_path}/test_collection"
        db_provider = ChromaDBProvider(connection_string)

        try:
            # Read records from YAML
            records = []
            async for record in read_yaml(test_data_yaml):
                records.append(record)

            assert len(records) == 100, "Should read 100 records"

            # Generate embeddings
            texts = [record["text"] for record in records]
            embeddings = await embedding_provider.embed_texts(texts)

            assert len(embeddings) == 100, "Should generate 100 embeddings"
            assert all(len(emb) == 384 for emb in embeddings), "All embeddings should have dimension 384"

            # Prepare records for insertion
            db_records = [
                {
                    "id": record["id"],
                    "text": record["text"],
                    "embedding": embedding,
                }
                for record, embedding in zip(records, embeddings, strict=False)
            ]

            # Insert into ChromaDB
            await db_provider.insert("test_collection", db_records)

            # Verify collection exists and has correct count
            info = await db_provider.get_collection_info("test_collection")
            assert info["name"] == "test_collection"
            assert info["dimension"] == 384
            assert info["count"] == 100

        finally:
            # Cleanup
            await db_provider.delete_collection("test_collection")
            await db_provider.close()


@pytest.mark.e2e
@pytest.mark.critical
class TestE2E002ManualQueryTest:
    """E2E-002: Manual Query Test."""

    async def test_query_returns_relevant_document(
        self,
        test_data_yaml: Path,
        embedding_provider: LocalEmbeddingProvider,
        chroma_persistent_path: Path,
    ) -> None:
        """Query for 'What is machine learning?' should return doc42."""
        # Setup and load data
        connection_string = f"chromadb:///{chroma_persistent_path}/test_collection"
        db_provider = ChromaDBProvider(connection_string)

        try:
            # Load data
            records = []
            async for record in read_yaml(test_data_yaml):
                records.append(record)

            texts = [record["text"] for record in records]
            embeddings = await embedding_provider.embed_texts(texts)

            db_records = [
                {"id": record["id"], "text": record["text"], "embedding": embedding}
                for record, embedding in zip(records, embeddings, strict=False)
            ]

            await db_provider.insert("test_collection", db_records)

            # Query
            query_text = "What is machine learning?"
            query_embedding = await embedding_provider.embed_texts([query_text])

            results = await db_provider.query("test_collection", query_embedding[0], top_k=3)

            # Verify results
            assert len(results) == 3, "Should return 3 results"
            assert results[0]["id"] == "doc42", "First result should be doc42"
            assert results[0]["score"] > 0.7, f"Score should be > 0.7, got {results[0]['score']}"

            # Verify results are sorted by score descending
            scores = [r["score"] for r in results]
            assert scores == sorted(scores, reverse=True), "Results should be sorted by score descending"

        finally:
            await db_provider.delete_collection("test_collection")
            await db_provider.close()


@pytest.mark.e2e
@pytest.mark.critical
class TestE2E025AutoCreateCollection:
    """E2E-025: Auto-Create Collection."""

    async def test_auto_create_collection_on_insert(
        self,
        embedding_provider: LocalEmbeddingProvider,
        chroma_persistent_path: Path,
    ) -> None:
        """Collection should be auto-created when inserting records."""
        connection_string = f"chromadb:///{chroma_persistent_path}/new_collection"
        db_provider = ChromaDBProvider(connection_string)

        try:
            # Verify collection doesn't exist
            exists_before = await db_provider.collection_exists("new_collection")
            assert not exists_before, "Collection should not exist initially"

            # Insert records (should auto-create collection)
            texts = [f"Document {i}" for i in range(10)]
            embeddings = await embedding_provider.embed_texts(texts)

            records = [
                {"id": f"doc{i}", "text": text, "embedding": embedding}
                for i, (text, embedding) in enumerate(zip(texts, embeddings, strict=False))
            ]

            await db_provider.insert("new_collection", records)

            # Verify collection was created
            exists_after = await db_provider.collection_exists("new_collection")
            assert exists_after, "Collection should exist after insert"

            info = await db_provider.get_collection_info("new_collection")
            assert info["dimension"] == 384
            assert info["count"] == 10

        finally:
            await db_provider.delete_collection("new_collection")
            await db_provider.close()


@pytest.mark.e2e
@pytest.mark.critical
class TestE2E069ChromaDBHTTPMode:
    """E2E-069: ChromaDB HTTP Mode."""

    @pytest.mark.skip(reason="Requires ChromaDB HTTP server running at localhost:8000")
    async def test_http_mode_operations(
        self,
        embedding_provider: LocalEmbeddingProvider,
    ) -> None:
        """Test ChromaDB HTTP mode operations."""
        connection_string = "chromadb://localhost:8000/http_test_collection"
        db_provider = ChromaDBProvider(connection_string)

        try:
            # Verify HTTP mode
            assert db_provider._mode == "http"
            assert db_provider._host == "localhost"
            assert db_provider._port == 8000

            # Test insert and query
            texts = ["Test document 1", "Test document 2"]
            embeddings = await embedding_provider.embed_texts(texts)

            records = [
                {"id": f"doc{i}", "text": text, "embedding": embedding}
                for i, (text, embedding) in enumerate(zip(texts, embeddings, strict=False))
            ]

            await db_provider.insert("http_test_collection", records)

            # Query
            query_embedding = embeddings[0]
            results = await db_provider.query("http_test_collection", query_embedding, top_k=2)

            assert len(results) == 2

        finally:
            await db_provider.delete_collection("http_test_collection")
            await db_provider.close()


@pytest.mark.e2e
@pytest.mark.high
class TestE2E070ChromaDBPersistentMode:
    """E2E-070: ChromaDB Persistent Mode."""

    async def test_persistent_mode_survives_restart(
        self,
        embedding_provider: LocalEmbeddingProvider,
        chroma_persistent_path: Path,
    ) -> None:
        """Test that data survives provider restart in persistent mode."""
        connection_string = f"chromadb:///{chroma_persistent_path}/persistent_collection"

        # First provider instance - insert data
        db_provider1 = ChromaDBProvider(connection_string)
        try:
            assert db_provider1._mode == "persistent"

            texts = ["Persistent document 1", "Persistent document 2"]
            embeddings = await embedding_provider.embed_texts(texts)

            records = [
                {"id": f"doc{i}", "text": text, "embedding": embedding}
                for i, (text, embedding) in enumerate(zip(texts, embeddings, strict=False))
            ]

            await db_provider1.insert("persistent_collection", records)

            # Verify data was inserted
            info1 = await db_provider1.get_collection_info("persistent_collection")
            assert info1["count"] == 2

        finally:
            await db_provider1.close()

        # Second provider instance - verify data persisted
        db_provider2 = ChromaDBProvider(connection_string)
        try:
            # Query the data
            query_embedding = embeddings[0]
            results = await db_provider2.query("persistent_collection", query_embedding, top_k=2)

            assert len(results) == 2, "Data should persist across provider restarts"
            assert results[0]["text"] in texts

        finally:
            await db_provider2.delete_collection("persistent_collection")
            await db_provider2.close()


@pytest.mark.e2e
@pytest.mark.high
class TestE2E077LoadThenTest:
    """E2E-077: Load Then Test (cross-scenario)."""

    async def test_load_and_query_workflow(
        self,
        test_data_yaml: Path,
        embedding_provider: LocalEmbeddingProvider,
        chroma_persistent_path: Path,
    ) -> None:
        """Test complete workflow: load data then query it."""
        connection_string = f"chromadb:///{chroma_persistent_path}/integration_test"
        db_provider = ChromaDBProvider(connection_string)

        try:
            # Load data
            records = []
            async for record in read_yaml(test_data_yaml):
                records.append(record)

            texts = [record["text"] for record in records]
            embeddings = await embedding_provider.embed_texts(texts)

            db_records = [
                {"id": record["id"], "text": record["text"], "embedding": embedding}
                for record, embedding in zip(records, embeddings, strict=False)
            ]

            await db_provider.insert("integration_test", db_records)

            # Find doc5 (should contain "Document 5")
            doc5_text = next(r["text"] for r in records if r["id"] == "doc5")

            # Query for similar content
            query_embedding = await embedding_provider.embed_texts([doc5_text])
            results = await db_provider.query("integration_test", query_embedding[0], top_k=5)

            # Verify doc5 is in results with high similarity
            doc5_result = next((r for r in results if r["id"] == "doc5"), None)
            assert doc5_result is not None, "doc5 should be in results"
            assert doc5_result["score"] > 0.9, "doc5 should have very high similarity to itself"

        finally:
            await db_provider.delete_collection("integration_test")
            await db_provider.close()


@pytest.mark.e2e
@pytest.mark.high
class TestE2EPROV001EmptyInputFile:
    """E2E-PROV-001: Empty Input File."""

    async def test_empty_yaml_raises_validation_error(self, tmp_path: Path) -> None:
        """Empty YAML file should raise ValidationError."""
        empty_file = tmp_path / "empty.yaml"
        empty_file.write_text("")

        with pytest.raises(ValidationError, match="Input file is empty or has no records"):
            async for _ in read_yaml(empty_file):
                pass


@pytest.mark.e2e
@pytest.mark.critical
class TestE2EPROV002MissingRequiredFields:
    """E2E-PROV-002: Missing Required Fields."""

    async def test_missing_text_field_raises_validation_error(self, tmp_path: Path) -> None:
        """Record missing 'text' field should raise ValidationError."""
        invalid_file = tmp_path / "invalid.yaml"
        data = {"records": [{"id": "doc1"}]}  # Missing 'text'
        with open(invalid_file, "w") as f:
            yaml.dump(data, f)

        with pytest.raises(ValidationError, match="Missing required field 'text' in record 'doc1'"):
            async for _ in read_yaml(invalid_file):
                pass


@pytest.mark.e2e
@pytest.mark.critical
class TestE2EPROV003DimensionMismatch:
    """E2E-PROV-003: Dimension Mismatch."""

    async def test_dimension_mismatch_raises_error(
        self,
        embedding_provider: LocalEmbeddingProvider,
        chroma_persistent_path: Path,
    ) -> None:
        """Inserting records with wrong dimension should raise DimensionMismatchError."""
        from rag_tester.providers.databases.base import DimensionMismatchError

        connection_string = f"chromadb:///{chroma_persistent_path}/existing_collection"
        db_provider = ChromaDBProvider(connection_string)

        try:
            # Create collection with dimension 768
            await db_provider.create_collection("existing_collection", dimension=768)

            # Try to insert records with dimension 384
            texts = ["Test document"]
            embeddings = await embedding_provider.embed_texts(texts)  # Returns 384-dim embeddings

            records = [{"id": "doc1", "text": texts[0], "embedding": embeddings[0]}]

            with pytest.raises(DimensionMismatchError, match="Dimension mismatch: model=384, database=768"):
                await db_provider.insert("existing_collection", records)

        finally:
            await db_provider.delete_collection("existing_collection")
            await db_provider.close()


@pytest.mark.e2e
@pytest.mark.critical
class TestE2EPROV004DatabaseUnreachable:
    """E2E-PROV-004: Database Unreachable."""

    @pytest.mark.skip(reason="Retry logic makes this test slow (5 attempts with backoff)")
    async def test_connection_error_with_retry(self) -> None:
        """Connection to unreachable database should fail after retries."""
        from rag_tester.providers.databases.base import ConnectionError as DBConnectionError

        # Use wrong port
        connection_string = "chromadb://localhost:9999/test_collection"
        db_provider = ChromaDBProvider(connection_string)

        try:
            # This should fail after retry attempts
            with pytest.raises((DBConnectionError, Exception)):
                await db_provider.insert("test_collection", [{"id": "doc1", "text": "test", "embedding": [0.1] * 384}])
        finally:
            await db_provider.close()


@pytest.mark.e2e
@pytest.mark.critical
class TestE2EPROV005ModelLoadingError:
    """E2E-PROV-005: Model Loading Error."""

    def test_invalid_model_name_raises_error(self) -> None:
        """Invalid model name should raise ModelLoadError."""
        from rag_tester.providers.embeddings.base import ModelLoadError

        with pytest.raises(ModelLoadError, match="Failed to load model: invalid/model-name"):
            LocalEmbeddingProvider("invalid/model-name", device="cpu")


@pytest.mark.e2e
@pytest.mark.medium
class TestE2EPROV006SingleRecordLoad:
    """E2E-PROV-006: Single Record Load."""

    async def test_single_record_workflow(
        self,
        embedding_provider: LocalEmbeddingProvider,
        chroma_persistent_path: Path,
        tmp_path: Path,
    ) -> None:
        """Test loading and querying a single record."""
        # Create single record file
        single_file = tmp_path / "single.yaml"
        data = {"records": [{"id": "doc1", "text": "Test"}]}
        with open(single_file, "w") as f:
            yaml.dump(data, f)

        connection_string = f"chromadb:///{chroma_persistent_path}/single_collection"
        db_provider = ChromaDBProvider(connection_string)

        try:
            # Load single record
            records = []
            async for record in read_yaml(single_file):
                records.append(record)

            assert len(records) == 1

            embeddings = await embedding_provider.embed_texts([records[0]["text"]])
            db_records = [{"id": records[0]["id"], "text": records[0]["text"], "embedding": embeddings[0]}]

            await db_provider.insert("single_collection", db_records)

            # Verify
            info = await db_provider.get_collection_info("single_collection")
            assert info["count"] == 1

            # Query
            results = await db_provider.query("single_collection", embeddings[0], top_k=1)
            assert len(results) == 1
            assert results[0]["id"] == "doc1"

        finally:
            await db_provider.delete_collection("single_collection")
            await db_provider.close()


@pytest.mark.e2e
@pytest.mark.medium
class TestE2EPROV007VeryLongText:
    """E2E-PROV-007: Very Long Text (10K chars)."""

    async def test_long_text_handling(
        self,
        embedding_provider: LocalEmbeddingProvider,
        chroma_persistent_path: Path,
    ) -> None:
        """Test handling of very long text (10K characters)."""
        connection_string = f"chromadb:///{chroma_persistent_path}/long_text_collection"
        db_provider = ChromaDBProvider(connection_string)

        try:
            # Create 10K character text
            long_text = "a" * 10000

            # Generate embedding
            embeddings = await embedding_provider.embed_texts([long_text])
            assert len(embeddings[0]) == 384

            # Insert
            records = [{"id": "doc1", "text": long_text, "embedding": embeddings[0]}]
            await db_provider.insert("long_text_collection", records)

            # Query and verify text round-trips correctly
            results = await db_provider.query("long_text_collection", embeddings[0], top_k=1)
            assert len(results) == 1
            assert results[0]["text"] == long_text
            assert len(results[0]["text"]) == 10000

        finally:
            await db_provider.delete_collection("long_text_collection")
            await db_provider.close()


@pytest.mark.e2e
@pytest.mark.high
class TestE2EPROV008UnicodeAndEmoji:
    """E2E-PROV-008: Unicode and Emoji."""

    async def test_unicode_and_emoji_handling(
        self,
        embedding_provider: LocalEmbeddingProvider,
        chroma_persistent_path: Path,
    ) -> None:
        """Test handling of Unicode and emoji characters."""
        connection_string = f"chromadb:///{chroma_persistent_path}/unicode_collection"
        db_provider = ChromaDBProvider(connection_string)

        try:
            # Text with Unicode and emoji
            unicode_text = "Hello 世界 🌍 مرحبا"

            # Generate embedding
            embeddings = await embedding_provider.embed_texts([unicode_text])
            assert len(embeddings[0]) == 384

            # Insert
            records = [{"id": "doc1", "text": unicode_text, "embedding": embeddings[0]}]
            await db_provider.insert("unicode_collection", records)

            # Query and verify exact match
            results = await db_provider.query("unicode_collection", embeddings[0], top_k=1)
            assert len(results) == 1
            assert results[0]["text"] == unicode_text

            # Query with similar text
            similar_text = "Hello world 🌍"
            similar_embedding = await embedding_provider.embed_texts([similar_text])
            results2 = await db_provider.query("unicode_collection", similar_embedding[0], top_k=1)
            assert len(results2) == 1
            assert results2[0]["id"] == "doc1"

        finally:
            await db_provider.delete_collection("unicode_collection")
            await db_provider.close()


@pytest.mark.e2e
@pytest.mark.high
class TestE2EPROV009JSONFileFormat:
    """E2E-PROV-009: JSON File Format."""

    async def test_json_file_format(
        self,
        test_data_json: Path,
        embedding_provider: LocalEmbeddingProvider,
        chroma_persistent_path: Path,
    ) -> None:
        """Test reading JSON file format and loading into database."""
        connection_string = f"chromadb:///{chroma_persistent_path}/json_collection"
        db_provider = ChromaDBProvider(connection_string)

        try:
            # Read JSON file
            records = []
            async for record in read_json(test_data_json):
                records.append(record)

            assert len(records) == 50

            # Generate embeddings and insert
            texts = [record["text"] for record in records]
            embeddings = await embedding_provider.embed_texts(texts)

            db_records = [
                {"id": record["id"], "text": record["text"], "embedding": embedding}
                for record, embedding in zip(records, embeddings, strict=False)
            ]

            await db_provider.insert("json_collection", db_records)

            # Verify
            info = await db_provider.get_collection_info("json_collection")
            assert info["count"] == 50

        finally:
            await db_provider.delete_collection("json_collection")
            await db_provider.close()


@pytest.mark.e2e
@pytest.mark.medium
class TestE2EPROV010PersistentStorageDirectoryCreation:
    """E2E-PROV-010: Persistent Storage Directory Creation."""

    async def test_directory_auto_creation(
        self,
        embedding_provider: LocalEmbeddingProvider,
        tmp_path: Path,
    ) -> None:
        """Test that persistent storage directory is auto-created."""
        # Use non-existent directory
        new_dir = tmp_path / "new_chroma_dir"
        assert not new_dir.exists(), "Directory should not exist initially"

        connection_string = f"chromadb:///{new_dir}/collection"
        db_provider = ChromaDBProvider(connection_string)

        try:
            # Directory should be created during initialization
            assert new_dir.exists(), "Directory should be auto-created"

            # Insert data to verify it works
            texts = ["Test document"]
            embeddings = await embedding_provider.embed_texts(texts)
            records = [{"id": "doc1", "text": texts[0], "embedding": embeddings[0]}]

            await db_provider.insert("collection", records)

            # Verify
            info = await db_provider.get_collection_info("collection")
            assert info["count"] == 1

        finally:
            await db_provider.delete_collection("collection")
            await db_provider.close()
            # Cleanup
            if new_dir.exists():
                shutil.rmtree(new_dir)
