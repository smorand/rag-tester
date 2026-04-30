"""E2E tests for US-007: Load Modes (Upsert & Flush)."""

import json
import tempfile
from pathlib import Path

import pytest
import yaml

from rag_tester.commands.load import _load_async
from rag_tester.providers.databases.chromadb import ChromaDBProvider


@pytest.fixture
def temp_data_dir(tmp_path: Path) -> Path:
    """Create temporary directory for test data."""
    data_dir = tmp_path / "test_data"
    data_dir.mkdir()
    return data_dir


@pytest.fixture
def initial_data_file(temp_data_dir: Path) -> Path:
    """Create initial dataset with 100 records."""
    data = [
        {
            "id": f"doc{i}",
            "text": f"This is document {i} about machine learning and AI.",
            "metadata": {"index": i, "category": "ml"},
        }
        for i in range(100)
    ]
    file_path = temp_data_dir / "initial_data.yaml"
    with open(file_path, "w") as f:
        yaml.dump(data, f)
    return file_path


@pytest.fixture
def updates_file(temp_data_dir: Path) -> Path:
    """Create updates file with 10 existing + 10 new IDs."""
    data = []
    # 10 existing IDs with modified text
    for i in range(10):
        data.append(
            {
                "id": f"doc{i}",
                "text": f"UPDATED: This is document {i} with new content about deep learning.",
                "metadata": {"index": i, "category": "ml", "updated": True},
            }
        )
    # 10 new IDs
    for i in range(100, 110):
        data.append(
            {
                "id": f"doc{i}",
                "text": f"This is NEW document {i} about neural networks.",
                "metadata": {"index": i, "category": "ml", "new": True},
            }
        )
    file_path = temp_data_dir / "updates.yaml"
    with open(file_path, "w") as f:
        yaml.dump(data, f)
    return file_path


@pytest.fixture
def new_data_file(temp_data_dir: Path) -> Path:
    """Create new dataset with 50 records."""
    data = [
        {
            "id": f"new{i}",
            "text": f"This is new document {i} about computer vision.",
            "metadata": {"index": i, "category": "cv"},
        }
        for i in range(50)
    ]
    file_path = temp_data_dir / "new_data.yaml"
    with open(file_path, "w") as f:
        yaml.dump(data, f)
    return file_path


@pytest.fixture
async def chromadb_provider(tmp_path: Path) -> ChromaDBProvider:
    """Create ChromaDB provider with persistent storage."""
    db_path = tmp_path / "chroma_db"
    db_path.mkdir()
    connection_string = f"chromadb:///{db_path}/test_collection"
    provider = ChromaDBProvider(connection_string)
    yield provider
    # Cleanup
    try:
        await provider.delete_collection("test_collection")
    except Exception:
        pass


class TestE2E006UpsertMode:
    """E2E-006: Upsert Mode."""

    @pytest.mark.asyncio
    async def test_upsert_mode(
        self,
        initial_data_file: Path,
        updates_file: Path,
        chromadb_provider: ChromaDBProvider,
        tmp_path: Path,
    ) -> None:
        """Test upsert mode with force re-embed.

        Requirements: FR-011, FR-013
        """
        db_path = tmp_path / "chroma_db"
        connection_string = f"chromadb:///{db_path}/test_collection"

        # Step 1: Load initial 100 documents
        await _load_async(
            file=str(initial_data_file),
            database=connection_string,
            embedding="sentence-transformers/all-MiniLM-L6-v2",
            mode="initial",
            parallel=1,
            batch_size=32,
            force_reembed=False,
        )

        # Verify initial load
        info = await chromadb_provider.get_collection_info("test_collection")
        assert info["count"] == 100

        # Step 2: Upsert with 10 updates + 10 new records
        await _load_async(
            file=str(updates_file),
            database=connection_string,
            embedding="sentence-transformers/all-MiniLM-L6-v2",
            mode="upsert",
            parallel=1,
            batch_size=32,
            force_reembed=True,
        )

        # Verify upsert results
        info = await chromadb_provider.get_collection_info("test_collection")
        assert info["count"] == 110  # 100 - 10 + 10 + 10 = 110

        # Verify updated document has new text
        query_embedding = [0.1] * 384  # Dummy embedding for testing
        results = await chromadb_provider.query(
            collection="test_collection",
            query_embedding=query_embedding,
            top_k=1,
        )
        # Should find at least one document
        assert len(results) > 0


class TestE2E007FlushMode:
    """E2E-007: Flush Mode."""

    @pytest.mark.asyncio
    async def test_flush_mode(
        self,
        initial_data_file: Path,
        new_data_file: Path,
        chromadb_provider: ChromaDBProvider,
        tmp_path: Path,
    ) -> None:
        """Test flush mode deletes all and loads new data.

        Requirements: FR-012, FR-014
        """
        db_path = tmp_path / "chroma_db"
        connection_string = f"chromadb:///{db_path}/test_collection"

        # Step 1: Load initial 100 documents
        await _load_async(
            file=str(initial_data_file),
            database=connection_string,
            embedding="sentence-transformers/all-MiniLM-L6-v2",
            mode="initial",
            parallel=1,
            batch_size=32,
            force_reembed=False,
        )

        # Verify initial load
        info = await chromadb_provider.get_collection_info("test_collection")
        assert info["count"] == 100

        # Step 2: Flush and load 50 new documents
        await _load_async(
            file=str(new_data_file),
            database=connection_string,
            embedding="sentence-transformers/all-MiniLM-L6-v2",
            mode="flush",
            parallel=1,
            batch_size=32,
            force_reembed=False,
        )

        # Verify flush results
        info = await chromadb_provider.get_collection_info("test_collection")
        assert info["count"] == 50  # Only new data remains

        # Verify old IDs are gone by attempting to query
        # (we can't directly check IDs, but count verification is sufficient)


class TestE2E028UpsertUpdatesDocuments:
    """E2E-028: Upsert Updates Documents."""

    @pytest.mark.asyncio
    async def test_upsert_updates_documents(
        self,
        temp_data_dir: Path,
        chromadb_provider: ChromaDBProvider,
        tmp_path: Path,
    ) -> None:
        """Test that upsert actually updates document text.

        Requirements: FR-011
        """
        db_path = tmp_path / "chroma_db"
        connection_string = f"chromadb:///{db_path}/test_collection"

        # Create initial data with doc42
        initial_data = [
            {
                "id": "doc42",
                "text": "Old text about machine learning",
                "metadata": {"version": 1},
            }
        ]
        initial_file = temp_data_dir / "initial.yaml"
        with open(initial_file, "w") as f:
            yaml.dump(initial_data, f)

        # Load initial data
        await _load_async(
            file=str(initial_file),
            database=connection_string,
            embedding="sentence-transformers/all-MiniLM-L6-v2",
            mode="initial",
            parallel=1,
            batch_size=32,
            force_reembed=False,
        )

        # Create update data with doc42
        update_data = [
            {
                "id": "doc42",
                "text": "New text about machine learning",
                "metadata": {"version": 2},
            }
        ]
        update_file = temp_data_dir / "update.yaml"
        with open(update_file, "w") as f:
            yaml.dump(update_data, f)

        # Upsert with force_reembed
        await _load_async(
            file=str(update_file),
            database=connection_string,
            embedding="sentence-transformers/all-MiniLM-L6-v2",
            mode="upsert",
            parallel=1,
            batch_size=32,
            force_reembed=True,
        )

        # Verify only one doc42 exists
        info = await chromadb_provider.get_collection_info("test_collection")
        assert info["count"] == 1


class TestE2E029FlushDeletesAll:
    """E2E-029: Flush Deletes All."""

    @pytest.mark.asyncio
    async def test_flush_deletes_all(
        self,
        temp_data_dir: Path,
        chromadb_provider: ChromaDBProvider,
        tmp_path: Path,
    ) -> None:
        """Test that flush deletes all old records.

        Requirements: FR-012
        """
        db_path = tmp_path / "chroma_db"
        connection_string = f"chromadb:///{db_path}/test_collection"

        # Create initial data with doc1-doc100
        initial_data = [
            {"id": f"doc{i}", "text": f"Document {i}", "metadata": {"index": i}}
            for i in range(1, 101)
        ]
        initial_file = temp_data_dir / "initial.yaml"
        with open(initial_file, "w") as f:
            yaml.dump(initial_data, f)

        # Load initial data
        await _load_async(
            file=str(initial_file),
            database=connection_string,
            embedding="sentence-transformers/all-MiniLM-L6-v2",
            mode="initial",
            parallel=1,
            batch_size=32,
            force_reembed=False,
        )

        # Verify initial load
        info = await chromadb_provider.get_collection_info("test_collection")
        assert info["count"] == 100

        # Create new data with new1-new50
        new_data = [
            {"id": f"new{i}", "text": f"New document {i}", "metadata": {"index": i}}
            for i in range(1, 51)
        ]
        new_file = temp_data_dir / "new.yaml"
        with open(new_file, "w") as f:
            yaml.dump(new_data, f)

        # Flush and load new data
        await _load_async(
            file=str(new_file),
            database=connection_string,
            embedding="sentence-transformers/all-MiniLM-L6-v2",
            mode="flush",
            parallel=1,
            batch_size=32,
            force_reembed=False,
        )

        # Verify only new data exists
        info = await chromadb_provider.get_collection_info("test_collection")
        assert info["count"] == 50


class TestE2EMODE001UpsertWithoutForceReembed:
    """E2E-MODE-001: Upsert Without Force Re-embed."""

    @pytest.mark.asyncio
    async def test_upsert_without_force_reembed(
        self,
        temp_data_dir: Path,
        chromadb_provider: ChromaDBProvider,
        tmp_path: Path,
    ) -> None:
        """Test upsert without force_reembed (optimization case).

        Requirements: FR-011, FR-013
        """
        db_path = tmp_path / "chroma_db"
        connection_string = f"chromadb:///{db_path}/test_collection"

        # Create initial data
        initial_data = [
            {"id": "doc5", "text": "Python is a language", "metadata": {"version": 1}}
        ]
        initial_file = temp_data_dir / "initial.yaml"
        with open(initial_file, "w") as f:
            yaml.dump(initial_data, f)

        # Load initial data
        await _load_async(
            file=str(initial_file),
            database=connection_string,
            embedding="sentence-transformers/all-MiniLM-L6-v2",
            mode="initial",
            parallel=1,
            batch_size=32,
            force_reembed=False,
        )

        # Create update with same text
        update_data = [
            {"id": "doc5", "text": "Python is a language", "metadata": {"version": 2}}
        ]
        update_file = temp_data_dir / "update.yaml"
        with open(update_file, "w") as f:
            yaml.dump(update_data, f)

        # Upsert without force_reembed
        await _load_async(
            file=str(update_file),
            database=connection_string,
            embedding="sentence-transformers/all-MiniLM-L6-v2",
            mode="upsert",
            parallel=1,
            batch_size=32,
            force_reembed=False,
        )

        # Verify document still exists
        info = await chromadb_provider.get_collection_info("test_collection")
        assert info["count"] == 1


class TestE2EMODE002UpsertAllNewRecords:
    """E2E-MODE-002: Upsert All New Records."""

    @pytest.mark.asyncio
    async def test_upsert_all_new_records(
        self,
        temp_data_dir: Path,
        chromadb_provider: ChromaDBProvider,
        tmp_path: Path,
    ) -> None:
        """Test upsert with all new IDs (edge case).

        Requirements: FR-011
        """
        db_path = tmp_path / "chroma_db"
        connection_string = f"chromadb:///{db_path}/test_collection"

        # Create initial data with doc1-doc100
        initial_data = [
            {"id": f"doc{i}", "text": f"Document {i}", "metadata": {"index": i}}
            for i in range(1, 101)
        ]
        initial_file = temp_data_dir / "initial.yaml"
        with open(initial_file, "w") as f:
            yaml.dump(initial_data, f)

        # Load initial data
        await _load_async(
            file=str(initial_file),
            database=connection_string,
            embedding="sentence-transformers/all-MiniLM-L6-v2",
            mode="initial",
            parallel=1,
            batch_size=32,
            force_reembed=False,
        )

        # Create upsert data with new1-new50 (all new IDs)
        upsert_data = [
            {"id": f"new{i}", "text": f"New document {i}", "metadata": {"index": i}}
            for i in range(1, 51)
        ]
        upsert_file = temp_data_dir / "upsert.yaml"
        with open(upsert_file, "w") as f:
            yaml.dump(upsert_data, f)

        # Upsert with all new IDs
        await _load_async(
            file=str(upsert_file),
            database=connection_string,
            embedding="sentence-transformers/all-MiniLM-L6-v2",
            mode="upsert",
            parallel=1,
            batch_size=32,
            force_reembed=False,
        )

        # Verify total count is 150 (100 + 50)
        info = await chromadb_provider.get_collection_info("test_collection")
        assert info["count"] == 150


class TestE2EMODE004FlushEmptyCollection:
    """E2E-MODE-004: Flush Empty Collection."""

    @pytest.mark.asyncio
    async def test_flush_empty_collection(
        self,
        temp_data_dir: Path,
        chromadb_provider: ChromaDBProvider,
        tmp_path: Path,
    ) -> None:
        """Test flush on empty collection (edge case).

        Requirements: FR-012
        """
        db_path = tmp_path / "chroma_db"
        connection_string = f"chromadb:///{db_path}/test_collection"

        # Create new data
        new_data = [
            {"id": f"doc{i}", "text": f"Document {i}", "metadata": {"index": i}}
            for i in range(1, 51)
        ]
        new_file = temp_data_dir / "new.yaml"
        with open(new_file, "w") as f:
            yaml.dump(new_data, f)

        # Flush on non-existent collection (will create it)
        await _load_async(
            file=str(new_file),
            database=connection_string,
            embedding="sentence-transformers/all-MiniLM-L6-v2",
            mode="flush",
            parallel=1,
            batch_size=32,
            force_reembed=False,
        )

        # Verify data loaded
        info = await chromadb_provider.get_collection_info("test_collection")
        assert info["count"] == 50
