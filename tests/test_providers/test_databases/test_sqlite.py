"""Tests for SQLiteProvider using a real on-disk SQLite database."""

from pathlib import Path

import pytest

from rag_tester.providers.databases.base import (
    ConnectionError,
    DatabaseError,
    DimensionMismatchError,
)
from rag_tester.providers.databases.sqlite import SQLiteProvider


def _conn_string(tmp_path: Path, table: str = "test_table") -> str:
    """Build a connection string targeting an isolated DB file."""
    return f"sqlite:///{tmp_path}/test.db/{table}"


class TestSQLiteParsing:
    """Connection string parsing and security validation."""

    def test_parse_valid_connection_string(self, tmp_path: Path) -> None:
        provider = SQLiteProvider(_conn_string(tmp_path))
        assert provider._table_name == "test_table"
        assert str(provider._db_path).endswith("test.db")

    def test_invalid_prefix_raises(self) -> None:
        with pytest.raises(ValueError, match="must start with 'sqlite:///'"):
            SQLiteProvider("postgresql://localhost/test")

    def test_missing_table_raises(self) -> None:
        with pytest.raises(ValueError, match="must be sqlite:///path"):
            SQLiteProvider("sqlite:///onlyfile")

    def test_path_traversal_rejected(self) -> None:
        with pytest.raises(ValueError, match="path traversal"):
            SQLiteProvider("sqlite:///../etc/passwd/test")

    @pytest.mark.parametrize(
        "table_name",
        [
            "valid_name",
            "name123",
            "_underscore",
        ],
    )
    def test_valid_table_names(self, tmp_path: Path, table_name: str) -> None:
        provider = SQLiteProvider(_conn_string(tmp_path, table_name))
        assert provider._table_name == table_name

    @pytest.mark.parametrize(
        "table_name",
        [
            "name with space",
            "name; DROP TABLE",
            "name'OR'1'='1",
            "name-dash",
            "name.dot",
            "name(paren)",
        ],
    )
    def test_hostile_table_names_rejected(self, tmp_path: Path, table_name: str) -> None:
        """Whitelist regex must reject any non-alphanumeric/underscore name."""
        with pytest.raises(ValueError, match="alphanumeric with underscores"):
            SQLiteProvider(_conn_string(tmp_path, table_name))


class TestSQLiteCRUD:
    """CRUD operations against a real SQLite database in tmp_path."""

    @pytest.fixture
    async def provider(self, tmp_path: Path) -> SQLiteProvider:
        prov = SQLiteProvider(_conn_string(tmp_path))
        yield prov
        if prov._conn is not None:
            await prov._conn.close()

    async def test_create_collection_and_exists(self, provider: SQLiteProvider) -> None:
        assert await provider.collection_exists("test_table") is False
        await provider.create_collection("test_table", dimension=4)
        assert await provider.collection_exists("test_table") is True

    async def test_create_collection_idempotent(self, provider: SQLiteProvider) -> None:
        await provider.create_collection("test_table", dimension=4)
        # Second call should be a no-op (already exists), not raise.
        await provider.create_collection("test_table", dimension=4)
        assert await provider.collection_exists("test_table") is True

    async def test_create_collection_table_name_mismatch(self, provider: SQLiteProvider) -> None:
        with pytest.raises(DatabaseError, match="Table name mismatch"):
            await provider.create_collection("other_table", dimension=4)

    async def test_insert_and_query(self, provider: SQLiteProvider) -> None:
        records = [
            {"id": "doc1", "text": "first", "embedding": [1.0, 0.0, 0.0, 0.0]},
            {"id": "doc2", "text": "second", "embedding": [0.0, 1.0, 0.0, 0.0]},
            {"id": "doc3", "text": "third", "embedding": [0.0, 0.0, 1.0, 0.0]},
        ]
        await provider.insert("test_table", records)

        results = await provider.query("test_table", [1.0, 0.0, 0.0, 0.0], top_k=2)
        assert len(results) == 2
        assert results[0]["id"] == "doc1"  # Closest match to itself
        assert results[0]["score"] == pytest.approx(1.0, abs=1e-6)

    async def test_insert_empty_records_noop(self, provider: SQLiteProvider) -> None:
        await provider.insert("test_table", [])
        # Table is auto-created on first insert; with empty list, no creation.
        assert await provider.collection_exists("test_table") is False

    async def test_insert_dimension_mismatch_raises(self, provider: SQLiteProvider) -> None:
        await provider.create_collection("test_table", dimension=4)
        await provider.insert(
            "test_table",
            [{"id": "x", "text": "x", "embedding": [1.0, 0.0, 0.0, 0.0]}],
        )
        with pytest.raises(DimensionMismatchError):
            await provider.insert(
                "test_table",
                [{"id": "y", "text": "y", "embedding": [1.0, 0.0]}],
            )

    async def test_insert_with_metadata(self, provider: SQLiteProvider) -> None:
        records = [
            {
                "id": "doc1",
                "text": "hello",
                "embedding": [1.0, 0.0, 0.0, 0.0],
                "metadata": {"source": "kb", "lang": "en"},
            }
        ]
        await provider.insert("test_table", records)
        results = await provider.query("test_table", [1.0, 0.0, 0.0, 0.0], top_k=1)
        assert results[0]["metadata"]["source"] == "kb"

    async def test_get_collection_info(self, provider: SQLiteProvider) -> None:
        await provider.insert(
            "test_table",
            [{"id": "doc1", "text": "x", "embedding": [1.0, 2.0, 3.0]}],
        )
        info = await provider.get_collection_info("test_table")
        assert info["name"] == "test_table"
        assert info["dimension"] == 3
        assert info["count"] == 1

    async def test_delete_all(self, provider: SQLiteProvider) -> None:
        await provider.insert(
            "test_table",
            [
                {"id": "a", "text": "a", "embedding": [1.0, 0.0]},
                {"id": "b", "text": "b", "embedding": [0.0, 1.0]},
            ],
        )
        deleted = await provider.delete_all("test_table")
        assert deleted == 2
        info = await provider.get_collection_info("test_table")
        assert info["count"] == 0

    async def test_delete_all_on_empty_collection(self, provider: SQLiteProvider) -> None:
        await provider.create_collection("test_table", dimension=2)
        deleted = await provider.delete_all("test_table")
        assert deleted == 0

    async def test_delete_by_ids(self, provider: SQLiteProvider) -> None:
        await provider.insert(
            "test_table",
            [
                {"id": "a", "text": "a", "embedding": [1.0, 0.0]},
                {"id": "b", "text": "b", "embedding": [0.0, 1.0]},
                {"id": "c", "text": "c", "embedding": [1.0, 1.0]},
            ],
        )
        deleted = await provider.delete_by_ids("test_table", ["a", "c"])
        assert deleted == 2
        info = await provider.get_collection_info("test_table")
        assert info["count"] == 1

    async def test_delete_by_ids_empty_list(self, provider: SQLiteProvider) -> None:
        await provider.insert(
            "test_table",
            [{"id": "a", "text": "a", "embedding": [1.0, 0.0]}],
        )
        deleted = await provider.delete_by_ids("test_table", [])
        assert deleted == 0

    async def test_delete_collection(self, provider: SQLiteProvider) -> None:
        await provider.create_collection("test_table", dimension=2)
        assert await provider.collection_exists("test_table") is True
        await provider.delete_collection("test_table")
        assert await provider.collection_exists("test_table") is False

    async def test_query_with_filter_metadata(self, provider: SQLiteProvider) -> None:
        await provider.insert(
            "test_table",
            [
                {"id": "1", "text": "a", "embedding": [1.0, 0.0], "metadata": {"src": "kb"}},
                {"id": "2", "text": "b", "embedding": [0.0, 1.0], "metadata": {"src": "doc"}},
            ],
        )
        results = await provider.query(
            "test_table",
            [1.0, 0.0],
            top_k=5,
            filter_metadata={"src": "kb"},
        )
        assert len(results) == 1
        assert results[0]["id"] == "1"

    async def test_serialize_deserialize_embedding_roundtrip(self, provider: SQLiteProvider) -> None:
        original = [0.1, -0.2, 3.4, 0.0, 1e-3]
        data = provider._serialize_embedding(original)
        recovered = provider._deserialize_embedding(data)
        assert recovered == pytest.approx(original, abs=1e-6)


class TestSQLiteErrorHandling:
    """Error handling paths."""

    async def test_insert_table_name_mismatch(self, tmp_path: Path) -> None:
        provider = SQLiteProvider(_conn_string(tmp_path))
        try:
            with pytest.raises(DatabaseError, match="Table name mismatch"):
                await provider.insert(
                    "wrong_table",
                    [{"id": "a", "text": "a", "embedding": [1.0]}],
                )
        finally:
            if provider._conn is not None:
                await provider._conn.close()

    async def test_get_collection_info_missing_table_raises(self, tmp_path: Path) -> None:
        provider = SQLiteProvider(_conn_string(tmp_path))
        try:
            with pytest.raises(DatabaseError):
                await provider.get_collection_info("test_table")
        finally:
            if provider._conn is not None:
                await provider._conn.close()

    async def test_connection_failure_raises(self, tmp_path: Path) -> None:
        # Point at a path inside a regular file to force a failure.
        blocker = tmp_path / "blocker"
        blocker.write_text("not a directory")
        provider = SQLiteProvider(f"sqlite:///{blocker}/sub/db.sqlite/tbl")
        with pytest.raises(ConnectionError):
            await provider._get_connection()
