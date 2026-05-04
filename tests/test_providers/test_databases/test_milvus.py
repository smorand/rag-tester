"""Tests for MilvusProvider with mocked pymilvus dependencies."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from rag_tester.providers.databases.base import (
    ConnectionError,
    DatabaseError,
    DimensionMismatchError,
)

VALID_CONN = "milvus://localhost:19530/test_collection"


@pytest.fixture(autouse=True)
def patch_pymilvus():
    """Stub pymilvus globals used by MilvusProvider."""
    with (
        patch("rag_tester.providers.databases.milvus.connections") as connections,
        patch("rag_tester.providers.databases.milvus.utility") as utility,
        patch("rag_tester.providers.databases.milvus.Collection") as Collection,
    ):
        connections.connect = MagicMock()
        connections.disconnect = MagicMock()
        utility.has_collection = MagicMock(return_value=False)
        utility.drop_collection = MagicMock()
        yield connections, utility, Collection


@pytest.fixture
def provider(patch_pymilvus):
    from rag_tester.providers.databases.milvus import MilvusProvider

    return MilvusProvider(VALID_CONN)


class TestMilvusParsing:
    def test_valid_connection_string(self, provider) -> None:
        assert provider._host == "localhost"
        assert provider._port == 19530
        assert provider._collection_name == "test_collection"

    def test_missing_prefix_rejected(self) -> None:
        from rag_tester.providers.databases.milvus import MilvusProvider

        with pytest.raises(ValueError, match="must start with 'milvus://'"):
            MilvusProvider("postgresql://localhost/test")

    def test_invalid_format_rejected(self) -> None:
        from rag_tester.providers.databases.milvus import MilvusProvider

        with pytest.raises(ValueError, match="must be milvus"):
            MilvusProvider("milvus://localhost/test")

    @pytest.mark.parametrize("name", ["bad name", "bad-dash", "bad;DROP", "bad'OR'1", "bad.dot"])
    def test_hostile_collection_name_rejected(self, name: str) -> None:
        from rag_tester.providers.databases.milvus import MilvusProvider

        with pytest.raises(ValueError, match="alphanumeric"):
            MilvusProvider(f"milvus://localhost:19530/{name}")

    def test_connect_failure_raises(self) -> None:
        from rag_tester.providers.databases.milvus import MilvusProvider

        with patch("rag_tester.providers.databases.milvus.connections") as conn_mod:
            conn_mod.connect = MagicMock(side_effect=Exception("net down"))
            with pytest.raises(ConnectionError, match="Failed to connect"):
                MilvusProvider(VALID_CONN)


class TestMilvusOperations:
    async def test_collection_exists_true(self, provider, patch_pymilvus) -> None:
        _, utility, _ = patch_pymilvus
        utility.has_collection.return_value = True
        assert await provider.collection_exists("test_collection") is True

    async def test_collection_exists_false(self, provider, patch_pymilvus) -> None:
        _, utility, _ = patch_pymilvus
        utility.has_collection.return_value = False
        assert await provider.collection_exists("test_collection") is False

    async def test_collection_exists_swallows_errors(self, provider, patch_pymilvus) -> None:
        _, utility, _ = patch_pymilvus
        utility.has_collection.side_effect = Exception("boom")
        assert await provider.collection_exists("test_collection") is False

    async def test_create_collection_already_exists(self, provider, patch_pymilvus) -> None:
        _, utility, _ = patch_pymilvus
        utility.has_collection.return_value = True
        await provider.create_collection("test_collection", dimension=4)

    async def test_create_collection_new(self, provider, patch_pymilvus) -> None:
        _, utility, Collection = patch_pymilvus
        utility.has_collection.return_value = False
        col_instance = MagicMock()
        Collection.return_value = col_instance
        await provider.create_collection("test_collection", dimension=4)
        Collection.assert_called_once()
        col_instance.create_index.assert_called_once()

    async def test_create_collection_with_metadata(self, provider, patch_pymilvus) -> None:
        _, utility, Collection = patch_pymilvus
        utility.has_collection.return_value = False
        Collection.return_value = MagicMock()
        await provider.create_collection("test_collection", dimension=4, metadata={"k": "v"})

    async def test_create_collection_failure_raises_database_error(self, provider, patch_pymilvus) -> None:
        _, utility, Collection = patch_pymilvus
        utility.has_collection.return_value = False
        Collection.side_effect = RuntimeError("schema invalid")
        with pytest.raises(DatabaseError):
            await provider.create_collection("test_collection", dimension=4)

    async def test_insert_empty_records_noop(self, provider) -> None:
        await provider.insert("test_collection", [])

    async def test_insert_dimension_mismatch_raises(self, provider, patch_pymilvus) -> None:
        _, utility, Collection = patch_pymilvus
        utility.has_collection.return_value = True
        col_instance = MagicMock()
        embedding_field = MagicMock()
        embedding_field.name = "embedding"
        embedding_field.params = {"dim": 8}
        text_field = MagicMock()
        text_field.name = "text"
        col_instance.schema.fields = [text_field, embedding_field]
        Collection.return_value = col_instance
        with pytest.raises(DimensionMismatchError):
            await provider.insert(
                "test_collection",
                [{"id": "x", "text": "x", "embedding": [1.0, 0.0]}],
            )

    async def test_insert_success(self, provider, patch_pymilvus) -> None:
        _, utility, Collection = patch_pymilvus
        utility.has_collection.return_value = True
        col_instance = MagicMock()
        embedding_field = MagicMock()
        embedding_field.name = "embedding"
        embedding_field.params = {"dim": 4}
        text_field = MagicMock()
        text_field.name = "text"
        col_instance.schema.fields = [text_field, embedding_field]
        Collection.return_value = col_instance
        await provider.insert(
            "test_collection",
            [{"id": "doc1", "text": "hello", "embedding": [1.0, 0.0, 0.0, 0.0]}],
        )
        col_instance.insert.assert_called()
        col_instance.flush.assert_called()

    async def test_query_returns_results(self, provider, patch_pymilvus) -> None:
        _, _, Collection = patch_pymilvus
        col_instance = MagicMock()
        text_field = MagicMock()
        text_field.name = "text"
        col_instance.schema.fields = [text_field]
        Collection.return_value = col_instance

        hit1 = MagicMock()
        hit1.score = 0.9
        hit1.entity.get = MagicMock(side_effect=lambda key: {"id": "doc1", "text": "first", "metadata": None}[key])
        col_instance.search.return_value = [[hit1]]

        results = await provider.query("test_collection", [1.0, 0.0, 0.0, 0.0], top_k=1)
        assert len(results) == 1
        assert results[0]["id"] == "doc1"

    async def test_query_with_filter_metadata(self, provider, patch_pymilvus) -> None:
        _, _, Collection = patch_pymilvus
        col_instance = MagicMock()
        meta_field = MagicMock()
        meta_field.name = "metadata"
        col_instance.schema.fields = [meta_field]
        Collection.return_value = col_instance
        col_instance.search.return_value = [[]]
        await provider.query(
            "test_collection",
            [1.0, 0.0],
            top_k=5,
            filter_metadata={"src": "kb", "year": 2024},
        )
        col_instance.search.assert_called_once()

    async def test_delete_collection(self, provider, patch_pymilvus) -> None:
        _, utility, _ = patch_pymilvus
        utility.has_collection.return_value = True
        await provider.delete_collection("test_collection")
        utility.drop_collection.assert_called_once()

    async def test_delete_collection_when_missing(self, provider, patch_pymilvus) -> None:
        _, utility, _ = patch_pymilvus
        utility.has_collection.return_value = False
        await provider.delete_collection("test_collection")
        utility.drop_collection.assert_not_called()

    async def test_delete_all_returns_count(self, provider, patch_pymilvus) -> None:
        _, _, Collection = patch_pymilvus
        col_instance = MagicMock()
        col_instance.num_entities = 5
        Collection.return_value = col_instance
        deleted = await provider.delete_all("test_collection")
        assert deleted == 5

    async def test_delete_by_ids_empty(self, provider) -> None:
        assert await provider.delete_by_ids("test_collection", []) == 0

    async def test_delete_by_ids_returns_count(self, provider, patch_pymilvus) -> None:
        _, _, Collection = patch_pymilvus
        col_instance = MagicMock()
        delete_result = MagicMock()
        delete_result.delete_count = 2
        col_instance.delete.return_value = delete_result
        Collection.return_value = col_instance
        deleted = await provider.delete_by_ids("test_collection", ["a", "b"])
        assert deleted == 2

    async def test_get_collection_info(self, provider, patch_pymilvus) -> None:
        _, utility, Collection = patch_pymilvus
        utility.has_collection.return_value = True
        col_instance = MagicMock()
        col_instance.num_entities = 10
        embedding_field = MagicMock()
        embedding_field.name = "embedding"
        embedding_field.params = {"dim": 4}
        col_instance.schema.fields = [embedding_field]
        col_instance.schema.description = "Test"
        Collection.return_value = col_instance
        info = await provider.get_collection_info("test_collection")
        assert info["name"] == "test_collection"
        assert info["count"] == 10

    async def test_get_collection_info_missing_raises(self, provider, patch_pymilvus) -> None:
        _, utility, _ = patch_pymilvus
        utility.has_collection.return_value = False
        with pytest.raises(DatabaseError):
            await provider.get_collection_info("test_collection")

    async def test_close(self, provider, patch_pymilvus) -> None:
        connections, _, _ = patch_pymilvus
        await provider.close()
        connections.disconnect.assert_called()
