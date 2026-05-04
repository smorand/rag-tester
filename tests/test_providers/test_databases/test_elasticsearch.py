"""Tests for ElasticsearchProvider with mocked AsyncElasticsearch."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from rag_tester.providers.databases.base import (
    ConnectionError,
    DatabaseError,
    DimensionMismatchError,
)
from rag_tester.providers.databases.elasticsearch import ElasticsearchProvider

VALID_CONN = "elasticsearch://localhost:9200/test_index"


def _build_client() -> AsyncMock:
    client = AsyncMock()
    client.ping = AsyncMock(return_value=True)
    client.bulk = AsyncMock(return_value={"errors": False, "items": []})
    client.search = AsyncMock(return_value={"hits": {"hits": []}})
    client.count = AsyncMock(return_value={"count": 0})
    client.delete_by_query = AsyncMock(return_value={})
    client.indices = AsyncMock()
    client.indices.exists = AsyncMock(return_value=False)
    client.indices.create = AsyncMock()
    client.indices.delete = AsyncMock()
    client.indices.get_mapping = AsyncMock()
    client.indices.get_settings = AsyncMock()
    return client


class TestElasticsearchParsing:
    def test_valid_connection_string(self) -> None:
        p = ElasticsearchProvider(VALID_CONN)
        assert p._host == "localhost"
        assert p._port == 9200
        assert p._index_name == "test_index"

    def test_missing_prefix_rejected(self) -> None:
        with pytest.raises(ValueError, match="must start with 'elasticsearch://'"):
            ElasticsearchProvider("postgresql://localhost/test")

    def test_invalid_format_rejected(self) -> None:
        with pytest.raises(ValueError, match="must be elasticsearch"):
            ElasticsearchProvider("elasticsearch://localhost/test")

    @pytest.mark.parametrize("name", ["BadName", "bad name", "bad;DROP", "bad'OR", "BAD"])
    def test_hostile_index_name_rejected(self, name: str) -> None:
        with pytest.raises(ValueError, match="lowercase"):
            ElasticsearchProvider(f"elasticsearch://localhost:9200/{name}")


class TestElasticsearchOperations:
    @pytest.fixture
    def provider(self) -> ElasticsearchProvider:
        return ElasticsearchProvider(VALID_CONN)

    @pytest.fixture
    def client(self, provider: ElasticsearchProvider) -> AsyncMock:
        c = _build_client()
        provider._client = c
        return c

    async def test_get_client_creates_once(self, provider: ElasticsearchProvider) -> None:
        fake = _build_client()
        with patch(
            "rag_tester.providers.databases.elasticsearch.AsyncElasticsearch",
            return_value=fake,
        ) as ctor:
            await provider._get_client()
            await provider._get_client()
            ctor.assert_called_once()
            fake.ping.assert_awaited()

    async def test_get_client_failure_raises(self, provider: ElasticsearchProvider) -> None:
        fake = _build_client()
        fake.ping = AsyncMock(side_effect=Exception("network down"))
        with (
            patch(
                "rag_tester.providers.databases.elasticsearch.AsyncElasticsearch",
                return_value=fake,
            ),
            pytest.raises(ConnectionError, match="Failed to connect"),
        ):
            await provider._get_client()

    async def test_collection_exists_true(self, provider: ElasticsearchProvider, client: AsyncMock) -> None:
        client.indices.exists = AsyncMock(return_value=True)
        assert await provider.collection_exists("test_index") is True

    async def test_collection_exists_false(self, provider: ElasticsearchProvider, client: AsyncMock) -> None:
        client.indices.exists = AsyncMock(return_value=False)
        assert await provider.collection_exists("test_index") is False

    async def test_collection_exists_swallows_errors(self, provider: ElasticsearchProvider, client: AsyncMock) -> None:
        client.indices.exists = AsyncMock(side_effect=Exception("boom"))
        assert await provider.collection_exists("test_index") is False

    async def test_create_collection_already_exists(self, provider: ElasticsearchProvider, client: AsyncMock) -> None:
        client.indices.exists = AsyncMock(return_value=True)
        await provider.create_collection("test_index", dimension=4)
        client.indices.create.assert_not_called()

    async def test_create_collection_new(self, provider: ElasticsearchProvider, client: AsyncMock) -> None:
        client.indices.exists = AsyncMock(return_value=False)
        await provider.create_collection("test_index", dimension=4)
        client.indices.create.assert_awaited_once()

    async def test_create_collection_with_metadata(self, provider: ElasticsearchProvider, client: AsyncMock) -> None:
        client.indices.exists = AsyncMock(return_value=False)
        await provider.create_collection("test_index", dimension=4, metadata={"k": "v"})
        client.indices.create.assert_awaited_once()

    async def test_create_collection_failure_raises_database_error(
        self, provider: ElasticsearchProvider, client: AsyncMock
    ) -> None:
        client.indices.exists = AsyncMock(return_value=False)
        client.indices.create = AsyncMock(side_effect=Exception("schema invalid"))
        with pytest.raises(DatabaseError):
            await provider.create_collection("test_index", dimension=4)

    async def test_insert_empty_records_noop(self, provider: ElasticsearchProvider) -> None:
        await provider.insert("test_index", [])

    async def test_insert_dimension_mismatch_raises(self, provider: ElasticsearchProvider, client: AsyncMock) -> None:
        client.indices.exists = AsyncMock(return_value=True)
        client.indices.get_mapping = AsyncMock(
            return_value={"test_index": {"mappings": {"properties": {"embedding": {"dims": 8}}}}}
        )
        client.count = AsyncMock(return_value={"count": 0})
        client.indices.get_settings = AsyncMock(return_value={"test_index": {"settings": {}}})
        with pytest.raises(DimensionMismatchError):
            await provider.insert(
                "test_index",
                [{"id": "x", "text": "x", "embedding": [1.0, 0.0]}],
            )

    async def test_insert_success(self, provider: ElasticsearchProvider, client: AsyncMock) -> None:
        client.indices.exists = AsyncMock(return_value=True)
        client.indices.get_mapping = AsyncMock(
            return_value={"test_index": {"mappings": {"properties": {"embedding": {"dims": 4}}}}}
        )
        client.count = AsyncMock(return_value={"count": 0})
        client.indices.get_settings = AsyncMock(return_value={"test_index": {"settings": {}}})
        await provider.insert(
            "test_index",
            [{"id": "doc1", "text": "hello", "embedding": [1.0, 0.0, 0.0, 0.0]}],
        )
        client.bulk.assert_awaited()

    async def test_insert_with_bulk_errors_raises(self, provider: ElasticsearchProvider, client: AsyncMock) -> None:
        client.indices.exists = AsyncMock(return_value=True)
        client.indices.get_mapping = AsyncMock(
            return_value={"test_index": {"mappings": {"properties": {"embedding": {"dims": 4}}}}}
        )
        client.count = AsyncMock(return_value={"count": 0})
        client.indices.get_settings = AsyncMock(return_value={"test_index": {"settings": {}}})
        client.bulk = AsyncMock(
            return_value={
                "errors": True,
                "items": [{"index": {"error": {"type": "mapper_parsing_exception"}}}],
            }
        )
        with pytest.raises(DatabaseError):
            await provider.insert(
                "test_index",
                [{"id": "doc1", "text": "hello", "embedding": [1.0, 0.0, 0.0, 0.0]}],
            )

    async def test_query_returns_results(self, provider: ElasticsearchProvider, client: AsyncMock) -> None:
        client.search = AsyncMock(
            return_value={
                "hits": {
                    "hits": [
                        {
                            "_score": 0.9,
                            "_source": {"id": "doc1", "text": "first", "metadata": None},
                        },
                        {
                            "_score": 0.5,
                            "_source": {
                                "id": "doc2",
                                "text": "second",
                                "metadata": {"k": "v"},
                            },
                        },
                    ]
                }
            }
        )
        results = await provider.query("test_index", [1.0, 0.0], top_k=2)
        assert len(results) == 2
        assert results[0]["id"] == "doc1"
        assert results[1]["metadata"] == {"k": "v"}

    async def test_query_with_filter_metadata(self, provider: ElasticsearchProvider, client: AsyncMock) -> None:
        client.search = AsyncMock(return_value={"hits": {"hits": []}})
        await provider.query(
            "test_index",
            [1.0, 0.0],
            top_k=5,
            filter_metadata={"k": "v"},
        )
        client.search.assert_awaited_once()

    async def test_delete_collection(self, provider: ElasticsearchProvider, client: AsyncMock) -> None:
        client.indices.exists = AsyncMock(return_value=True)
        await provider.delete_collection("test_index")
        client.indices.delete.assert_awaited_once()

    async def test_delete_collection_when_missing(self, provider: ElasticsearchProvider, client: AsyncMock) -> None:
        client.indices.exists = AsyncMock(return_value=False)
        await provider.delete_collection("test_index")
        client.indices.delete.assert_not_called()

    async def test_get_collection_info(self, provider: ElasticsearchProvider, client: AsyncMock) -> None:
        client.indices.exists = AsyncMock(return_value=True)
        client.indices.get_mapping = AsyncMock(
            return_value={"test_index": {"mappings": {"properties": {"embedding": {"dims": 4}}}}}
        )
        client.count = AsyncMock(return_value={"count": 5})
        client.indices.get_settings = AsyncMock(
            return_value={"test_index": {"settings": {"index": {"metadata": {"k": "v"}}}}}
        )
        info = await provider.get_collection_info("test_index")
        assert info["name"] == "test_index"
        assert info["dimension"] == 4
        assert info["count"] == 5
        assert info["metadata"] == {"k": "v"}

    async def test_get_collection_info_missing_raises(self, provider: ElasticsearchProvider, client: AsyncMock) -> None:
        client.indices.exists = AsyncMock(return_value=False)
        with pytest.raises(DatabaseError):
            await provider.get_collection_info("test_index")

    async def test_delete_all_returns_count(self, provider: ElasticsearchProvider, client: AsyncMock) -> None:
        client.count = AsyncMock(return_value={"count": 7})
        deleted = await provider.delete_all("test_index")
        assert deleted == 7
        client.delete_by_query.assert_awaited_once()

    async def test_delete_all_empty(self, provider: ElasticsearchProvider, client: AsyncMock) -> None:
        client.count = AsyncMock(return_value={"count": 0})
        deleted = await provider.delete_all("test_index")
        assert deleted == 0
        client.delete_by_query.assert_not_called()

    async def test_delete_by_ids_empty(self, provider: ElasticsearchProvider) -> None:
        assert await provider.delete_by_ids("test_index", []) == 0

    async def test_delete_by_ids_returns_count(self, provider: ElasticsearchProvider, client: AsyncMock) -> None:
        client.delete_by_query = AsyncMock(return_value={"deleted": 3})
        deleted = await provider.delete_by_ids("test_index", ["a", "b", "c"])
        assert deleted == 3

    async def test_close(self, provider: ElasticsearchProvider, client: AsyncMock) -> None:
        await provider.close()
        client.close.assert_awaited()
