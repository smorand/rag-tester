"""Tests for ChromaDBProvider."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from rag_tester.providers.databases.base import DimensionMismatchError
from rag_tester.providers.databases.chromadb import ChromaDBProvider


class TestChromaDBProvider:
    """Tests for ChromaDBProvider."""

    @patch("rag_tester.providers.databases.chromadb.chromadb")
    def test_parse_http_connection_string(self, mock_chromadb: MagicMock) -> None:
        """Test parsing HTTP connection string."""
        mock_chromadb.HttpClient.return_value = MagicMock()
        provider = ChromaDBProvider("chromadb://localhost:8000/test_collection")

        assert provider._mode == "http"
        assert provider._host == "localhost"
        assert provider._port == 8000
        assert provider._collection_name == "test_collection"

    @patch("rag_tester.providers.databases.chromadb.chromadb")
    def test_parse_persistent_connection_string(self, mock_chromadb: MagicMock, tmp_path: Path) -> None:
        """Test parsing persistent connection string."""
        mock_chromadb.PersistentClient.return_value = MagicMock()
        connection_string = f"chromadb://{tmp_path}/test_collection"
        provider = ChromaDBProvider(connection_string)

        assert provider._mode == "persistent"
        assert provider._path == str(tmp_path)
        assert provider._collection_name == "test_collection"

    def test_invalid_connection_string_no_prefix(self) -> None:
        """Test that connection string without chromadb:// prefix raises ValueError."""
        with pytest.raises(ValueError, match="Invalid connection string: must start with 'chromadb://'"):
            ChromaDBProvider("http://localhost:8000/collection")

    def test_invalid_http_connection_string(self) -> None:
        """Test that invalid HTTP connection string raises ValueError."""
        with pytest.raises(ValueError, match="Invalid HTTP connection string"):
            ChromaDBProvider("chromadb://localhost/collection")  # Missing port

    def test_invalid_persistent_connection_string(self) -> None:
        """Test that invalid persistent connection string raises ValueError."""
        with pytest.raises(ValueError, match="Invalid persistent connection string"):
            ChromaDBProvider("chromadb:///")  # Missing collection name

    @patch("rag_tester.providers.databases.chromadb.chromadb")
    async def test_create_collection_success(self, mock_chromadb: MagicMock) -> None:
        """Test successful collection creation."""
        # Setup mocks
        mock_client = MagicMock()
        mock_chromadb.HttpClient.return_value = mock_client
        mock_client.list_collections.return_value = []
        mock_client.create_collection.return_value = MagicMock()

        provider = ChromaDBProvider("chromadb://localhost:8000/test_collection")

        await provider.create_collection("test_collection", dimension=384)

        mock_client.create_collection.assert_called_once()
        call_args = mock_client.create_collection.call_args
        assert call_args[1]["name"] == "test_collection"
        assert call_args[1]["metadata"]["dimension"] == 384

    @patch("rag_tester.providers.databases.chromadb.chromadb")
    async def test_create_collection_already_exists(self, mock_chromadb: MagicMock) -> None:
        """Test that creating existing collection is a no-op."""
        # Setup mocks
        mock_client = MagicMock()
        mock_chromadb.HttpClient.return_value = mock_client

        mock_collection = MagicMock()
        mock_collection.name = "test_collection"
        mock_client.list_collections.return_value = [mock_collection]

        provider = ChromaDBProvider("chromadb://localhost:8000/test_collection")

        await provider.create_collection("test_collection", dimension=384)

        # Should not call create_collection since it already exists
        mock_client.create_collection.assert_not_called()

    @patch("rag_tester.providers.databases.chromadb.chromadb")
    async def test_collection_exists_true(self, mock_chromadb: MagicMock) -> None:
        """Test checking if collection exists (true case)."""
        mock_client = MagicMock()
        mock_chromadb.HttpClient.return_value = mock_client

        mock_collection = MagicMock()
        mock_collection.name = "test_collection"
        mock_client.list_collections.return_value = [mock_collection]

        provider = ChromaDBProvider("chromadb://localhost:8000/test_collection")

        exists = await provider.collection_exists("test_collection")
        assert exists is True

    @patch("rag_tester.providers.databases.chromadb.chromadb")
    async def test_collection_exists_false(self, mock_chromadb: MagicMock) -> None:
        """Test checking if collection exists (false case)."""
        mock_client = MagicMock()
        mock_chromadb.HttpClient.return_value = mock_client
        mock_client.list_collections.return_value = []

        provider = ChromaDBProvider("chromadb://localhost:8000/test_collection")

        exists = await provider.collection_exists("test_collection")
        assert exists is False

    @patch("rag_tester.providers.databases.chromadb.chromadb")
    async def test_insert_success(self, mock_chromadb: MagicMock) -> None:
        """Test successful record insertion."""
        mock_client = MagicMock()
        mock_chromadb.HttpClient.return_value = mock_client

        mock_collection = MagicMock()
        mock_collection.name = "test_collection"
        mock_collection.metadata = {"dimension": 384}
        mock_client.list_collections.return_value = [mock_collection]
        mock_client.get_collection.return_value = mock_collection

        provider = ChromaDBProvider("chromadb://localhost:8000/test_collection")

        records = [
            {"id": "doc1", "text": "First document", "embedding": [0.1] * 384},
            {"id": "doc2", "text": "Second document", "embedding": [0.2] * 384},
        ]

        await provider.insert("test_collection", records)

        mock_collection.add.assert_called_once()
        call_args = mock_collection.add.call_args[1]
        assert call_args["ids"] == ["doc1", "doc2"]
        assert len(call_args["embeddings"]) == 2
        assert call_args["documents"] == ["First document", "Second document"]

    @patch("rag_tester.providers.databases.chromadb.chromadb")
    async def test_insert_dimension_mismatch(self, mock_chromadb: MagicMock) -> None:
        """Test that dimension mismatch raises DimensionMismatchError."""
        mock_client = MagicMock()
        mock_chromadb.HttpClient.return_value = mock_client

        mock_collection = MagicMock()
        mock_collection.name = "test_collection"
        mock_collection.metadata = {"dimension": 768}  # Different dimension
        mock_client.list_collections.return_value = [mock_collection]
        mock_client.get_collection.return_value = mock_collection

        provider = ChromaDBProvider("chromadb://localhost:8000/test_collection")

        records = [
            {"id": "doc1", "text": "First document", "embedding": [0.1] * 384},  # Wrong dimension
        ]

        with pytest.raises(DimensionMismatchError, match="Dimension mismatch: model=384, database=768"):
            await provider.insert("test_collection", records)

    @patch("rag_tester.providers.databases.chromadb.chromadb")
    async def test_insert_empty_records(self, mock_chromadb: MagicMock) -> None:
        """Test that inserting empty records is a no-op."""
        mock_client = MagicMock()
        mock_chromadb.HttpClient.return_value = mock_client

        provider = ChromaDBProvider("chromadb://localhost:8000/test_collection")

        await provider.insert("test_collection", [])

        # Should not call any collection methods
        mock_client.get_collection.assert_not_called()

    @patch("rag_tester.providers.databases.chromadb.chromadb")
    async def test_query_success(self, mock_chromadb: MagicMock) -> None:
        """Test successful query."""
        mock_client = MagicMock()
        mock_chromadb.HttpClient.return_value = mock_client

        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            "ids": [["doc1", "doc2"]],
            "documents": [["First document", "Second document"]],
            "distances": [[0.1, 0.3]],
            "metadatas": [[{}, {}]],
        }
        mock_client.get_collection.return_value = mock_collection

        provider = ChromaDBProvider("chromadb://localhost:8000/test_collection")

        query_embedding = [0.5] * 384
        results = await provider.query("test_collection", query_embedding, top_k=2)

        assert len(results) == 2
        assert results[0]["id"] == "doc1"
        assert results[0]["text"] == "First document"
        assert results[0]["score"] == pytest.approx(0.9)  # 1.0 - 0.1
        assert results[1]["id"] == "doc2"
        assert results[1]["score"] == pytest.approx(0.7)  # 1.0 - 0.3

    @patch("rag_tester.providers.databases.chromadb.chromadb")
    async def test_query_no_results(self, mock_chromadb: MagicMock) -> None:
        """Test query with no results."""
        mock_client = MagicMock()
        mock_chromadb.HttpClient.return_value = mock_client

        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            "ids": [[]],
            "documents": [[]],
            "distances": [[]],
            "metadatas": [[]],
        }
        mock_client.get_collection.return_value = mock_collection

        provider = ChromaDBProvider("chromadb://localhost:8000/test_collection")

        query_embedding = [0.5] * 384
        results = await provider.query("test_collection", query_embedding, top_k=5)

        assert len(results) == 0

    @patch("rag_tester.providers.databases.chromadb.chromadb")
    async def test_delete_collection_success(self, mock_chromadb: MagicMock) -> None:
        """Test successful collection deletion."""
        mock_client = MagicMock()
        mock_chromadb.HttpClient.return_value = mock_client

        provider = ChromaDBProvider("chromadb://localhost:8000/test_collection")

        await provider.delete_collection("test_collection")

        mock_client.delete_collection.assert_called_once_with(name="test_collection")

    @patch("rag_tester.providers.databases.chromadb.chromadb")
    async def test_get_collection_info_success(self, mock_chromadb: MagicMock) -> None:
        """Test getting collection info."""
        mock_client = MagicMock()
        mock_chromadb.HttpClient.return_value = mock_client

        mock_collection = MagicMock()
        mock_collection.count.return_value = 100
        mock_collection.metadata = {"dimension": 384, "custom": "value"}
        mock_client.get_collection.return_value = mock_collection

        provider = ChromaDBProvider("chromadb://localhost:8000/test_collection")

        info = await provider.get_collection_info("test_collection")

        assert info["name"] == "test_collection"
        assert info["dimension"] == 384
        assert info["count"] == 100
        assert info["metadata"]["custom"] == "value"

    @patch("rag_tester.providers.databases.chromadb.chromadb")
    async def test_close_http_client(self, mock_chromadb: MagicMock) -> None:
        """Test closing HTTP client."""
        mock_client = MagicMock()
        mock_chromadb.HttpClient.return_value = mock_client

        with patch("httpx.AsyncClient") as mock_async_client:
            mock_http = AsyncMock()
            mock_async_client.return_value = mock_http

            provider = ChromaDBProvider("chromadb://localhost:8000/test_collection")
            await provider.close()

            mock_http.aclose.assert_called_once()

    @patch("chromadb.HttpClient")
    async def test_delete_all_success(self, mock_http_client: MagicMock) -> None:
        """Test successful deletion of all records."""
        mock_client = MagicMock()
        mock_http_client.return_value = mock_client

        mock_collection = MagicMock()
        mock_collection.count.return_value = 100
        mock_collection.get.return_value = {"ids": [f"doc{i}" for i in range(100)]}
        mock_client.get_collection.return_value = mock_collection

        provider = ChromaDBProvider("chromadb://localhost:8000/test_collection")

        deleted_count = await provider.delete_all("test_collection")

        assert deleted_count == 100
        mock_collection.delete.assert_called_once()
        call_args = mock_collection.delete.call_args[1]
        assert len(call_args["ids"]) == 100

    @patch("chromadb.HttpClient")
    async def test_delete_all_empty_collection(self, mock_http_client: MagicMock) -> None:
        """Test deleting from empty collection."""
        mock_client = MagicMock()
        mock_http_client.return_value = mock_client

        mock_collection = MagicMock()
        mock_collection.count.return_value = 0
        mock_client.get_collection.return_value = mock_collection

        provider = ChromaDBProvider("chromadb://localhost:8000/test_collection")

        deleted_count = await provider.delete_all("test_collection")

        assert deleted_count == 0
        mock_collection.delete.assert_not_called()

    @patch("chromadb.HttpClient")
    async def test_delete_by_ids_success(self, mock_http_client: MagicMock) -> None:
        """Test successful deletion of specific records."""
        mock_client = MagicMock()
        mock_http_client.return_value = mock_client

        mock_collection = MagicMock()
        mock_client.get_collection.return_value = mock_collection

        provider = ChromaDBProvider("chromadb://localhost:8000/test_collection")

        ids_to_delete = ["doc1", "doc2", "doc3"]
        deleted_count = await provider.delete_by_ids("test_collection", ids_to_delete)

        assert deleted_count == 3
        mock_collection.delete.assert_called_once_with(ids=ids_to_delete)

    @patch("chromadb.HttpClient")
    async def test_delete_by_ids_empty_list(self, mock_http_client: MagicMock) -> None:
        """Test deleting with empty ID list."""
        mock_client = MagicMock()
        mock_http_client.return_value = mock_client

        provider = ChromaDBProvider("chromadb://localhost:8000/test_collection")

        deleted_count = await provider.delete_by_ids("test_collection", [])

        assert deleted_count == 0
        mock_client.get_collection.assert_not_called()
