"""ChromaDB vector database provider."""

import logging
import re
from pathlib import Path
from typing import Any

import httpx
from opentelemetry import trace

from rag_tester.providers.databases.base import (
    ConnectionError,
    DatabaseError,
    DimensionMismatchError,
    VectorDatabase,
)
from rag_tester.utils.retry import retry_with_backoff as retry

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)


class ChromaDBProvider(VectorDatabase):
    """ChromaDB vector database provider.

    Supports both HTTP and persistent storage modes:
    - HTTP: chromadb://host:port/collection_name
    - Persistent: chromadb:///path/to/data/collection_name

    Args:
        connection_string: Connection string specifying mode and location

    Raises:
        ConnectionError: If connection fails
        ValueError: If connection string is invalid
    """

    def __init__(self, connection_string: str) -> None:
        """Initialize ChromaDB provider.

        Args:
            connection_string: Connection string (HTTP or persistent mode)
        """
        self._connection_string = connection_string
        self._mode: str
        self._client: Any = None
        self._http_client: httpx.AsyncClient | None = None
        self._parse_connection_string()
        self._initialize_client()

    def _parse_connection_string(self) -> None:
        """Parse connection string to determine mode and parameters.

        Raises:
            ValueError: If connection string format is invalid
        """
        if not self._connection_string.startswith("chromadb://"):
            msg = "Invalid connection string: must start with 'chromadb://'"
            raise ValueError(msg)

        # Remove chromadb:// prefix
        remainder = self._connection_string[len("chromadb://") :]

        # Check if it's persistent mode (starts with /)
        if remainder.startswith("/"):
            self._mode = "persistent"
            # Extract path and collection name
            # Format: chromadb:///path/to/data/collection_name
            parts = remainder.rsplit("/", 1)
            if len(parts) != 2:
                msg = "Invalid persistent connection string: must be chromadb:///path/to/data/collection_name"
                raise ValueError(msg)
            self._path = parts[0] if parts[0] else "/"
            self._collection_name = parts[1]
            logger.info(f"Persistent mode: path={self._path}, collection={self._collection_name}")
        else:
            self._mode = "http"
            # Extract host, port, and collection name
            # Format: chromadb://host:port/collection_name
            match = re.match(r"^([^:]+):(\d+)/(.+)$", remainder)
            if not match:
                msg = "Invalid HTTP connection string: must be chromadb://host:port/collection_name"
                raise ValueError(msg)
            self._host = match.group(1)
            self._port = int(match.group(2))
            self._collection_name = match.group(3)
            logger.info(f"HTTP mode: host={self._host}, port={self._port}, collection={self._collection_name}")

    def _initialize_client(self) -> None:
        """Initialize ChromaDB client based on mode.

        Raises:
            ConnectionError: If client initialization fails
        """
        try:
            import chromadb

            if self._mode == "http":
                # HTTP mode: use httpx for async operations
                base_url = f"http://{self._host}:{self._port}"
                self._http_client = httpx.AsyncClient(base_url=base_url, timeout=30.0)
                # Also create sync client for chromadb operations
                self._client = chromadb.HttpClient(host=self._host, port=self._port)
                logger.info(f"Initialized HTTP client: {base_url}")
            else:
                # Persistent mode: create directory if needed
                path = Path(self._path)
                path.mkdir(parents=True, exist_ok=True)
                self._client = chromadb.PersistentClient(path=str(path))
                logger.info(f"Initialized persistent client: {self._path}")

        except Exception as e:
            error_msg = f"Failed to initialize ChromaDB client: {e}"
            logger.error(error_msg)
            raise ConnectionError(error_msg) from e

    @retry(max_attempts=5, initial_delay=1.0, backoff_multiplier=2.0)
    async def create_collection(self, name: str, dimension: int, metadata: dict[str, Any] | None = None) -> None:
        """Create a collection for embeddings.

        Args:
            name: Collection name
            dimension: Embedding dimension
            metadata: Optional collection metadata

        Raises:
            DatabaseError: If collection creation fails
        """
        with tracer.start_as_current_span("create_collection") as span:
            span.set_attribute("collection.name", name)
            span.set_attribute("collection.dimension", dimension)

            try:
                # Check if collection already exists
                if await self.collection_exists(name):
                    logger.debug(f"Collection already exists: {name}")
                    return

                # Create collection with metadata
                collection_metadata = metadata or {}
                collection_metadata["dimension"] = dimension

                self._client.create_collection(
                    name=name,
                    metadata=collection_metadata,
                )

                logger.info(f"Collection created: {name} (dimension: {dimension})")

            except Exception as e:
                error_msg = f"Failed to create collection {name}: {e}"
                logger.error(error_msg)
                span.record_exception(e)
                raise DatabaseError(error_msg) from e

    async def collection_exists(self, name: str) -> bool:
        """Check if a collection exists.

        Args:
            name: Collection name

        Returns:
            True if collection exists, False otherwise
        """
        try:
            collections = self._client.list_collections()
            return any(col.name == name for col in collections)
        except Exception as e:
            logger.error(f"Failed to check collection existence: {e}")
            return False

    @retry(max_attempts=5, initial_delay=1.0, backoff_multiplier=2.0)
    async def insert(self, collection: str, records: list[dict[str, Any]]) -> None:
        """Insert records with embeddings.

        ChromaDB v2 has a maximum batch size limit. This method automatically
        splits large batches into smaller chunks of 2000 records.

        Args:
            collection: Collection name
            records: List of records with id, text, embedding, and optional metadata

        Raises:
            DatabaseError: If insertion fails
            DimensionMismatchError: If embedding dimension doesn't match
        """
        if not records:
            return

        with tracer.start_as_current_span("database_insert") as span:
            span.set_attribute("collection.name", collection)
            span.set_attribute("records.count", len(records))

            try:
                # Get or create collection
                if not await self.collection_exists(collection):
                    # Auto-create with dimension from first record
                    dimension = len(records[0]["embedding"])
                    await self.create_collection(collection, dimension)

                # Get collection
                col = self._client.get_collection(name=collection)

                # Verify dimension compatibility
                expected_dimension = col.metadata.get("dimension")
                if expected_dimension:
                    actual_dimension = len(records[0]["embedding"])
                    if actual_dimension != expected_dimension:
                        msg = f"Dimension mismatch: model={actual_dimension}, database={expected_dimension}"
                        raise DimensionMismatchError(msg)

                # ChromaDB v2 has a batch size limit, split into chunks of 2000
                batch_size = 2000
                total_inserted = 0
                
                for i in range(0, len(records), batch_size):
                    batch = records[i:i + batch_size]
                    
                    # Prepare data for ChromaDB
                    ids = [record["id"] for record in batch]
                    embeddings = [record["embedding"] for record in batch]
                    documents = [record["text"] for record in batch]
                    # ChromaDB requires non-empty metadata or None
                    metadatas = [record.get("metadata") or None for record in batch]

                    # Insert batch into ChromaDB
                    col.add(
                        ids=ids,
                        embeddings=embeddings,
                        documents=documents,
                        metadatas=metadatas,
                    )
                    
                    total_inserted += len(batch)
                    logger.debug(f"Inserted batch of {len(batch)} records ({total_inserted}/{len(records)})")

                logger.info(f"Inserted {total_inserted} records into {collection}")
                span.set_attribute("records.inserted", total_inserted)

            except DimensionMismatchError:
                raise
            except Exception as e:
                error_msg = f"Failed to insert records into {collection}: {e}"
                logger.error(error_msg)
                span.record_exception(e)
                raise DatabaseError(error_msg) from e

    @retry(max_attempts=5, initial_delay=1.0, backoff_multiplier=2.0)
    async def query(
        self,
        collection: str,
        query_embedding: list[float],
        top_k: int = 5,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Query for similar embeddings.

        Args:
            collection: Collection name
            query_embedding: Query embedding vector
            top_k: Number of results to return
            filter_metadata: Optional metadata filters

        Returns:
            List of records sorted by similarity (descending)

        Raises:
            DatabaseError: If query fails
        """
        with tracer.start_as_current_span("database_search") as span:
            span.set_attribute("collection.name", collection)
            span.set_attribute("query.top_k", top_k)

            try:
                col = self._client.get_collection(name=collection)

                # Query ChromaDB
                results = col.query(
                    query_embeddings=[query_embedding],
                    n_results=top_k,
                    where=filter_metadata,
                )

                # Transform results to standard format
                output = []
                if results["ids"] and results["ids"][0]:
                    for i, doc_id in enumerate(results["ids"][0]):
                        record = {
                            "id": doc_id,
                            "text": results["documents"][0][i] if results["documents"] else "",
                            "score": 1.0 - results["distances"][0][i] if results["distances"] else 0.0,
                        }
                        if results["metadatas"] and results["metadatas"][0][i]:
                            record["metadata"] = results["metadatas"][0][i]
                        output.append(record)

                logger.debug(f"Query returned {len(output)} results from {collection}")
                span.set_attribute("results.count", len(output))

                return output

            except Exception as e:
                error_msg = f"Failed to query {collection}: {e}"
                logger.error(error_msg)
                span.record_exception(e)
                raise DatabaseError(error_msg) from e

    async def delete_collection(self, name: str) -> None:
        """Delete a collection and all its data.

        Args:
            name: Collection name

        Raises:
            DatabaseError: If deletion fails
        """
        with tracer.start_as_current_span("delete_collection") as span:
            span.set_attribute("collection.name", name)

            try:
                self._client.delete_collection(name=name)
                logger.info(f"Deleted collection: {name}")

            except Exception as e:
                error_msg = f"Failed to delete collection {name}: {e}"
                logger.error(error_msg)
                span.record_exception(e)
                raise DatabaseError(error_msg) from e

    async def get_collection_info(self, name: str) -> dict[str, Any]:
        """Get information about a collection.

        Args:
            name: Collection name

        Returns:
            Dictionary with collection metadata

        Raises:
            DatabaseError: If collection doesn't exist or info retrieval fails
        """
        try:
            col = self._client.get_collection(name=name)
            count = col.count()

            return {
                "name": name,
                "dimension": col.metadata.get("dimension", 0),
                "count": count,
                "metadata": col.metadata,
            }

        except Exception as e:
            error_msg = f"Failed to get info for collection {name}: {e}"
            logger.error(error_msg)
            raise DatabaseError(error_msg) from e

    @retry(max_attempts=5, initial_delay=1.0, backoff_multiplier=2.0)
    async def delete_all(self, collection: str) -> int:
        """Delete all records from a collection.

        Args:
            collection: Collection name

        Returns:
            Number of records deleted

        Raises:
            DatabaseError: If deletion fails
        """
        with tracer.start_as_current_span("database_delete_all") as span:
            span.set_attribute("collection.name", collection)

            try:
                col = self._client.get_collection(name=collection)

                # Get count before deletion
                count: int = col.count()

                if count == 0:
                    logger.debug(f"Collection {collection} is already empty")
                    span.set_attribute("records.deleted", 0)
                    return 0

                # Get all IDs
                results = col.get()
                ids = results["ids"]

                # Delete all records
                if ids:
                    col.delete(ids=ids)
                    logger.info(f"Deleted {count} records from {collection}")
                    span.set_attribute("records.deleted", count)
                    return count

                return 0

            except Exception as e:
                error_msg = f"Failed to delete all records from {collection}: {e}"
                logger.error(error_msg)
                span.record_exception(e)
                raise DatabaseError(error_msg) from e

    @retry(max_attempts=5, initial_delay=1.0, backoff_multiplier=2.0)
    async def delete_by_ids(self, collection: str, ids: list[str]) -> int:
        """Delete specific records by their IDs.

        Args:
            collection: Collection name
            ids: List of record IDs to delete

        Returns:
            Number of records deleted

        Raises:
            DatabaseError: If deletion fails
        """
        if not ids:
            return 0

        with tracer.start_as_current_span("database_delete_by_ids") as span:
            span.set_attribute("collection.name", collection)
            span.set_attribute("ids.count", len(ids))

            try:
                col = self._client.get_collection(name=collection)

                # Delete records by IDs
                col.delete(ids=ids)

                logger.info(f"Deleted {len(ids)} records from {collection}")
                span.set_attribute("records.deleted", len(ids))

                return len(ids)

            except Exception as e:
                error_msg = f"Failed to delete records from {collection}: {e}"
                logger.error(error_msg)
                span.record_exception(e)
                raise DatabaseError(error_msg) from e

    async def close(self) -> None:
        """Close the database connection and cleanup resources."""
        if self._http_client:
            await self._http_client.aclose()
            logger.debug("Closed HTTP client")
