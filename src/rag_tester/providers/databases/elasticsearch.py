"""Elasticsearch vector database provider."""

import logging
import re
from typing import Any

from elasticsearch import AsyncElasticsearch
from opentelemetry import trace

from rag_tester.providers.databases._elasticsearch_helpers import (
    delete_all_documents,
    delete_documents_by_ids,
    drop_index_if_exists,
    fetch_index_info,
)
from rag_tester.providers.databases.base import (
    ConnectionError,
    DatabaseError,
    DimensionMismatchError,
    VectorDatabase,
)
from rag_tester.utils.retry import retry_with_backoff as retry

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)


class ElasticsearchProvider(VectorDatabase):
    """Elasticsearch vector database provider.

    Connection string format: elasticsearch://host:port/index_name

    Args:
        connection_string: Elasticsearch connection string

    Raises:
        ConnectionError: If connection fails
        ValueError: If connection string is invalid
    """

    def __init__(self, connection_string: str) -> None:
        """Initialize Elasticsearch provider.

        Args:
            connection_string: Elasticsearch connection string
        """
        self._connection_string = connection_string
        self._client: AsyncElasticsearch | None = None
        self._parse_connection_string()

    def _parse_connection_string(self) -> None:
        """Parse connection string to extract parameters.

        Raises:
            ValueError: If connection string format is invalid
        """
        if not self._connection_string.startswith("elasticsearch://"):
            msg = "Invalid connection string: must start with 'elasticsearch://'"
            raise ValueError(msg)

        # Format: elasticsearch://host:port/index_name
        pattern = r"^([^:]+):(\d+)/(.+)$"
        remainder = self._connection_string[len("elasticsearch://") :]
        match = re.match(pattern, remainder)

        if not match:
            msg = "Invalid Elasticsearch connection string: must be elasticsearch://host:port/index_name"
            raise ValueError(msg)

        self._host = match.group(1)
        self._port = int(match.group(2))
        self._index_name = match.group(3)

        # Validate index name (lowercase alphanumeric + underscore + hyphen)
        if not re.match(r"^[a-z0-9_-]+$", self._index_name):
            msg = "Invalid index name: must be lowercase alphanumeric with underscores and hyphens only"
            raise ValueError(msg)

        logger.info("Elasticsearch mode: host=%s, port=%s, index=%s", self._host, self._port, self._index_name)

    async def _get_client(self) -> AsyncElasticsearch:
        """Get or create Elasticsearch client.

        Returns:
            Active Elasticsearch client

        Raises:
            ConnectionError: If connection fails
        """
        if self._client is None:
            try:
                self._client = AsyncElasticsearch(
                    hosts=[{"host": self._host, "port": self._port, "scheme": "http"}],
                    request_timeout=30,
                    max_retries=3,
                    retry_on_timeout=True,
                )

                # Test connection
                await self._client.ping()
                logger.info("Connected to Elasticsearch at %s:%s", self._host, self._port)

            except Exception as e:
                error_msg = f"Failed to connect to Elasticsearch: {e}"
                logger.error(error_msg)
                raise ConnectionError(error_msg) from e

        return self._client

    @retry(max_attempts=5, initial_delay=1.0, backoff_multiplier=2.0)
    async def create_collection(self, name: str, dimension: int, metadata: dict[str, Any] | None = None) -> None:
        """Create an index for embeddings.

        Args:
            name: Index name
            dimension: Embedding dimension
            metadata: Optional index metadata

        Raises:
            DatabaseError: If index creation fails
        """
        with tracer.start_as_current_span("create_collection") as span:
            span.set_attribute("collection.name", name)
            span.set_attribute("collection.dimension", dimension)

            try:
                # Check if index already exists
                if await self.collection_exists(name):
                    logger.debug("Index already exists: %s", name)
                    return

                client = await self._get_client()

                # Define index mapping with dense_vector field
                mapping = {
                    "mappings": {
                        "properties": {
                            "id": {"type": "keyword"},
                            "text": {"type": "text"},
                            "embedding": {
                                "type": "dense_vector",
                                "dims": dimension,
                                "index": True,
                                "similarity": "cosine",
                            },
                            "metadata": {"type": "object", "enabled": True},
                        }
                    },
                    "settings": {
                        "number_of_shards": 1,
                        "number_of_replicas": 0,
                    },
                }

                # Add metadata to settings if provided
                if metadata:
                    mapping["settings"]["index"] = {"metadata": metadata}  # type: ignore[index]

                # Create index
                await client.indices.create(index=name, body=mapping)

                logger.info("Index created: %s (dimension: %s)", name, dimension)

            except Exception as e:
                error_msg = f"Failed to create index {name}: {e}"
                logger.error(error_msg)
                span.record_exception(e)
                raise DatabaseError(error_msg) from e

    async def collection_exists(self, name: str) -> bool:
        """Check if an index exists.

        Args:
            name: Index name

        Returns:
            True if index exists, False otherwise
        """
        try:
            client = await self._get_client()
            result: bool = await client.indices.exists(index=name)  # type: ignore[assignment]
            return result
        except Exception as e:
            logger.error("Failed to check index existence: %s", e)
            return False

    @retry(max_attempts=5, initial_delay=1.0, backoff_multiplier=2.0)
    async def insert(self, collection: str, records: list[dict[str, Any]]) -> None:
        """Insert records with embeddings.

        Args:
            collection: Index name
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
                # Get or create index
                if not await self.collection_exists(collection):
                    # Auto-create with dimension from first record
                    dimension = len(records[0]["embedding"])
                    await self.create_collection(collection, dimension)

                # Verify dimension compatibility
                info = await self.get_collection_info(collection)
                expected_dimension = info["dimension"]
                actual_dimension = len(records[0]["embedding"])
                if actual_dimension != expected_dimension:
                    msg = f"Dimension mismatch: model={actual_dimension}, database={expected_dimension}"
                    raise DimensionMismatchError(msg)

                client = await self._get_client()

                # Insert records in batches using bulk API
                batch_size = 1000
                total_inserted = 0

                for i in range(0, len(records), batch_size):
                    batch = records[i : i + batch_size]

                    # Prepare bulk operations
                    operations = []
                    for record in batch:
                        # Index operation
                        operations.append({"index": {"_index": collection, "_id": record["id"]}})

                        # Document
                        doc = {
                            "id": record["id"],
                            "text": record["text"],
                            "embedding": record["embedding"],
                        }
                        if record.get("metadata"):
                            doc["metadata"] = record["metadata"]

                        operations.append(doc)

                    # Execute bulk operation
                    response = await client.bulk(operations=operations, refresh=True)

                    # Check for errors
                    if response.get("errors"):
                        error_items = [item for item in response["items"] if "error" in item.get("index", {})]
                        if error_items:
                            error_msg = f"Bulk insert errors: {error_items[:5]}"  # Show first 5 errors
                            raise DatabaseError(error_msg)

                    total_inserted += len(batch)
                    logger.debug("Inserted batch of %s records (%s/%s)", len(batch), total_inserted, len(records))

                logger.info("Inserted %s records into %s", total_inserted, collection)
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
        """Query for similar embeddings using kNN search.

        Args:
            collection: Index name
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
                client = await self._get_client()

                # Build kNN query
                knn_query = {
                    "field": "embedding",
                    "query_vector": query_embedding,
                    "k": top_k,
                    "num_candidates": top_k * 10,  # Oversample for better results
                }

                # Add filter if provided
                if filter_metadata:
                    knn_query["filter"] = {"term": {"metadata": filter_metadata}}

                # Execute kNN search
                response = await client.search(
                    index=collection,
                    knn=knn_query,
                    source=["id", "text", "metadata"],  # type: ignore[arg-type]
                )

                # Transform results to standard format
                output = []
                for hit in response["hits"]["hits"]:
                    source = hit["_source"]
                    record = {
                        "id": source["id"],
                        "text": source["text"],
                        "score": float(hit["_score"]),
                    }
                    if source.get("metadata"):
                        record["metadata"] = source["metadata"]
                    output.append(record)

                logger.debug("Query returned %s results from %s", len(output), collection)
                span.set_attribute("results.count", len(output))

                return output

            except Exception as e:
                error_msg = f"Failed to query {collection}: {e}"
                logger.error(error_msg)
                span.record_exception(e)
                raise DatabaseError(error_msg) from e

    async def delete_collection(self, name: str) -> None:
        """Delete the index if it exists. ``DatabaseError`` on failure."""
        await drop_index_if_exists(await self._get_client(), name)

    async def get_collection_info(self, name: str) -> dict[str, Any]:
        """Return ``{name, dimension, count, metadata}`` for the index."""
        return await fetch_index_info(await self._get_client(), name)

    @retry(max_attempts=5, initial_delay=1.0, backoff_multiplier=2.0)
    async def delete_all(self, collection: str) -> int:
        """Delete every document in the index."""
        return await delete_all_documents(await self._get_client(), collection)

    @retry(max_attempts=5, initial_delay=1.0, backoff_multiplier=2.0)
    async def delete_by_ids(self, collection: str, ids: list[str]) -> int:
        """Delete documents whose ``_id`` is in ``ids``."""
        if not ids:
            return 0
        return await delete_documents_by_ids(await self._get_client(), collection, ids)

    async def close(self) -> None:
        """Close the Elasticsearch client."""
        if self._client:
            await self._client.close()
            logger.debug("Closed Elasticsearch connection")
