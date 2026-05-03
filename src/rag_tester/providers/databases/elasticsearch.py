"""Elasticsearch vector database provider."""

import logging
import re
from typing import Any

from elasticsearch import AsyncElasticsearch
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

        logger.info(f"Elasticsearch mode: host={self._host}, port={self._port}, index={self._index_name}")

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
                logger.info(f"Connected to Elasticsearch at {self._host}:{self._port}")

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
                    logger.debug(f"Index already exists: {name}")
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

                logger.info(f"Index created: {name} (dimension: {dimension})")

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
            logger.error(f"Failed to check index existence: {e}")
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

                logger.debug(f"Query returned {len(output)} results from {collection}")
                span.set_attribute("results.count", len(output))

                return output

            except Exception as e:
                error_msg = f"Failed to query {collection}: {e}"
                logger.error(error_msg)
                span.record_exception(e)
                raise DatabaseError(error_msg) from e

    async def delete_collection(self, name: str) -> None:
        """Delete an index and all its data.

        Args:
            name: Index name

        Raises:
            DatabaseError: If deletion fails
        """
        with tracer.start_as_current_span("delete_collection") as span:
            span.set_attribute("collection.name", name)

            try:
                client = await self._get_client()

                if await self.collection_exists(name):
                    await client.indices.delete(index=name)
                    logger.info(f"Deleted index: {name}")
                else:
                    logger.debug(f"Index does not exist: {name}")

            except Exception as e:
                error_msg = f"Failed to delete index {name}: {e}"
                logger.error(error_msg)
                span.record_exception(e)
                raise DatabaseError(error_msg) from e

    async def get_collection_info(self, name: str) -> dict[str, Any]:
        """Get information about an index.

        Args:
            name: Index name

        Returns:
            Dictionary with index metadata

        Raises:
            DatabaseError: If index doesn't exist or info retrieval fails
        """
        try:
            if not await self.collection_exists(name):
                msg = f"Index does not exist: {name}"
                raise DatabaseError(msg)

            client = await self._get_client()

            # Get index mapping
            mapping_response = await client.indices.get_mapping(index=name)
            mapping = mapping_response[name]["mappings"]["properties"]

            # Get dimension from embedding field
            dimension = 0
            if "embedding" in mapping:
                dimension = mapping["embedding"].get("dims", 0)

            # Get document count
            count_response = await client.count(index=name)
            count = count_response["count"]

            # Get index settings
            settings_response = await client.indices.get_settings(index=name)
            settings = settings_response[name]["settings"]
            metadata = settings.get("index", {}).get("metadata", {})

            return {
                "name": name,
                "dimension": dimension,
                "count": count,
                "metadata": metadata,
            }

        except DatabaseError:
            raise
        except Exception as e:
            error_msg = f"Failed to get info for index {name}: {e}"
            logger.error(error_msg)
            raise DatabaseError(error_msg) from e

    @retry(max_attempts=5, initial_delay=1.0, backoff_multiplier=2.0)
    async def delete_all(self, collection: str) -> int:
        """Delete all records from an index.

        Args:
            collection: Index name

        Returns:
            Number of records deleted

        Raises:
            DatabaseError: If deletion fails
        """
        with tracer.start_as_current_span("database_delete_all") as span:
            span.set_attribute("collection.name", collection)

            try:
                client = await self._get_client()

                # Get count before deletion
                count_response = await client.count(index=collection)
                count: int = count_response["count"]

                if count == 0:
                    logger.debug(f"Index {collection} is already empty")
                    span.set_attribute("records.deleted", 0)
                    return 0

                # Delete all documents using delete_by_query
                await client.delete_by_query(
                    index=collection,
                    body={"query": {"match_all": {}}},
                    refresh=True,
                )

                logger.info(f"Deleted {count} records from {collection}")
                span.set_attribute("records.deleted", count)

                return count

            except Exception as e:
                error_msg = f"Failed to delete all records from {collection}: {e}"
                logger.error(error_msg)
                span.record_exception(e)
                raise DatabaseError(error_msg) from e

    @retry(max_attempts=5, initial_delay=1.0, backoff_multiplier=2.0)
    async def delete_by_ids(self, collection: str, ids: list[str]) -> int:
        """Delete specific records by their IDs.

        Args:
            collection: Index name
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
                client = await self._get_client()

                # Delete documents using delete_by_query with terms filter
                response = await client.delete_by_query(
                    index=collection,
                    body={"query": {"terms": {"_id": ids}}},
                    refresh=True,
                )

                deleted_count: int = response["deleted"]

                logger.info(f"Deleted {deleted_count} records from {collection}")
                span.set_attribute("records.deleted", deleted_count)

                return deleted_count

            except Exception as e:
                error_msg = f"Failed to delete records from {collection}: {e}"
                logger.error(error_msg)
                span.record_exception(e)
                raise DatabaseError(error_msg) from e

    async def close(self) -> None:
        """Close the Elasticsearch client."""
        if self._client:
            await self._client.close()
            logger.debug("Closed Elasticsearch connection")
