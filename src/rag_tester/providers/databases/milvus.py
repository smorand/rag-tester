"""Milvus vector database provider."""

import logging
import re
from typing import Any

from opentelemetry import trace
from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, connections, utility

from rag_tester.providers.databases.base import (
    ConnectionError,
    DatabaseError,
    DimensionMismatchError,
    VectorDatabase,
)
from rag_tester.utils.retry import retry_with_backoff as retry

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)


class MilvusProvider(VectorDatabase):
    """Milvus vector database provider.

    Connection string format: milvus://host:port/collection_name

    Args:
        connection_string: Milvus connection string

    Raises:
        ConnectionError: If connection fails
        ValueError: If connection string is invalid
    """

    def __init__(self, connection_string: str) -> None:
        """Initialize Milvus provider.

        Args:
            connection_string: Milvus connection string
        """
        self._connection_string = connection_string
        self._alias = "default"
        self._parse_connection_string()
        self._initialize_connection()

    def _parse_connection_string(self) -> None:
        """Parse connection string to extract parameters.

        Raises:
            ValueError: If connection string format is invalid
        """
        if not self._connection_string.startswith("milvus://"):
            msg = "Invalid connection string: must start with 'milvus://'"
            raise ValueError(msg)

        # Format: milvus://host:port/collection_name
        pattern = r"^([^:]+):(\d+)/(.+)$"
        remainder = self._connection_string[len("milvus://"):]
        match = re.match(pattern, remainder)

        if not match:
            msg = "Invalid Milvus connection string: must be milvus://host:port/collection_name"
            raise ValueError(msg)

        self._host = match.group(1)
        self._port = int(match.group(2))
        self._collection_name = match.group(3)

        # Validate collection name (alphanumeric + underscore only)
        if not re.match(r"^[a-zA-Z0-9_]+$", self._collection_name):
            msg = "Invalid collection name: must be alphanumeric with underscores only"
            raise ValueError(msg)

        logger.info(f"Milvus mode: host={self._host}, port={self._port}, collection={self._collection_name}")

    def _initialize_connection(self) -> None:
        """Initialize Milvus connection.

        Raises:
            ConnectionError: If connection fails
        """
        try:
            # Connect to Milvus
            connections.connect(
                alias=self._alias,
                host=self._host,
                port=str(self._port),
            )
            logger.info(f"Connected to Milvus at {self._host}:{self._port}")

        except Exception as e:
            error_msg = f"Failed to connect to Milvus: {e}"
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

                # Define collection schema
                fields = [
                    FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=512),
                    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
                    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dimension),
                ]

                # Add metadata field if needed
                if metadata:
                    fields.append(FieldSchema(name="metadata", dtype=DataType.JSON))

                schema = CollectionSchema(
                    fields=fields,
                    description=f"RAG collection with {dimension}-dimensional embeddings",
                )

                # Create collection
                collection = Collection(
                    name=name,
                    schema=schema,
                    using=self._alias,
                )

                # Create IVF_FLAT index for similarity search
                index_params = {
                    "metric_type": "COSINE",
                    "index_type": "IVF_FLAT",
                    "params": {"nlist": 128},
                }
                collection.create_index(field_name="embedding", index_params=index_params)

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
            return utility.has_collection(name, using=self._alias)
        except Exception as e:
            logger.error(f"Failed to check collection existence: {e}")
            return False

    @retry(max_attempts=5, initial_delay=1.0, backoff_multiplier=2.0)
    async def insert(self, collection: str, records: list[dict[str, Any]]) -> None:
        """Insert records with embeddings.

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
                    has_metadata = any("metadata" in record for record in records)
                    metadata_config = {"has_metadata": True} if has_metadata else None
                    await self.create_collection(collection, dimension, metadata_config)

                # Get collection
                col = Collection(name=collection, using=self._alias)

                # Verify dimension compatibility
                schema = col.schema
                embedding_field = next((f for f in schema.fields if f.name == "embedding"), None)
                if embedding_field:
                    expected_dimension = embedding_field.params.get("dim", 0)
                    actual_dimension = len(records[0]["embedding"])
                    if actual_dimension != expected_dimension:
                        msg = f"Dimension mismatch: model={actual_dimension}, database={expected_dimension}"
                        raise DimensionMismatchError(msg)

                # Prepare data for Milvus
                ids = [record["id"] for record in records]
                texts = [record["text"] for record in records]
                embeddings = [record["embedding"] for record in records]

                # Check if collection has metadata field
                has_metadata_field = any(f.name == "metadata" for f in schema.fields)

                if has_metadata_field:
                    metadatas = [record.get("metadata", {}) for record in records]
                    data = [ids, texts, embeddings, metadatas]
                else:
                    data = [ids, texts, embeddings]

                # Insert data in batches
                batch_size = 1000
                total_inserted = 0

                for i in range(0, len(records), batch_size):
                    batch_data = [field[i : i + batch_size] for field in data]
                    col.insert(batch_data)
                    total_inserted += len(batch_data[0])
                    logger.debug(f"Inserted batch of {len(batch_data[0])} records ({total_inserted}/{len(records)})")

                # Flush to ensure data is persisted
                col.flush()

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
        """Query for similar embeddings using cosine similarity.

        Args:
            collection: Collection name
            query_embedding: Query embedding vector
            top_k: Number of results to return
            filter_metadata: Optional metadata filters (JSON expressions)

        Returns:
            List of records sorted by similarity (descending)

        Raises:
            DatabaseError: If query fails
        """
        with tracer.start_as_current_span("database_search") as span:
            span.set_attribute("collection.name", collection)
            span.set_attribute("query.top_k", top_k)

            try:
                col = Collection(name=collection, using=self._alias)

                # Load collection into memory for search
                col.load()

                # Prepare search parameters
                search_params = {
                    "metric_type": "COSINE",
                    "params": {"nprobe": 10},
                }

                # Build filter expression if metadata filters provided
                expr = None
                if filter_metadata:
                    # Build JSON filter expression
                    conditions = []
                    for key, value in filter_metadata.items():
                        if isinstance(value, str):
                            conditions.append(f'metadata["{key}"] == "{value}"')
                        else:
                            conditions.append(f'metadata["{key}"] == {value}')
                    expr = " && ".join(conditions)

                # Determine output fields based on schema
                output_fields = ["id", "text"]
                schema = col.schema
                if any(f.name == "metadata" for f in schema.fields):
                    output_fields.append("metadata")

                # Search for similar vectors
                results = col.search(
                    data=[query_embedding],
                    anns_field="embedding",
                    param=search_params,
                    limit=top_k,
                    expr=expr,
                    output_fields=output_fields,
                )

                # Transform results to standard format
                output = []
                if results and len(results) > 0:
                    for hit in results[0]:
                        record = {
                            "id": hit.entity.get("id"),
                            "text": hit.entity.get("text"),
                            "score": float(hit.score),
                        }
                        if "metadata" in output_fields and hit.entity.get("metadata"):
                            record["metadata"] = hit.entity.get("metadata")
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
                if await self.collection_exists(name):
                    utility.drop_collection(name, using=self._alias)
                    logger.info(f"Deleted collection: {name}")
                else:
                    logger.debug(f"Collection does not exist: {name}")

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
            if not await self.collection_exists(name):
                msg = f"Collection does not exist: {name}"
                raise DatabaseError(msg)

            col = Collection(name=name, using=self._alias)

            # Get dimension from schema
            schema = col.schema
            embedding_field = next((f for f in schema.fields if f.name == "embedding"), None)
            dimension = embedding_field.params.get("dim", 0) if embedding_field else 0

            # Get count
            col.flush()
            count = col.num_entities

            return {
                "name": name,
                "dimension": dimension,
                "count": count,
                "metadata": {"description": schema.description},
            }

        except DatabaseError:
            raise
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
                col = Collection(name=collection, using=self._alias)

                # Get count before deletion
                col.flush()
                count = col.num_entities

                if count == 0:
                    logger.debug(f"Collection {collection} is already empty")
                    span.set_attribute("records.deleted", 0)
                    return 0

                # Delete all entities using expression (delete all)
                col.delete(expr="id != ''")
                col.flush()

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
                col = Collection(name=collection, using=self._alias)

                # Build delete expression for IDs
                # Milvus uses 'in' operator for multiple IDs
                ids_str = ", ".join(f'"{id_}"' for id_ in ids)
                expr = f"id in [{ids_str}]"

                # Delete records
                col.delete(expr=expr)
                col.flush()

                logger.info(f"Deleted {len(ids)} records from {collection}")
                span.set_attribute("records.deleted", len(ids))

                return len(ids)

            except Exception as e:
                error_msg = f"Failed to delete records from {collection}: {e}"
                logger.error(error_msg)
                span.record_exception(e)
                raise DatabaseError(error_msg) from e

    async def close(self) -> None:
        """Close the database connection."""
        try:
            connections.disconnect(alias=self._alias)
            logger.debug("Closed Milvus connection")
        except Exception as e:
            logger.warning(f"Error closing Milvus connection: {e}")
