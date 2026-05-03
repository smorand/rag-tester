"""SQLite vector database provider with sqlite-vec extension."""

import logging
import re
import struct
from pathlib import Path
from typing import Any

import aiosqlite
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


class SQLiteProvider(VectorDatabase):
    """SQLite vector database provider with sqlite-vec extension.

    Connection string format: sqlite:///path/to/db.db/table_name

    Args:
        connection_string: SQLite connection string

    Raises:
        ConnectionError: If connection fails
        ValueError: If connection string is invalid
    """

    def __init__(self, connection_string: str) -> None:
        """Initialize SQLite provider.

        Args:
            connection_string: SQLite connection string
        """
        self._connection_string = connection_string
        self._conn: aiosqlite.Connection | None = None
        self._parse_connection_string()

    def _parse_connection_string(self) -> None:
        """Parse connection string to extract parameters.

        Raises:
            ValueError: If connection string format is invalid
        """
        if not self._connection_string.startswith("sqlite:///"):
            msg = "Invalid connection string: must start with 'sqlite:///'"
            raise ValueError(msg)

        # Format: sqlite:///path/to/db.db/table_name
        remainder = self._connection_string[len("sqlite:///") :]
        parts = remainder.rsplit("/", 1)

        if len(parts) != 2:
            msg = "Invalid SQLite connection string: must be sqlite:///path/to/db.db/table_name"
            raise ValueError(msg)

        db_path = parts[0]
        self._table_name = parts[1]

        # Validate table name (alphanumeric + underscore only)
        if not re.match(r"^[a-zA-Z0-9_]+$", self._table_name):
            msg = "Invalid table name: must be alphanumeric with underscores only"
            raise ValueError(msg)

        # Check for path traversal
        if ".." in db_path:
            msg = "Invalid database path: path traversal detected"
            raise ValueError(msg)

        self._db_path = Path(db_path)

        logger.info(f"SQLite mode: path={self._db_path}, table={self._table_name}")

    async def _get_connection(self) -> aiosqlite.Connection:
        """Get or create database connection.

        Returns:
            Active database connection

        Raises:
            ConnectionError: If connection fails
        """
        if self._conn is None:
            try:
                # Create parent directory if it doesn't exist
                self._db_path.parent.mkdir(parents=True, exist_ok=True)

                self._conn = await aiosqlite.connect(str(self._db_path))

                # Enable foreign keys
                await self._conn.execute("PRAGMA foreign_keys = ON")

                # Try to load sqlite-vec extension
                try:
                    await self._conn.enable_load_extension(True)
                    # Try common extension names
                    for ext_name in ["vec0", "vector0", "sqlite_vec"]:
                        try:
                            await self._conn.load_extension(ext_name)
                            logger.info(f"Loaded sqlite-vec extension: {ext_name}")
                            break
                        except Exception:
                            continue
                    await self._conn.enable_load_extension(False)
                except Exception as e:
                    logger.warning(f"Could not load sqlite-vec extension: {e}")
                    logger.warning("Vector similarity search may not work correctly")

                logger.info(f"Connected to SQLite database: {self._db_path}")

            except Exception as e:
                error_msg = f"Failed to connect to SQLite: {e}"
                logger.error(error_msg)
                raise ConnectionError(error_msg) from e

        return self._conn

    @retry(max_attempts=5, initial_delay=1.0, backoff_multiplier=2.0)
    async def create_collection(self, name: str, dimension: int, metadata: dict[str, Any] | None = None) -> None:
        """Create a table for embeddings.

        Args:
            name: Table name (must match table_name from connection string)
            dimension: Embedding dimension
            metadata: Optional metadata (not used in SQLite)

        Raises:
            DatabaseError: If table creation fails
        """
        _ = metadata  # Unused parameter for interface compatibility
        with tracer.start_as_current_span("create_collection") as span:
            span.set_attribute("collection.name", name)
            span.set_attribute("collection.dimension", dimension)

            try:
                # Validate table name matches connection string
                if name != self._table_name:
                    msg = f"Table name mismatch: expected {self._table_name}, got {name}"
                    raise ValueError(msg)

                # Check if table already exists
                if await self.collection_exists(name):
                    logger.debug(f"Table already exists: {name}")
                    return

                conn = await self._get_connection()

                # Create table with BLOB column for embeddings
                await conn.execute(
                    f"""
                    CREATE TABLE {name} (
                        id TEXT PRIMARY KEY,
                        text TEXT NOT NULL,
                        embedding BLOB NOT NULL,
                        dimension INTEGER NOT NULL,
                        metadata TEXT
                    )
                    """
                )

                # Create index on id for faster lookups
                await conn.execute(f"CREATE INDEX idx_{name}_id ON {name}(id)")

                await conn.commit()
                logger.info(f"Table created: {name} (dimension: {dimension})")

            except Exception as e:
                error_msg = f"Failed to create table {name}: {e}"
                logger.error(error_msg)
                span.record_exception(e)
                raise DatabaseError(error_msg) from e

    async def collection_exists(self, name: str) -> bool:
        """Check if a table exists.

        Args:
            name: Table name

        Returns:
            True if table exists, False otherwise
        """
        try:
            conn = await self._get_connection()
            async with conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                (name,),
            ) as cursor:
                result = await cursor.fetchone()
                return result is not None

        except Exception as e:
            logger.error(f"Failed to check table existence: {e}")
            return False

    def _serialize_embedding(self, embedding: list[float]) -> bytes:
        """Serialize embedding to bytes.

        Args:
            embedding: Embedding vector

        Returns:
            Serialized embedding as bytes
        """
        # Pack as array of floats (little-endian)
        return struct.pack(f"<{len(embedding)}f", *embedding)

    def _deserialize_embedding(self, data: bytes) -> list[float]:
        """Deserialize embedding from bytes.

        Args:
            data: Serialized embedding

        Returns:
            Embedding vector
        """
        # Unpack array of floats (little-endian)
        count = len(data) // 4  # 4 bytes per float
        return list(struct.unpack(f"<{count}f", data))

    @retry(max_attempts=5, initial_delay=1.0, backoff_multiplier=2.0)
    async def insert(self, collection: str, records: list[dict[str, Any]]) -> None:
        """Insert records with embeddings.

        Args:
            collection: Table name
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
                # Validate table name
                if collection != self._table_name:
                    msg = f"Table name mismatch: expected {self._table_name}, got {collection}"
                    raise ValueError(msg)

                # Get or create table
                if not await self.collection_exists(collection):
                    dimension = len(records[0]["embedding"])
                    await self.create_collection(collection, dimension)

                # Verify dimension compatibility
                info = await self.get_collection_info(collection)
                expected_dimension = info["dimension"]
                actual_dimension = len(records[0]["embedding"])
                if expected_dimension > 0 and actual_dimension != expected_dimension:
                    msg = f"Dimension mismatch: model={actual_dimension}, database={expected_dimension}"
                    raise DimensionMismatchError(msg)

                conn = await self._get_connection()

                # Insert records in batches
                batch_size = 1000
                total_inserted = 0

                for i in range(0, len(records), batch_size):
                    batch = records[i : i + batch_size]

                    for record in batch:
                        # Serialize embedding
                        embedding_bytes = self._serialize_embedding(record["embedding"])
                        dimension = len(record["embedding"])

                        # Serialize metadata if present
                        metadata_str = None
                        if record.get("metadata"):
                            import json

                            metadata_str = json.dumps(record["metadata"])

                        # Insert or replace record
                        await conn.execute(
                            f"""
                            INSERT OR REPLACE INTO {collection}
                            (id, text, embedding, dimension, metadata)
                            VALUES (?, ?, ?, ?, ?)
                            """,
                            (
                                record["id"],
                                record["text"],
                                embedding_bytes,
                                dimension,
                                metadata_str,
                            ),
                        )

                    await conn.commit()
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

    def _cosine_similarity(self, vec1: list[float], vec2: list[float]) -> float:
        """Calculate cosine similarity between two vectors.

        Args:
            vec1: First vector
            vec2: Second vector

        Returns:
            Cosine similarity score (0 to 1)
        """
        import math

        # Calculate dot product
        dot_product = sum(a * b for a, b in zip(vec1, vec2, strict=True))

        # Calculate magnitudes
        magnitude1 = math.sqrt(sum(a * a for a in vec1))
        magnitude2 = math.sqrt(sum(b * b for b in vec2))

        # Avoid division by zero
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0

        # Calculate cosine similarity
        similarity = dot_product / (magnitude1 * magnitude2)

        # Normalize to [0, 1] range
        return (similarity + 1) / 2

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
            collection: Table name
            query_embedding: Query embedding vector
            top_k: Number of results to return
            filter_metadata: Optional metadata filters (JSON matching)

        Returns:
            List of records sorted by similarity (descending)

        Raises:
            DatabaseError: If query fails
        """
        with tracer.start_as_current_span("database_search") as span:
            span.set_attribute("collection.name", collection)
            span.set_attribute("query.top_k", top_k)

            try:
                conn = await self._get_connection()

                # Fetch all records (SQLite doesn't have native vector search without extension)
                query = f"SELECT id, text, embedding, metadata FROM {collection}"
                params: tuple[str, ...] = ()

                if filter_metadata:
                    # Add metadata filter
                    import json

                    filter_json = json.dumps(filter_metadata)
                    query += " WHERE metadata LIKE ?"
                    params = (f"%{filter_json}%",)

                async with conn.execute(query, params) as cursor:
                    rows = await cursor.fetchall()

                # Calculate similarity for each record
                results = []
                for row in rows:
                    record_id, text, embedding_bytes, metadata_str = row

                    # Deserialize embedding
                    embedding = self._deserialize_embedding(embedding_bytes)

                    # Calculate cosine similarity
                    score = self._cosine_similarity(query_embedding, embedding)

                    record = {
                        "id": record_id,
                        "text": text,
                        "score": score,
                    }

                    if metadata_str:
                        import contextlib
                        import json

                        with contextlib.suppress(json.JSONDecodeError):
                            record["metadata"] = json.loads(metadata_str)

                    results.append(record)

                # Sort by similarity (descending) and take top_k
                results.sort(key=lambda x: x["score"], reverse=True)
                output = results[:top_k]

                logger.debug(f"Query returned {len(output)} results from {collection}")
                span.set_attribute("results.count", len(output))

                return output

            except Exception as e:
                error_msg = f"Failed to query {collection}: {e}"
                logger.error(error_msg)
                span.record_exception(e)
                raise DatabaseError(error_msg) from e

    async def delete_collection(self, name: str) -> None:
        """Delete a table and all its data.

        Args:
            name: Table name

        Raises:
            DatabaseError: If deletion fails
        """
        with tracer.start_as_current_span("delete_collection") as span:
            span.set_attribute("collection.name", name)

            try:
                conn = await self._get_connection()

                await conn.execute(f"DROP TABLE IF EXISTS {name}")
                await conn.commit()

                logger.info(f"Deleted table: {name}")

            except Exception as e:
                error_msg = f"Failed to delete table {name}: {e}"
                logger.error(error_msg)
                span.record_exception(e)
                raise DatabaseError(error_msg) from e

    async def get_collection_info(self, name: str) -> dict[str, Any]:
        """Get information about a table.

        Args:
            name: Table name

        Returns:
            Dictionary with table metadata

        Raises:
            DatabaseError: If table doesn't exist or info retrieval fails
        """
        try:
            conn = await self._get_connection()

            # Get dimension from first record
            async with conn.execute(f"SELECT dimension FROM {name} LIMIT 1") as cursor:
                dim_result = await cursor.fetchone()
                dimension = dim_result[0] if dim_result else 0

            # Get row count
            async with conn.execute(f"SELECT COUNT(*) FROM {name}") as cursor:
                count_result = await cursor.fetchone()
                count = count_result[0] if count_result else 0

            return {
                "name": name,
                "dimension": dimension,
                "count": count,
                "metadata": {},
            }

        except Exception as e:
            error_msg = f"Failed to get info for table {name}: {e}"
            logger.error(error_msg)
            raise DatabaseError(error_msg) from e

    @retry(max_attempts=5, initial_delay=1.0, backoff_multiplier=2.0)
    async def delete_all(self, collection: str) -> int:
        """Delete all records from a table.

        Args:
            collection: Table name

        Returns:
            Number of records deleted

        Raises:
            DatabaseError: If deletion fails
        """
        with tracer.start_as_current_span("database_delete_all") as span:
            span.set_attribute("collection.name", collection)

            try:
                conn = await self._get_connection()

                # Get count before deletion
                async with conn.execute(f"SELECT COUNT(*) FROM {collection}") as cursor:
                    count_result = await cursor.fetchone()
                    count = count_result[0] if count_result else 0

                if count == 0:
                    logger.debug(f"Table {collection} is already empty")
                    span.set_attribute("records.deleted", 0)
                    return 0

                # Delete all records
                await conn.execute(f"DELETE FROM {collection}")
                await conn.commit()

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
            collection: Table name
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
                conn = await self._get_connection()

                # Build placeholders for IN clause
                placeholders = ",".join("?" * len(ids))

                await conn.execute(
                    f"DELETE FROM {collection} WHERE id IN ({placeholders})",
                    ids,
                )
                await conn.commit()

                # Get number of deleted rows
                deleted_count = conn.total_changes

                logger.info(f"Deleted {deleted_count} records from {collection}")
                span.set_attribute("records.deleted", deleted_count)

                return deleted_count

            except Exception as e:
                error_msg = f"Failed to delete records from {collection}: {e}"
                logger.error(error_msg)
                span.record_exception(e)
                raise DatabaseError(error_msg) from e

    async def close(self) -> None:
        """Close the database connection."""
        if self._conn:
            await self._conn.close()
            logger.debug("Closed SQLite connection")
