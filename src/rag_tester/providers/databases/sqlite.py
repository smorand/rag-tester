"""SQLite vector database provider with sqlite-vec extension."""

import contextlib
import json
import logging
import re
from pathlib import Path
from typing import Any

import aiosqlite
from opentelemetry import trace

from rag_tester.providers.databases._sqlite_helpers import (
    cosine_similarity,
    delete_all_rows,
    delete_rows_by_ids,
    deserialize_embedding,
    drop_table,
    get_table_info,
    serialize_embedding,
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

        logger.info("SQLite mode: path=%s, table=%s", self._db_path, self._table_name)

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
                            logger.info("Loaded sqlite-vec extension: %s", ext_name)
                            break
                        except Exception:  # nosec B112  # fallback intentionnel pour extension sqlite-vec optionnelle
                            continue
                    await self._conn.enable_load_extension(False)
                except Exception as e:
                    logger.warning("Could not load sqlite-vec extension: %s", e)
                    logger.warning("Vector similarity search may not work correctly")

                logger.info("Connected to SQLite database: %s", self._db_path)

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
                    logger.debug("Table already exists: %s", name)
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
                logger.info("Table created: %s (dimension: %s)", name, dimension)

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
            logger.error("Failed to check table existence: %s", e)
            return False

    # Compatibility shims so existing tests can keep using the bound-method form.
    _serialize_embedding = staticmethod(serialize_embedding)
    _deserialize_embedding = staticmethod(deserialize_embedding)
    _cosine_similarity = staticmethod(cosine_similarity)

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
                        embedding_bytes = serialize_embedding(record["embedding"])
                        dimension = len(record["embedding"])

                        # Serialize metadata if present
                        metadata_str = None
                        if record.get("metadata"):
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

                # Fetch all records (SQLite doesn't have native vector search without extension).
                # Note: collection name is validated by regex ^[a-zA-Z0-9_]+$ at construction (line 69).
                query = f"SELECT id, text, embedding, metadata FROM {collection}"  # nosec B608
                params: tuple[str, ...] = ()

                if filter_metadata:
                    # Add metadata filter
                    filter_json = json.dumps(filter_metadata)
                    query += " WHERE metadata LIKE ?"
                    params = (f"%{filter_json}%",)

                async with conn.execute(query, params) as cursor:
                    rows = await cursor.fetchall()

                # Calculate similarity for each record
                results = []
                for row in rows:
                    record_id, text, embedding_bytes, metadata_str = row
                    embedding = deserialize_embedding(embedding_bytes)
                    score = cosine_similarity(query_embedding, embedding)

                    record = {
                        "id": record_id,
                        "text": text,
                        "score": score,
                    }

                    if metadata_str:
                        with contextlib.suppress(json.JSONDecodeError):
                            record["metadata"] = json.loads(metadata_str)

                    results.append(record)

                # Sort by similarity (descending) and take top_k
                results.sort(key=lambda x: x["score"], reverse=True)
                output = results[:top_k]

                logger.debug("Query returned %s results from %s", len(output), collection)
                span.set_attribute("results.count", len(output))

                return output

            except Exception as e:
                error_msg = f"Failed to query {collection}: {e}"
                logger.error(error_msg)
                span.record_exception(e)
                raise DatabaseError(error_msg) from e

    async def delete_collection(self, name: str) -> None:
        """Drop the table backing the collection. Raises ``DatabaseError`` on failure."""
        await drop_table(await self._get_connection(), name)

    async def get_collection_info(self, name: str) -> dict[str, Any]:
        """Return ``{name, dimension, count, metadata}`` for the table."""
        return await get_table_info(await self._get_connection(), name)

    @retry(max_attempts=5, initial_delay=1.0, backoff_multiplier=2.0)
    async def delete_all(self, collection: str) -> int:
        """Delete every row from the table; returns the count that was deleted."""
        return await delete_all_rows(await self._get_connection(), collection)

    @retry(max_attempts=5, initial_delay=1.0, backoff_multiplier=2.0)
    async def delete_by_ids(self, collection: str, ids: list[str]) -> int:
        """Delete rows whose ``id`` is in ``ids``; returns the affected rowcount."""
        return await delete_rows_by_ids(await self._get_connection(), collection, ids)

    async def close(self) -> None:
        """Close the database connection if it is still open."""
        if self._conn:
            await self._conn.close()
            logger.debug("Closed SQLite connection")
