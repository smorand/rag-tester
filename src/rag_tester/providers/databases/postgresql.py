"""PostgreSQL vector database provider with pgvector extension."""

import json
import logging
import re
from typing import Any

import psycopg
from opentelemetry import trace
from psycopg import sql
from psycopg.rows import dict_row

from rag_tester.providers.databases._postgresql_helpers import (
    delete_all_rows,
    delete_rows_by_ids,
    drop_table,
    fetch_table_info,
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


class PostgreSQLProvider(VectorDatabase):
    """PostgreSQL vector database provider with pgvector extension.

    Connection string format: postgresql://user:pass@host:port/dbname/table_name

    Args:
        connection_string: PostgreSQL connection string

    Raises:
        ConnectionError: If connection fails
        ValueError: If connection string is invalid
    """

    def __init__(self, connection_string: str) -> None:
        """Initialize PostgreSQL provider.

        Args:
            connection_string: PostgreSQL connection string
        """
        self._connection_string = connection_string
        self._conn: psycopg.AsyncConnection | None = None
        self._parse_connection_string()

    def _parse_connection_string(self) -> None:
        """Parse connection string to extract parameters.

        Raises:
            ValueError: If connection string format is invalid
        """
        if not self._connection_string.startswith("postgresql://"):
            msg = "Invalid connection string: must start with 'postgresql://'"
            raise ValueError(msg)

        # Format: postgresql://user:pass@host:port/dbname/table_name
        pattern = r"^postgresql://([^:]+):([^@]+)@([^:]+):(\d+)/([^/]+)/(.+)$"
        match = re.match(pattern, self._connection_string)

        if not match:
            msg = "Invalid PostgreSQL connection string: must be postgresql://user:pass@host:port/dbname/table_name"
            raise ValueError(msg)

        self._user = match.group(1)
        self._password = match.group(2)
        self._host = match.group(3)
        self._port = int(match.group(4))
        self._dbname = match.group(5)
        self._table_name = match.group(6)

        # Validate table name (alphanumeric + underscore only)
        if not re.match(r"^[a-zA-Z0-9_]+$", self._table_name):
            msg = "Invalid table name: must be alphanumeric with underscores only"
            raise ValueError(msg)

        # Build connection string without table name (for psycopg)
        self._psycopg_connstring = (
            f"postgresql://{self._user}:{self._password}@{self._host}:{self._port}/{self._dbname}"
        )

        logger.info(
            "PostgreSQL mode: host=%s, port=%s, dbname=%s, table=%s",
            self._host,
            self._port,
            self._dbname,
            self._table_name,
        )

    async def _get_connection(self) -> psycopg.AsyncConnection:
        """Get or create database connection.

        Returns:
            Active database connection

        Raises:
            ConnectionError: If connection fails
        """
        if self._conn is None or self._conn.closed:
            try:
                self._conn = await psycopg.AsyncConnection.connect(
                    self._psycopg_connstring,
                    row_factory=dict_row,  # type: ignore[arg-type]
                )
                # Enable pgvector extension
                async with self._conn.cursor() as cur:
                    await cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
                await self._conn.commit()
                logger.info("Connected to PostgreSQL with pgvector extension")
            except Exception as e:
                error_msg = f"Failed to connect to PostgreSQL: {e}"
                logger.error(error_msg)
                raise ConnectionError(error_msg) from e

        return self._conn

    @retry(max_attempts=5, initial_delay=1.0, backoff_multiplier=2.0)
    async def create_collection(self, name: str, dimension: int, metadata: dict[str, Any] | None = None) -> None:
        """Create a table for embeddings.

        Args:
            name: Table name (must match table_name from connection string)
            dimension: Embedding dimension
            metadata: Optional metadata (stored in table comment)

        Raises:
            DatabaseError: If table creation fails
        """
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

                # Create table with vector column
                async with conn.cursor() as cur:
                    # Use sql.Identifier for safe table name interpolation
                    create_table_query = sql.SQL(
                        """
                        CREATE TABLE {table} (
                            id TEXT PRIMARY KEY,
                            text TEXT NOT NULL,
                            embedding vector({dimension}) NOT NULL,
                            metadata JSONB
                        )
                        """
                    ).format(
                        table=sql.Identifier(name),
                        dimension=sql.Literal(dimension),
                    )
                    await cur.execute(create_table_query)

                    # Create IVFFlat index for similarity search
                    create_index_query = sql.SQL(
                        """
                        CREATE INDEX {index_name} ON {table}
                        USING ivfflat (embedding vector_cosine_ops)
                        WITH (lists = 100)
                        """
                    ).format(
                        index_name=sql.Identifier(f"{name}_embedding_idx"),
                        table=sql.Identifier(name),
                    )
                    await cur.execute(create_index_query)

                    # Store metadata in table comment if provided
                    if metadata:
                        comment_query = sql.SQL("COMMENT ON TABLE {table} IS {comment}").format(
                            table=sql.Identifier(name), comment=sql.Literal(json.dumps(metadata))
                        )
                        await cur.execute(comment_query)

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
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables
                        WHERE table_schema = 'public'
                        AND table_name = %s
                    )
                    """,
                    (name,),
                )
                result = await cur.fetchone()
                return bool(result["exists"]) if result else False  # type: ignore[call-overload]

        except Exception as e:
            logger.error("Failed to check table existence: %s", e)
            return False

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
                if actual_dimension != expected_dimension:
                    msg = f"Dimension mismatch: model={actual_dimension}, database={expected_dimension}"
                    raise DimensionMismatchError(msg)

                conn = await self._get_connection()

                # Insert records in batches
                batch_size = 1000
                total_inserted = 0

                for i in range(0, len(records), batch_size):
                    batch = records[i : i + batch_size]

                    async with conn.cursor() as cur:
                        # Use COPY for efficient bulk insert
                        insert_query = sql.SQL(
                            """
                            INSERT INTO {table} (id, text, embedding, metadata)
                            VALUES (%s, %s, %s, %s)
                            ON CONFLICT (id) DO UPDATE
                            SET text = EXCLUDED.text,
                                embedding = EXCLUDED.embedding,
                                metadata = EXCLUDED.metadata
                            """
                        ).format(table=sql.Identifier(collection))

                        for record in batch:
                            # Convert embedding to string format for pgvector
                            embedding_str = "[" + ",".join(str(x) for x in record["embedding"]) + "]"
                            metadata_json = record.get("metadata")

                            await cur.execute(
                                insert_query,
                                (
                                    record["id"],
                                    record["text"],
                                    embedding_str,
                                    metadata_json,
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
            filter_metadata: Optional metadata filters (JSONB queries)

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

                # Convert embedding to string format
                embedding_str = "[" + ",".join(str(x) for x in query_embedding) + "]"

                # Build query with optional metadata filter
                query_parts = [
                    sql.SQL("SELECT id, text, metadata, 1 - (embedding <=> %s::vector) AS score"),
                    sql.SQL("FROM {table}").format(table=sql.Identifier(collection)),
                ]

                params: list[Any] = [embedding_str]

                if filter_metadata:
                    # Add JSONB filter
                    query_parts.append(sql.SQL("WHERE metadata @> %s::jsonb"))
                    params.append(json.dumps(filter_metadata))

                query_parts.extend(
                    [
                        sql.SQL("ORDER BY embedding <=> %s::vector"),
                        sql.SQL("LIMIT %s"),
                    ]
                )
                params.extend([embedding_str, top_k])

                query = sql.SQL(" ").join(query_parts)

                async with conn.cursor() as cur:
                    await cur.execute(query, params)
                    results = await cur.fetchall()

                # Transform results to standard format
                output = []
                for row in results:
                    record = {
                        "id": str(row["id"]),  # type: ignore[call-overload]
                        "text": str(row["text"]),  # type: ignore[call-overload]
                        "score": float(row["score"]),  # type: ignore[call-overload]
                    }
                    if row["metadata"]:  # type: ignore[call-overload]
                        record["metadata"] = row["metadata"]  # type: ignore[call-overload]
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
        """Drop the table backing the collection. Raises ``DatabaseError`` on failure."""
        await drop_table(await self._get_connection(), name)

    async def get_collection_info(self, name: str) -> dict[str, Any]:
        """Return ``{name, dimension, count, metadata}`` for the table."""
        return await fetch_table_info(await self._get_connection(), name)

    @retry(max_attempts=5, initial_delay=1.0, backoff_multiplier=2.0)
    async def delete_all(self, collection: str) -> int:
        """Delete every row from the table; returns the count that was deleted."""
        return await delete_all_rows(await self._get_connection(), collection)

    @retry(max_attempts=5, initial_delay=1.0, backoff_multiplier=2.0)
    async def delete_by_ids(self, collection: str, ids: list[str]) -> int:
        """Delete rows whose ``id`` is in ``ids``; returns the affected rowcount."""
        if not ids:
            return 0
        return await delete_rows_by_ids(await self._get_connection(), collection, ids)

    async def close(self) -> None:
        """Close the database connection."""
        if self._conn and not self._conn.closed:
            await self._conn.close()
            logger.debug("Closed PostgreSQL connection")
