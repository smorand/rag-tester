"""Helpers for the PostgreSQL/pgvector backend.

Module-level coroutines that take an open ``psycopg.AsyncConnection`` and a
pre-validated table name. They keep ``postgresql.py`` focused on the provider
class itself.
"""

from __future__ import annotations

import contextlib
import json
import logging
from typing import Any

import psycopg
from opentelemetry import trace
from psycopg import sql

from rag_tester.providers.databases.base import DatabaseError

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)


async def drop_table(conn: psycopg.AsyncConnection, name: str) -> None:
    """``DROP TABLE IF EXISTS <name> CASCADE`` (identifier quoted)."""
    with tracer.start_as_current_span("delete_collection") as span:
        span.set_attribute("collection.name", name)
        try:
            async with conn.cursor() as cur:
                drop_query = sql.SQL("DROP TABLE IF EXISTS {table} CASCADE").format(table=sql.Identifier(name))
                await cur.execute(drop_query)
            await conn.commit()
            logger.info("Deleted table: %s", name)
        except Exception as e:
            error_msg = f"Failed to delete table {name}: {e}"
            logger.error(error_msg)
            span.record_exception(e)
            raise DatabaseError(error_msg) from e


async def fetch_table_info(conn: psycopg.AsyncConnection, name: str) -> dict[str, Any]:
    """Return ``{name, dimension, count, metadata}`` for the table."""
    try:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                SELECT atttypmod - 4 AS dimension
                FROM pg_attribute
                WHERE attrelid = %s::regclass
                AND attname = 'embedding'
                """,
                (name,),
            )
            dim_result = await cur.fetchone()
            dimension = int(dim_result["dimension"]) if dim_result else 0  # type: ignore[call-overload]

            count_query = sql.SQL("SELECT COUNT(*) as count FROM {table}").format(table=sql.Identifier(name))
            await cur.execute(count_query)
            count_result = await cur.fetchone()
            count = int(count_result["count"]) if count_result else 0  # type: ignore[call-overload]

            await cur.execute(
                "SELECT obj_description(%s::regclass, 'pg_class') AS comment",
                (name,),
            )
            comment_result = await cur.fetchone()
            metadata: dict[str, Any] = {}
            if comment_result and comment_result["comment"]:  # type: ignore[call-overload]
                with contextlib.suppress(json.JSONDecodeError):
                    metadata = json.loads(str(comment_result["comment"]))  # type: ignore[call-overload]

            return {"name": name, "dimension": dimension, "count": count, "metadata": metadata}
    except Exception as e:
        error_msg = f"Failed to get info for table {name}: {e}"
        logger.error(error_msg)
        raise DatabaseError(error_msg) from e


async def delete_all_rows(conn: psycopg.AsyncConnection, collection: str) -> int:
    """``DELETE FROM <collection>``; returns the count taken before deletion."""
    with tracer.start_as_current_span("database_delete_all") as span:
        span.set_attribute("collection.name", collection)
        try:
            async with conn.cursor() as cur:
                count_query = sql.SQL("SELECT COUNT(*) as count FROM {table}").format(table=sql.Identifier(collection))
                await cur.execute(count_query)
                count_result = await cur.fetchone()
                count = int(count_result["count"]) if count_result else 0  # type: ignore[call-overload]
                if count == 0:
                    span.set_attribute("records.deleted", 0)
                    return 0
                delete_query = sql.SQL("DELETE FROM {table}").format(table=sql.Identifier(collection))
                await cur.execute(delete_query)
            await conn.commit()
            logger.info("Deleted %s records from %s", count, collection)
            span.set_attribute("records.deleted", count)
            return count
        except Exception as e:
            error_msg = f"Failed to delete all records from {collection}: {e}"
            logger.error(error_msg)
            span.record_exception(e)
            raise DatabaseError(error_msg) from e


async def delete_rows_by_ids(conn: psycopg.AsyncConnection, collection: str, ids: list[str]) -> int:
    """``DELETE FROM <collection> WHERE id = ANY(...)``; returns rowcount."""
    if not ids:
        return 0
    with tracer.start_as_current_span("database_delete_by_ids") as span:
        span.set_attribute("collection.name", collection)
        span.set_attribute("ids.count", len(ids))
        try:
            async with conn.cursor() as cur:
                delete_query = sql.SQL("DELETE FROM {table} WHERE id = ANY(%s)").format(
                    table=sql.Identifier(collection)
                )
                await cur.execute(delete_query, (ids,))
                deleted_count = cur.rowcount
            await conn.commit()
            logger.info("Deleted %s records from %s", deleted_count, collection)
            span.set_attribute("records.deleted", deleted_count)
            return int(deleted_count)
        except Exception as e:
            error_msg = f"Failed to delete records from {collection}: {e}"
            logger.error(error_msg)
            span.record_exception(e)
            raise DatabaseError(error_msg) from e
