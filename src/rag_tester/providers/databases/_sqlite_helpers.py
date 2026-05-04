"""Pure helpers for the SQLite vector backend.

The math/serialization helpers are pure functions; the DDL helpers take an
``aiosqlite.Connection`` and operate against an already-validated table name.
Module-level so they can be unit-tested without instantiating a provider.
"""

from __future__ import annotations

import logging
import math
import struct
from typing import TYPE_CHECKING, Any

from opentelemetry import trace

from rag_tester.providers.databases.base import DatabaseError

if TYPE_CHECKING:
    import aiosqlite

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)


def serialize_embedding(embedding: list[float]) -> bytes:
    """Pack an embedding vector as little-endian float32 bytes."""
    return struct.pack(f"<{len(embedding)}f", *embedding)


def deserialize_embedding(data: bytes) -> list[float]:
    """Unpack little-endian float32 bytes back into a list of floats."""
    count = len(data) // 4  # 4 bytes per float32
    return list(struct.unpack(f"<{count}f", data))


def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """Cosine similarity normalized to [0, 1] (1 = identical direction)."""
    dot_product = sum(a * b for a, b in zip(vec1, vec2, strict=True))
    magnitude1 = math.sqrt(sum(a * a for a in vec1))
    magnitude2 = math.sqrt(sum(b * b for b in vec2))
    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0
    similarity = dot_product / (magnitude1 * magnitude2)
    return (similarity + 1) / 2


# `name` and `collection` arguments below are pre-validated by the regex
# ^[a-zA-Z0-9_]+$ enforced at SQLiteProvider construction time, which is why
# direct interpolation into f-strings is safe (annotated # nosec B608).


async def get_table_info(conn: aiosqlite.Connection, name: str) -> dict[str, Any]:
    """Return ``{name, dimension, count, metadata}`` for an existing table."""
    try:
        async with conn.execute(f"SELECT dimension FROM {name} LIMIT 1") as cursor:  # nosec B608
            dim_result = await cursor.fetchone()
            dimension = dim_result[0] if dim_result else 0
        async with conn.execute(f"SELECT COUNT(*) FROM {name}") as cursor:  # nosec B608
            count_result = await cursor.fetchone()
            count = count_result[0] if count_result else 0
        return {"name": name, "dimension": dimension, "count": count, "metadata": {}}
    except Exception as e:
        error_msg = f"Failed to get info for table {name}: {e}"
        logger.error(error_msg)
        raise DatabaseError(error_msg) from e


async def delete_all_rows(conn: aiosqlite.Connection, collection: str) -> int:
    """``DELETE FROM <collection>`` with up-front count for the return value."""
    with tracer.start_as_current_span("database_delete_all") as span:
        span.set_attribute("collection.name", collection)
        try:
            async with conn.execute(f"SELECT COUNT(*) FROM {collection}") as cursor:  # nosec B608
                count_result = await cursor.fetchone()
                count = count_result[0] if count_result else 0
            if count == 0:
                span.set_attribute("records.deleted", 0)
                return 0
            await conn.execute(f"DELETE FROM {collection}")  # nosec B608
            await conn.commit()
            logger.info("Deleted %s records from %s", count, collection)
            span.set_attribute("records.deleted", count)
            return int(count)
        except Exception as e:
            error_msg = f"Failed to delete all records from {collection}: {e}"
            logger.error(error_msg)
            span.record_exception(e)
            raise DatabaseError(error_msg) from e


async def delete_rows_by_ids(conn: aiosqlite.Connection, collection: str, ids: list[str]) -> int:
    """``DELETE FROM <collection> WHERE id IN (?,?,...)`` returning rowcount."""
    if not ids:
        return 0
    with tracer.start_as_current_span("database_delete_by_ids") as span:
        span.set_attribute("collection.name", collection)
        span.set_attribute("ids.count", len(ids))
        try:
            placeholders = ",".join("?" * len(ids))
            cursor = await conn.execute(
                f"DELETE FROM {collection} WHERE id IN ({placeholders})",  # nosec B608
                ids,
            )
            deleted_count = cursor.rowcount
            await conn.commit()
            logger.info("Deleted %s records from %s", deleted_count, collection)
            span.set_attribute("records.deleted", deleted_count)
            return int(deleted_count)
        except Exception as e:
            error_msg = f"Failed to delete records from {collection}: {e}"
            logger.error(error_msg)
            span.record_exception(e)
            raise DatabaseError(error_msg) from e


async def drop_table(conn: aiosqlite.Connection, name: str) -> None:
    """``DROP TABLE IF EXISTS <name>`` and commit."""
    with tracer.start_as_current_span("delete_collection") as span:
        span.set_attribute("collection.name", name)
        try:
            await conn.execute(f"DROP TABLE IF EXISTS {name}")  # nosec B608
            await conn.commit()
            logger.info("Deleted table: %s", name)
        except Exception as e:
            error_msg = f"Failed to delete table {name}: {e}"
            logger.error(error_msg)
            span.record_exception(e)
            raise DatabaseError(error_msg) from e
