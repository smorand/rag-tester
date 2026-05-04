"""Helpers for the ChromaDB backend.

Functions take the chromadb client and operate on a single collection.
"""

from __future__ import annotations

import logging
from typing import Any

from opentelemetry import trace

from rag_tester.providers.databases.base import DatabaseError

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)


async def drop_collection(client: Any, name: str) -> None:
    """Delete a collection. ``DatabaseError`` on failure."""
    with tracer.start_as_current_span("delete_collection") as span:
        span.set_attribute("collection.name", name)
        try:
            client.delete_collection(name=name)
            logger.info("Deleted collection: %s", name)
        except Exception as e:
            error_msg = f"Failed to delete collection {name}: {e}"
            logger.error(error_msg)
            span.record_exception(e)
            raise DatabaseError(error_msg) from e


async def fetch_collection_info(client: Any, name: str) -> dict[str, Any]:
    """Return ``{name, dimension, count, metadata}`` for the collection."""
    try:
        col = client.get_collection(name=name)
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


async def delete_all_records(client: Any, collection: str) -> int:
    """Delete every record by listing IDs and bulk-deleting."""
    with tracer.start_as_current_span("database_delete_all") as span:
        span.set_attribute("collection.name", collection)
        try:
            col = client.get_collection(name=collection)
            count: int = col.count()
            if count == 0:
                span.set_attribute("records.deleted", 0)
                return 0
            results = col.get()
            ids = results["ids"]
            if ids:
                col.delete(ids=ids)
                logger.info("Deleted %s records from %s", count, collection)
                span.set_attribute("records.deleted", count)
                return count
            return 0
        except Exception as e:
            error_msg = f"Failed to delete all records from {collection}: {e}"
            logger.error(error_msg)
            span.record_exception(e)
            raise DatabaseError(error_msg) from e


async def delete_records_by_ids(client: Any, collection: str, ids: list[str]) -> int:
    """Delete records whose ID is in ``ids`` and return the requested count."""
    if not ids:
        return 0
    with tracer.start_as_current_span("database_delete_by_ids") as span:
        span.set_attribute("collection.name", collection)
        span.set_attribute("ids.count", len(ids))
        try:
            col = client.get_collection(name=collection)
            col.delete(ids=ids)
            logger.info("Deleted %s records from %s", len(ids), collection)
            span.set_attribute("records.deleted", len(ids))
            return len(ids)
        except Exception as e:
            error_msg = f"Failed to delete records from {collection}: {e}"
            logger.error(error_msg)
            span.record_exception(e)
            raise DatabaseError(error_msg) from e
