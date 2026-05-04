"""Helpers for the Milvus backend.

These functions take the Milvus globals (``Collection``, ``utility``) and an
``alias``, so the provider class stays focused on routing.
"""

from __future__ import annotations

import logging
from typing import Any

from opentelemetry import trace

from rag_tester.providers.databases.base import DatabaseError

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)


async def drop_collection_if_exists(
    name: str,
    *,
    alias: str,
    has_collection: Any,
    drop_collection: Any,
) -> None:
    """Drop the named collection if it exists; no-op otherwise."""
    with tracer.start_as_current_span("delete_collection") as span:
        span.set_attribute("collection.name", name)
        try:
            if has_collection(name, using=alias):
                drop_collection(name, using=alias)
                logger.info("Deleted collection: %s", name)
            else:
                logger.debug("Collection does not exist: %s", name)
        except Exception as e:
            error_msg = f"Failed to delete collection {name}: {e}"
            logger.error(error_msg)
            span.record_exception(e)
            raise DatabaseError(error_msg) from e


async def fetch_collection_info(
    name: str,
    *,
    alias: str,
    has_collection: Any,
    collection_cls: Any,
) -> dict[str, Any]:
    """Return ``{name, dimension, count, metadata}`` for the collection."""
    try:
        if not has_collection(name, using=alias):
            raise DatabaseError(f"Collection does not exist: {name}")
        col = collection_cls(name=name, using=alias)
        schema = col.schema
        embedding_field = next((f for f in schema.fields if f.name == "embedding"), None)
        dimension = embedding_field.params.get("dim", 0) if embedding_field else 0
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


async def delete_all_entities(collection: str, *, alias: str, collection_cls: Any) -> int:
    """Delete every entity (count taken before deletion)."""
    with tracer.start_as_current_span("database_delete_all") as span:
        span.set_attribute("collection.name", collection)
        try:
            col = collection_cls(name=collection, using=alias)
            col.flush()
            count: int = col.num_entities
            if count == 0:
                span.set_attribute("records.deleted", 0)
                return 0
            col.delete(expr="id != ''")
            col.flush()
            logger.info("Deleted %s records from %s", count, collection)
            span.set_attribute("records.deleted", count)
            return count
        except Exception as e:
            error_msg = f"Failed to delete all records from {collection}: {e}"
            logger.error(error_msg)
            span.record_exception(e)
            raise DatabaseError(error_msg) from e


async def delete_entities_by_ids(
    collection: str,
    ids: list[str],
    *,
    alias: str,
    collection_cls: Any,
) -> int:
    """``DELETE WHERE id in [...]`` returning the requested count."""
    if not ids:
        return 0
    with tracer.start_as_current_span("database_delete_by_ids") as span:
        span.set_attribute("collection.name", collection)
        span.set_attribute("ids.count", len(ids))
        try:
            col = collection_cls(name=collection, using=alias)
            ids_str = ", ".join(f'"{id_}"' for id_ in ids)
            col.delete(expr=f"id in [{ids_str}]")
            col.flush()
            logger.info("Deleted %s records from %s", len(ids), collection)
            span.set_attribute("records.deleted", len(ids))
            return len(ids)
        except Exception as e:
            error_msg = f"Failed to delete records from {collection}: {e}"
            logger.error(error_msg)
            span.record_exception(e)
            raise DatabaseError(error_msg) from e
