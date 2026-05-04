"""Helpers for the Elasticsearch backend.

Module-level coroutines that take an open ``AsyncElasticsearch`` client and
the pre-validated index name.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from opentelemetry import trace

from rag_tester.providers.databases.base import DatabaseError

if TYPE_CHECKING:
    from elasticsearch import AsyncElasticsearch

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)


async def drop_index_if_exists(client: AsyncElasticsearch, name: str) -> None:
    """Delete the Elasticsearch index if it exists; no-op otherwise."""
    with tracer.start_as_current_span("delete_collection") as span:
        span.set_attribute("collection.name", name)
        try:
            if await client.indices.exists(index=name):
                await client.indices.delete(index=name)
                logger.info("Deleted index: %s", name)
            else:
                logger.debug("Index does not exist: %s", name)
        except Exception as e:
            error_msg = f"Failed to delete index {name}: {e}"
            logger.error(error_msg)
            span.record_exception(e)
            raise DatabaseError(error_msg) from e


async def fetch_index_info(client: AsyncElasticsearch, name: str) -> dict[str, Any]:
    """Return ``{name, dimension, count, metadata}`` for the index."""
    try:
        if not await client.indices.exists(index=name):
            raise DatabaseError(f"Index does not exist: {name}")

        mapping_response = await client.indices.get_mapping(index=name)
        mapping = mapping_response[name]["mappings"]["properties"]
        dimension = mapping["embedding"].get("dims", 0) if "embedding" in mapping else 0

        count_response = await client.count(index=name)
        count = count_response["count"]

        settings_response = await client.indices.get_settings(index=name)
        settings = settings_response[name]["settings"]
        metadata = settings.get("index", {}).get("metadata", {})

        return {"name": name, "dimension": dimension, "count": count, "metadata": metadata}
    except DatabaseError:
        raise
    except Exception as e:
        error_msg = f"Failed to get info for index {name}: {e}"
        logger.error(error_msg)
        raise DatabaseError(error_msg) from e


async def delete_all_documents(client: AsyncElasticsearch, collection: str) -> int:
    """``delete_by_query match_all`` returning the count taken before deletion."""
    with tracer.start_as_current_span("database_delete_all") as span:
        span.set_attribute("collection.name", collection)
        try:
            count_response = await client.count(index=collection)
            count: int = count_response["count"]
            if count == 0:
                span.set_attribute("records.deleted", 0)
                return 0
            await client.delete_by_query(
                index=collection,
                body={"query": {"match_all": {}}},
                refresh=True,
            )
            logger.info("Deleted %s records from %s", count, collection)
            span.set_attribute("records.deleted", count)
            return count
        except Exception as e:
            error_msg = f"Failed to delete all records from {collection}: {e}"
            logger.error(error_msg)
            span.record_exception(e)
            raise DatabaseError(error_msg) from e


async def delete_documents_by_ids(client: AsyncElasticsearch, collection: str, ids: list[str]) -> int:
    """``delete_by_query terms _id`` returning ``response['deleted']``."""
    if not ids:
        return 0
    with tracer.start_as_current_span("database_delete_by_ids") as span:
        span.set_attribute("collection.name", collection)
        span.set_attribute("ids.count", len(ids))
        try:
            response = await client.delete_by_query(
                index=collection,
                body={"query": {"terms": {"_id": ids}}},
                refresh=True,
            )
            deleted_count: int = response["deleted"]
            logger.info("Deleted %s records from %s", deleted_count, collection)
            span.set_attribute("records.deleted", deleted_count)
            return deleted_count
        except Exception as e:
            error_msg = f"Failed to delete records from {collection}: {e}"
            logger.error(error_msg)
            span.record_exception(e)
            raise DatabaseError(error_msg) from e
