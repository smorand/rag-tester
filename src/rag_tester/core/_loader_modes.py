"""Mode-specific helpers for ``load_records`` (flush, upsert, initial)."""

from __future__ import annotations

import logging
from typing import Any

from opentelemetry import trace

from rag_tester.providers.databases.base import DatabaseError, VectorDatabase
from rag_tester.providers.embeddings.base import EmbeddingError, EmbeddingProvider

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)


async def handle_flush_mode(database: VectorDatabase, collection_name: str) -> int:
    """Delete every record in ``collection_name`` if the collection exists.

    Returns the number of records deleted (0 if the collection did not exist).
    """
    with tracer.start_as_current_span("flush_operation") as flush_span:
        if await database.collection_exists(collection_name):
            logger.info("Flush mode: deleting all records from %s", collection_name)
            deleted_count = await database.delete_all(collection_name)
            logger.info("Deleted %s records", deleted_count)
            flush_span.set_attribute("deleted_count", deleted_count)
            return deleted_count
        logger.info("Flush mode: collection %s does not exist, will create", collection_name)
        return 0


async def find_existing_ids(
    database: VectorDatabase,
    collection_name: str,
    records: list[dict[str, Any]],
    dimension: int,
) -> set[str]:
    """Return the subset of record IDs that already exist in the database.

    Done by issuing one zero-vector query per record with a metadata id-filter;
    this is a workaround because the public ``VectorDatabase`` API has no
    direct ``has_id`` method.
    """
    existing_ids: set[str] = set()
    try:
        for record in records:
            try:
                results = await database.query(
                    collection=collection_name,
                    query_embedding=[0.0] * dimension,
                    top_k=1,
                    filter_metadata={"id": record["id"]} if "id" in record else None,
                )
                if results and any(r["id"] == record["id"] for r in results):
                    existing_ids.add(record["id"])
            except Exception:  # nosec B110
                # If query fails, assume ID doesn't exist; falling back to insert.
                pass
    except Exception as e:
        logger.warning("Failed to check existing IDs, treating all as new: %s", e)
        return set()
    return existing_ids


def to_db_record(record: dict[str, Any], embedding: list[float]) -> dict[str, Any]:
    """Build the dict shape expected by ``VectorDatabase.insert``."""
    db_record: dict[str, Any] = {
        "id": record["id"],
        "text": record["text"],
        "embedding": embedding,
    }
    if "metadata" in record:
        db_record["metadata"] = record["metadata"]
    return db_record


async def handle_upsert_mode(
    database: VectorDatabase,
    embedding_provider: EmbeddingProvider,
    collection_name: str,
    records_to_process: list[dict[str, Any]],
    batch_size: int,
    force_reembed: bool,
    generate_embeddings: Any,
) -> tuple[int, int, int]:
    """Run the upsert flow.

    Returns ``(updated, inserted, failed)``.
    """
    with tracer.start_as_current_span("upsert_operation") as upsert_span:
        existing_ids = await find_existing_ids(
            database,
            collection_name,
            records_to_process,
            embedding_provider.get_dimension(),
        )
        logger.info("Found %s existing records in database", len(existing_ids))
        upsert_span.set_attribute("existing_count", len(existing_ids))

        records_to_update = [r for r in records_to_process if r["id"] in existing_ids]
        records_to_insert = [r for r in records_to_process if r["id"] not in existing_ids]
        logger.info("Upsert: %s updates, %s inserts", len(records_to_update), len(records_to_insert))

        updated = 0
        inserted = 0
        failed = 0

        if records_to_update:
            if force_reembed:
                logger.info(
                    "Force re-embed enabled: generating new embeddings for %s updates",
                    len(records_to_update),
                )
            else:
                logger.info("Generating embeddings for %s updates", len(records_to_update))

            update_texts = [r["text"] for r in records_to_update]
            update_embeddings = await generate_embeddings(
                texts=update_texts,
                provider=embedding_provider,
                batch_size=batch_size,
            )
            update_db_records = [to_db_record(r, e) for r, e in zip(records_to_update, update_embeddings, strict=True)]

            try:
                with tracer.start_as_current_span("database_update"):
                    await database.delete_by_ids(collection_name, [r["id"] for r in records_to_update])
                    await database.insert(collection_name, update_db_records)
                updated = len(update_db_records)
                logger.info("Successfully updated %s records", updated)
            except DatabaseError as e:
                logger.error("Database update failed: %s", e)
                failed += len(update_db_records)
                raise

        if records_to_insert:
            logger.info("Generating embeddings for %s new records", len(records_to_insert))
            insert_texts = [r["text"] for r in records_to_insert]
            insert_embeddings = await generate_embeddings(
                texts=insert_texts,
                provider=embedding_provider,
                batch_size=batch_size,
            )
            insert_db_records = [to_db_record(r, e) for r, e in zip(records_to_insert, insert_embeddings, strict=True)]

            try:
                with tracer.start_as_current_span("database_insert"):
                    await database.insert(collection_name, insert_db_records)
                inserted = len(insert_db_records)
                logger.info("Successfully inserted %s new records", inserted)
            except DatabaseError as e:
                logger.error("Database insertion failed: %s", e)
                failed += len(insert_db_records)
                raise

        return updated, inserted, failed


async def handle_initial_or_flush_insert(
    database: VectorDatabase,
    embedding_provider: EmbeddingProvider,
    collection_name: str,
    records_to_process: list[dict[str, Any]],
    batch_size: int,
    generate_embeddings: Any,
) -> tuple[int, int]:
    """Insert every record from scratch (initial or post-flush mode).

    Returns ``(loaded, failed)``. Re-raises ``EmbeddingError`` and ``DatabaseError``
    after recording stats so the caller can attach them to its span.
    """
    texts = [record["text"] for record in records_to_process]
    logger.info("Generating embeddings for %s texts (batch size: %s)", len(texts), batch_size)

    try:
        embeddings = await generate_embeddings(
            texts=texts,
            provider=embedding_provider,
            batch_size=batch_size,
        )
    except EmbeddingError:
        logger.error("Embedding generation failed")
        raise

    db_records = [to_db_record(r, e) for r, e in zip(records_to_process, embeddings, strict=True)]

    logger.info("Inserting %s records into database", len(db_records))
    try:
        with tracer.start_as_current_span(
            "database_insert",
            attributes={"collection.name": collection_name, "record.count": len(db_records)},
        ):
            await database.insert(collection_name, db_records)
        loaded = len(db_records)
        logger.info("Successfully loaded %s records", loaded)
        return loaded, 0
    except DatabaseError:
        logger.error("Database insertion failed")
        raise
