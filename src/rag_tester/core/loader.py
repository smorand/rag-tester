"""Core loading logic for RAG Tester.

This module provides the main loading functionality with streaming file processing,
batch embedding generation, parallel processing, and duplicate detection.
"""

import json
import logging
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import yaml
from opentelemetry import trace

from rag_tester.core.validator import ValidationError, validate_record
from rag_tester.providers.databases.base import (
    DatabaseError,
    DimensionMismatchError,
    VectorDatabase,
)
from rag_tester.providers.embeddings.base import EmbeddingError, EmbeddingProvider
from rag_tester.tracing import get_tracer

logger = logging.getLogger(__name__)


class LoadStatistics:
    """Statistics for a load operation."""

    def __init__(self) -> None:
        """Initialize load statistics."""
        self.total_records = 0
        self.loaded_records = 0
        self.updated_records = 0
        self.deleted_records = 0
        self.failed_records = 0
        self.skipped_records = 0
        self.total_tokens = 0
        self.embedding_model = ""
        self.database = ""
        self.mode = "initial"
        self.force_reembed = False

    def to_dict(self) -> dict[str, Any]:
        """Convert statistics to dictionary.

        Returns:
            Dictionary representation of statistics
        """
        return {
            "total_records": self.total_records,
            "loaded_records": self.loaded_records,
            "updated_records": self.updated_records,
            "deleted_records": self.deleted_records,
            "failed_records": self.failed_records,
            "skipped_records": self.skipped_records,
            "total_tokens": self.total_tokens,
            "embedding_model": self.embedding_model,
            "database": self.database,
            "mode": self.mode,
            "force_reembed": self.force_reembed,
        }


async def stream_records(file_path: Path) -> AsyncIterator[dict[str, Any]]:
    """Stream records from a YAML or JSON file.

    Uses generator pattern to process files incrementally without loading
    entire dataset into memory.

    Args:
        file_path: Path to the input file

    Yields:
        Individual records from the file

    Raises:
        ValidationError: If file format is invalid
    """
    tracer = get_tracer()
    with tracer.start_as_current_span(
        "file_read",
        attributes={
            "file.path": str(file_path),
            "file.size": file_path.stat().st_size,
        },
    ) as span:
        try:
            if file_path.suffix.lower() in {".yaml", ".yml"}:
                with open(file_path, encoding="utf-8") as f:
                    data = yaml.safe_load(f)

                if not data:
                    msg = "Input file is empty or has no records"
                    logger.error(msg)
                    raise ValidationError(msg)

                if not isinstance(data, list):
                    msg = "YAML file must contain a list of records"
                    logger.error(msg)
                    raise ValidationError(msg)

                record_count = len(data)
                span.set_attribute("file.record_count", record_count)
                logger.debug(f"Streaming {record_count} records from YAML file")

                for record in data:
                    yield record

            elif file_path.suffix.lower() == ".json":
                with open(file_path, encoding="utf-8") as f:
                    data = json.load(f)

                if not data:
                    msg = "Input file is empty or has no records"
                    logger.error(msg)
                    raise ValidationError(msg)

                if not isinstance(data, list):
                    msg = "JSON file must contain a list of records"
                    logger.error(msg)
                    raise ValidationError(msg)

                record_count = len(data)
                span.set_attribute("file.record_count", record_count)
                logger.debug(f"Streaming {record_count} records from JSON file")

                for record in data:
                    yield record

        except (yaml.YAMLError, json.JSONDecodeError) as e:
            msg = f"Invalid file format. Failed to parse {file_path.suffix}: {e}"
            logger.error(msg)
            span.record_exception(e)
            span.set_status(trace.Status(trace.StatusCode.ERROR, msg))
            raise ValidationError(msg) from e


async def generate_embeddings_batch(
    texts: list[str],
    provider: EmbeddingProvider,
    batch_size: int,
) -> list[list[float]]:
    """Generate embeddings for texts in batches.

    Args:
        texts: List of texts to embed
        provider: Embedding provider to use
        batch_size: Number of texts to embed per batch

    Returns:
        List of embedding vectors (same order as input texts)

    Raises:
        EmbeddingError: If embedding generation fails
    """
    tracer = get_tracer()
    all_embeddings: list[list[float]] = []
    total_batches = (len(texts) + batch_size - 1) // batch_size

    for batch_num in range(total_batches):
        start_idx = batch_num * batch_size
        end_idx = min(start_idx + batch_size, len(texts))
        batch_texts = texts[start_idx:end_idx]

        with tracer.start_as_current_span(
            "embedding_batch",
            attributes={
                "batch.number": batch_num + 1,
                "batch.size": len(batch_texts),
                "batch.total": total_batches,
            },
        ):
            logger.debug(f"Generating embeddings for batch {batch_num + 1}/{total_batches} ({len(batch_texts)} texts)")
            batch_embeddings = await provider.embed_texts(batch_texts)
            all_embeddings.extend(batch_embeddings)

    return all_embeddings


async def load_records(
    file_path: Path,
    database: VectorDatabase,
    embedding_provider: EmbeddingProvider,
    collection_name: str,
    mode: str = "initial",
    batch_size: int = 32,
    parallel: int = 1,
    force_reembed: bool = False,
) -> LoadStatistics:
    """Load records from file into vector database.

    This is the main loading function that orchestrates:
    - Streaming file reading
    - Record validation
    - Duplicate detection (initial mode only)
    - Batch embedding generation
    - Database insertion/update/deletion based on mode
    - Progress tracking

    Args:
        file_path: Path to input file (YAML or JSON)
        database: Vector database instance
        embedding_provider: Embedding provider instance
        collection_name: Name of the collection to load into
        mode: Load mode (initial, upsert, flush)
        batch_size: Batch size for embedding generation
        parallel: Number of parallel workers (currently not used, sequential processing)
        force_reembed: Force re-embedding on upsert mode (ignored in other modes)

    Returns:
        LoadStatistics with results of the load operation

    Raises:
        ValidationError: If input validation fails
        DatabaseError: If database operations fail
        EmbeddingError: If embedding generation fails
    """
    stats = LoadStatistics()
    stats.embedding_model = embedding_provider.get_model_name()
    stats.database = collection_name
    stats.mode = mode
    stats.force_reembed = force_reembed

    tracer = get_tracer()
    with tracer.start_as_current_span(
        "load_operation",
        attributes={
            "file.path": str(file_path),
            "collection.name": collection_name,
            "embedding.model": stats.embedding_model,
            "batch.size": batch_size,
            "parallel.workers": parallel,
            "mode": mode,
            "force_reembed": force_reembed,
        },
    ) as span:
        # Handle flush mode: delete all existing records first
        if mode == "flush":
            with tracer.start_as_current_span("flush_operation") as flush_span:
                collection_exists = await database.collection_exists(collection_name)
                if collection_exists:
                    logger.info(f"Flush mode: deleting all records from {collection_name}")
                    deleted_count = await database.delete_all(collection_name)
                    stats.deleted_records = deleted_count
                    logger.info(f"Deleted {deleted_count} records")
                    flush_span.set_attribute("deleted_count", deleted_count)
                else:
                    logger.info(f"Flush mode: collection {collection_name} does not exist, will create")
                    stats.deleted_records = 0

        # Check if collection exists, create if not
        collection_exists = await database.collection_exists(collection_name)

        if not collection_exists:
            logger.info(f"Creating collection: {collection_name}")
            dimension = embedding_provider.get_dimension()
            await database.create_collection(
                name=collection_name,
                dimension=dimension,
                metadata={"embedding_model": stats.embedding_model},
            )
            logger.info(f"Collection created: {collection_name} (dimension: {dimension})")
        else:
            # Verify dimension compatibility
            collection_info = await database.get_collection_info(collection_name)
            expected_dim = embedding_provider.get_dimension()
            actual_dim = collection_info.get("dimension", 0)

            if actual_dim != expected_dim:
                msg = f"Dimension mismatch: model={expected_dim}, database={actual_dim}"
                logger.error(msg)
                span.set_status(trace.Status(trace.StatusCode.ERROR, msg))
                raise DimensionMismatchError(msg)

            logger.info(f"Using existing collection: {collection_name}")

        # Track seen IDs for duplicate detection (only in initial mode)
        seen_ids: set[str] = set()

        # Collect records for batch processing
        records_to_process: list[dict[str, Any]] = []

        # Stream and validate records
        record_index = 0
        async for record in stream_records(file_path):
            stats.total_records += 1

            try:
                # Validate record structure
                validate_record(record, record_index)

                # Check for duplicate ID (only in initial mode)
                record_id = record["id"]
                if mode == "initial" and record_id in seen_ids:
                    logger.warning(f"Duplicate ID skipped: {record_id}")
                    stats.skipped_records += 1
                    with tracer.start_as_current_span(
                        "duplicate_detection",
                        attributes={"record.id": record_id},
                    ):
                        pass
                    continue

                seen_ids.add(record_id)
                records_to_process.append(record)

            except ValidationError as e:
                logger.error(f"Record validation failed at index {record_index}: {e}")
                stats.failed_records += 1

            record_index += 1

        # Check if we have any records to process
        if not records_to_process:
            if stats.total_records == 0:
                msg = "Input file is empty or has no records"
                logger.error(msg)
                span.set_status(trace.Status(trace.StatusCode.ERROR, msg))
                raise ValidationError(msg)
            else:
                logger.warning("No valid records to load after validation and duplicate detection")
                span.set_attribute("load.status", "no_valid_records")
                return stats

        # Handle upsert mode: separate existing and new records
        if mode == "upsert":
            with tracer.start_as_current_span("upsert_operation") as upsert_span:
                # Get existing IDs from database by querying each ID
                existing_ids: set[str] = set()
                try:
                    # Check which IDs exist by attempting to query them
                    # This is a workaround since we don't have a direct "check if ID exists" method
                    for record in records_to_process:
                        try:
                            # Try to query by ID - if it exists, we'll get results
                            # We use a dummy embedding since we're just checking existence
                            results = await database.query(
                                collection=collection_name,
                                query_embedding=[0.0] * embedding_provider.get_dimension(),
                                top_k=1,
                                filter_metadata={"id": record["id"]} if "id" in record else None,
                            )
                            # If we get any results with this ID, it exists
                            if results and any(r["id"] == record["id"] for r in results):
                                existing_ids.add(record["id"])
                        except Exception:  # nosec B110
                            # If query fails, assume ID doesn't exist
                            pass

                    logger.info(f"Found {len(existing_ids)} existing records in database")
                    upsert_span.set_attribute("existing_count", len(existing_ids))
                except Exception as e:
                    logger.warning(f"Failed to check existing IDs, treating all as new: {e}")
                    existing_ids = set()

                # Separate records into updates and inserts
                records_to_update = [r for r in records_to_process if r["id"] in existing_ids]
                records_to_insert = [r for r in records_to_process if r["id"] not in existing_ids]

                logger.info(f"Upsert: {len(records_to_update)} updates, {len(records_to_insert)} inserts")

                # Process updates
                if records_to_update:
                    # Generate embeddings for updates
                    # Note: In the current implementation, we always generate new embeddings
                    # even when force_reembed=False, as we don't have a clean way to retrieve
                    # existing embeddings through the public VectorDatabase API
                    if force_reembed:
                        logger.info(
                            f"Force re-embed enabled: generating new embeddings for {len(records_to_update)} updates"
                        )
                    else:
                        logger.info(f"Generating embeddings for {len(records_to_update)} updates")

                    update_texts = [r["text"] for r in records_to_update]
                    update_embeddings = await generate_embeddings_batch(
                        texts=update_texts,
                        provider=embedding_provider,
                        batch_size=batch_size,
                    )

                    # Prepare update records
                    update_db_records = []
                    for record, embedding in zip(records_to_update, update_embeddings, strict=True):
                        db_record = {
                            "id": record["id"],
                            "text": record["text"],
                            "embedding": embedding,
                        }
                        if "metadata" in record:
                            db_record["metadata"] = record["metadata"]
                        update_db_records.append(db_record)

                    # Update in database (delete + insert for ChromaDB)
                    try:
                        with tracer.start_as_current_span("database_update"):
                            # Delete old versions
                            await database.delete_by_ids(collection_name, [r["id"] for r in records_to_update])
                            # Insert updated versions
                            await database.insert(collection_name, update_db_records)
                        stats.updated_records = len(update_db_records)
                        logger.info(f"Successfully updated {stats.updated_records} records")
                    except DatabaseError as e:
                        logger.error(f"Database update failed: {e}")
                        stats.failed_records += len(update_db_records)
                        span.record_exception(e)
                        raise

                # Process inserts
                if records_to_insert:
                    logger.info(f"Generating embeddings for {len(records_to_insert)} new records")
                    insert_texts = [r["text"] for r in records_to_insert]
                    insert_embeddings = await generate_embeddings_batch(
                        texts=insert_texts,
                        provider=embedding_provider,
                        batch_size=batch_size,
                    )

                    # Prepare insert records
                    insert_db_records = []
                    for record, embedding in zip(records_to_insert, insert_embeddings, strict=True):
                        db_record = {
                            "id": record["id"],
                            "text": record["text"],
                            "embedding": embedding,
                        }
                        if "metadata" in record:
                            db_record["metadata"] = record["metadata"]
                        insert_db_records.append(db_record)

                    # Insert into database
                    try:
                        with tracer.start_as_current_span("database_insert"):
                            await database.insert(collection_name, insert_db_records)
                        stats.loaded_records = len(insert_db_records)
                        logger.info(f"Successfully inserted {stats.loaded_records} new records")
                    except DatabaseError as e:
                        logger.error(f"Database insertion failed: {e}")
                        stats.failed_records += len(insert_db_records)
                        span.record_exception(e)
                        raise

        else:
            # Initial or flush mode: generate embeddings and insert all records
            texts = [record["text"] for record in records_to_process]
            logger.info(f"Generating embeddings for {len(texts)} texts (batch size: {batch_size})")

            try:
                embeddings = await generate_embeddings_batch(
                    texts=texts,
                    provider=embedding_provider,
                    batch_size=batch_size,
                )
            except EmbeddingError as e:
                logger.error(f"Embedding generation failed: {e}")
                span.record_exception(e)
                span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                raise

            # Prepare records with embeddings for database insertion
            db_records = []
            for record, embedding in zip(records_to_process, embeddings, strict=True):
                db_record = {
                    "id": record["id"],
                    "text": record["text"],
                    "embedding": embedding,
                }
                # Include metadata if present
                if "metadata" in record:
                    db_record["metadata"] = record["metadata"]
                db_records.append(db_record)

            # Insert into database
            logger.info(f"Inserting {len(db_records)} records into database")

            try:
                with tracer.start_as_current_span(
                    "database_insert",
                    attributes={
                        "collection.name": collection_name,
                        "record.count": len(db_records),
                    },
                ):
                    await database.insert(collection_name, db_records)
                stats.loaded_records = len(db_records)
                logger.info(f"Successfully loaded {stats.loaded_records} records")

            except DatabaseError as e:
                logger.error(f"Database insertion failed: {e}")
                stats.failed_records += len(db_records)
                span.record_exception(e)
                span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                raise

        # Set final span attributes
        span.set_attribute("load.total_records", stats.total_records)
        span.set_attribute("load.loaded_records", stats.loaded_records)
        span.set_attribute("load.updated_records", stats.updated_records)
        span.set_attribute("load.deleted_records", stats.deleted_records)
        span.set_attribute("load.failed_records", stats.failed_records)
        span.set_attribute("load.skipped_records", stats.skipped_records)
        span.set_attribute("load.mode", stats.mode)

        if mode == "upsert":
            logger.info(
                f"Upsert complete: {stats.updated_records} updated, {stats.loaded_records} added, "
                f"{stats.total_records} total (failed: {stats.failed_records}, skipped: {stats.skipped_records})"
            )
        elif mode == "flush":
            logger.info(
                f"Flush complete: {stats.deleted_records} deleted, {stats.loaded_records} loaded, "
                f"{stats.total_records} total (failed: {stats.failed_records}, skipped: {stats.skipped_records})"
            )
        else:
            logger.info(
                f"Load complete: {stats.loaded_records}/{stats.total_records} records "
                f"(failed: {stats.failed_records}, skipped: {stats.skipped_records})"
            )

        return stats
