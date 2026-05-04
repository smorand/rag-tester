"""Core loading logic for RAG Tester.

This module provides the main loading functionality with streaming file processing,
batch embedding generation, parallel processing, and duplicate detection.
"""

import json
import logging
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import aiofiles
import yaml
from opentelemetry import trace

from rag_tester.core._loader_modes import (
    handle_flush_mode,
    handle_initial_or_flush_insert,
    handle_upsert_mode,
)
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
                async with aiofiles.open(file_path, encoding="utf-8") as f:
                    content = await f.read()
                data = yaml.safe_load(content)

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
                logger.debug("Streaming %s records from YAML file", record_count)

                for record in data:
                    yield record

            elif file_path.suffix.lower() == ".json":
                async with aiofiles.open(file_path, encoding="utf-8") as f:
                    content = await f.read()
                data = json.loads(content)

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
                logger.debug("Streaming %s records from JSON file", record_count)

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
            logger.debug(
                "Generating embeddings for batch %s/%s (%s texts)", batch_num + 1, total_batches, len(batch_texts)
            )
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
            stats.deleted_records = await handle_flush_mode(database, collection_name)

        # Check if collection exists, create if not
        collection_exists = await database.collection_exists(collection_name)

        if not collection_exists:
            logger.info("Creating collection: %s", collection_name)
            dimension = embedding_provider.get_dimension()
            await database.create_collection(
                name=collection_name,
                dimension=dimension,
                metadata={"embedding_model": stats.embedding_model},
            )
            logger.info("Collection created: %s (dimension: %s)", collection_name, dimension)
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

            logger.info("Using existing collection: %s", collection_name)

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
                    logger.warning("Duplicate ID skipped: %s", record_id)
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
                logger.error("Record validation failed at index %s: %s", record_index, e)
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

        # Dispatch to mode-specific handler
        if mode == "upsert":
            try:
                updated, inserted, failed = await handle_upsert_mode(
                    database=database,
                    embedding_provider=embedding_provider,
                    collection_name=collection_name,
                    records_to_process=records_to_process,
                    batch_size=batch_size,
                    force_reembed=force_reembed,
                    generate_embeddings=generate_embeddings_batch,
                )
                stats.updated_records = updated
                stats.loaded_records = inserted
                stats.failed_records += failed
            except DatabaseError as e:
                span.record_exception(e)
                raise
        else:
            try:
                loaded, failed = await handle_initial_or_flush_insert(
                    database=database,
                    embedding_provider=embedding_provider,
                    collection_name=collection_name,
                    records_to_process=records_to_process,
                    batch_size=batch_size,
                    generate_embeddings=generate_embeddings_batch,
                )
                stats.loaded_records = loaded
                stats.failed_records += failed
            except (EmbeddingError, DatabaseError) as e:
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
                "Upsert complete: %s updated, %s added, %s total (failed: %s, skipped: %s)",
                stats.updated_records,
                stats.loaded_records,
                stats.total_records,
                stats.failed_records,
                stats.skipped_records,
            )
        elif mode == "flush":
            logger.info(
                "Flush complete: %s deleted, %s loaded, %s total (failed: %s, skipped: %s)",
                stats.deleted_records,
                stats.loaded_records,
                stats.total_records,
                stats.failed_records,
                stats.skipped_records,
            )
        else:
            logger.info(
                "Load complete: %s/%s records (failed: %s, skipped: %s)",
                stats.loaded_records,
                stats.total_records,
                stats.failed_records,
                stats.skipped_records,
            )

        return stats
