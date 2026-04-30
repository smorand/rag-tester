"""Load command implementation for RAG Tester.

This module provides the CLI command for loading records into a vector database.
"""

import asyncio
import logging
import time

import typer
from rich.console import Console

from rag_tester.core.loader import load_records
from rag_tester.core.validator import (
    ValidationError,
    validate_batch_size,
    validate_file_path,
    validate_load_mode,
    validate_parallel_workers,
)
from rag_tester.providers.databases.base import DatabaseError, DimensionMismatchError
from rag_tester.providers.databases.chromadb import ChromaDBProvider
from rag_tester.providers.embeddings.base import EmbeddingError
from rag_tester.providers.embeddings.local import LocalEmbeddingProvider
from rag_tester.utils.progress import ProgressTracker

logger = logging.getLogger(__name__)
console = Console()
error_console = Console(stderr=True)


def load_command(
    file: str = typer.Option(..., "--file", "-f", help="Path to input file (YAML or JSON)"),
    database: str = typer.Option(
        ..., "--database", "-d", help="Database connection string (e.g., chromadb://localhost:8000/collection)"
    ),
    embedding: str = typer.Option(
        ..., "--embedding", "-e", help="Embedding model identifier (e.g., sentence-transformers/all-MiniLM-L6-v2)"
    ),
    mode: str = typer.Option("initial", "--mode", "-m", help="Load mode: initial, upsert, or flush"),
    parallel: int = typer.Option(1, "--parallel", "-p", help="Number of parallel workers (1-16)"),
    batch_size: int = typer.Option(32, "--batch-size", "-b", help="Batch size for embedding generation (1-256)"),
    force_reembed: bool = typer.Option(False, "--force-reembed", help="Force re-embedding on upsert mode"),
) -> None:
    """Load records into a vector database.

    This command loads records from a YAML or JSON file into a vector database,
    generating embeddings and handling duplicates automatically.

    Examples:
        # Load 100 records with local embedding model
        rag-tester load --file data.yaml --database chromadb://localhost:8000/my_collection --embedding sentence-transformers/all-MiniLM-L6-v2

        # Load with parallel processing and custom batch size
        rag-tester load -f data.yaml -d chromadb://localhost:8000/my_collection -e sentence-transformers/all-MiniLM-L6-v2 --parallel 4 --batch-size 64
    """
    # Run async load function
    exit_code = asyncio.run(
        _load_async(
            file=file,
            database=database,
            embedding=embedding,
            mode=mode,
            parallel=parallel,
            batch_size=batch_size,
            force_reembed=force_reembed,
        )
    )

    if exit_code != 0:
        raise typer.Exit(code=exit_code)


async def _load_async(
    file: str,
    database: str,
    embedding: str,
    mode: str,
    parallel: int,
    batch_size: int,
    force_reembed: bool,
) -> int:
    """Async implementation of load command.

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    start_time = time.time()

    try:
        # Validate arguments
        logger.info("Validating arguments...")
        file_path = validate_file_path(file)
        validate_load_mode(mode)
        validate_parallel_workers(parallel)
        validate_batch_size(batch_size)

        # Validate mode is one of: initial, upsert, flush
        if mode not in {"initial", "upsert", "flush"}:
            error_console.print(f"[red]Error: Invalid mode '{mode}'. Must be one of: initial, upsert, flush[/red]")
            return 1

        # Warn if force_reembed is used with non-upsert mode
        if force_reembed and mode != "upsert":
            logger.warning(f"force-reembed flag ignored in {mode} mode")
            console.print(f"[yellow]Warning: force-reembed flag ignored in {mode} mode[/yellow]")

        # Parse database connection string
        # Format: chromadb://host:port/collection_name OR chromadb:///path/to/db/collection_name
        if not database.startswith("chromadb://"):
            error_console.print(
                "[red]Error: Only ChromaDB is currently supported. Use chromadb://host:port/collection or chromadb:///path/to/db/collection[/red]"
            )
            return 1

        # Extract collection name (last part after /)
        remainder = database.replace("chromadb://", "")
        parts = remainder.rsplit("/", 1)
        if len(parts) != 2:
            error_console.print(
                "[red]Error: Invalid database connection string. Expected format: chromadb://host:port/collection or chromadb:///path/to/db/collection[/red]"
            )
            return 1
        
        collection_name = parts[1]

        # Initialize providers
        logger.info(f"Initializing embedding provider: {embedding}")
        console.print(f"[blue]Embedding model:[/blue] {embedding}")

        try:
            embedding_provider = LocalEmbeddingProvider(model_name=embedding)
        except Exception as e:
            error_console.print(f"[red]Error: Failed to load embedding model: {e}[/red]")
            return 1

        logger.info(f"Connecting to database: {database}")
        console.print(f"[blue]Database:[/blue] {database}")

        try:
            db_provider = ChromaDBProvider(connection_string=database)
        except Exception as e:
            error_console.print(f"[red]Error: Database connection failed: {e}[/red]")
            return 1

        # Log configuration
        console.print(f"[blue]Mode:[/blue] {mode}")
        console.print(f"[blue]Parallel workers:[/blue] {parallel}")
        console.print(f"[blue]Batch size:[/blue] {batch_size}")
        if mode == "upsert":
            console.print(f"[blue]Force re-embed:[/blue] {force_reembed}")
        console.print()

        # Load records with progress tracking
        logger.info(f"Loading records from {file_path}")

        # Count records for progress bar (quick pre-scan)
        import json

        import yaml

        with open(file_path, encoding="utf-8") as f:
            data = yaml.safe_load(f) if file_path.suffix.lower() in {".yaml", ".yml"} else json.load(f)

        total_records = len(data) if isinstance(data, list) else 0

        with ProgressTracker("Loading records", total_records) as progress:
            stats = await load_records(
                file_path=file_path,
                database=db_provider,
                embedding_provider=embedding_provider,
                collection_name=collection_name,
                mode=mode,
                batch_size=batch_size,
                parallel=parallel,
                force_reembed=force_reembed,
            )
            progress.update(total_records)  # Complete progress bar

        # Calculate elapsed time
        elapsed_time = time.time() - start_time

        # Display results
        console.print()
        console.print("[green]✓[/green] Load complete!")

        if mode == "upsert":
            console.print(f"[blue]Records updated:[/blue] {stats.updated_records}")
            console.print(f"[blue]Records added:[/blue] {stats.loaded_records}")
        elif mode == "flush":
            console.print(f"[blue]Records deleted:[/blue] {stats.deleted_records}")
            console.print(f"[blue]Records loaded:[/blue] {stats.loaded_records}")
        else:
            console.print(f"[blue]Successfully loaded:[/blue] {stats.loaded_records} records")

        console.print(f"[blue]Failed records:[/blue] {stats.failed_records}")
        console.print(f"[blue]Skipped records:[/blue] {stats.skipped_records}")
        console.print(f"[blue]Total time:[/blue] {elapsed_time:.2f} seconds")

        if stats.failed_records > 0:
            console.print("[yellow]Warning: Some records failed to load. Check logs for details.[/yellow]")

        return 0

    except ValidationError as e:
        error_console.print(f"[red]Error: {e}[/red]")
        logger.error(f"Validation error: {e}")
        return 1

    except DimensionMismatchError as e:
        error_console.print(f"[red]Error: {e}[/red]")
        logger.error(f"Dimension mismatch: {e}")
        return 1

    except DatabaseError as e:
        error_console.print(f"[red]Error: Database operation failed: {e}[/red]")
        logger.error(f"Database error: {e}")
        return 1

    except EmbeddingError as e:
        error_console.print(f"[red]Error: Embedding generation failed: {e}[/red]")
        logger.error(f"Embedding error: {e}")
        return 1

    except Exception as e:
        error_console.print(f"[red]Error: Unexpected error: {e}[/red]")
        logger.exception("Unexpected error during load")
        return 1
