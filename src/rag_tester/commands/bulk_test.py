"""Bulk-test command implementation for RAG Tester.

The CLI surface lives here; parsing/validation helpers are in
``_bulk_test_parsing`` and execution helpers in ``_bulk_test_execution``.
"""

import asyncio
import logging
import time
from typing import TYPE_CHECKING

import typer
from rich.console import Console

from rag_tester.commands._bulk_test_execution import execute_test_suite, write_results_file
from rag_tester.commands._bulk_test_parsing import (
    BulkTestError,
    parse_database_connection,
    parse_test_file,
    validate_output_path,
    validate_parallel_workers,
)
from rag_tester.core.validator import ValidationError
from rag_tester.providers.databases import get_database_provider
from rag_tester.providers.embeddings.local import LocalEmbeddingProvider
from rag_tester.tracing import trace_span

if TYPE_CHECKING:
    from rag_tester.providers.databases.base import VectorDatabase

logger = logging.getLogger(__name__)
console = Console()
error_console = Console(stderr=True)


def bulk_test_command(
    file: str = typer.Argument(..., help="Path to test file (YAML or JSON)"),
    database: str = typer.Option(
        ..., "--database", "-d", help="Database connection string (e.g., chromadb://localhost:8000/collection)"
    ),
    embedding: str = typer.Option(
        ..., "--embedding", "-e", help="Embedding model identifier (e.g., sentence-transformers/all-MiniLM-L6-v2)"
    ),
    output: str = typer.Option(..., "--output", "-o", help="Output file path (YAML)"),
    parallel: int = typer.Option(1, "--parallel", "-p", help="Number of parallel workers (1-16)"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Include all tests in output (not just failures)"),
) -> None:
    """Run a test suite against the vector database.

    This command executes multiple test cases from a file and validates results
    against expected outcomes. Results are written to a YAML file with summary
    statistics and test details.

    Examples:
        # Run test suite with default settings
        rag-tester bulk-test tests.yaml --database chromadb://localhost:8000/my_collection --embedding sentence-transformers/all-MiniLM-L6-v2 --output results.yaml

        # Run with parallel execution
        rag-tester bulk-test tests.yaml -d chromadb://localhost:8000/my_collection -e sentence-transformers/all-MiniLM-L6-v2 -o results.yaml --parallel 4

        # Include all tests in output (not just failures)
        rag-tester bulk-test tests.yaml -d chromadb://localhost:8000/my_collection -e sentence-transformers/all-MiniLM-L6-v2 -o results.yaml --verbose
    """
    try:
        asyncio.get_running_loop()
        # An event loop is already running (e.g. when called from an async test).
        # Patch it via nest_asyncio so asyncio.run can be reused safely.
        import nest_asyncio

        nest_asyncio.apply()
    except RuntimeError:
        pass

    exit_code = asyncio.run(
        _bulk_test_async(
            file=file,
            database=database,
            embedding=embedding,
            output=output,
            parallel=parallel,
            verbose=verbose,
        )
    )

    if exit_code != 0:
        raise typer.Exit(code=exit_code)


async def _bulk_test_async(
    file: str,
    database: str,
    embedding: str,
    output: str,
    parallel: int,
    verbose: bool,
) -> int:
    """Async implementation of the bulk-test command. Returns an exit code."""
    start_time = time.time()

    try:
        # Validate inputs
        validate_parallel_workers(parallel)
        validate_output_path(output)

        # Parse test file
        with trace_span("bulk_test.parse_file", attributes={"file": file}):
            test_cases = parse_test_file(file)

        if not test_cases:
            error_console.print("[red]Error: Test file is empty or has no tests[/red]")
            return 1

        logger.info("Loaded %s test cases from %s", len(test_cases), file)

        # Surface a friendly error before instantiating a provider
        if parse_database_connection(database) is None:
            return 1

        # Initialize providers
        logger.info("Initializing embedding provider: %s", embedding)
        try:
            embedding_provider = LocalEmbeddingProvider(model_name=embedding)
        except Exception as e:
            error_console.print(f"[red]Error: Failed to load embedding model: {embedding}[/red]")
            logger.error("Failed to load embedding model: %s", e)
            return 1

        logger.info("Connecting to database: %s", database)
        try:
            db_provider: VectorDatabase = get_database_provider(database)
        except ValueError as e:
            error_console.print(f"[red]Error: {e}[/red]")
            return 1
        except Exception as e:
            error_console.print(f"[red]Error: Database connection failed: {e}[/red]")
            logger.error("Database connection failed: %s", e)
            return 1

        # Resolve collection name from URI tail
        collection_name = database.rsplit("/", 1)[-1]

        if not await db_provider.collection_exists(collection_name):
            error_console.print(f"[red]Error: Collection '{collection_name}' does not exist[/red]")
            return 1

        # Execute test suite
        logger.info("Using %s parallel workers", parallel)
        with trace_span(
            "bulk_test.execute_suite",
            attributes={"total_tests": len(test_cases), "parallel_workers": parallel},
        ):
            test_results = await execute_test_suite(
                test_cases=test_cases,
                db_provider=db_provider,
                embedding_provider=embedding_provider,
                collection_name=collection_name,
                parallel=parallel,
            )

        # Generate summary
        total_tests = len(test_results)
        passed = sum(1 for r in test_results if r["status"] == "passed")
        failed = sum(1 for r in test_results if r["status"] == "failed")
        errors = sum(1 for r in test_results if r["status"] == "error")
        total_time = time.time() - start_time

        summary = {
            "total_tests": total_tests,
            "passed": passed,
            "failed": failed,
            "errors": errors,
            "total_tokens": 0,  # Local models don't consume tokens
            "total_time": round(total_time, 2),
            "embedding_model": embedding,
            "database": database,
        }

        with trace_span("bulk_test.write_results", attributes={"output": output}):
            write_results_file(
                output=output,
                summary=summary,
                test_results=test_results,
                verbose=verbose,
            )

        pct = passed * 100 // total_tests if total_tests > 0 else 0
        logger.info("Test suite complete: %s/%s passed (%s%%)", passed, total_tests, pct)
        logger.info("Results written to: %s", output)

        console.print()
        console.print(f"[green]✓[/green] Test suite complete: {passed}/{total_tests} passed ({pct}%)")
        if failed > 0:
            console.print(f"[red]✗[/red] Failed: {failed}")
        if errors > 0:
            console.print(f"[yellow]![/yellow] Errors: {errors}")
        console.print(f"[blue]ℹ[/blue] Results written to: {output}")

        with trace_span(
            "bulk_test.complete",
            attributes={
                "total_tests": total_tests,
                "passed": passed,
                "failed": failed,
                "errors": errors,
                "duration": total_time,
            },
        ):
            pass

        return 0

    except ValidationError as e:
        error_console.print(f"[red]Error: {e}[/red]")
        logger.error("Validation error: %s", e)
        return 1

    except BulkTestError as e:
        error_console.print(f"[red]Error: {e}[/red]")
        logger.error("Bulk test error: %s", e)
        return 1

    except Exception as e:
        error_console.print(f"[red]Error: Unexpected error: {e}[/red]")
        logger.exception("Unexpected error during bulk test")
        return 1
