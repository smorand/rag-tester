"""Bulk-test command implementation for RAG Tester.

This module provides the CLI command for running test suites against a vector database.
"""

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any

import typer
import yaml
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from rag_tester.core.validator import ValidationError
from rag_tester.providers.databases.base import DatabaseError
from rag_tester.providers.databases.chromadb import ChromaDBProvider
from rag_tester.providers.embeddings.base import EmbeddingError
from rag_tester.providers.embeddings.local import LocalEmbeddingProvider
from rag_tester.tracing import trace_span

logger = logging.getLogger(__name__)
console = Console()
error_console = Console(stderr=True)


class BulkTestError(Exception):
    """Base exception for bulk test errors."""

    pass


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
    # Check if we're already in an event loop (e.g., called from async tests)
    try:
        asyncio.get_running_loop()
        # We're in an event loop, we need to use nest_asyncio or return a coroutine
        # For now, use a workaround with nest_asyncio
        import nest_asyncio
        nest_asyncio.apply()
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
    except RuntimeError:
        # No event loop running, use asyncio.run()
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
    """Async implementation of bulk-test command.

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    start_time = time.time()

    try:
        # Validate inputs
        _validate_parallel_workers(parallel)
        _validate_output_path(output)

        # Parse test file
        with trace_span("bulk_test.parse_file", attributes={"file": file}):
            test_cases = _parse_test_file(file)

        if not test_cases:
            error_console.print("[red]Error: Test file is empty or has no tests[/red]")
            return 1

        logger.info(f"Loaded {len(test_cases)} test cases from {file}")

        # Parse database connection string
        db_config = _parse_database_connection(database)
        if not db_config:
            return 1

        # Initialize providers
        logger.info(f"Initializing embedding provider: {embedding}")

        try:
            embedding_provider = LocalEmbeddingProvider(model_name=embedding)
        except Exception as e:
            error_console.print(f"[red]Error: Failed to load embedding model: {embedding}[/red]")
            logger.error(f"Failed to load embedding model: {e}")
            return 1

        logger.info(f"Connecting to database: {database}")

        try:
            # Instantiate the appropriate database provider
            if database.startswith("chromadb://"):
                db_provider = ChromaDBProvider(connection_string=database)
            elif database.startswith("postgresql://"):
                from rag_tester.providers.databases.postgresql import PostgreSQLProvider
                db_provider = PostgreSQLProvider(connection_string=database)
            else:
                error_console.print("[red]Error: Unsupported database provider[/red]")
                return 1
        except Exception as e:
            error_console.print(f"[red]Error: Database connection failed: {e}[/red]")
            logger.error(f"Database connection failed: {e}")
            return 1

        # Check collection exists
        collection_name = db_config["collection"]
        if not await db_provider.collection_exists(collection_name):
            error_console.print(f"[red]Error: Collection '{collection_name}' does not exist[/red]")
            return 1

        # Execute test suite
        logger.info(f"Using {parallel} parallel workers")
        with trace_span(
            "bulk_test.execute_suite",
            attributes={
                "total_tests": len(test_cases),
                "parallel_workers": parallel,
            },
        ):
            test_results = await _execute_test_suite(
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

        # Write results file
        with trace_span("bulk_test.write_results", attributes={"output": output}):
            _write_results_file(
                output=output,
                summary=summary,
                test_results=test_results,
                verbose=verbose,
            )

        # Log summary
        logger.info(
            f"Test suite complete: {passed}/{total_tests} passed ({passed * 100 // total_tests if total_tests > 0 else 0}%)"
        )
        logger.info(f"Results written to: {output}")

        # Display summary to user
        console.print()
        console.print(
            f"[green]✓[/green] Test suite complete: {passed}/{total_tests} passed ({passed * 100 // total_tests if total_tests > 0 else 0}%)"
        )
        if failed > 0:
            console.print(f"[red]✗[/red] Failed: {failed}")
        if errors > 0:
            console.print(f"[yellow]![/yellow] Errors: {errors}")
        console.print(f"[blue]ℹ[/blue] Results written to: {output}")

        # Trace completion
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
        logger.error(f"Validation error: {e}")
        return 1

    except BulkTestError as e:
        error_console.print(f"[red]Error: {e}[/red]")
        logger.error(f"Bulk test error: {e}")
        return 1

    except Exception as e:
        error_console.print(f"[red]Error: Unexpected error: {e}[/red]")
        logger.exception("Unexpected error during bulk test")
        return 1


def _parse_test_file(file_path: str) -> list[dict[str, Any]]:
    """Parse test file (YAML or JSON) and return test cases.

    Args:
        file_path: Path to test file

    Returns:
        List of test case dictionaries

    Raises:
        ValidationError: If file is invalid or malformed
    """
    path = Path(file_path)

    if not path.exists():
        raise ValidationError(f"File not found: {file_path}")

    if not path.is_file():
        raise ValidationError(f"Path is not a file: {file_path}")

    try:
        with open(path) as f:
            if path.suffix.lower() in {".yaml", ".yml"}:
                data = yaml.safe_load(f)
            elif path.suffix.lower() == ".json":
                data = json.load(f)
            else:
                raise ValidationError(f"Unsupported file format: {path.suffix}. Expected .yaml, .yml, or .json")
    except yaml.YAMLError as e:
        raise ValidationError(f"Invalid test file format. Failed to parse YAML: {e}")
    except json.JSONDecodeError as e:
        raise ValidationError(f"Invalid test file format. Failed to parse JSON: {e}")
    except Exception as e:
        raise ValidationError(f"Failed to read test file: {e}")

    if not isinstance(data, dict):
        raise ValidationError("Test file must contain a dictionary at root level")

    if "tests" not in data:
        raise ValidationError("Test file must contain a 'tests' key")

    tests = data["tests"]
    if not isinstance(tests, list):
        raise ValidationError("'tests' must be a list")

    # Validate each test case
    for i, test in enumerate(tests):
        _validate_test_case(test, i)

    return tests


def _validate_test_case(test: dict[str, Any], index: int) -> None:
    """Validate a single test case.

    Args:
        test: Test case dictionary
        index: Index of test in list (for error messages)

    Raises:
        ValidationError: If test case is invalid
    """
    if not isinstance(test, dict):
        raise ValidationError(f"Test at index {index} is not a dictionary")

    # Check required fields
    if "test_id" not in test:
        raise ValidationError(f"Missing required field 'test_id' in test at index {index}")

    if "query" not in test:
        raise ValidationError(f"Missing required field 'query' in test '{test.get('test_id', index)}'")

    if "expected" not in test:
        raise ValidationError(f"Missing required field 'expected' in test '{test['test_id']}'")

    # Validate expected results
    expected = test["expected"]
    if not isinstance(expected, list):
        raise ValidationError(f"'expected' must be a list in test '{test['test_id']}'")

    if not expected:
        raise ValidationError(f"'expected' cannot be empty in test '{test['test_id']}'")

    for i, exp in enumerate(expected):
        if not isinstance(exp, dict):
            raise ValidationError(f"Expected result at index {i} must be a dictionary in test '{test['test_id']}'")

        if "id" not in exp:
            raise ValidationError(f"Missing 'id' in expected result at index {i} in test '{test['test_id']}'")

        if "text" not in exp:
            raise ValidationError(f"Missing 'text' in expected result at index {i} in test '{test['test_id']}'")

        # Validate min_threshold if present
        if "min_threshold" in exp:
            threshold = exp["min_threshold"]
            if not isinstance(threshold, (int, float)):
                raise ValidationError(
                    f"'min_threshold' must be a number in expected result at index {i} in test '{test['test_id']}'"
                )
            if threshold < 0 or threshold > 1:
                raise ValidationError(
                    f"'min_threshold' must be between 0 and 1 in expected result at index {i} in test '{test['test_id']}'"
                )


def _validate_parallel_workers(parallel: int) -> None:
    """Validate number of parallel workers.

    Args:
        parallel: Number of parallel workers

    Raises:
        ValidationError: If parallel workers is out of range
    """
    if not isinstance(parallel, int):
        raise ValidationError(f"Parallel workers must be an integer, got {type(parallel).__name__}")

    if parallel < 1 or parallel > 16:
        raise ValidationError("Parallel workers must be between 1 and 16")


def _validate_output_path(output: str) -> None:
    """Validate output file path is writable.

    Args:
        output: Output file path

    Raises:
        ValidationError: If output path is not writable
    """
    path = Path(output)

    # Check parent directory exists and is writable
    parent = path.parent
    if not parent.exists():
        raise ValidationError(f"Output directory does not exist: {parent}")

    if not parent.is_dir():
        raise ValidationError(f"Output parent path is not a directory: {parent}")

    # Try to create/write to the file
    try:
        path.touch(exist_ok=True)
    except Exception as e:
        raise ValidationError(f"Cannot write output file: {output} ({e})")


def _parse_database_connection(database: str) -> dict[str, Any] | None:
    """Parse database connection string.

    Args:
        database: Database connection string

    Returns:
        Dictionary with host, port, collection or None if invalid
    """
    if database.startswith("chromadb://"):
        db_parts = database.replace("chromadb://", "").split("/")
        if len(db_parts) != 2:
            error_console.print(
                "[red]Error: Invalid database connection string. Expected format: chromadb://host:port/collection[/red]"
            )
            return None

        host_port = db_parts[0]
        collection_name = db_parts[1]

        if ":" not in host_port:
            error_console.print(
                "[red]Error: Invalid database connection string. Expected format: chromadb://host:port/collection[/red]"
            )
            return None

        host, port_str = host_port.rsplit(":", 1)
        try:
            port = int(port_str)
        except ValueError:
            error_console.print(f"[red]Error: Invalid port number: {port_str}[/red]")
            return None

        return {
            "host": host,
            "port": port,
            "collection": collection_name,
        }
        
    elif database.startswith("postgresql://"):
        # Extract table name (last part after /)
        remainder = database.replace("postgresql://", "")
        parts = remainder.rsplit("/", 1)
        if len(parts) != 2:
            error_console.print(
                "[red]Error: Invalid PostgreSQL connection string. Expected format: postgresql://user:pass@host:port/dbname/table_name[/red]"
            )
            return None
        
        # For PostgreSQL, we don't need to parse host/port separately
        # Just return the collection name
        return {
            "host": "postgresql",
            "port": 0,
            "collection": parts[1],
        }
        
    else:
        error_console.print(
            "[red]Error: Unsupported database. Use chromadb://... or postgresql://...[/red]"
        )
        return None


async def _execute_test_suite(
    test_cases: list[dict[str, Any]],
    db_provider: Any,
    embedding_provider: Any,
    collection_name: str,
    parallel: int,
) -> list[dict[str, Any]]:
    """Execute test suite with parallel workers.

    Args:
        test_cases: List of test case dictionaries
        db_provider: Database provider instance
        embedding_provider: Embedding provider instance
        collection_name: Name of collection to query
        parallel: Number of parallel workers

    Returns:
        List of test result dictionaries
    """
    # Create progress bar
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
    )

    with progress:
        task = progress.add_task("Testing", total=len(test_cases))

        # Execute tests in parallel
        if parallel == 1:
            # Sequential execution
            results = []
            for test_case in test_cases:
                result = await _execute_single_test(
                    test_case=test_case,
                    db_provider=db_provider,
                    embedding_provider=embedding_provider,
                    collection_name=collection_name,
                )
                results.append(result)
                progress.update(task, advance=1)
        else:
            # Parallel execution with semaphore
            semaphore = asyncio.Semaphore(parallel)
            results = []

            async def execute_with_semaphore(test_case: dict[str, Any]) -> dict[str, Any]:
                async with semaphore:
                    result = await _execute_single_test(
                        test_case=test_case,
                        db_provider=db_provider,
                        embedding_provider=embedding_provider,
                        collection_name=collection_name,
                    )
                    progress.update(task, advance=1)
                    return result

            # Execute all tests concurrently
            results = await asyncio.gather(*[execute_with_semaphore(tc) for tc in test_cases])

    return results


async def _execute_single_test(
    test_case: dict[str, Any],
    db_provider: Any,
    embedding_provider: Any,
    collection_name: str,
) -> dict[str, Any]:
    """Execute a single test case.

    Args:
        test_case: Test case dictionary
        db_provider: Database provider instance
        embedding_provider: Embedding provider instance
        collection_name: Name of collection to query

    Returns:
        Test result dictionary with status, actual results, and validation details
    """
    test_id = test_case["test_id"]
    query = test_case["query"]
    expected = test_case["expected"]

    start_time = time.time()

    try:
        with trace_span("bulk_test.test_execution", attributes={"test_id": test_id, "query": query[:50]}):
            # Generate query embedding
            embeddings = await embedding_provider.embed_texts([query])
            query_embedding = embeddings[0]

            # Query database (get more results than expected to ensure we have enough)
            top_k = max(len(expected) * 2, 10)
            raw_results = await db_provider.query(
                collection=collection_name,
                query_embedding=query_embedding,
                top_k=top_k,
            )

            # Format actual results
            actual = [{"id": r["id"], "text": r["text"], "score": r["score"]} for r in raw_results]

            # Validate results
            with trace_span("bulk_test.validation", attributes={"test_id": test_id}):
                validation_result = _validate_results(expected=expected, actual=actual)

            duration = time.time() - start_time

            result = {
                "test_id": test_id,
                "query": query,
                "expected": expected,
                "actual": actual,
                "status": validation_result["status"],
                "duration": round(duration, 3),
            }

            if validation_result["status"] == "failed":
                result["reason"] = validation_result["reason"]

            logger.debug(f"Test {test_id}: {validation_result['status']}")

            return result

    except DatabaseError as e:
        duration = time.time() - start_time
        logger.error(f"Database error in test {test_id}: {e}")
        return {
            "test_id": test_id,
            "query": query,
            "expected": expected,
            "actual": [],
            "status": "error",
            "reason": f"Database connection failed: {e}",
            "duration": round(duration, 3),
        }

    except EmbeddingError as e:
        duration = time.time() - start_time
        logger.error(f"Embedding error in test {test_id}: {e}")
        return {
            "test_id": test_id,
            "query": query,
            "expected": expected,
            "actual": [],
            "status": "error",
            "reason": f"Embedding generation failed: {e}",
            "duration": round(duration, 3),
        }

    except Exception as e:
        duration = time.time() - start_time
        logger.exception(f"Unexpected error in test {test_id}: {e}")
        return {
            "test_id": test_id,
            "query": query,
            "expected": expected,
            "actual": [],
            "status": "error",
            "reason": f"Unexpected error: {e}",
            "duration": round(duration, 3),
        }


def _validate_results(expected: list[dict[str, Any]], actual: list[dict[str, Any]]) -> dict[str, Any]:
    """Validate actual results against expected results.

    Validation rules:
    1. Expected IDs must appear in exact order in actual results
    2. If min_threshold is specified, actual score must be >= threshold
    3. Missing expected IDs cause failure

    Args:
        expected: List of expected result dictionaries
        actual: List of actual result dictionaries

    Returns:
        Dictionary with status ("passed" or "failed") and optional reason
    """
    # Extract actual IDs in order
    actual_ids = [r["id"] for r in actual]

    # Build map of actual results by ID
    actual_map = {r["id"]: r for r in actual}

    # Check order and presence of expected IDs
    expected_ids = [e["id"] for e in expected]
    last_index = -1

    for exp in expected:
        exp_id = exp["id"]

        # Check if ID exists in actual results
        if exp_id not in actual_map:
            return {
                "status": "failed",
                "reason": f"Expected ID '{exp_id}' not found in results",
            }

        # Check order
        try:
            current_index = actual_ids.index(exp_id)
        except ValueError:
            return {
                "status": "failed",
                "reason": f"Expected ID '{exp_id}' not found in results",
            }

        if current_index <= last_index:
            return {
                "status": "failed",
                "reason": f"Expected order not matched: expected {expected_ids}, got {actual_ids}",
            }

        last_index = current_index

        # Check threshold if specified
        if "min_threshold" in exp:
            min_threshold = exp["min_threshold"]
            actual_score = actual_map[exp_id]["score"]

            if actual_score < min_threshold:
                return {
                    "status": "failed",
                    "reason": f"Threshold not met for '{exp_id}': expected >= {min_threshold}, got {actual_score:.4f}",
                }

    return {"status": "passed"}


def _write_results_file(
    output: str,
    summary: dict[str, Any],
    test_results: list[dict[str, Any]],
    verbose: bool,
) -> None:
    """Write results to YAML file.

    Args:
        output: Output file path
        summary: Summary statistics dictionary
        test_results: List of test result dictionaries
        verbose: If True, include all tests; if False, only failed/error tests

    Raises:
        BulkTestError: If file write fails
    """
    # Filter tests based on verbose flag
    tests_to_write = test_results if verbose else [r for r in test_results if r["status"] in {"failed", "error"}]

    results_data = {
        "summary": summary,
        "tests": tests_to_write,
    }

    try:
        with open(output, "w") as f:
            yaml.dump(results_data, f, default_flow_style=False, sort_keys=False)
    except Exception as e:
        raise BulkTestError(f"Failed to write results file: {e}")

    logger.info(f"Results written to: {output}")
