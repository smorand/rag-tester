"""Test execution and results helpers for the bulk-test command."""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import Any

import yaml
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from rag_tester.commands._bulk_test_parsing import BulkTestError
from rag_tester.providers.databases.base import DatabaseError
from rag_tester.providers.embeddings.base import EmbeddingError
from rag_tester.tracing import trace_span

logger = logging.getLogger(__name__)
console = Console()


async def execute_test_suite(
    test_cases: list[dict[str, Any]],
    db_provider: Any,
    embedding_provider: Any,
    collection_name: str,
    parallel: int,
) -> list[dict[str, Any]]:
    """Run every test case and return the per-case result list.

    Sequential when ``parallel == 1``, otherwise dispatches via
    ``asyncio.TaskGroup`` bounded by ``asyncio.Semaphore(parallel)``.
    """
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

        if parallel == 1:
            results: list[dict[str, Any]] = []
            for test_case in test_cases:
                result = await execute_single_test(
                    test_case=test_case,
                    db_provider=db_provider,
                    embedding_provider=embedding_provider,
                    collection_name=collection_name,
                )
                results.append(result)
                progress.update(task, advance=1)
        else:
            semaphore = asyncio.Semaphore(parallel)

            async def execute_with_semaphore(test_case: dict[str, Any]) -> dict[str, Any]:
                async with semaphore:
                    result = await execute_single_test(
                        test_case=test_case,
                        db_provider=db_provider,
                        embedding_provider=embedding_provider,
                        collection_name=collection_name,
                    )
                    progress.update(task, advance=1)
                    return result

            async with asyncio.TaskGroup() as tg:
                tasks = [tg.create_task(execute_with_semaphore(tc)) for tc in test_cases]
            results = [t.result() for t in tasks]

    return results


async def execute_single_test(
    test_case: dict[str, Any],
    db_provider: Any,
    embedding_provider: Any,
    collection_name: str,
) -> dict[str, Any]:
    """Execute one test case and translate driver errors into a result row."""
    test_id = test_case["test_id"]
    query = test_case["query"]
    expected = test_case["expected"]

    start_time = time.time()

    try:
        # Avoid putting raw user query text into traces; record metadata only.
        with trace_span(
            "bulk_test.test_execution",
            attributes={"test_id": test_id, "query.length": len(query)},
        ):
            embeddings = await embedding_provider.embed_texts([query])
            query_embedding = embeddings[0]

            top_k = max(len(expected) * 2, 10)
            raw_results = await db_provider.query(
                collection=collection_name,
                query_embedding=query_embedding,
                top_k=top_k,
            )

            actual = [{"id": r["id"], "text": r["text"], "score": r["score"]} for r in raw_results]

            with trace_span("bulk_test.validation", attributes={"test_id": test_id}):
                validation_result = validate_results(expected=expected, actual=actual)

            duration = time.time() - start_time

            result: dict[str, Any] = {
                "test_id": test_id,
                "query": query,
                "expected": expected,
                "actual": actual,
                "status": validation_result["status"],
                "duration": round(duration, 3),
            }
            if validation_result["status"] == "failed":
                result["reason"] = validation_result["reason"]

            logger.debug("Test %s: %s", test_id, validation_result["status"])
            return result

    except DatabaseError as e:
        return _error_result(test_id, query, expected, start_time, "Database connection failed", e)
    except EmbeddingError as e:
        return _error_result(test_id, query, expected, start_time, "Embedding generation failed", e)
    except Exception as e:
        logger.exception("Unexpected error in test %s: %s", test_id, e)
        return _error_result(test_id, query, expected, start_time, "Unexpected error", e)


def _error_result(
    test_id: str,
    query: str,
    expected: list[dict[str, Any]],
    start_time: float,
    label: str,
    err: Exception,
) -> dict[str, Any]:
    """Build the standard error result row."""
    duration = time.time() - start_time
    logger.error("%s in test %s: %s", label, test_id, err)
    return {
        "test_id": test_id,
        "query": query,
        "expected": expected,
        "actual": [],
        "status": "error",
        "reason": f"{label}: {err}",
        "duration": round(duration, 3),
    }


def validate_results(expected: list[dict[str, Any]], actual: list[dict[str, Any]]) -> dict[str, Any]:
    """Compare actual vs expected results.

    Validation rules:
        1. Every expected ID must appear in the actual results.
        2. Expected IDs must appear in the same relative order.
        3. If ``min_threshold`` is set on an expected entry, the actual score
           must be greater than or equal to it.
    """
    actual_ids = [r["id"] for r in actual]
    actual_map = {r["id"]: r for r in actual}
    expected_ids = [e["id"] for e in expected]
    last_index = -1

    for exp in expected:
        exp_id = exp["id"]

        if exp_id not in actual_map:
            return {"status": "failed", "reason": f"Expected ID '{exp_id}' not found in results"}

        try:
            current_index = actual_ids.index(exp_id)
        except ValueError:
            return {"status": "failed", "reason": f"Expected ID '{exp_id}' not found in results"}

        if current_index <= last_index:
            return {
                "status": "failed",
                "reason": f"Expected order not matched: expected {expected_ids}, got {actual_ids}",
            }

        last_index = current_index

        if "min_threshold" in exp:
            min_threshold = exp["min_threshold"]
            actual_score = actual_map[exp_id]["score"]
            if actual_score < min_threshold:
                return {
                    "status": "failed",
                    "reason": (
                        f"Threshold not met for '{exp_id}': expected >= {min_threshold}, got {actual_score:.4f}"
                    ),
                }

    return {"status": "passed"}


def write_results_file(
    output: str,
    summary: dict[str, Any],
    test_results: list[dict[str, Any]],
    verbose: bool,
) -> None:
    """Write a YAML results file (always summary, full or failed-only tests)."""
    tests_to_write = test_results if verbose else [r for r in test_results if r["status"] in {"failed", "error"}]

    results_data = {"summary": summary, "tests": tests_to_write}

    try:
        with Path(output).open("w", encoding="utf-8") as f:
            yaml.dump(results_data, f, default_flow_style=False, sort_keys=False)
    except Exception as e:
        raise BulkTestError(f"Failed to write results file: {e}") from e

    logger.info("Results written to: %s", output)
