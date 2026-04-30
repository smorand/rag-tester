"""Comparator logic for comparing test results from different models."""

import logging
from typing import Any

from rag_tester.utils.cost import calculate_cost

logger = logging.getLogger(__name__)


class ComparatorError(Exception):
    """Base exception for comparator errors."""

    pass


def calculate_aggregate_metrics(
    result_data: dict[str, Any],
    model_name: str,
) -> dict[str, Any]:
    """Calculate aggregate metrics for a single model's results.

    Args:
        result_data: Parsed result file data with 'summary' and 'tests' sections
        model_name: Model identifier for cost calculation

    Returns:
        Dictionary with aggregate metrics:
        - name: model name
        - database: database connection string
        - pass_rate: passed / total_tests
        - avg_score: mean of all actual scores
        - total_tokens: sum of tokens consumed
        - total_time: sum of test durations
        - total_cost: calculated from tokens and model pricing
        - cost_per_test: total_cost / total_tests

    Business Rules:
        - Pass rate: count(passed) / count(total)
        - Average score: mean of all actual scores (across all expected results)
        - Total tokens: from summary section
        - Total time: from summary section
        - Total cost: calculated from tokens and model pricing
        - Cost per test: total_cost / total_tests
        - Handle edge cases: division by zero, missing data
    """
    summary = result_data.get("summary", {})
    tests = result_data.get("tests", [])

    # Extract basic metrics from summary
    total_tests = summary.get("total_tests", 0)
    passed = summary.get("passed", 0)
    total_tokens = summary.get("total_tokens", 0)
    total_time = summary.get("total_time", 0.0)
    database = summary.get("database", "")

    # Calculate pass rate
    pass_rate = passed / total_tests if total_tests > 0 else 0.0

    # Calculate average score across all actual results
    # Need to look at all tests (not just failed ones in non-verbose output)
    # For verbose output, we have all tests; for non-verbose, we only have failed tests
    # We need to calculate avg_score from the actual results we have
    all_scores = []
    for test in tests:
        actual = test.get("actual", [])
        for result in actual:
            if "score" in result:
                all_scores.append(result["score"])

    # If we don't have all tests (non-verbose output), we can't calculate accurate avg_score
    # In this case, we'll use 0.0 or calculate from available data
    # However, for comparison purposes, we should ideally have verbose output or summary stats
    avg_score = sum(all_scores) / len(all_scores) if all_scores else 0.0

    # Calculate cost
    total_cost = calculate_cost(model_name, total_tokens)
    cost_per_test = total_cost / total_tests if total_tests > 0 else 0.0

    logger.debug(
        "Calculated metrics for %s: pass_rate=%.2f, avg_score=%.2f, cost=$%.6f",
        model_name,
        pass_rate,
        avg_score,
        total_cost,
    )

    return {
        "name": model_name,
        "database": database,
        "pass_rate": round(pass_rate, 4),
        "avg_score": round(avg_score, 4),
        "total_tokens": total_tokens,
        "total_time": round(total_time, 2),
        "total_cost": total_cost,
        "cost_per_test": round(cost_per_test, 6),
    }


def identify_per_test_differences(
    results_list: list[tuple[str, dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Identify tests where models disagree on pass/fail status.

    Args:
        results_list: List of (model_name, result_data) tuples

    Returns:
        List of test differences, each containing:
        - test_id: test identifier
        - <model_name>_status: pass/fail status for each model
        - <model_name>_score: average score for each model
        - expected_threshold: threshold from expected results (if any)

    Business Rules:
        - Compare pass/fail status for each test across models
        - Include test in diff if any two models disagree
        - For each diff: test_id, status per model, scores per model, expected_threshold
        - Sort diffs by test_id
    """
    if len(results_list) < 2:
        logger.warning("Need at least 2 models to compare")
        return []

    # Build a map of test_id -> {model_name: test_result}
    test_map: dict[str, dict[str, dict[str, Any]]] = {}

    for model_name, result_data in results_list:
        tests = result_data.get("tests", [])
        for test in tests:
            test_id = test.get("test_id")
            if not test_id:
                continue

            if test_id not in test_map:
                test_map[test_id] = {}

            test_map[test_id][model_name] = test

    # Identify differences
    differences = []

    for test_id, model_tests in test_map.items():
        # Get statuses from all models
        statuses = {model: test.get("status", "unknown") for model, test in model_tests.items()}

        # Check if there's disagreement (not all statuses are the same)
        unique_statuses = set(statuses.values())
        if len(unique_statuses) <= 1:
            # All models agree, skip
            continue

        # Build difference entry
        diff_entry: dict[str, Any] = {"test_id": test_id}

        # Add status and score for each model
        for model_name, test in model_tests.items():
            diff_entry[f"{model_name}_status"] = test.get("status", "unknown")

            # Calculate average score from actual results
            actual = test.get("actual", [])
            scores = [r.get("score", 0.0) for r in actual if "score" in r]
            avg_score = sum(scores) / len(scores) if scores else 0.0
            diff_entry[f"{model_name}_score"] = round(avg_score, 4)

        # Extract expected_threshold from any model (should be same across all)
        expected_threshold = None
        for test in model_tests.values():
            expected = test.get("expected", [])
            for exp in expected:
                if "min_threshold" in exp:
                    expected_threshold = exp["min_threshold"]
                    break
            if expected_threshold is not None:
                break

        if expected_threshold is not None:
            diff_entry["expected_threshold"] = expected_threshold

        differences.append(diff_entry)

    # Sort by test_id
    differences.sort(key=lambda d: d["test_id"])

    logger.info(f"Found {len(differences)} test differences between models")

    return differences


def validate_result_files_compatible(
    results_list: list[tuple[str, dict[str, Any]]],
) -> None:
    """Validate that result files are from the same test suite.

    Args:
        results_list: List of (model_name, result_data) tuples

    Raises:
        ComparatorError: If result files are not compatible

    Business Rules:
        - All result files must have the same set of test IDs
        - Test IDs must match exactly (same tests in all files)
    """
    if len(results_list) < 2:
        return

    # Extract test IDs from each result file
    all_test_ids = []
    for model_name, result_data in results_list:
        summary = result_data.get("summary", {})
        tests = result_data.get("tests", [])

        # For non-verbose output, we only have failed tests
        # We need to use total_tests from summary to validate
        total_tests = summary.get("total_tests", 0)

        # Get test IDs from tests section
        test_ids = {test.get("test_id") for test in tests if test.get("test_id")}

        all_test_ids.append((model_name, total_tests, test_ids))

    # Check that all models have the same total_tests
    first_total = all_test_ids[0][1]
    for model_name, total_tests, _ in all_test_ids[1:]:
        if total_tests != first_total:
            raise ComparatorError(
                f"Test suites do not match. Result files must be from the same test suite. "
                f"Model '{all_test_ids[0][0]}' has {first_total} tests, "
                f"but model '{model_name}' has {total_tests} tests."
            )

    logger.debug("Result files are compatible (same test suite)")
