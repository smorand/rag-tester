"""Compare command implementation for RAG Tester.

This module provides the CLI command for comparing test results from different models.
"""

import logging
from pathlib import Path
from typing import Any

import typer
import yaml
from rich.console import Console

from rag_tester.core.comparator import (
    ComparatorError,
    calculate_aggregate_metrics,
    identify_per_test_differences,
    validate_result_files_compatible,
)
from rag_tester.tracing import trace_span

logger = logging.getLogger(__name__)
console = Console()
error_console = Console(stderr=True)


def compare_command(
    results: list[str] = typer.Argument(
        ...,
        help="Paths to 2+ result files (YAML)",
    ),
    output: str = typer.Option(
        ...,
        "--output",
        "-o",
        help="Output file path (YAML)",
    ),
) -> None:
    """Compare test results from different models or configurations.

    This command analyzes test results from multiple models, calculates aggregate
    metrics (pass rate, average score, cost), and identifies per-test differences
    where models disagree on pass/fail status.

    Examples:
        # Compare two models
        rag-tester compare results_model_a.yaml results_model_b.yaml --output comparison.yaml

        # Compare three models
        rag-tester compare results_a.yaml results_b.yaml results_c.yaml -o comparison.yaml
    """
    try:
        # Validate inputs
        if len(results) < 2:
            error_console.print("[red]Error: At least 2 result files required for comparison[/red]")
            raise typer.Exit(code=1)

        logger.info(f"Comparing {len(results)} result files")

        # Parse result files
        with trace_span("compare.parse_files", attributes={"num_files": len(results)}):
            results_data = []
            for result_file in results:
                result_data = _parse_result_file(result_file)
                model_name = result_data.get("summary", {}).get("embedding_model", "unknown")
                results_data.append((model_name, result_data))

        # Validate compatibility
        with trace_span("compare.validate_compatibility"):
            validate_result_files_compatible(results_data)

        # Calculate aggregate metrics for each model
        with trace_span("compare.calculate_metrics"):
            model_metrics: dict[str, dict[str, Any]] = {}
            for model_name, result_data in results_data:
                metrics = calculate_aggregate_metrics(result_data, model_name)
                # Use a safe key name (model_a, model_b, etc.)
                key = f"model_{chr(97 + len(model_metrics))}"  # 97 is 'a'
                model_metrics[key] = metrics

        # Identify per-test differences
        with trace_span("compare.identify_differences"):
            per_test_diff = identify_per_test_differences(results_data)

        # Generate comparison file
        comparison_data = {**model_metrics, "per_test_diff": per_test_diff}

        with trace_span("compare.write_output", attributes={"output": output}):
            _write_comparison_file(output, comparison_data)

        # Log summary
        num_diffs = len(per_test_diff)
        total_tests = model_metrics.get("model_a", {}).get("pass_rate", 0)  # Get from first model
        logger.info(f"Comparison complete: {len(results)} models, {total_tests} tests, {num_diffs} differences")

        # Display summary to user
        console.print()
        console.print(f"[green]✓[/green] Comparison complete: {len(results)} models compared")
        if num_diffs > 0:
            console.print(f"[yellow]![/yellow] Found {num_diffs} test differences between models")
        else:
            console.print("[green]✓[/green] No test differences found between models")
        console.print(f"[blue]ℹ[/blue] Comparison written to: {output}")

        # Trace completion
        with trace_span(
            "compare.complete",
            attributes={
                "models_compared": len(results),
                "total_tests": total_tests,
                "differences": num_diffs,
            },
        ):
            pass

    except ComparatorError as e:
        error_console.print(f"[red]Error: {e}[/red]")
        logger.error(f"Comparator error: {e}")
        raise typer.Exit(code=1)

    except Exception as e:
        error_console.print(f"[red]Error: Unexpected error: {e}[/red]")
        logger.exception("Unexpected error during comparison")
        raise typer.Exit(code=1)


def _parse_result_file(file_path: str) -> dict[str, Any]:
    """Parse a result file (YAML).

    Args:
        file_path: Path to result file

    Returns:
        Parsed result data dictionary

    Raises:
        ComparatorError: If file is invalid or malformed
    """
    path = Path(file_path)

    if not path.exists():
        raise ComparatorError(f"Result file not found: {file_path}")

    if not path.is_file():
        raise ComparatorError(f"Path is not a file: {file_path}")

    try:
        with open(path) as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ComparatorError(f"Invalid result file format: {file_path}. Failed to parse YAML: {e}")
    except Exception as e:
        raise ComparatorError(f"Failed to read result file: {file_path}. Error: {e}")

    if not isinstance(data, dict):
        raise ComparatorError(f"Invalid result file format: {file_path}. Expected dictionary at root level")

    if "summary" not in data:
        raise ComparatorError(f"Invalid result file format: {file_path}. Missing 'summary' section")

    if "tests" not in data:
        raise ComparatorError(f"Invalid result file format: {file_path}. Missing 'tests' section")

    logger.debug(f"Parsed result file: {file_path}")

    return data


def _write_comparison_file(output: str, comparison_data: dict[str, Any]) -> None:
    """Write comparison data to YAML file.

    Args:
        output: Output file path
        comparison_data: Comparison data dictionary

    Raises:
        ComparatorError: If file write fails
    """
    path = Path(output)

    # Check parent directory exists and is writable
    parent = path.parent
    if not parent.exists():
        raise ComparatorError(f"Output directory does not exist: {parent}")

    try:
        with open(path, "w") as f:
            yaml.dump(comparison_data, f, default_flow_style=False, sort_keys=False)
    except Exception as e:
        raise ComparatorError(f"Failed to write comparison file: {output}. Error: {e}")

    logger.info(f"Comparison written to: {output}")
