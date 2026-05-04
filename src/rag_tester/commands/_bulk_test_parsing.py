"""Parsing and validation helpers for the bulk-test command.

Kept private (leading underscore) because the public surface is
``rag_tester.commands.bulk_test.bulk_test_command``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml
from rich.console import Console

from rag_tester.core.validator import ValidationError

error_console = Console(stderr=True)


class BulkTestError(Exception):
    """Raised by the bulk-test command when a non-validation error occurs."""


def parse_test_file(file_path: str) -> list[dict[str, Any]]:
    """Parse a YAML/JSON test file and return validated test cases.

    Args:
        file_path: Path to the test definition file.

    Returns:
        List of test case dictionaries (each already validated).

    Raises:
        ValidationError: If the file is missing, malformed, or any test case
            fails validation.
    """
    path = Path(file_path)

    if not path.exists():
        raise ValidationError(f"File not found: {file_path}")

    if not path.is_file():
        raise ValidationError(f"Path is not a file: {file_path}")

    try:
        with path.open(encoding="utf-8") as f:
            if path.suffix.lower() in {".yaml", ".yml"}:
                data = yaml.safe_load(f)
            elif path.suffix.lower() == ".json":
                data = json.load(f)
            else:
                raise ValidationError(f"Unsupported file format: {path.suffix}. Expected .yaml, .yml, or .json")
    except yaml.YAMLError as e:
        raise ValidationError(f"Invalid test file format. Failed to parse YAML: {e}") from e
    except json.JSONDecodeError as e:
        raise ValidationError(f"Invalid test file format. Failed to parse JSON: {e}") from e
    except Exception as e:
        raise ValidationError(f"Failed to read test file: {e}") from e

    if not isinstance(data, dict):
        raise ValidationError("Test file must contain a dictionary at root level")

    if "tests" not in data:
        raise ValidationError("Test file must contain a 'tests' key")

    tests = data["tests"]
    if not isinstance(tests, list):
        raise ValidationError("'tests' must be a list")

    for i, test in enumerate(tests):
        validate_test_case(test, i)

    return tests


def validate_test_case(test: dict[str, Any], index: int) -> None:
    """Validate the schema of a single test case.

    Args:
        test: Test case dictionary.
        index: Position of the test in the list (for error messages).

    Raises:
        ValidationError: If a required field is missing or has the wrong type.
    """
    if not isinstance(test, dict):
        raise ValidationError(f"Test at index {index} is not a dictionary")

    if "test_id" not in test:
        raise ValidationError(f"Missing required field 'test_id' in test at index {index}")

    if "query" not in test:
        raise ValidationError(f"Missing required field 'query' in test '{test.get('test_id', index)}'")

    if "expected" not in test:
        raise ValidationError(f"Missing required field 'expected' in test '{test['test_id']}'")

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


def validate_parallel_workers(parallel: int) -> None:
    """Ensure the parallel-workers flag is between 1 and 16."""
    if not isinstance(parallel, int):
        raise ValidationError(f"Parallel workers must be an integer, got {type(parallel).__name__}")

    if parallel < 1 or parallel > 16:
        raise ValidationError("Parallel workers must be between 1 and 16")


def validate_output_path(output: str) -> None:
    """Ensure ``output`` is in an existing, writable directory."""
    path = Path(output)

    parent = path.parent
    if not parent.exists():
        raise ValidationError(f"Output directory does not exist: {parent}")

    if not parent.is_dir():
        raise ValidationError(f"Output parent path is not a directory: {parent}")

    try:
        path.touch(exist_ok=True)
    except Exception as e:
        raise ValidationError(f"Cannot write output file: {output} ({e})") from e


def parse_database_connection(database: str) -> dict[str, Any] | None:
    """Best-effort parse of a database connection string for display only.

    Returns ``None`` and prints to stderr when the URI is invalid; this is
    used to surface a helpful error message before instantiating a provider.
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

        return {"host": host, "port": port, "collection": collection_name}

    if database.startswith("postgresql://"):
        remainder = database.replace("postgresql://", "")
        parts = remainder.rsplit("/", 1)
        if len(parts) != 2:
            error_console.print(
                "[red]Error: Invalid PostgreSQL connection string. Expected format: postgresql://user:pass@host:port/dbname/table_name[/red]"
            )
            return None
        return {"host": "postgresql", "port": 0, "collection": parts[1]}

    if database.startswith("sqlite://"):
        remainder = database.replace("sqlite://", "")
        parts = remainder.rsplit("/", 1)
        if len(parts) != 2:
            error_console.print(
                "[red]Error: Invalid SQLite connection string. Expected format: sqlite:///path/to/db.db/table_name[/red]"
            )
            return None
        return {"host": "sqlite", "port": 0, "collection": parts[1]}

    error_console.print("[red]Error: Unsupported database. Use chromadb://..., postgresql://..., or sqlite://...[/red]")
    return None
