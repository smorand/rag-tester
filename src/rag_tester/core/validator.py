"""Input validation for RAG Tester load command.

This module provides validation functions for load command arguments
and input data records.
"""

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Raised when validation fails."""

    pass


def validate_file_path(file_path: str) -> Path:
    """Validate that file exists and is readable.

    Args:
        file_path: Path to the file to validate

    Returns:
        Path object for the validated file

    Raises:
        ValidationError: If file doesn't exist or is not readable
    """
    path = Path(file_path)

    if not path.exists():
        msg = f"File not found: {file_path}"
        logger.error(msg)
        raise ValidationError(msg)

    if not path.is_file():
        msg = f"Path is not a file: {file_path}"
        logger.error(msg)
        raise ValidationError(msg)

    if path.suffix.lower() not in {".yaml", ".yml", ".json"}:
        msg = f"Unsupported file format: {path.suffix}. Expected .yaml, .yml, or .json"
        logger.error(msg)
        raise ValidationError(msg)

    logger.debug("File validated: %s", file_path)
    return path


def validate_batch_size(batch_size: int) -> None:
    """Validate batch size is within allowed range.

    Args:
        batch_size: Batch size to validate

    Raises:
        ValidationError: If batch size is out of range
    """
    if not 1 <= batch_size <= 256:
        msg = f"Batch size must be between 1 and 256, got: {batch_size}"
        logger.error(msg)
        raise ValidationError(msg)

    logger.debug("Batch size validated: %s", batch_size)


def validate_parallel_workers(parallel: int) -> None:
    """Validate number of parallel workers is within allowed range.

    Args:
        parallel: Number of parallel workers to validate

    Raises:
        ValidationError: If parallel workers is out of range
    """
    if not 1 <= parallel <= 16:
        msg = f"Parallel workers must be between 1 and 16, got: {parallel}"
        logger.error(msg)
        raise ValidationError(msg)

    logger.debug("Parallel workers validated: %s", parallel)


def validate_record(record: dict[str, Any], record_index: int) -> None:
    """Validate that a record has required fields.

    Args:
        record: Record dictionary to validate
        record_index: Index of the record in the file (for error messages)

    Raises:
        ValidationError: If record is missing required fields
    """
    if not isinstance(record, dict):
        msg = f"Record at index {record_index} is not a dictionary"
        logger.error(msg)
        raise ValidationError(msg)

    # Check for required 'id' field
    if "id" not in record:
        msg = f"Missing required field 'id' in record at index {record_index}"
        logger.error(msg)
        raise ValidationError(msg)

    # Check for required 'text' field
    if "text" not in record:
        record_id = record.get("id", f"index {record_index}")
        msg = f"Missing required field 'text' in record '{record_id}'"
        logger.error(msg)
        raise ValidationError(msg)

    # Validate field types
    if not isinstance(record["id"], str):
        msg = f"Field 'id' must be a string in record at index {record_index}"
        logger.error(msg)
        raise ValidationError(msg)

    if not isinstance(record["text"], str):
        record_id = record["id"]
        msg = f"Field 'text' must be a string in record '{record_id}'"
        logger.error(msg)
        raise ValidationError(msg)

    logger.debug("Record validated: %s", record["id"])


def validate_load_mode(mode: str) -> None:
    """Validate load mode is supported.

    Args:
        mode: Load mode to validate

    Raises:
        ValidationError: If mode is not supported
    """
    valid_modes = {"initial", "upsert", "flush"}
    if mode not in valid_modes:
        msg = f"Invalid load mode: {mode}. Must be one of: {', '.join(valid_modes)}"
        logger.error(msg)
        raise ValidationError(msg)

    logger.debug("Load mode validated: %s", mode)
