"""File I/O utilities for reading records from YAML and JSON files."""

import json
import logging
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any

import aiofiles
import yaml
from opentelemetry import trace

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)


class ValidationError(Exception):
    """Raised when file or record validation fails."""

    pass


async def read_yaml(file_path: str | Path) -> AsyncGenerator[dict[str, Any]]:
    """Read records from a YAML file with streaming support.

    Args:
        file_path: Path to the YAML file

    Yields:
        Records with 'id' and 'text' fields

    Raises:
        ValidationError: If file is empty, invalid, or records are missing required fields
        FileNotFoundError: If file doesn't exist
    """
    path = Path(file_path)

    with tracer.start_as_current_span("file_read") as span:
        span.set_attribute("file.path", str(path))
        span.set_attribute("file.format", "yaml")

        try:
            # Get file size for tracing
            file_size = path.stat().st_size
            span.set_attribute("file.size", file_size)

            logger.debug("Reading YAML file: %s (%s bytes)", path, file_size)

            # Read file content
            async with aiofiles.open(path, encoding="utf-8") as f:
                content = await f.read()

            # Parse YAML
            data = yaml.safe_load(content)

            if not data:
                msg = "Input file is empty or has no records"
                raise ValidationError(msg)

            # Extract records
            records = data.get("records", [])
            if not records:
                msg = "Input file is empty or has no records"
                raise ValidationError(msg)

            span.set_attribute("records.count", len(records))

            # Validate and yield records
            for record in records:
                _validate_record(record)
                yield record

            logger.debug("Read %s records from %s", len(records), path)

        except ValidationError:
            raise
        except FileNotFoundError:
            logger.error("File not found: %s", path)
            raise
        except yaml.YAMLError as e:
            error_msg = f"Invalid YAML format in {path}: {e}"
            logger.error(error_msg)
            span.record_exception(e)
            raise ValidationError(error_msg) from e
        except Exception as e:
            error_msg = f"Failed to read YAML file {path}: {e}"
            logger.error(error_msg)
            span.record_exception(e)
            raise


async def read_json(file_path: str | Path) -> AsyncGenerator[dict[str, Any]]:
    """Read records from a JSON file with streaming support.

    Args:
        file_path: Path to the JSON file

    Yields:
        Records with 'id' and 'text' fields

    Raises:
        ValidationError: If file is empty, invalid, or records are missing required fields
        FileNotFoundError: If file doesn't exist
    """
    path = Path(file_path)

    with tracer.start_as_current_span("file_read") as span:
        span.set_attribute("file.path", str(path))
        span.set_attribute("file.format", "json")

        try:
            # Get file size for tracing
            file_size = path.stat().st_size
            span.set_attribute("file.size", file_size)

            logger.debug("Reading JSON file: %s (%s bytes)", path, file_size)

            # Read file content
            async with aiofiles.open(path, encoding="utf-8") as f:
                content = await f.read()

            # Parse JSON
            data = json.loads(content)

            if not data:
                msg = "Input file is empty or has no records"
                raise ValidationError(msg)

            # Extract records
            records = data.get("records", [])
            if not records:
                msg = "Input file is empty or has no records"
                raise ValidationError(msg)

            span.set_attribute("records.count", len(records))

            # Validate and yield records
            for record in records:
                _validate_record(record)
                yield record

            logger.debug("Read %s records from %s", len(records), path)

        except ValidationError:
            raise
        except FileNotFoundError:
            logger.error("File not found: %s", path)
            raise
        except json.JSONDecodeError as e:
            error_msg = f"Invalid JSON format in {path}: {e}"
            logger.error(error_msg)
            span.record_exception(e)
            raise ValidationError(error_msg) from e
        except Exception as e:
            error_msg = f"Failed to read JSON file {path}: {e}"
            logger.error(error_msg)
            span.record_exception(e)
            raise


def _validate_record(record: dict[str, Any]) -> None:
    """Validate that a record has required fields.

    Args:
        record: Record to validate

    Raises:
        ValidationError: If required fields are missing
    """
    if "id" not in record:
        msg = "Missing required field 'id' in record"
        raise ValidationError(msg)

    if "text" not in record:
        record_id = record.get("id", "unknown")
        msg = f"Missing required field 'text' in record '{record_id}'"
        raise ValidationError(msg)
