"""OpenTelemetry tracing with JSONL file export.

Provides distributed tracing for all operations with:
- JSONL file export (one span per line)
- Optional OTLP endpoint support
- Context manager for span creation
- Automatic attribute sanitization (no sensitive data)
"""

import json
import logging
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter, SpanExportResult

try:
    from opentelemetry.sdk.trace import ReadableSpan
except ImportError:
    from opentelemetry.sdk.trace.export import ReadableSpan  # type: ignore[attr-defined]

from rag_tester.config import Settings

logger = logging.getLogger(__name__)

# Sensitive data patterns to filter from attributes
# These are exact matches or compound patterns (e.g., "api_key", "auth_token")
SENSITIVE_PATTERNS = [
    "api_key",
    "apikey",
    "api-key",
    "authorization",
    "bearer",
    "token",
    "password",
    "secret",
    "credential",
]

# Additional patterns that need word boundaries (not standalone "key")
SENSITIVE_COMPOUND_PATTERNS = ["_key", "-key", "key_", "key-"]

# Minimum length for a string to be considered potentially sensitive
MIN_SENSITIVE_STRING_LENGTH = 20


class JSONLSpanExporter(SpanExporter):
    """Export spans to a JSONL file (one JSON object per line)."""

    def __init__(self, file_path: str) -> None:
        """Initialize the JSONL exporter.

        Args:
            file_path: Path to the JSONL output file
        """
        self.file_path = Path(file_path)
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("JSONL span exporter initialized: file=%s", self.file_path)

    def export(self, spans: list[ReadableSpan]) -> SpanExportResult:  # type: ignore[override]
        """Export spans to JSONL file.

        Args:
            spans: List of spans to export

        Returns:
            SpanExportResult indicating success or failure
        """
        try:
            with self.file_path.open("a", encoding="utf-8") as f:
                for span in spans:
                    span_dict = self._span_to_dict(span)
                    f.write(json.dumps(span_dict) + "\n")
            return SpanExportResult.SUCCESS
        except Exception as e:
            logger.error("Failed to export spans: %s", e)
            return SpanExportResult.FAILURE

    def _span_to_dict(self, span: ReadableSpan) -> dict[str, Any]:
        """Convert a span to a dictionary for JSON serialization.

        Args:
            span: The span to convert

        Returns:
            Dictionary representation of the span
        """
        # Extract context
        span_context = span.get_span_context()
        parent_context = span.parent

        # Convert timestamps to ISO format
        start_time_ns = span.start_time if span.start_time is not None else 0
        end_time_ns = span.end_time if span.end_time is not None else 0
        start_time = datetime.fromtimestamp(start_time_ns / 1e9, tz=UTC).isoformat()
        end_time = datetime.fromtimestamp(end_time_ns / 1e9, tz=UTC).isoformat()

        # Sanitize attributes to remove sensitive data
        attributes = self._sanitize_attributes(dict(span.attributes) if span.attributes else {})

        parent_span_id = None
        if parent_context is not None:
            parent_span_id = format(parent_context.span_id, "016x")
        
        # Handle potential None span_context (should not happen in practice)
        trace_id = format(span_context.trace_id, "032x") if span_context else "0" * 32
        span_id = format(span_context.span_id, "016x") if span_context else "0" * 16
        
        return {
            "trace_id": trace_id,
            "span_id": span_id,
            "parent_span_id": parent_span_id,
            "name": span.name,
            "start_time": start_time,
            "end_time": end_time,
            "attributes": attributes,
            "status": span.status.status_code.name if span.status else "UNSET",
        }

    def _sanitize_attributes(self, attributes: dict[str, Any]) -> dict[str, Any]:
        """Remove sensitive data from span attributes.

        Args:
            attributes: Original attributes dictionary

        Returns:
            Sanitized attributes dictionary
        """
        sanitized = {}
        for key, value in attributes.items():
            key_lower = key.lower()
            
            # Check exact matches with main patterns
            is_sensitive = key_lower in SENSITIVE_PATTERNS
            
            # Check compound patterns (e.g., "api_key", "auth_token")
            if not is_sensitive:
                is_sensitive = any(pattern in key_lower for pattern in SENSITIVE_COMPOUND_PATTERNS)
            
            if is_sensitive:
                sanitized[key] = "[REDACTED]"
            # Check if value looks like a sensitive string
            elif isinstance(value, str) and len(value) > MIN_SENSITIVE_STRING_LENGTH:
                value_lower = value.lower()
                if any(pattern in value_lower for pattern in ["bearer", "sk-", "key-"]):
                    sanitized[key] = "[REDACTED]"
                else:
                    sanitized[key] = value
            else:
                sanitized[key] = value
        return sanitized

    def shutdown(self) -> None:
        """Shutdown the exporter."""
        logger.info("JSONL span exporter shutdown")

    def force_flush(self, _timeout_millis: int = 30000) -> bool:
        """Force flush any buffered spans.

        Args:
            timeout_millis: Timeout in milliseconds

        Returns:
            True if successful, False otherwise
        """
        return True


_tracer_provider: TracerProvider | None = None
_tracer: trace.Tracer | None = None


def setup_tracing(settings: Settings) -> None:
    """Configure OpenTelemetry tracing with JSONL export.

    Args:
        settings: Application settings containing trace_file and otel_endpoint
    """
    global _tracer_provider, _tracer  # noqa: PLW0603

    # Create resource with service information
    resource = Resource.create(
        {
            "service.name": "rag-tester",
            "service.version": "0.1.0",
        }
    )

    # Create tracer provider
    _tracer_provider = TracerProvider(resource=resource)

    # Add JSONL exporter
    jsonl_exporter = JSONLSpanExporter(settings.trace_file)
    _tracer_provider.add_span_processor(SimpleSpanProcessor(jsonl_exporter))

    # Set as global tracer provider
    trace.set_tracer_provider(_tracer_provider)

    # Get tracer
    _tracer = trace.get_tracer(__name__)

    logger.info("Tracing configured: file=%s", settings.trace_file)


def get_tracer() -> trace.Tracer:
    """Get the global tracer instance.

    Returns:
        The global tracer instance

    Raises:
        RuntimeError: If tracing has not been set up
    """
    if _tracer is None:
        raise RuntimeError("Tracing not initialized. Call setup_tracing() first.")
    return _tracer


@contextmanager
def trace_span(name: str, attributes: dict[str, Any] | None = None) -> Iterator[trace.Span]:
    """Context manager for creating a traced span.

    Args:
        name: Name of the span
        attributes: Optional attributes to attach to the span

    Yields:
        The created span

    Example:
        with trace_span("database.query", {"table": "users", "operation": "select"}):
            # traced code
            pass
    """
    tracer = get_tracer()
    with tracer.start_as_current_span(name) as span:
        if attributes:
            for key, value in attributes.items():
                span.set_attribute(key, value)
        yield span
