"""Tests for OpenTelemetry tracing."""

import json
import time
from pathlib import Path

import pytest
from opentelemetry import trace

from rag_tester.config import Settings
from rag_tester.tracing import setup_tracing, trace_span


class TestTracingConfiguration:
    """Tests for tracing setup."""

    def test_trace_directory_auto_created(self, tmp_path: Path) -> None:
        """Test that trace directory is automatically created.

        Corresponds to: E2E-INFRA-005
        """
        trace_file = tmp_path / "new_traces" / "test.jsonl"
        settings = Settings(trace_file=str(trace_file))

        # Directory should not exist yet
        assert not trace_file.parent.exists()

        # Setup tracing
        setup_tracing(settings)

        # Create a span
        with trace_span("test_operation", {"key": "value"}):
            pass

        # Force flush
        tracer_provider = trace.get_tracer_provider()
        if hasattr(tracer_provider, "force_flush"):
            tracer_provider.force_flush()  # type: ignore[attr-defined]

        # Give it a moment to write
        time.sleep(0.1)

        # Directory and file should now exist
        assert trace_file.parent.exists()
        assert trace_file.exists()

    def test_trace_file_written(self, tmp_path: Path) -> None:
        """Test that spans are written to JSONL file.

        Corresponds to: E2E-026
        """
        trace_file = tmp_path / "test_traces" / "test.jsonl"
        settings = Settings(trace_file=str(trace_file))

        setup_tracing(settings)

        # Create a span with attributes
        with trace_span("test_operation", {"key": "value", "number": 42}):
            pass

        # Force flush
        tracer_provider = trace.get_tracer_provider()
        if hasattr(tracer_provider, "force_flush"):
            tracer_provider.force_flush()  # type: ignore[attr-defined]

        # Give it a moment to write
        time.sleep(0.1)

        # Read and verify trace file
        assert trace_file.exists()
        content = trace_file.read_text()
        lines = [line for line in content.strip().split("\n") if line]

        assert len(lines) >= 1

        # Parse first span
        span_data = json.loads(lines[0])

        # Verify required fields
        assert "trace_id" in span_data
        assert "span_id" in span_data
        assert "name" in span_data
        assert span_data["name"] == "test_operation"
        assert "start_time" in span_data
        assert "end_time" in span_data
        assert "attributes" in span_data

        # Verify attributes
        attributes = span_data["attributes"]
        assert attributes["key"] == "value"
        assert attributes["number"] == 42

    def test_api_key_not_logged(self, tmp_path: Path) -> None:
        """Test that sensitive data is not logged in traces.

        Corresponds to: E2E-056
        """
        trace_file = tmp_path / "test.jsonl"
        settings = Settings(trace_file=str(trace_file))

        setup_tracing(settings)

        # Create a span with sensitive data
        sensitive_key = "sk-test-key-12345"
        with trace_span(
            "api_call",
            {
                "api_call": "true",
                "model": "test-model",
                "authorization": f"Bearer {sensitive_key}",
                "api_key": sensitive_key,
            },
        ):
            pass

        # Force flush
        tracer_provider = trace.get_tracer_provider()
        if hasattr(tracer_provider, "force_flush"):
            tracer_provider.force_flush()  # type: ignore[attr-defined]

        time.sleep(0.1)

        # Read trace file
        content = trace_file.read_text()

        # Verify sensitive data is NOT present
        assert sensitive_key not in content
        assert f"Bearer {sensitive_key}" not in content

        # Verify non-sensitive data IS present
        assert "api_call" in content
        assert "test-model" in content

        # Parse and verify redaction
        lines = [line for line in content.strip().split("\n") if line]
        span_data = json.loads(lines[0])
        attributes = span_data["attributes"]

        assert attributes["api_call"] == "true"
        assert attributes["model"] == "test-model"
        assert attributes["authorization"] == "[REDACTED]"
        assert attributes["api_key"] == "[REDACTED]"

    def test_nested_spans(self, tmp_path: Path) -> None:
        """Test that nested spans are properly traced."""
        trace_file = tmp_path / "test.jsonl"
        settings = Settings(trace_file=str(trace_file))

        setup_tracing(settings)

        # Create nested spans
        with trace_span("parent_operation", {"level": "parent"}):  # noqa: SIM117
            with trace_span("child_operation", {"level": "child"}):
                pass

        # Force flush
        tracer_provider = trace.get_tracer_provider()
        if hasattr(tracer_provider, "force_flush"):
            tracer_provider.force_flush()  # type: ignore[attr-defined]

        time.sleep(0.1)

        # Read and parse spans
        content = trace_file.read_text()
        lines = [line for line in content.strip().split("\n") if line]

        assert len(lines) >= 2

        spans = [json.loads(line) for line in lines]

        # Find parent and child spans
        parent_span = next(s for s in spans if s["name"] == "parent_operation")
        child_span = next(s for s in spans if s["name"] == "child_operation")

        # Verify parent-child relationship
        assert child_span["parent_span_id"] == parent_span["span_id"]
        assert child_span["trace_id"] == parent_span["trace_id"]

    def test_span_attributes(self, tmp_path: Path) -> None:
        """Test that span attributes are properly recorded."""
        trace_file = tmp_path / "test.jsonl"
        settings = Settings(trace_file=str(trace_file))

        setup_tracing(settings)

        # Create span with various attribute types
        with trace_span(
            "test_operation",
            {
                "string_attr": "value",
                "int_attr": 42,
                "float_attr": 3.14,
                "bool_attr": True,
            },
        ):
            pass

        # Force flush
        tracer_provider = trace.get_tracer_provider()
        if hasattr(tracer_provider, "force_flush"):
            tracer_provider.force_flush()  # type: ignore[attr-defined]

        time.sleep(0.1)

        # Read and verify
        content = trace_file.read_text()
        lines = [line for line in content.strip().split("\n") if line]
        span_data = json.loads(lines[0])

        attributes = span_data["attributes"]
        assert attributes["string_attr"] == "value"
        assert attributes["int_attr"] == 42
        assert attributes["float_attr"] == 3.14
        assert attributes["bool_attr"] is True

    def test_get_tracer_before_setup_raises_error(self) -> None:
        """Test that get_tracer raises error if tracing not initialized."""
        # Reset global tracer by importing fresh
        import importlib  # noqa: PLC0415

        import rag_tester.tracing  # noqa: PLC0415

        importlib.reload(rag_tester.tracing)

        with pytest.raises(RuntimeError, match="Tracing not initialized"):
            rag_tester.tracing.get_tracer()

    def test_span_timestamps(self, tmp_path: Path) -> None:
        """Test that span timestamps are in ISO format."""
        trace_file = tmp_path / "test.jsonl"
        settings = Settings(trace_file=str(trace_file))

        setup_tracing(settings)

        with trace_span("test_operation"):
            time.sleep(0.01)  # Small delay to ensure different timestamps

        # Force flush
        tracer_provider = trace.get_tracer_provider()
        if hasattr(tracer_provider, "force_flush"):
            tracer_provider.force_flush()  # type: ignore[attr-defined]

        time.sleep(0.1)

        # Read and verify
        content = trace_file.read_text()
        lines = [line for line in content.strip().split("\n") if line]
        span_data = json.loads(lines[0])

        # Verify ISO format timestamps
        start_time = span_data["start_time"]
        end_time = span_data["end_time"]

        # Should be valid ISO format
        from datetime import datetime  # noqa: PLC0415

        datetime.fromisoformat(start_time.replace("Z", "+00:00"))
        datetime.fromisoformat(end_time.replace("Z", "+00:00"))

        # End time should be after start time
        assert end_time > start_time

    def test_multiple_spans_appended(self, tmp_path: Path) -> None:
        """Test that multiple spans are appended to the same file."""
        trace_file = tmp_path / "test.jsonl"
        settings = Settings(trace_file=str(trace_file))

        setup_tracing(settings)

        # Create multiple spans
        for i in range(3):
            with trace_span(f"operation_{i}", {"index": i}):
                pass

        # Force flush
        tracer_provider = trace.get_tracer_provider()
        if hasattr(tracer_provider, "force_flush"):
            tracer_provider.force_flush()  # type: ignore[attr-defined]

        time.sleep(0.1)

        # Read and verify
        content = trace_file.read_text()
        lines = [line for line in content.strip().split("\n") if line]

        assert len(lines) >= 3

        # Verify each span
        for i, line in enumerate(lines[:3]):
            span_data = json.loads(line)
            assert span_data["name"] == f"operation_{i}"
            assert span_data["attributes"]["index"] == i
