"""Shared test fixtures and configuration."""

import tempfile
from pathlib import Path

import pytest
from opentelemetry import trace

from rag_tester.config import Settings
from rag_tester.tracing import setup_tracing


@pytest.fixture(autouse=True)
def setup_test_tracing(tmp_path: Path) -> None:
    """Set up tracing for tests with a temporary trace file."""
    # Create a temporary trace file for this test
    trace_file = tmp_path / "test_traces.jsonl"
    
    # Create test settings with the temporary trace file
    settings = Settings(trace_file=str(trace_file))
    
    # Initialize tracing
    setup_tracing(settings)
    
    yield
    
    # After test, reset to a new no-op provider to clear state
    trace._TRACER_PROVIDER = None  # type: ignore[attr-defined]
    trace._TRACER_PROVIDER_SET_ONCE = trace.Once()  # type: ignore[attr-defined]
