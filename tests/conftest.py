"""Shared test fixtures and configuration."""

import os
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


@pytest.fixture(scope="session")
def chromadb_server():
    """Provide ChromaDB server connection details.

    This fixture returns the host and port for a running ChromaDB instance.
    By default, it uses localhost:8000, but can be overridden via environment variables.

    Note: This fixture assumes ChromaDB is already running. For CI/CD, you should
    start ChromaDB before running tests (e.g., via docker-compose).
    """
    host = os.getenv("CHROMADB_HOST", "localhost")
    port = int(os.getenv("CHROMADB_PORT", "8000"))

    return host, port
