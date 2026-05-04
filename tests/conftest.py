"""Shared test fixtures and configuration."""

import os
from collections.abc import Iterator
from pathlib import Path

import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from rag_tester.config import Settings, get_settings
from rag_tester.tracing import reset_tracing, setup_tracing


@pytest.fixture(autouse=True)
def _clear_settings_cache() -> Iterator[None]:
    """Clear the Settings lru_cache before and after each test.

    Tests that mutate environment variables via monkeypatch must see the
    updated values, which the lru_cache would otherwise hide.
    """
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture(autouse=True)
def setup_test_tracing(tmp_path: Path) -> Iterator[None]:
    """Initialize tracing for each test with a temporary trace file.

    Resets OpenTelemetry global state before and after each test so tests
    can call ``setup_tracing`` with their own settings without conflict.
    """
    # Defensive reset in case a prior test left state behind
    reset_tracing()

    # Create a temporary trace file for this test
    trace_file = tmp_path / "test_traces.jsonl"

    # Create test settings with the temporary trace file
    settings = Settings(trace_file=str(trace_file))

    # Initialize tracing
    setup_tracing(settings)

    yield

    # Clean shutdown after test
    reset_tracing()


@pytest.fixture()
def tracing_memory(tmp_path: Path) -> Iterator[InMemorySpanExporter]:
    """Initialize tracing with an in-memory span exporter for assertions.

    Yields the exporter so tests can call ``get_finished_spans()`` to verify
    span content without parsing JSONL files.
    """
    reset_tracing()

    exporter = InMemorySpanExporter()
    settings = Settings(trace_file=str(tmp_path / "unused.jsonl"))
    setup_tracing(settings, exporter=exporter)

    yield exporter

    reset_tracing()


@pytest.fixture(scope="session")
def chromadb_server() -> tuple[str, int]:
    """Provide ChromaDB server connection details.

    This fixture returns the host and port for a running ChromaDB instance.
    By default, it uses localhost:8000, but can be overridden via environment variables.

    Note: This fixture assumes ChromaDB is already running. For CI/CD, you should
    start ChromaDB before running tests (e.g., via docker-compose).
    """
    host = os.getenv("CHROMADB_HOST", "localhost")
    port = int(os.getenv("CHROMADB_PORT", "8000"))

    return host, port
