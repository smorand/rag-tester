"""Shared test fixtures and configuration."""

import pytest
from opentelemetry import trace


@pytest.fixture(autouse=True)
def reset_tracer() -> None:
    """Reset the global tracer provider between tests to avoid state pollution."""
    # Store the original provider
    trace.get_tracer_provider()

    yield

    # After test, reset to a new no-op provider to clear state
    # This allows each test to set its own provider
    trace._TRACER_PROVIDER = None  # type: ignore[attr-defined]
    trace._TRACER_PROVIDER_SET_ONCE = trace.Once()  # type: ignore[attr-defined]
