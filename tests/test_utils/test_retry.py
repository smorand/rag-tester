"""Tests for retry decorator with exponential backoff."""

import asyncio
import time
from pathlib import Path

import pytest

from rag_tester.config import Settings
from rag_tester.tracing import setup_tracing
from rag_tester.utils.retry import RetryError, retry_with_backoff


class TestRetryDecorator:
    """Tests for retry_with_backoff decorator."""

    def test_successful_operation_no_retry(self) -> None:
        """Test that successful operations don't trigger retries."""
        call_count = 0

        @retry_with_backoff(max_attempts=3)
        def successful_operation() -> str:
            nonlocal call_count
            call_count += 1
            return "success"

        result = successful_operation()

        assert result == "success"
        assert call_count == 1

    def test_transient_error_with_retry(self, tmp_path: Path) -> None:
        """Test that transient errors trigger retries.

        Corresponds to: E2E-030
        """
        # Setup tracing and logging
        settings = Settings(
            trace_file=str(tmp_path / "trace.jsonl"),
            log_file=str(tmp_path / "test.log"),
        )
        setup_tracing(settings)

        call_count = 0

        @retry_with_backoff(max_attempts=3, initial_delay=0.1)
        def failing_then_success() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ConnectionError("Transient error")
            return "success"

        result = failing_then_success()

        assert result == "success"
        assert call_count == 2

    def test_max_attempts_exhausted(self, tmp_path: Path) -> None:
        """Test that RetryError is raised after max attempts.

        Corresponds to: E2E-INFRA-003
        """
        settings = Settings(
            trace_file=str(tmp_path / "trace.jsonl"),
            log_file=str(tmp_path / "test.log"),
        )
        setup_tracing(settings)

        call_count = 0

        @retry_with_backoff(max_attempts=3, initial_delay=0.1)
        def always_fails() -> str:
            nonlocal call_count
            call_count += 1
            raise ConnectionError("Always fails")

        with pytest.raises(RetryError, match="Max retry attempts \\(3\\) exceeded"):
            always_fails()

        assert call_count == 3

    def test_permanent_error_no_retry(self, tmp_path: Path) -> None:
        """Test that permanent errors are not retried.

        Corresponds to: E2E-INFRA-004
        """
        settings = Settings(
            trace_file=str(tmp_path / "trace.jsonl"),
            log_file=str(tmp_path / "test.log"),
        )
        setup_tracing(settings)

        call_count = 0

        @retry_with_backoff(max_attempts=5, initial_delay=0.1)
        def permanent_error() -> str:
            nonlocal call_count
            call_count += 1
            raise ValueError("Invalid input")

        with pytest.raises(ValueError, match="Invalid input"):
            permanent_error()

        # Should only be called once (no retries)
        assert call_count == 1

    def test_exponential_backoff_timing(self, tmp_path: Path) -> None:
        """Test that exponential backoff timing is correct.

        Corresponds to: E2E-INFRA-007
        """
        settings = Settings(
            trace_file=str(tmp_path / "trace.jsonl"),
            log_file=str(tmp_path / "test.log"),
        )
        setup_tracing(settings)

        call_times: list[float] = []

        @retry_with_backoff(max_attempts=4, initial_delay=0.1, backoff_multiplier=2.0)
        def failing_operation() -> str:
            call_times.append(time.time())
            if len(call_times) < 4:
                raise ConnectionError("Transient error")
            return "success"

        result = failing_operation()

        assert result == "success"
        assert len(call_times) == 4

        # Calculate delays between attempts
        delays = [call_times[i + 1] - call_times[i] for i in range(len(call_times) - 1)]

        # Expected delays: 0.1, 0.2, 0.4 (with some tolerance)
        assert 0.08 <= delays[0] <= 0.15  # ~0.1s
        assert 0.18 <= delays[1] <= 0.25  # ~0.2s
        assert 0.38 <= delays[2] <= 0.45  # ~0.4s

    def test_retry_with_custom_transient_errors(self) -> None:
        """Test retry with custom transient error types."""
        call_count = 0

        class CustomError(Exception):
            pass

        @retry_with_backoff(max_attempts=3, initial_delay=0.1, transient_errors=(CustomError,))
        def custom_error_operation() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise CustomError("Custom transient error")
            return "success"

        result = custom_error_operation()

        assert result == "success"
        assert call_count == 2

    def test_retry_with_default_settings(self, tmp_path: Path) -> None:
        """Test that retry uses settings defaults when not specified."""
        settings = Settings(
            trace_file=str(tmp_path / "trace.jsonl"),
            max_retry_attempts=3,
            retry_initial_delay=0.1,
            retry_backoff_multiplier=2.0,
        )
        setup_tracing(settings)

        call_count = 0

        @retry_with_backoff()  # No parameters - should use settings
        def operation() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ConnectionError("Transient error")
            return "success"

        result = operation()

        assert result == "success"
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_async_function_retry(self, tmp_path: Path) -> None:
        """Test that async functions are properly retried."""
        settings = Settings(
            trace_file=str(tmp_path / "trace.jsonl"),
            log_file=str(tmp_path / "test.log"),
        )
        setup_tracing(settings)

        call_count = 0

        @retry_with_backoff(max_attempts=3, initial_delay=0.1)
        async def async_operation() -> str:
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.01)
            if call_count < 2:
                raise ConnectionError("Transient error")
            return "success"

        result = await async_operation()

        assert result == "success"
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_async_function_max_attempts(self, tmp_path: Path) -> None:
        """Test that async functions respect max attempts."""
        settings = Settings(
            trace_file=str(tmp_path / "trace.jsonl"),
            log_file=str(tmp_path / "test.log"),
        )
        setup_tracing(settings)

        call_count = 0

        @retry_with_backoff(max_attempts=3, initial_delay=0.1)
        async def async_always_fails() -> str:
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.01)
            raise ConnectionError("Always fails")

        with pytest.raises(RetryError, match="Max retry attempts \\(3\\) exceeded"):
            await async_always_fails()

        assert call_count == 3

    def test_retry_traces_recorded(self, tmp_path: Path) -> None:
        """Test that retry attempts are traced.

        Corresponds to: E2E-031
        """
        import json

        trace_file = tmp_path / "trace.jsonl"
        settings = Settings(trace_file=str(trace_file))
        setup_tracing(settings)

        call_count = 0

        @retry_with_backoff(max_attempts=5, initial_delay=0.1)
        def operation_with_retries() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Transient error")
            return "success"

        result = operation_with_retries()

        assert result == "success"

        # Force flush traces
        from opentelemetry import trace as otel_trace

        tracer_provider = otel_trace.get_tracer_provider()
        if hasattr(tracer_provider, "force_flush"):
            tracer_provider.force_flush()  # type: ignore[attr-defined]

        time.sleep(0.1)

        # Read and verify traces
        content = trace_file.read_text()
        lines = [line for line in content.strip().split("\n") if line]

        # Should have parent span + retry attempt spans
        assert len(lines) >= 4  # 1 parent + 3 attempts

        spans = [json.loads(line) for line in lines]

        # Find parent span
        parent_spans = [s for s in spans if "with_retry" in s["name"]]
        assert len(parent_spans) >= 1
        parent_span = parent_spans[0]

        # Verify parent span attributes
        assert parent_span["attributes"]["total_attempts"] == 3
        assert parent_span["attributes"]["status"] == "success"

        # Find retry attempt spans
        retry_spans = [s for s in spans if s["name"] == "retry_attempt"]
        assert len(retry_spans) >= 3

        # Verify attempt 1 and 2 failed, attempt 3 succeeded
        for i, span in enumerate(retry_spans[:3], 1):
            assert span["attributes"]["attempt_number"] == i
            if i < 3:
                assert span["attributes"]["status"] == "failed"
                assert "error" in span["attributes"]
            else:
                assert span["attributes"]["status"] == "success"

    def test_function_with_arguments(self) -> None:
        """Test that decorated functions with arguments work correctly."""
        call_count = 0

        @retry_with_backoff(max_attempts=3, initial_delay=0.1)
        def operation_with_args(x: int, y: int, z: str = "default") -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ConnectionError("Transient error")
            return f"{x + y} {z}"

        result = operation_with_args(1, 2, z="test")

        assert result == "3 test"
        assert call_count == 2

    def test_function_with_return_type(self) -> None:
        """Test that return types are preserved."""

        @retry_with_backoff(max_attempts=3, initial_delay=0.1)
        def returns_dict() -> dict[str, int]:
            return {"key": 42}

        result = returns_dict()

        assert isinstance(result, dict)
        assert result == {"key": 42}
