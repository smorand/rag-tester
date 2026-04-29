"""Retry decorator with exponential backoff.

Provides automatic retry logic for transient failures with:
- Exponential backoff (configurable multiplier and initial delay)
- Tracing of retry attempts
- Logging of retry attempts and failures
- Distinction between transient and permanent errors
"""

import asyncio
import logging
import time
from collections.abc import Callable
from functools import wraps
from typing import Any, TypeVar

from opentelemetry import trace

from rag_tester.config import Settings

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Transient errors that should be retried
TRANSIENT_ERRORS = (
    ConnectionError,
    TimeoutError,
    OSError,
)


class RetryError(Exception):
    """Raised when max retry attempts are exhausted."""

    pass


def retry_with_backoff(  # noqa: PLR0915
    max_attempts: int | None = None,
    initial_delay: float | None = None,
    backoff_multiplier: float | None = None,
    transient_errors: tuple[type[Exception], ...] = TRANSIENT_ERRORS,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator to retry a function with exponential backoff.

    Args:
        max_attempts: Maximum number of retry attempts (default: from settings)
        initial_delay: Initial delay in seconds (default: from settings)
        backoff_multiplier: Backoff multiplier for exponential delay (default: from settings)
        transient_errors: Tuple of exception types to retry on

    Returns:
        Decorated function with retry logic

    Raises:
        RetryError: When max attempts are exhausted
        Exception: Permanent errors are raised immediately without retry

    Example:
        @retry_with_backoff(max_attempts=3, initial_delay=1.0)
        def fetch_data():
            # code that might fail transiently
            pass
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:  # noqa: PLR0915
        @wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> T:
            # Load settings for defaults
            settings = Settings()
            attempts = max_attempts if max_attempts is not None else settings.max_retry_attempts
            delay = initial_delay if initial_delay is not None else settings.retry_initial_delay
            multiplier = backoff_multiplier if backoff_multiplier is not None else settings.retry_backoff_multiplier

            tracer = trace.get_tracer(__name__)
            last_exception: Exception | None = None

            with tracer.start_as_current_span(f"{func.__name__}_with_retry") as parent_span:
                parent_span.set_attribute("max_attempts", attempts)

                for attempt in range(1, attempts + 1):
                    with tracer.start_as_current_span("retry_attempt") as span:
                        span.set_attribute("attempt_number", attempt)
                        
                        try:
                            result = func(*args, **kwargs)

                            span.set_attribute("status", "success")
                            parent_span.set_attribute("total_attempts", attempt)
                            parent_span.set_attribute("status", "success")

                            if attempt > 1:
                                logger.info("Operation succeeded on attempt %d", attempt)

                            return result

                        except transient_errors as e:
                            last_exception = e
                            span.set_attribute("status", "failed")
                            span.set_attribute("error", str(e))
                            logger.warning("Retry attempt %d failed: %s", attempt, str(e))

                            if attempt < attempts:
                                backoff_delay = delay * (multiplier ** (attempt - 1))
                                span.set_attribute("backoff_delay", backoff_delay)
                                logger.info("Retrying in %.2f seconds...", backoff_delay)
                                time.sleep(backoff_delay)
                            else:
                                span.set_attribute("final_attempt", True)

                        except Exception as e:
                            # Permanent error - don't retry
                            logger.error("Permanent error (not retrying): %s", str(e))
                            with tracer.start_as_current_span("permanent_error") as error_span:
                                error_span.set_attribute("error", str(e))
                                error_span.set_attribute("error_type", type(e).__name__)
                            parent_span.set_attribute("status", "failed")
                            parent_span.set_attribute("error_type", "permanent")
                            raise

                # All attempts exhausted
                parent_span.set_attribute("total_attempts", attempts)
                parent_span.set_attribute("status", "failed")
                error_msg = f"Max retry attempts ({attempts}) exceeded"
                logger.error(error_msg)
                raise RetryError(error_msg) from last_exception

        @wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> T:
            # Load settings for defaults
            settings = Settings()
            attempts = max_attempts if max_attempts is not None else settings.max_retry_attempts
            delay = initial_delay if initial_delay is not None else settings.retry_initial_delay
            multiplier = backoff_multiplier if backoff_multiplier is not None else settings.retry_backoff_multiplier

            tracer = trace.get_tracer(__name__)
            last_exception: Exception | None = None

            with tracer.start_as_current_span(f"{func.__name__}_with_retry") as parent_span:
                parent_span.set_attribute("max_attempts", attempts)

                for attempt in range(1, attempts + 1):
                    with tracer.start_as_current_span("retry_attempt") as span:
                        span.set_attribute("attempt_number", attempt)
                        
                        try:
                            result: T = await func(*args, **kwargs)  # type: ignore[misc]

                            span.set_attribute("status", "success")
                            parent_span.set_attribute("total_attempts", attempt)
                            parent_span.set_attribute("status", "success")

                            if attempt > 1:
                                logger.info("Operation succeeded on attempt %d", attempt)

                            return result

                        except transient_errors as e:
                            last_exception = e
                            span.set_attribute("status", "failed")
                            span.set_attribute("error", str(e))
                            logger.warning("Retry attempt %d failed: %s", attempt, str(e))

                            if attempt < attempts:
                                backoff_delay = delay * (multiplier ** (attempt - 1))
                                span.set_attribute("backoff_delay", backoff_delay)
                                logger.info("Retrying in %.2f seconds...", backoff_delay)
                                await asyncio.sleep(backoff_delay)
                            else:
                                span.set_attribute("final_attempt", True)

                        except Exception as e:
                            # Permanent error - don't retry
                            logger.error("Permanent error (not retrying): %s", str(e))
                            with tracer.start_as_current_span("permanent_error") as error_span:
                                error_span.set_attribute("error", str(e))
                                error_span.set_attribute("error_type", type(e).__name__)
                            parent_span.set_attribute("status", "failed")
                            parent_span.set_attribute("error_type", "permanent")
                            raise

                # All attempts exhausted
                parent_span.set_attribute("total_attempts", attempts)
                parent_span.set_attribute("status", "failed")
                error_msg = f"Max retry attempts ({attempts}) exceeded"
                logger.error(error_msg)
                raise RetryError(error_msg) from last_exception

        # Return appropriate wrapper based on whether function is async
        if asyncio.iscoroutinefunction(func):
            return async_wrapper  # type: ignore[return-value]
        return sync_wrapper

    return decorator
