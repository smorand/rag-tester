# US-001: Core Infrastructure & Configuration

> Parent Spec: specs/2026-04-29_18:26:42-rag-testing-framework.md
> Status: ready
> Priority: 1
> Depends On: none
> Complexity: M

## Objective

Establish the foundational infrastructure for the RAG Testing Framework: configuration management, structured logging with rich console output, OpenTelemetry tracing to JSONL files, and automatic retry logic with exponential backoff. This story creates the plumbing that all other features depend on, ensuring observability, reliability, and proper error handling from day one.

## Technical Context

### Stack
- **Language:** Python 3.13+
- **CLI Framework:** Typer (for command structure)
- **Config:** pydantic-settings (environment variable management)
- **Logging:** Python logging + rich (colored console output)
- **Tracing:** OpenTelemetry API + SDK (JSONL file exporter)
- **Testing:** pytest, pytest-asyncio, pytest-cov

### Relevant File Structure
```
rag-tester/
├── src/rag_tester/
│   ├── __init__.py           # Package exports
│   ├── config.py             # Settings class (NEW)
│   ├── logging_config.py     # Logging setup (NEW)
│   ├── tracing.py            # OpenTelemetry setup (NEW)
│   └── utils/
│       ├── __init__.py
│       └── retry.py          # Retry decorator (NEW)
├── tests/
│   ├── test_config.py        # Config tests (NEW)
│   ├── test_logging.py       # Logging tests (NEW)
│   ├── test_tracing.py       # Tracing tests (NEW)
│   └── test_utils/
│       └── test_retry.py     # Retry tests (NEW)
├── traces/                   # Trace output directory (created at runtime)
└── logs/                     # Log output directory (created at runtime)
```

### Existing Patterns
This is a greenfield project. Establish patterns here that will be followed throughout:
- **Config:** Single `Settings` class using pydantic-settings, all env vars prefixed `RAG_TESTER_`
- **Logging:** Module-level loggers (`logger = logging.getLogger(__name__)`), rich handler for console
- **Tracing:** Context manager pattern for spans, attributes dict for metadata
- **Retry:** Decorator pattern with configurable attempts and backoff

### Data Model (excerpt)

**Settings (config.py):**
```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Logging
    log_level: str = "INFO"
    log_file: str = "logs/rag-tester.log"
    
    # Tracing
    trace_file: str = "traces/rag-tester.jsonl"
    otel_endpoint: str | None = None  # Optional OTLP endpoint
    
    # Retry
    max_retry_attempts: int = 5
    retry_backoff_multiplier: float = 2.0
    retry_initial_delay: float = 1.0
    
    class Config:
        env_prefix = "RAG_TESTER_"
```

**Trace Span (JSONL format):**
```json
{
  "trace_id": "abc123...",
  "span_id": "def456...",
  "parent_span_id": "ghi789...",
  "name": "operation_name",
  "start_time": "2026-04-29T16:00:00.000Z",
  "end_time": "2026-04-29T16:00:00.150Z",
  "attributes": {
    "key": "value",
    "status": "success"
  }
}
```

## Functional Requirements

### FR-032: Retry Logic with Exponential Backoff
- **Description:** Automatically retry failed operations up to 5 times with exponential backoff
- **Inputs:** 
  - Operation to retry (callable)
  - Max attempts (default: 5)
  - Initial delay (default: 1.0 seconds)
  - Backoff multiplier (default: 2.0)
- **Outputs:** 
  - Operation result on success
  - Exception raised after max attempts exhausted
- **Business Rules:**
  - Retry on transient errors: network errors, rate limits, temporary unavailability
  - Do NOT retry on permanent errors: authentication failures, invalid inputs, not found
  - Backoff delay: attempt_n_delay = initial_delay * (multiplier ^ (n-1))
  - Log each retry attempt with attempt number and delay
  - Trace each retry attempt as a separate span with `attempt_number` attribute

### FR-045: OpenTelemetry Tracing
- **Description:** Trace all operations with OpenTelemetry, output to JSONL file
- **Inputs:** 
  - Trace file path (default: traces/rag-tester.jsonl, configurable via RAG_TESTER_TRACE_FILE)
  - Optional OTLP endpoint (configurable via RAG_TESTER_OTEL_ENDPOINT)
- **Outputs:** 
  - JSONL file with one span per line
  - Each span has: trace_id, span_id, parent_span_id, name, start_time, end_time, attributes
- **Business Rules:**
  - Trace levels:
    - INFO: API calls (embedding API calls with model, tokens, duration, cost)
    - DEBUG: DB operations (insert, update, delete, query with duration)
    - DEBUG: File I/O (read operations with file size, duration)
    - INFO: Test execution (per-test spans with test_id, status, duration)
    - WARNING: Retry attempts (retry spans with attempt_number, backoff_delay, error)
    - ERROR: Errors (error spans with error message, stack trace)
  - Span attributes: always include operation-specific metadata (model, tokens, file_size, etc.)
  - **NEVER log sensitive data:** API keys, credentials, full text content (only IDs and metadata)
  - Auto-create trace directory if it doesn't exist
  - Append to existing trace file (don't overwrite)

### FR-046: Logging Configuration
- **Description:** Configure logging with rich console output and file output
- **Inputs:** 
  - Log level (default: INFO, configurable via RAG_TESTER_LOG_LEVEL)
  - Log file path (default: logs/rag-tester.log, configurable via RAG_TESTER_LOG_FILE)
- **Outputs:** 
  - Console: rich logging with colors, timestamps, log levels
  - File: plain text logs with format "[timestamp] [level] [module] message"
- **Business Rules:**
  - Log levels: DEBUG, INFO, WARNING, ERROR, CRITICAL
  - Console handler: rich.logging.RichHandler with colors and formatting
  - File handler: RotatingFileHandler (max 10MB, 5 backups)
  - Format: "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
  - Auto-create log directory if it doesn't exist
  - Module-level loggers: `logger = logging.getLogger(__name__)`

## Acceptance Tests

> **Acceptance tests are mandatory: 100% must pass.**
> A user story is NOT considered implemented until **every single acceptance test below passes**.
> The implementing agent MUST loop (fix code → run tests → check results → repeat) until all acceptance tests pass with zero failures. Do not stop or declare the story "done" while any test is failing.
> Tests MUST be validated through the project's CI/CD chain (generally `make test` or equivalent Makefile target). No other method of running or validating tests is acceptable: do not run test files directly, do not use ad hoc commands. Use the Makefile.

### Test Data

> All test data for this story is auto-generated or uses mock objects. No user-provided data required.

| Data | Description | Source | Status |
|------|-------------|--------|--------|
| Mock API responses | Simulated API responses for retry testing | auto-generated (pytest fixtures) | ready |
| Sample trace spans | Example span data for tracing tests | auto-generated (test code) | ready |
| Log messages | Test log messages at various levels | auto-generated (test code) | ready |

### Happy Path Tests

### E2E-026: Trace File Written
- **Category:** happy
- **Scenario:** Infrastructure validation
- **Requirements:** FR-045
- **Preconditions:**
  - Clean test environment (no existing trace file)
  - Settings configured with trace_file = "test_traces/test.jsonl"
- **Steps:**
  - Given a tracer instance initialized with test settings
  - When a span is created with name="test_operation" and attributes={"key": "value"}
  - And the span is ended
  - Then the trace file "test_traces/test.jsonl" exists
  - And the file contains exactly 1 line of valid JSON
  - And the JSON has fields: trace_id, span_id, name="test_operation", start_time, end_time, attributes={"key": "value"}
  - And attributes does NOT contain any sensitive data patterns (API keys, passwords)
- **Cleanup:** Delete test_traces directory
- **Priority:** Critical

### E2E-030: Failed Records Logged
- **Category:** happy
- **Scenario:** Retry and error handling
- **Requirements:** FR-032
- **Preconditions:**
  - Logging configured with log_level=INFO
  - Mock operation that fails once then succeeds
- **Steps:**
  - Given a retry decorator with max_attempts=3
  - When the decorated operation is called
  - And it fails on attempt 1 with a transient error
  - And succeeds on attempt 2
  - Then the log file contains: "Retry attempt 1 failed: <error>"
  - And the log file contains: "Operation succeeded on attempt 2"
  - And the operation returns the expected result
- **Cleanup:** Delete log file
- **Priority:** High

### E2E-031: Retry Attempts Traced
- **Category:** happy
- **Scenario:** Retry with tracing
- **Requirements:** FR-032, FR-045
- **Preconditions:**
  - Tracing enabled
  - Mock operation that requires 3 attempts (2 failures + 1 success)
- **Steps:**
  - Given a retry decorator with max_attempts=5
  - When the decorated operation is called
  - And it fails on attempts 1 and 2 with transient errors
  - And succeeds on attempt 3
  - Then the trace file contains 4 spans:
    - 1 parent span: "operation_with_retry"
    - 3 child spans: "retry_attempt" with attempt_number=1,2,3
  - And attempt 1 span has attributes: {attempt_number: 1, status: "failed", error: "<error>"}
  - And attempt 2 span has attributes: {attempt_number: 2, status: "failed", error: "<error>"}
  - And attempt 3 span has attributes: {attempt_number: 3, status: "success"}
  - And parent span has attributes: {total_attempts: 3, status: "success"}
- **Cleanup:** Delete trace file
- **Priority:** Medium

### E2E-056: API Key Not Logged
- **Category:** happy (security validation)
- **Scenario:** Sensitive data protection
- **Requirements:** FR-045
- **Preconditions:**
  - Tracing enabled
  - Mock API call with API key in headers
- **Steps:**
  - Given a traced operation that makes an API call
  - When the operation is executed with headers={"Authorization": "Bearer sk-test-key-12345"}
  - And the span is created with attributes including the API call metadata
  - Then the trace file does NOT contain "sk-test-key-12345"
  - And the trace file does NOT contain "Bearer sk-test-key-12345"
  - And the span attributes contain {"api_call": "true", "model": "test-model"} (non-sensitive metadata only)
- **Cleanup:** Delete trace file
- **Priority:** Critical

### Edge Case and Error Tests

> **Edge case and error tests are equally mandatory.** These tests verify that the system correctly rejects invalid inputs, enforces boundaries, and returns proper error responses. Each test MUST specify the **exact expected error** (error code, HTTP status, error message). A test that says "should fail" without specifying the exact error is incomplete and will be rejected.

### E2E-INFRA-001: Config Loads from Environment Variables
- **Category:** edge
- **Scenario:** Configuration validation
- **Requirements:** FR-046
- **Preconditions:**
  - Environment variables set: RAG_TESTER_LOG_LEVEL=DEBUG, RAG_TESTER_TRACE_FILE=custom/trace.jsonl
- **Steps:**
  - Given environment variables are set
  - When Settings() is instantiated
  - Then settings.log_level == "DEBUG"
  - And settings.trace_file == "custom/trace.jsonl"
  - And settings.max_retry_attempts == 5 (default value)
- **Cleanup:** Unset environment variables
- **Priority:** High

### E2E-INFRA-002: Config Uses Defaults When No Env Vars
- **Category:** edge
- **Scenario:** Default configuration
- **Requirements:** FR-046
- **Preconditions:**
  - No RAG_TESTER_* environment variables set
- **Steps:**
  - Given clean environment
  - When Settings() is instantiated
  - Then settings.log_level == "INFO"
  - And settings.log_file == "logs/rag-tester.log"
  - And settings.trace_file == "traces/rag-tester.jsonl"
  - And settings.max_retry_attempts == 5
- **Cleanup:** None
- **Priority:** High

### E2E-INFRA-003: Retry Exhausts Max Attempts
- **Category:** failure
- **Scenario:** Retry failure after max attempts
- **Requirements:** FR-032
- **Preconditions:**
  - Mock operation that always fails with transient error
- **Steps:**
  - Given a retry decorator with max_attempts=3
  - When the decorated operation is called
  - And it fails on all 3 attempts
  - Then a RetryError is raised with message "Max retry attempts (3) exceeded"
  - And the log contains 3 retry attempt messages
  - And the trace contains 4 spans (1 parent + 3 retry attempts)
  - And the parent span has attributes: {total_attempts: 3, status: "failed"}
- **Cleanup:** Delete log and trace files
- **Priority:** Critical

### E2E-INFRA-004: Retry Does Not Retry Permanent Errors
- **Category:** failure
- **Scenario:** Permanent error handling
- **Requirements:** FR-032
- **Preconditions:**
  - Mock operation that fails with permanent error (e.g., ValueError)
- **Steps:**
  - Given a retry decorator with max_attempts=5
  - When the decorated operation is called
  - And it raises ValueError("Invalid input")
  - Then the ValueError is raised immediately (no retries)
  - And the log does NOT contain any retry attempt messages
  - And the trace contains 1 span with status="failed" and error="Invalid input"
- **Cleanup:** Delete log and trace files
- **Priority:** Critical

### E2E-INFRA-005: Trace Directory Auto-Created
- **Category:** edge
- **Scenario:** Directory creation
- **Requirements:** FR-045
- **Preconditions:**
  - Trace directory does not exist
  - Settings configured with trace_file = "new_traces/test.jsonl"
- **Steps:**
  - Given the directory "new_traces" does not exist
  - When a tracer is initialized
  - And a span is created and ended
  - Then the directory "new_traces" is created
  - And the file "new_traces/test.jsonl" exists with valid span data
- **Cleanup:** Delete new_traces directory
- **Priority:** Medium

### E2E-INFRA-006: Log Directory Auto-Created
- **Category:** edge
- **Scenario:** Directory creation
- **Requirements:** FR-046
- **Preconditions:**
  - Log directory does not exist
  - Settings configured with log_file = "new_logs/test.log"
- **Steps:**
  - Given the directory "new_logs" does not exist
  - When logging is configured
  - And a log message is written
  - Then the directory "new_logs" is created
  - And the file "new_logs/test.log" exists with the log message
- **Cleanup:** Delete new_logs directory
- **Priority:** Medium

### E2E-INFRA-007: Exponential Backoff Timing
- **Category:** edge
- **Scenario:** Backoff calculation validation
- **Requirements:** FR-032
- **Preconditions:**
  - Mock operation that fails 3 times
  - Settings: initial_delay=1.0, backoff_multiplier=2.0
- **Steps:**
  - Given a retry decorator with the above settings
  - When the decorated operation is called
  - And it fails on attempts 1, 2, 3
  - Then the delays between attempts are approximately:
    - Attempt 1 → 2: 1.0 seconds (1.0 * 2^0)
    - Attempt 2 → 3: 2.0 seconds (1.0 * 2^1)
    - Attempt 3 → 4: 4.0 seconds (1.0 * 2^2)
  - And the trace spans have attributes: {backoff_delay: 1.0}, {backoff_delay: 2.0}, {backoff_delay: 4.0}
- **Cleanup:** Delete trace file
- **Priority:** Medium

## Constraints

### Files Not to Touch
- `src/rag_tester/rag_tester.py` (CLI entry point - will be implemented in later stories)
- `src/rag_tester/version.py` (already exists)
- Any files in `src/rag_tester/commands/` (not created yet)
- Any files in `src/rag_tester/providers/` (not created yet)
- Any files in `src/rag_tester/core/` (not created yet)

### Dependencies Not to Add
- Only these packages are allowed (already in pyproject.toml):
  - typer, rich, httpx, pydantic, pydantic-settings
  - opentelemetry-api, opentelemetry-sdk
  - pytest, pytest-asyncio, pytest-cov (dev)
- Do NOT add: structlog, loguru, sentry-sdk, or any other logging/tracing libraries

### Patterns to Avoid
- Do NOT use `print()` for logging (use `logger.info()`, etc.)
- Do NOT use bare `except:` clauses (specify exception types)
- Do NOT use mutable default arguments (e.g., `def func(items=[]):`)
- Do NOT use `.format()` for string formatting (use f-strings)
- Do NOT use `assert` for error handling (raise proper exceptions)
- Do NOT create blocking I/O in async contexts (use async libraries)

### Scope Boundary
- This story does NOT implement any CLI commands (load, test, bulk-test, compare)
- This story does NOT implement any embedding providers or database backends
- This story does NOT implement any business logic (loading data, running tests, etc.)
- This story ONLY implements infrastructure: config, logging, tracing, retry

## Non Regression

### Existing Tests That Must Pass
- `tests/test_version.py` (already exists, must continue to pass)
- `tests/test_cli.py` (already exists, must continue to pass)
- `tests/test_config.py` (already exists, must continue to pass)

### Behaviors That Must Not Change
- Package structure: `src/rag_tester/` layout must be preserved
- Entry point: `rag-tester` command must remain defined in pyproject.toml
- Version handling: `version.py` must remain unchanged

### API Contracts to Preserve
- None (this is the first story, no existing contracts)
