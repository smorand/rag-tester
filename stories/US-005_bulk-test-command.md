# US-005: Bulk-Test Command - Validation & Results

> Parent Spec: specs/2026-04-29_18:26:42-rag-testing-framework.md
> Status: ready
> Priority: 5
> Depends On: US-004
> Complexity: L

## Objective

Implement the `bulk-test` command for automated test suite execution with comprehensive validation logic. This command enables systematic testing of RAG systems by running multiple test cases, validating results against expected outcomes (exact order matching and similarity thresholds), and generating detailed reports. It's the core quality assurance tool for RAG pipelines, supporting both regression testing and model evaluation.

## Technical Context

### Stack
- **Language:** Python 3.13+
- **CLI Framework:** Typer (command definition)
- **Async:** asyncio for parallel test execution
- **Progress:** rich.progress for progress tracking
- **Output:** YAML for results file
- **Testing:** pytest, pytest-asyncio, pytest-cov

### Relevant File Structure
```
rag-tester/
├── src/rag_tester/
│   ├── rag_tester.py             # CLI entry point (UPDATE: add bulk-test command)
│   ├── commands/
│   │   ├── __init__.py           # Command exports (UPDATE)
│   │   └── bulk_test.py          # Bulk-test command implementation (NEW)
│   └── core/
│       ├── __init__.py           # Core exports (UPDATE)
│       ├── tester.py             # Test logic (UPDATE: add validation)
│       └── validator.py          # Input validation (UPDATE: add test file validation)
├── tests/
│   ├── test_commands/
│   │   └── test_bulk_test.py     # Bulk-test command tests (NEW)
│   └── test_core/
│       └── test_validation.py    # Validation tests (NEW)
└── pyproject.toml                # No new dependencies
```

### Existing Patterns
- **Config:** Use Settings from US-001
- **Logging:** Use module-level loggers from US-001
- **Tracing:** Use tracer from US-001 for all operations
- **Providers:** Use EmbeddingProvider and VectorDatabase from US-002
- **Parallel:** Use asyncio patterns from US-003

### Data Model (excerpt)

**Bulk-Test Command Arguments:**
```python
@app.command()
async def bulk_test(
    file: str,                    # Path to test file (YAML or JSON)
    database: str,                # Database connection string
    embedding: str,               # Embedding model identifier
    output: str,                  # Output file path (YAML)
    parallel: int = 1,            # Number of parallel workers
    verbose: bool = False,        # Include all tests in output (not just failures)
) -> None:
    """Run a test suite against the vector database."""
```

**Test File Format (YAML):**
```yaml
tests:
  - test_id: "test1"
    query: "What is machine learning?"
    expected:
      - id: "doc1"
        text: "Machine learning is..."
        min_threshold: 0.85  # optional
      - id: "doc3"
        text: "ML algorithms..."
        # no threshold = any score passes
```

**Results File Format (YAML):**
```yaml
summary:
  total_tests: 10
  passed: 7
  failed: 3
  total_tokens: 1250
  total_time: 15.3
  embedding_model: "sentence-transformers/all-MiniLM-L6-v2"
  database: "chromadb://localhost:8000/test_collection"

tests:  # all tests if --verbose, only failed otherwise
  - test_id: "test3"
    query: "What is deep learning?"
    expected:
      - id: "doc2"
        text: "Deep learning uses..."
        min_threshold: 0.85
    actual:
      - id: "doc2"
        text: "Deep learning uses..."
        score: 0.78
    status: "failed"
    reason: "Threshold not met: expected >= 0.85, got 0.78"
    duration: 0.12
```

## Functional Requirements

### FR-023: Bulk-Test Command
- **Description:** Execute a test suite and validate results against expected outcomes
- **Inputs:** 
  - Test file path (YAML or JSON)
  - Database connection string
  - Embedding model identifier
  - Output file path
  - Optional: parallel workers, verbose flag
- **Outputs:** 
  - Results file (YAML) with summary and test details
  - Exit code 0 (always, even if tests fail - failures are in results file)
- **Business Rules:**
  - Read test file and parse test cases
  - For each test: generate query embedding, query database, validate results
  - Validation logic:
    - **Exact order matching:** expected IDs must appear in exact order in actual results
    - **Threshold checking:** if min_threshold specified, actual score must be >= threshold
    - **Missing IDs:** if expected ID not in actual results, test fails
  - Generate results file with summary and test details
  - By default, only include failed tests in results (use --verbose for all tests)
  - Display progress bar during execution
  - Trace entire test suite with: total_tests, passed, failed, duration
  - Log summary at INFO level

### FR-024: Parallel Test Execution
- **Description:** Run multiple tests concurrently for faster execution
- **Inputs:** 
  - Number of parallel workers (default: 1, max: 16)
- **Outputs:** 
  - Faster test suite execution
- **Business Rules:**
  - Use asyncio.gather() or asyncio.TaskGroup for parallel execution
  - Each worker processes a subset of tests
  - Workers share the same embedding provider and database connection
  - Trace parallel execution with: worker_id, tests_processed, duration
  - Log parallel processing at INFO level

### FR-025: Test Validation - Exact Order Matching
- **Description:** Validate that expected IDs appear in exact order in actual results
- **Inputs:** 
  - Expected results: list of {id, text, min_threshold?}
  - Actual results: list of {id, text, score}
- **Outputs:** 
  - Pass/fail status
  - Failure reason if failed
- **Business Rules:**
  - Expected IDs must appear in exact order in actual results
  - Example: expected=[doc1, doc3, doc5], actual=[doc1, doc2, doc3, doc5] → PASS (order preserved)
  - Example: expected=[doc1, doc3], actual=[doc3, doc1] → FAIL (wrong order)
  - Example: expected=[doc1, doc3], actual=[doc1, doc2] → FAIL (doc3 missing)
  - Ignore IDs in actual results that are not in expected (extra results are OK)
  - Log validation details at DEBUG level

### FR-026: Test Validation - Threshold Checking
- **Description:** Validate that actual scores meet minimum thresholds
- **Inputs:** 
  - Expected result with min_threshold
  - Actual result with score
- **Outputs:** 
  - Pass/fail status
  - Failure reason if failed
- **Business Rules:**
  - If min_threshold specified, actual score must be >= threshold
  - If min_threshold not specified, any score passes (threshold check skipped)
  - Example: expected={id: doc1, min_threshold: 0.85}, actual={id: doc1, score: 0.87} → PASS
  - Example: expected={id: doc1, min_threshold: 0.85}, actual={id: doc1, score: 0.78} → FAIL
  - Failure reason: "Threshold not met: expected >= 0.85, got 0.78"
  - Log threshold checks at DEBUG level

### FR-027: Results File Generation
- **Description:** Generate YAML results file with summary and test details
- **Inputs:** 
  - Test results (passed and failed)
  - Summary statistics
- **Outputs:** 
  - YAML file with summary and tests sections
- **Business Rules:**
  - Summary section: total_tests, passed, failed, total_tokens, total_time, embedding_model, database
  - Tests section: by default only failed tests, all tests if --verbose
  - Each test: test_id, query, expected, actual, status, reason (if failed), duration
  - File must be valid YAML
  - Overwrite existing file if it exists
  - Log file generation at INFO level

### FR-029: Progress Tracking
- **Description:** Display progress updates during test execution
- **Inputs:** 
  - Total number of tests
  - Current test number
- **Outputs:** 
  - Progress bar or text updates
- **Business Rules:**
  - Display progress bar for test suites with > 10 tests
  - Update progress after each test completion
  - Show: "Testing: 25/100 (25%)"
  - Progress bar disappears on completion
  - Log progress at DEBUG level

## Acceptance Tests

> **Acceptance tests are mandatory: 100% must pass.**
> A user story is NOT considered implemented until **every single acceptance test below passes**.
> The implementing agent MUST loop (fix code → run tests → check results → repeat) until all acceptance tests pass with zero failures. Do not stop or declare the story "done" while any test is failing.
> Tests MUST be validated through the project's CI/CD chain (generally `make test` or equivalent Makefile target). No other method of running or validating tests is acceptable: do not run test files directly, do not use ad hoc commands. Use the Makefile.

### Test Data

| Data | Description | Source | Status |
|------|-------------|--------|--------|
| test_suite.yaml | 10 tests (7 pass, 2 fail ID order, 1 fail threshold) | auto-generated (pytest fixture) | ready |
| test_suite_large.yaml | 100 tests for parallel execution | auto-generated (pytest fixture) | ready |
| Loaded collection | Collection with known documents | existing (from US-003) | ready |

### Happy Path Tests

### E2E-003: Bulk Test with Pass/Fail Cases
- **Category:** happy
- **Scenario:** SC-003
- **Requirements:** FR-023, FR-025, FR-026, FR-027
- **Preconditions:**
  - Collection loaded with known documents
  - test_suite.yaml with 10 tests (7 pass, 2 fail order, 1 fail threshold)
- **Steps:**
  - Given collection "test_collection" with documents
  - When: `rag-tester bulk-test --file test_suite.yaml --database chromadb://localhost:8000/test_collection --embedding sentence-transformers/all-MiniLM-L6-v2 --output results.yaml --parallel 2`
  - Then exit code 0
  - And results.yaml is created
  - And summary section: {total_tests: 10, passed: 7, failed: 3, tokens: 0, time: X}
  - And tests section contains only 3 failed tests (not verbose)
  - And each failed test has: test_id, query, expected, actual, status="failed", reason, duration
  - And trace has spans: test_suite, test_execution (10 spans), validation (10 spans)
  - And log contains: "Test suite complete: 7/10 passed (70%)"
- **Cleanup:** Delete results.yaml
- **Priority:** Critical

### E2E-004: Bulk Test Verbose
- **Category:** happy
- **Scenario:** SC-003
- **Requirements:** FR-023, FR-027
- **Preconditions:**
  - Collection loaded
  - test_suite.yaml with 10 tests
- **Steps:**
  - Given collection with documents
  - When: `rag-tester bulk-test --file test_suite.yaml --database chromadb://localhost:8000/test_collection --embedding sentence-transformers/all-MiniLM-L6-v2 --output results_verbose.yaml --verbose`
  - Then exit code 0
  - And results_verbose.yaml contains ALL 10 test results (passed + failed)
  - And passed tests have: test_id, query, expected, actual, status="passed", duration
  - And failed tests have: test_id, query, expected, actual, status="failed", reason, duration
- **Cleanup:** Delete results_verbose.yaml
- **Priority:** High

### E2E-021: Progress Indicator
- **Category:** happy
- **Scenario:** SC-003
- **Requirements:** FR-029
- **Preconditions:**
  - Collection loaded
  - test_suite_50.yaml with 50 tests
- **Steps:**
  - Given test suite with 50 tests
  - When: `rag-tester bulk-test --file test_suite_50.yaml --database chromadb://localhost:8000/test_collection --embedding sentence-transformers/all-MiniLM-L6-v2 --output results.yaml`
  - Then progress bar is displayed
  - And progress updates show: "Testing: 10/50 (20%)", "Testing: 25/50 (50%)", etc.
  - And progress bar disappears on completion
- **Cleanup:** Delete results.yaml
- **Priority:** Medium

### E2E-022: Parallel Execution
- **Category:** happy
- **Scenario:** SC-003
- **Requirements:** FR-024
- **Preconditions:**
  - Collection loaded
  - test_suite_large.yaml with 100 tests
- **Steps:**
  - Given test suite with 100 tests
  - When: `rag-tester bulk-test --file test_suite_large.yaml --database chromadb://localhost:8000/test_collection --embedding sentence-transformers/all-MiniLM-L6-v2 --output results.yaml --parallel 4`
  - Then exit code 0
  - And trace shows concurrent test execution (4 workers with overlapping time ranges)
  - And total time < 50% of sequential execution (--parallel 1)
  - And log contains: "Using 4 parallel workers"
- **Cleanup:** Delete results.yaml
- **Priority:** High

### E2E-027: Results File Written
- **Category:** happy (side effect)
- **Scenario:** SC-003
- **Requirements:** FR-027
- **Preconditions:**
  - Collection loaded
  - test_suite.yaml
- **Steps:**
  - Given test suite
  - When: `rag-tester bulk-test --file test_suite.yaml --database chromadb://localhost:8000/test_collection --embedding sentence-transformers/all-MiniLM-L6-v2 --output results.yaml`
  - Then results.yaml is created
  - And file is valid YAML
  - And file has summary and tests sections
  - And log contains: "Results written to: results.yaml"
- **Cleanup:** Delete results.yaml
- **Priority:** Critical

### E2E-062: Test Result Accuracy
- **Category:** happy (data integrity)
- **Scenario:** SC-003
- **Requirements:** FR-025, FR-026
- **Preconditions:**
  - Collection with known documents
  - Test suite with known expected results
- **Steps:**
  - Given test suite with deterministic expected results
  - When bulk-test is executed
  - Then pass/fail determinations are correct:
    - Test with correct order and threshold → passed
    - Test with wrong order → failed with reason "Expected order not matched"
    - Test with score below threshold → failed with reason "Threshold not met"
  - And validation logic is accurate
- **Cleanup:** Delete results.yaml
- **Priority:** Critical

### E2E-067: Bulk Test Throughput
- **Category:** happy (performance baseline)
- **Scenario:** SC-003
- **Requirements:** NFR-004
- **Preconditions:**
  - Collection loaded
  - test_suite_large.yaml with 100 tests
- **Steps:**
  - Given test suite with 100 tests
  - When: `rag-tester bulk-test --file test_suite_large.yaml --database chromadb://localhost:8000/test_collection --embedding sentence-transformers/all-MiniLM-L6-v2 --output results.yaml --parallel 4`
  - Then throughput > 10 tests/second
  - And total time is displayed in summary
- **Cleanup:** Delete results.yaml
- **Priority:** Low

### E2E-078: Load, Test, Bulk-Test
- **Category:** happy (cross-scenario)
- **Scenario:** SC-001 + SC-002 + SC-003
- **Requirements:** FR-023
- **Preconditions:**
  - test_data.yaml for loading
  - test_suite.yaml for testing
- **Steps:**
  - Given clean environment
  - When data is loaded via load command
  - And manual test is executed via test command
  - And bulk-test is executed via bulk-test command
  - Then all operations succeed
  - And results are consistent across operations
- **Cleanup:** Delete collection and results.yaml
- **Priority:** High

### Edge Case and Error Tests

### E2E-086: Zero Threshold
- **Category:** edge
- **Scenario:** Threshold validation
- **Requirements:** FR-026
- **Preconditions:**
  - Test with expected result having min_threshold: 0.0
- **Steps:**
  - Given test with min_threshold: 0.0
  - When bulk-test is executed
  - Then any score >= 0.0 passes (effectively no threshold)
  - And test passes with any positive score
- **Cleanup:** Delete results.yaml
- **Priority:** Low

### E2E-BULK-001: Malformed Test File
- **Category:** failure
- **Scenario:** File validation
- **Requirements:** FR-023
- **Preconditions:**
  - malformed_tests.yaml with invalid YAML syntax
- **Steps:**
  - Given malformed test file
  - When: `rag-tester bulk-test --file malformed_tests.yaml --database chromadb://localhost:8000/test_collection --embedding sentence-transformers/all-MiniLM-L6-v2 --output results.yaml`
  - Then exit code 1
  - And stderr contains: "Error: Invalid test file format. Failed to parse YAML: <error details>"
  - And no results file created
- **Cleanup:** None
- **Priority:** Critical

### E2E-BULK-002: Missing Required Test Fields
- **Category:** failure
- **Scenario:** Test validation
- **Requirements:** FR-023
- **Preconditions:**
  - invalid_tests.yaml with test missing "query" field
- **Steps:**
  - Given test file with invalid test: {test_id: "test1", expected: [...]}
  - When: `rag-tester bulk-test --file invalid_tests.yaml --database chromadb://localhost:8000/test_collection --embedding sentence-transformers/all-MiniLM-L6-v2 --output results.yaml`
  - Then exit code 1
  - And stderr contains: "Error: Missing required field 'query' in test 'test1'"
  - And no results file created
- **Cleanup:** None
- **Priority:** Critical

### E2E-BULK-003: Database Unavailable Mid-Suite
- **Category:** failure
- **Scenario:** Connection error
- **Requirements:** FR-023
- **Preconditions:**
  - test_suite.yaml with 10 tests
  - ChromaDB goes down after 5 tests (mock)
- **Steps:**
  - Given test suite with 10 tests
  - When bulk-test is executed
  - And database becomes unavailable after 5 tests
  - Then exit code 0 (continue on error)
  - And results.yaml contains: 5 passed/failed tests, 5 error tests
  - And error tests have status="error", reason="Database connection failed"
  - And log contains: "Database error during test execution, continuing..."
- **Cleanup:** Delete results.yaml
- **Priority:** High

### E2E-BULK-004: Output File Write Failure
- **Category:** failure
- **Scenario:** File I/O error
- **Requirements:** FR-027
- **Preconditions:**
  - test_suite.yaml
  - Output path is unwritable (permissions)
- **Steps:**
  - Given unwritable output path
  - When: `rag-tester bulk-test --file test_suite.yaml --database chromadb://localhost:8000/test_collection --embedding sentence-transformers/all-MiniLM-L6-v2 --output /root/results.yaml`
  - Then exit code 1
  - And stderr contains: "Error: Cannot write output file: /root/results.yaml (permission denied)"
- **Cleanup:** None
- **Priority:** High

### E2E-BULK-005: Empty Test Suite
- **Category:** edge
- **Scenario:** Validation
- **Requirements:** FR-023
- **Preconditions:**
  - empty_tests.yaml with no tests
- **Steps:**
  - Given test file with no tests
  - When: `rag-tester bulk-test --file empty_tests.yaml --database chromadb://localhost:8000/test_collection --embedding sentence-transformers/all-MiniLM-L6-v2 --output results.yaml`
  - Then exit code 1
  - And stderr contains: "Error: Test file is empty or has no tests"
- **Cleanup:** None
- **Priority:** High

### E2E-BULK-006: Invalid Parallel Workers
- **Category:** failure
- **Scenario:** Argument validation
- **Requirements:** FR-024
- **Preconditions:**
  - test_suite.yaml
- **Steps:**
  - Given parallel workers 0 (invalid)
  - When: `rag-tester bulk-test --file test_suite.yaml --database chromadb://localhost:8000/test_collection --embedding sentence-transformers/all-MiniLM-L6-v2 --output results.yaml --parallel 0`
  - Then exit code 1
  - And stderr contains: "Error: Parallel workers must be between 1 and 16"
- **Cleanup:** None
- **Priority:** High

### E2E-BULK-007: Order Validation - Wrong Order
- **Category:** edge
- **Scenario:** Validation logic
- **Requirements:** FR-025
- **Preconditions:**
  - Test with expected=[doc1, doc3], actual=[doc3, doc1]
- **Steps:**
  - Given test with wrong order
  - When bulk-test is executed
  - Then test fails with reason: "Expected order not matched: expected [doc1, doc3], got [doc3, doc1]"
  - And status="failed"
- **Cleanup:** Delete results.yaml
- **Priority:** High

### E2E-BULK-008: Order Validation - Missing ID
- **Category:** edge
- **Scenario:** Validation logic
- **Requirements:** FR-025
- **Preconditions:**
  - Test with expected=[doc1, doc3], actual=[doc1, doc2]
- **Steps:**
  - Given test with missing expected ID
  - When bulk-test is executed
  - Then test fails with reason: "Expected ID 'doc3' not found in results"
  - And status="failed"
- **Cleanup:** Delete results.yaml
- **Priority:** High

### E2E-BULK-009: Threshold Validation - Below Threshold
- **Category:** edge
- **Scenario:** Validation logic
- **Requirements:** FR-026
- **Preconditions:**
  - Test with expected={id: doc1, min_threshold: 0.85}, actual={id: doc1, score: 0.78}
- **Steps:**
  - Given test with score below threshold
  - When bulk-test is executed
  - Then test fails with reason: "Threshold not met for 'doc1': expected >= 0.85, got 0.78"
  - And status="failed"
- **Cleanup:** Delete results.yaml
- **Priority:** High

## Constraints

### Files Not to Touch
- `src/rag_tester/providers/` (from US-002)
- `src/rag_tester/commands/load.py` (from US-003)
- `src/rag_tester/commands/test.py` (from US-004)

### Dependencies Not to Add
- No new dependencies required (all needed packages already in pyproject.toml)

### Patterns to Avoid
- Do NOT load entire test suite into memory before execution (stream tests)
- Do NOT accumulate all results before writing file (write incrementally if possible)
- Do NOT fail entire suite on single test error (continue on error, report in results)

### Scope Boundary
- This story does NOT implement compare command (that's US-006)
- This story does NOT implement cost calculation (that's US-006)
- This story does NOT implement upsert or flush modes (that's US-007)
- This story ONLY implements: bulk-test command with validation logic and results generation

## Non Regression

### Existing Tests That Must Pass
- All tests from US-001 (infrastructure)
- All tests from US-002 (providers)
- All tests from US-003 (load command)
- All tests from US-004 (test command)

### Behaviors That Must Not Change
- Configuration system from US-001
- Logging and tracing from US-001
- Provider interfaces from US-002
- Load command from US-003
- Test command from US-004

### API Contracts to Preserve
- EmbeddingProvider interface from US-002
- VectorDatabase interface from US-002
