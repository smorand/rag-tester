# US-006: Compare Command - Model Analysis

> Parent Spec: specs/2026-04-29_18:26:42-rag-testing-framework.md
> Status: ready
> Priority: 6
> Depends On: US-005
> Complexity: M

## Objective

Implement the `compare` command for comparative analysis of two embedding models or configurations. This command enables data scientists and ML engineers to systematically compare RAG system performance by analyzing test results from different models, calculating aggregate metrics (pass rate, average score, cost), and identifying per-test differences. It's the primary tool for model selection and A/B testing.

## Technical Context

### Stack
- **Language:** Python 3.13+
- **CLI Framework:** Typer (command definition)
- **Output:** YAML for comparison file
- **Testing:** pytest, pytest-asyncio, pytest-cov

### Relevant File Structure
```
rag-tester/
├── src/rag_tester/
│   ├── rag_tester.py             # CLI entry point (UPDATE: add compare command)
│   ├── commands/
│   │   ├── __init__.py           # Command exports (UPDATE)
│   │   └── compare.py            # Compare command implementation (NEW)
│   ├── core/
│   │   ├── __init__.py           # Core exports (UPDATE)
│   │   └── comparator.py         # Comparison logic (NEW)
│   └── utils/
│       └── cost.py               # Cost calculation (NEW)
├── tests/
│   ├── test_commands/
│   │   └── test_compare.py       # Compare command tests (NEW)
│   ├── test_core/
│   │   └── test_comparator.py    # Comparator tests (NEW)
│   └── test_utils/
│       └── test_cost.py          # Cost calculation tests (NEW)
└── pyproject.toml                # No new dependencies
```

### Existing Patterns
- **Config:** Use Settings from US-001
- **Logging:** Use module-level loggers from US-001
- **Tracing:** Use tracer from US-001 for all operations

### Data Model (excerpt)

**Compare Command Arguments:**
```python
@app.command()
async def compare(
    results: list[str],           # Paths to 2+ result files (YAML)
    output: str,                  # Output file path (YAML)
) -> None:
    """Compare test results from different models or configurations."""
```

**Comparison File Format (YAML):**
```yaml
model_a:
  name: "BAAI/bge-small-en-v1.5"
  database: "chromadb://localhost:8000/collection_a"
  pass_rate: 0.8
  avg_score: 0.85
  total_tokens: 0
  total_time: 5.2
  total_cost: 0.0
  cost_per_test: 0.0

model_b:
  name: "openai/text-embedding-3-small"
  database: "chromadb://localhost:8000/collection_b"
  pass_rate: 0.7
  avg_score: 0.82
  total_tokens: 15000
  total_time: 8.1
  total_cost: 0.0003
  cost_per_test: 0.00003

per_test_diff:
  - test_id: "test3"
    model_a_status: "passed"
    model_b_status: "failed"
    model_a_score: 0.87
    model_b_score: 0.62
    expected_threshold: 0.85
  - test_id: "test7"
    model_a_status: "failed"
    model_b_status: "passed"
    model_a_score: 0.72
    model_b_score: 0.88
    expected_threshold: 0.80
```

## Functional Requirements

### FR-028: Compare Command
- **Description:** Compare test results from two or more models/configurations
- **Inputs:** 
  - List of result file paths (2+ files, YAML format)
  - Output file path
- **Outputs:** 
  - Comparison file (YAML) with aggregate metrics and per-test differences
  - Exit code 0 on success, 1 on failure
- **Business Rules:**
  - Read and parse all result files
  - Validate that result files are from the same test suite (same test IDs)
  - Calculate aggregate metrics for each model:
    - Pass rate: passed / total_tests
    - Average score: mean of all actual scores
    - Total tokens: sum of tokens consumed
    - Total time: sum of test durations
    - Total cost: calculated from tokens and model pricing
    - Cost per test: total_cost / total_tests
  - Identify per-test differences (tests where models disagree on pass/fail)
  - Generate comparison file with all metrics
  - Trace comparison operation with: models_compared, total_tests, duration
  - Log comparison summary at INFO level

### FR-029: Cost Calculation
- **Description:** Calculate embedding costs for API-based models
- **Inputs:** 
  - Model identifier (e.g., "openai/text-embedding-3-small")
  - Total tokens consumed
- **Outputs:** 
  - Total cost in USD
- **Business Rules:**
  - Use pricing table for known models:
    - openai/text-embedding-3-small: $0.02 / 1M tokens
    - openai/text-embedding-3-large: $0.13 / 1M tokens
    - openai/text-embedding-ada-002: $0.10 / 1M tokens
    - voyage-ai/voyage-2: $0.10 / 1M tokens
    - cohere/embed-english-v3.0: $0.10 / 1M tokens
  - For unknown models: cost = 0.0 (log warning)
  - Cost = (tokens / 1,000,000) * price_per_million
  - Round to 6 decimal places
  - Log cost calculation at DEBUG level

### FR-030: Aggregate Metrics
- **Description:** Calculate summary statistics for each model
- **Inputs:** 
  - Test results from a single model
- **Outputs:** 
  - Aggregate metrics: pass_rate, avg_score, total_tokens, total_time, total_cost, cost_per_test
- **Business Rules:**
  - Pass rate: count(passed) / count(total)
  - Average score: mean of all actual scores (across all expected results)
  - Total tokens: sum from result file summary
  - Total time: sum from result file summary
  - Total cost: calculated from tokens and model pricing
  - Cost per test: total_cost / total_tests
  - Handle edge cases: division by zero, missing data
  - Log metrics calculation at DEBUG level

### FR-031: Per-Test Differences
- **Description:** Identify tests where models disagree on pass/fail status
- **Inputs:** 
  - Test results from multiple models
- **Outputs:** 
  - List of tests with different outcomes
- **Business Rules:**
  - Compare pass/fail status for each test across models
  - Include test in diff if any two models disagree
  - For each diff: test_id, status per model, scores per model, expected_threshold
  - Sort diffs by test_id
  - Log number of differences at INFO level

## Acceptance Tests

> **Acceptance tests are mandatory: 100% must pass.**
> A user story is NOT considered implemented until **every single acceptance test below passes**.
> The implementing agent MUST loop (fix code → run tests → check results → repeat) until all acceptance tests pass with zero failures. Do not stop or declare the story "done" while any test is failing.
> Tests MUST be validated through the project's CI/CD chain (generally `make test` or equivalent Makefile target). No other method of running or validating tests is acceptable: do not run test files directly, do not use ad hoc commands. Use the Makefile.

### Test Data

| Data | Description | Source | Status |
|------|-------------|--------|--------|
| results_model_a.yaml | Results from model A (8/10 pass) | auto-generated (pytest fixture) | ready |
| results_model_b.yaml | Results from model B (7/10 pass) | auto-generated (pytest fixture) | ready |
| results_api_model.yaml | Results from API model with token counts | auto-generated (pytest fixture) | ready |

### Happy Path Tests

### E2E-005: Compare Two Embedding Models
- **Category:** happy
- **Scenario:** SC-004
- **Requirements:** FR-028, FR-030, FR-031
- **Preconditions:**
  - results_model_a.yaml (8/10 pass, avg_score: 0.85)
  - results_model_b.yaml (7/10 pass, avg_score: 0.82)
- **Steps:**
  - Given two result files from different models
  - When: `rag-tester compare --results results_model_a.yaml results_model_b.yaml --output comparison.yaml`
  - Then exit code 0
  - And comparison.yaml is created
  - And model_a section: {pass_rate: 0.8, avg_score: 0.85, total_tokens: 0, total_time: X, total_cost: 0.0}
  - And model_b section: {pass_rate: 0.7, avg_score: 0.82, total_tokens: 0, total_time: Y, total_cost: 0.0}
  - And per_test_diff section lists tests where models disagree (3 tests)
  - And trace has spans: comparison, metrics_calculation, diff_analysis
  - And log contains: "Comparison complete: 2 models, 10 tests, 3 differences"
- **Cleanup:** Delete comparison.yaml
- **Priority:** High

### E2E-023: Cost Calculation
- **Category:** happy
- **Scenario:** SC-004
- **Requirements:** FR-029
- **Preconditions:**
  - results_api_model.yaml with OpenRouter model, 15000 tokens
- **Steps:**
  - Given result file with API model and token count
  - When: `rag-tester compare --results results_api_model.yaml results_model_a.yaml --output comparison.yaml`
  - Then comparison.yaml contains cost estimates
  - And API model section: {total_tokens: 15000, total_cost: 0.0003, cost_per_test: 0.00003}
  - And cost calculated as: (15000 / 1000000) * $0.02 = $0.0003
  - And log contains: "Cost calculated: $0.0003 for 15000 tokens"
- **Cleanup:** Delete comparison.yaml
- **Priority:** Medium

### E2E-024: Per-Test Diff
- **Category:** happy
- **Scenario:** SC-004
- **Requirements:** FR-031
- **Preconditions:**
  - results_model_a.yaml and results_model_b.yaml with different outcomes
- **Steps:**
  - Given two result files with disagreements
  - When compare is executed
  - Then per_test_diff section contains:
    - test_id for each disagreement
    - status per model (passed/failed)
    - scores per model
    - expected_threshold (if applicable)
  - And diffs are sorted by test_id
  - And log contains: "Found 3 test differences between models"
- **Cleanup:** Delete comparison.yaml
- **Priority:** High

### E2E-032: Comparison File Generated
- **Category:** happy (side effect)
- **Scenario:** SC-004
- **Requirements:** FR-028
- **Preconditions:**
  - Two result files
- **Steps:**
  - Given result files
  - When compare is executed
  - Then comparison.yaml is created
  - And file is valid YAML
  - And file has model_a, model_b, per_test_diff sections
  - And log contains: "Comparison written to: comparison.yaml"
- **Cleanup:** Delete comparison.yaml
- **Priority:** High

### Edge Case and Error Tests

### E2E-COMP-001: Missing Result File
- **Category:** failure
- **Scenario:** File validation
- **Requirements:** FR-028
- **Preconditions:**
  - File "nonexistent.yaml" does not exist
- **Steps:**
  - Given non-existent result file
  - When: `rag-tester compare --results nonexistent.yaml results_model_a.yaml --output comparison.yaml`
  - Then exit code 1
  - And stderr contains: "Error: Result file not found: nonexistent.yaml"
  - And no comparison file created
- **Cleanup:** None
- **Priority:** Critical

### E2E-COMP-002: Invalid Result File Format
- **Category:** failure
- **Scenario:** File validation
- **Requirements:** FR-028
- **Preconditions:**
  - malformed_results.yaml with invalid YAML
- **Steps:**
  - Given malformed result file
  - When: `rag-tester compare --results malformed_results.yaml results_model_a.yaml --output comparison.yaml`
  - Then exit code 1
  - And stderr contains: "Error: Invalid result file format: malformed_results.yaml"
  - And no comparison file created
- **Cleanup:** None
- **Priority:** High

### E2E-COMP-003: Incompatible Result Files
- **Category:** failure
- **Scenario:** Validation error
- **Requirements:** FR-028
- **Preconditions:**
  - results_suite_a.yaml with test IDs: test1-test10
  - results_suite_b.yaml with test IDs: test11-test20
- **Steps:**
  - Given result files from different test suites
  - When: `rag-tester compare --results results_suite_a.yaml results_suite_b.yaml --output comparison.yaml`
  - Then exit code 1
  - And stderr contains: "Error: Test suites do not match. Result files must be from the same test suite."
  - And no comparison file created
- **Cleanup:** None
- **Priority:** Medium

### E2E-COMP-004: Single Result File
- **Category:** failure
- **Scenario:** Argument validation
- **Requirements:** FR-028
- **Preconditions:**
  - results_model_a.yaml
- **Steps:**
  - Given only one result file
  - When: `rag-tester compare --results results_model_a.yaml --output comparison.yaml`
  - Then exit code 1
  - And stderr contains: "Error: At least 2 result files required for comparison"
- **Cleanup:** None
- **Priority:** High

### E2E-COMP-005: Unknown Model Pricing
- **Category:** edge
- **Scenario:** Cost calculation
- **Requirements:** FR-029
- **Preconditions:**
  - results_custom_model.yaml with unknown model "custom/my-model"
- **Steps:**
  - Given result file with unknown model
  - When compare is executed
  - Then comparison.yaml is created
  - And custom model section: {total_cost: 0.0, cost_per_test: 0.0}
  - And log contains warning: "Unknown model pricing for 'custom/my-model', cost set to 0.0"
- **Cleanup:** Delete comparison.yaml
- **Priority:** Medium

### E2E-COMP-006: Zero Tests
- **Category:** edge
- **Scenario:** Edge case
- **Requirements:** FR-030
- **Preconditions:**
  - results_empty.yaml with 0 tests
- **Steps:**
  - Given result file with no tests
  - When compare is executed
  - Then comparison.yaml is created
  - And metrics: {pass_rate: 0.0, avg_score: 0.0, cost_per_test: 0.0}
  - And no division by zero errors
- **Cleanup:** Delete comparison.yaml
- **Priority:** Medium

### E2E-COMP-007: All Tests Agree
- **Category:** edge
- **Scenario:** No differences
- **Requirements:** FR-031
- **Preconditions:**
  - Two result files with identical pass/fail outcomes
- **Steps:**
  - Given result files with no disagreements
  - When compare is executed
  - Then comparison.yaml is created
  - And per_test_diff section is empty list
  - And log contains: "No test differences found between models"
- **Cleanup:** Delete comparison.yaml
- **Priority:** Low

### E2E-COMP-008: Three Models Comparison
- **Category:** edge
- **Scenario:** Multiple models
- **Requirements:** FR-028
- **Preconditions:**
  - results_model_a.yaml, results_model_b.yaml, results_model_c.yaml
- **Steps:**
  - Given three result files
  - When: `rag-tester compare --results results_model_a.yaml results_model_b.yaml results_model_c.yaml --output comparison.yaml`
  - Then comparison.yaml is created
  - And file has model_a, model_b, model_c sections
  - And per_test_diff shows disagreements across all three models
- **Cleanup:** Delete comparison.yaml
- **Priority:** Low

## Constraints

### Files Not to Touch
- `src/rag_tester/providers/` (from US-002)
- `src/rag_tester/commands/load.py` (from US-003)
- `src/rag_tester/commands/test.py` (from US-004)
- `src/rag_tester/commands/bulk_test.py` (from US-005)

### Dependencies Not to Add
- No new dependencies required (all needed packages already in pyproject.toml)

### Patterns to Avoid
- Do NOT load entire result files into memory if they're large (stream if possible)
- Do NOT hardcode model pricing (use a pricing table that can be updated)
- Do NOT fail comparison if one model has missing data (handle gracefully)

### Scope Boundary
- This story does NOT implement load, test, or bulk-test commands (already done)
- This story does NOT implement upsert or flush modes (that's US-007)
- This story does NOT implement API embedding providers (that's US-008)
- This story ONLY implements: compare command with cost calculation and metrics analysis

## Non Regression

### Existing Tests That Must Pass
- All tests from US-001 (infrastructure)
- All tests from US-002 (providers)
- All tests from US-003 (load command)
- All tests from US-004 (test command)
- All tests from US-005 (bulk-test command)

### Behaviors That Must Not Change
- Configuration system from US-001
- Logging and tracing from US-001
- Provider interfaces from US-002
- Load command from US-003
- Test command from US-004
- Bulk-test command from US-005

### API Contracts to Preserve
- Result file format from US-005 (must remain compatible)
