# US-004: Test Command - Query & Output Formats

> Parent Spec: specs/2026-04-29_18:26:42-rag-testing-framework.md
> Status: ready
> Priority: 4
> Depends On: US-002
> Complexity: M

## Objective

Implement the `test` command for manual query testing with multiple output formats (table, JSON, text). This command enables users to interactively test their RAG system by querying the vector database with natural language questions and viewing results in their preferred format. It provides immediate feedback on retrieval quality and helps validate that the loaded data is working correctly.

## Technical Context

### Stack
- **Language:** Python 3.13+
- **CLI Framework:** Typer (command definition)
- **Output:** rich.table for table format, json for JSON format, plain text for text format
- **Testing:** pytest, pytest-asyncio, pytest-cov

### Relevant File Structure
```
rag-tester/
├── src/rag_tester/
│   ├── rag_tester.py             # CLI entry point (UPDATE: add test command)
│   ├── commands/
│   │   ├── __init__.py           # Command exports (UPDATE)
│   │   └── test.py               # Test command implementation (NEW)
│   └── core/
│       ├── __init__.py           # Core exports (UPDATE)
│       └── tester.py             # Test logic (NEW)
├── tests/
│   ├── test_commands/
│   │   └── test_test.py          # Test command tests (NEW)
│   └── test_core/
│       └── test_tester.py        # Tester tests (NEW)
└── pyproject.toml                # No new dependencies
```

### Existing Patterns
- **Config:** Use Settings from US-001
- **Logging:** Use module-level loggers from US-001
- **Tracing:** Use tracer from US-001 for all operations
- **Providers:** Use EmbeddingProvider and VectorDatabase from US-002

### Data Model (excerpt)

**Test Command Arguments:**
```python
@app.command()
async def test(
    query: str,                   # Query text
    database: str,                # Database connection string
    embedding: str,               # Embedding model identifier
    top_k: int = 5,              # Number of results to return
    format: str = "table",       # Output format: table, json, text
) -> None:
    """Test a query against the vector database."""
```

**Query Result:**
```python
{
    "query": "What is machine learning?",
    "results": [
        {
            "rank": 1,
            "id": "doc42",
            "text": "Machine learning is...",
            "score": 0.87
        },
        # ... more results
    ],
    "tokens": 0,      # for API models
    "time": 0.12      # seconds
}
```

## Functional Requirements

### FR-021: Test Command with Output Formats
- **Description:** Execute a single query and display results in specified format
- **Inputs:** 
  - Query text (natural language question)
  - Database connection string
  - Embedding model identifier
  - Top-K (default: 5)
  - Output format (default: table, options: table, json, text)
- **Outputs:** 
  - Results in specified format
  - Exit code 0 on success, 1 on failure
- **Business Rules:**
  - Generate query embedding using specified model
  - Query database for top-K similar documents
  - Sort results by similarity score (descending)
  - Display results in requested format:
    - **table:** rich table with columns: Rank, ID, Text (truncated to 80 chars), Score
    - **json:** valid JSON with query, results array, tokens, time
    - **text:** plain text with numbered results, full text, scores
  - Include metadata: tokens consumed (for API models), time taken
  - Trace query operation with: query (first 50 chars), top_k, duration, result_count
  - Log query at INFO level

### FR-022: Custom Top-K
- **Description:** Allow users to specify number of results to return
- **Inputs:** 
  - Top-K value (default: 5, min: 1, max: 100)
- **Outputs:** 
  - Exactly top-K results (or fewer if database has fewer documents)
- **Business Rules:**
  - Validate top-K is within allowed range
  - If top-K > database size, return all documents (no error)
  - Results always sorted by score descending
  - Log top-K value at DEBUG level

## Acceptance Tests

> **Acceptance tests are mandatory: 100% must pass.**
> A user story is NOT considered implemented until **every single acceptance test below passes**.
> The implementing agent MUST loop (fix code → run tests → check results → repeat) until all acceptance tests pass with zero failures. Do not stop or declare the story "done" while any test is failing.
> Tests MUST be validated through the project's CI/CD chain (generally `make test` or equivalent Makefile target). No other method of running or validating tests is acceptable: do not run test files directly, do not use ad hoc commands. Use the Makefile.

### Test Data

| Data | Description | Source | Status |
|------|-------------|--------|--------|
| Loaded collection | Collection with 100 documents from US-003 | existing (from US-003 tests) | ready |
| Known document | doc42 with text about machine learning | existing (from US-003 tests) | ready |

### Happy Path Tests

### E2E-002: Manual Query Test
- **Category:** happy
- **Scenario:** SC-002
- **Requirements:** FR-021
- **Preconditions:**
  - Collection "test_collection" loaded with 100 docs
  - doc42 has text: "Machine learning is a subset of artificial intelligence that enables computers to learn from data."
- **Steps:**
  - Given collection with 100 documents
  - When: `rag-tester test "What is machine learning?" --database chromadb://localhost:8000/test_collection --embedding sentence-transformers/all-MiniLM-L6-v2 --top-k 3 --format table`
  - Then exit code 0
  - And stdout shows rich table with 3 rows
  - And table has columns: Rank, ID, Text, Score
  - And row 1 has: rank=1, id="doc42", score > 0.7
  - And results are sorted by score descending
  - And trace has spans: embedding_query, database_search
  - And log contains: "Query: What is machine learning?", "Results: 3"
- **Cleanup:** None (uses existing collection)
- **Priority:** Critical

### E2E-018: JSON Output Format
- **Category:** happy
- **Scenario:** SC-002
- **Requirements:** FR-021
- **Preconditions:**
  - Collection "test_collection" loaded
- **Steps:**
  - Given collection with documents
  - When: `rag-tester test "machine learning" --database chromadb://localhost:8000/test_collection --embedding sentence-transformers/all-MiniLM-L6-v2 --format json`
  - Then exit code 0
  - And stdout is valid JSON
  - And JSON structure: {"query": "machine learning", "results": [...], "tokens": 0, "time": X}
  - And results array has 5 items (default top-k)
  - And each result has: rank, id, text, score
- **Cleanup:** None
- **Priority:** Medium

### E2E-019: Text Output Format
- **Category:** happy
- **Scenario:** SC-002
- **Requirements:** FR-021
- **Preconditions:**
  - Collection "test_collection" loaded
- **Steps:**
  - Given collection with documents
  - When: `rag-tester test "machine learning" --database chromadb://localhost:8000/test_collection --embedding sentence-transformers/all-MiniLM-L6-v2 --format text`
  - Then exit code 0
  - And stdout is plain text with format:
    ```
    Query: machine learning
    
    1. [doc42] (score: 0.87)
    Machine learning is...
    
    2. [doc15] (score: 0.82)
    ...
    
    Tokens: 0
    Time: 0.12s
    ```
  - And results are numbered 1-5
- **Cleanup:** None
- **Priority:** Medium

### E2E-020: Custom Top-K
- **Category:** happy
- **Scenario:** SC-002
- **Requirements:** FR-022
- **Preconditions:**
  - Collection "test_collection" loaded
- **Steps:**
  - Given collection with documents
  - When: `rag-tester test "machine learning" --database chromadb://localhost:8000/test_collection --embedding sentence-transformers/all-MiniLM-L6-v2 --top-k 10`
  - Then exit code 0
  - And exactly 10 results returned
  - And results sorted by score descending
  - And log contains: "Top-K: 10"
- **Cleanup:** None
- **Priority:** Medium

### E2E-066: Query Latency
- **Category:** happy (performance baseline)
- **Scenario:** SC-002
- **Requirements:** NFR-003
- **Preconditions:**
  - Collection "test_collection" loaded
- **Steps:**
  - Given collection with documents
  - When: `rag-tester test "machine learning" --database chromadb://localhost:8000/test_collection --embedding sentence-transformers/all-MiniLM-L6-v2`
  - Then exit code 0
  - And total time < 1 second (local model)
  - And time is displayed in output
- **Cleanup:** None
- **Priority:** Medium

### Edge Case and Error Tests

### E2E-085: Top-K Exceeds Collection Size
- **Category:** edge
- **Scenario:** Boundary condition
- **Requirements:** FR-022
- **Preconditions:**
  - Collection "small_collection" with 50 documents
- **Steps:**
  - Given collection with 50 documents
  - When: `rag-tester test "test query" --database chromadb://localhost:8000/small_collection --embedding sentence-transformers/all-MiniLM-L6-v2 --top-k 100`
  - Then exit code 0
  - And exactly 50 results returned (all documents)
  - And no error or warning
  - And log contains: "Requested top-k (100) exceeds collection size (50), returning all documents"
- **Cleanup:** Delete collection
- **Priority:** Medium

### E2E-087: Perfect Score (1.0)
- **Category:** edge
- **Scenario:** Exact match
- **Requirements:** FR-021
- **Preconditions:**
  - Collection with document: id="doc1", text="Python is a programming language"
- **Steps:**
  - Given collection with known document
  - When: `rag-tester test "Python is a programming language" --database chromadb://localhost:8000/exact_match --embedding sentence-transformers/all-MiniLM-L6-v2`
  - Then exit code 0
  - And result 1 has id="doc1" with score ≈ 1.0 (within 0.01)
- **Cleanup:** Delete collection
- **Priority:** Low

### E2E-TEST-001: Empty Database
- **Category:** failure
- **Scenario:** Validation error
- **Requirements:** FR-021
- **Preconditions:**
  - Empty collection "empty_collection"
- **Steps:**
  - Given empty collection
  - When: `rag-tester test "test query" --database chromadb://localhost:8000/empty_collection --embedding sentence-transformers/all-MiniLM-L6-v2`
  - Then exit code 1
  - And stderr contains: "Error: Database is empty. No documents to query."
- **Cleanup:** Delete collection
- **Priority:** Critical

### E2E-TEST-002: Database Unreachable
- **Category:** failure
- **Scenario:** Connection error
- **Requirements:** FR-021
- **Preconditions:**
  - ChromaDB server down or unreachable
- **Steps:**
  - Given ChromaDB at wrong port
  - When: `rag-tester test "test query" --database chromadb://localhost:9999/test --embedding sentence-transformers/all-MiniLM-L6-v2`
  - Then exit code 1
  - And stderr contains: "Error: Database connection failed: <error details>"
  - And trace shows retry attempts
- **Cleanup:** None
- **Priority:** Critical

### E2E-TEST-003: Embedding API Failure
- **Category:** failure
- **Scenario:** Embedding error
- **Requirements:** FR-021
- **Preconditions:**
  - Invalid embedding model name
- **Steps:**
  - Given invalid model "invalid/model"
  - When: `rag-tester test "test query" --database chromadb://localhost:8000/test_collection --embedding invalid/model`
  - Then exit code 1
  - And stderr contains: "Error: Failed to load embedding model: invalid/model"
- **Cleanup:** None
- **Priority:** Critical

### E2E-TEST-004: Dimension Mismatch
- **Category:** failure
- **Scenario:** Validation error
- **Requirements:** FR-021
- **Preconditions:**
  - Collection with dimension 768
  - Model with dimension 384
- **Steps:**
  - Given collection with dimension 768
  - When: `rag-tester test "test query" --database chromadb://localhost:8000/dim768_collection --embedding sentence-transformers/all-MiniLM-L6-v2`
  - Then exit code 1
  - And stderr contains: "Error: Dimension mismatch: model=384, database=768"
- **Cleanup:** Delete collection
- **Priority:** High

### E2E-TEST-005: Invalid Top-K (Zero)
- **Category:** failure
- **Scenario:** Argument validation
- **Requirements:** FR-022
- **Preconditions:**
  - Collection loaded
- **Steps:**
  - Given top-k value 0
  - When: `rag-tester test "test query" --database chromadb://localhost:8000/test_collection --embedding sentence-transformers/all-MiniLM-L6-v2 --top-k 0`
  - Then exit code 1
  - And stderr contains: "Error: Top-K must be between 1 and 100"
- **Cleanup:** None
- **Priority:** High

### E2E-TEST-006: Invalid Top-K (Too Large)
- **Category:** failure
- **Scenario:** Argument validation
- **Requirements:** FR-022
- **Preconditions:**
  - Collection loaded
- **Steps:**
  - Given top-k value 101
  - When: `rag-tester test "test query" --database chromadb://localhost:8000/test_collection --embedding sentence-transformers/all-MiniLM-L6-v2 --top-k 101`
  - Then exit code 1
  - And stderr contains: "Error: Top-K must be between 1 and 100"
- **Cleanup:** None
- **Priority:** High

### E2E-TEST-007: Invalid Output Format
- **Category:** failure
- **Scenario:** Argument validation
- **Requirements:** FR-021
- **Preconditions:**
  - Collection loaded
- **Steps:**
  - Given invalid format "xml"
  - When: `rag-tester test "test query" --database chromadb://localhost:8000/test_collection --embedding sentence-transformers/all-MiniLM-L6-v2 --format xml`
  - Then exit code 1
  - And stderr contains: "Error: Invalid format. Must be one of: table, json, text"
- **Cleanup:** None
- **Priority:** High

### E2E-TEST-008: Empty Query String
- **Category:** failure
- **Scenario:** Input validation
- **Requirements:** FR-021
- **Preconditions:**
  - Collection loaded
- **Steps:**
  - Given empty query string ""
  - When: `rag-tester test "" --database chromadb://localhost:8000/test_collection --embedding sentence-transformers/all-MiniLM-L6-v2`
  - Then exit code 1
  - And stderr contains: "Error: Query cannot be empty"
- **Cleanup:** None
- **Priority:** High

## Constraints

### Files Not to Touch
- `src/rag_tester/providers/` (from US-002)
- `src/rag_tester/commands/load.py` (from US-003)
- `src/rag_tester/core/loader.py` (from US-003)

### Dependencies Not to Add
- No new dependencies required (rich already in pyproject.toml)

### Patterns to Avoid
- Do NOT load all results into memory before displaying (stream if possible)
- Do NOT use print() for output (use rich for table, json.dumps for JSON, formatted strings for text)
- Do NOT create new database connections per query (reuse connection)

### Scope Boundary
- This story does NOT implement bulk-test command (that's US-005)
- This story does NOT implement compare command (that's US-006)
- This story does NOT implement validation logic for test suites (that's US-005)
- This story ONLY implements: single query testing with multiple output formats

## Non Regression

### Existing Tests That Must Pass
- All tests from US-001 (infrastructure)
- All tests from US-002 (providers)
- All tests from US-003 (load command)

### Behaviors That Must Not Change
- Configuration system from US-001
- Logging and tracing from US-001
- Provider interfaces from US-002
- Load command from US-003

### API Contracts to Preserve
- EmbeddingProvider interface from US-002
- VectorDatabase interface from US-002
