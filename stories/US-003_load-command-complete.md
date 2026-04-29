# US-003: Load Command - Streaming & Parallel Processing

> Parent Spec: specs/2026-04-29_18:26:42-rag-testing-framework.md
> Status: ready
> Priority: 3
> Depends On: US-002
> Complexity: L

## Objective

Implement the complete `load` command with streaming file processing, parallel embedding generation, batch optimization, and robust error handling. This story transforms the basic loading capability from US-002 into a production-ready command that can handle datasets of any size efficiently, with progress tracking, duplicate detection, and comprehensive observability.

## Technical Context

### Stack
- **Language:** Python 3.13+
- **CLI Framework:** Typer (command definition and argument parsing)
- **Async:** asyncio for parallel processing, aiofiles for streaming
- **Progress:** rich.progress for progress bars
- **Testing:** pytest, pytest-asyncio, pytest-cov

### Relevant File Structure
```
rag-tester/
├── src/rag_tester/
│   ├── rag_tester.py             # CLI entry point (UPDATE: add load command)
│   ├── commands/
│   │   ├── __init__.py           # Command exports (NEW)
│   │   └── load.py               # Load command implementation (NEW)
│   ├── core/
│   │   ├── __init__.py           # Core exports (NEW)
│   │   ├── loader.py             # Load logic (NEW)
│   │   └── validator.py          # Input validation (NEW)
│   └── utils/
│       └── progress.py           # Progress tracking (NEW)
├── tests/
│   ├── test_commands/
│   │   └── test_load.py          # Load command tests (NEW)
│   └── test_core/
│       ├── test_loader.py        # Loader tests (NEW)
│       └── test_validator.py     # Validator tests (NEW)
└── pyproject.toml                # No new dependencies
```

### Existing Patterns
- **Config:** Use Settings from US-001
- **Logging:** Use module-level loggers from US-001
- **Tracing:** Use tracer from US-001 for all operations
- **Retry:** Use retry decorator from US-001 for transient failures
- **Providers:** Use EmbeddingProvider and VectorDatabase from US-002

### Data Model (excerpt)

**Load Command Arguments:**
```python
@app.command()
async def load(
    file: str,                    # Path to input file (YAML or JSON)
    database: str,                # Database connection string
    embedding: str,               # Embedding model identifier
    mode: str = "initial",        # Load mode: initial, upsert, flush
    parallel: int = 1,            # Number of parallel workers
    batch_size: int = 32,         # Batch size for embedding
    force_reembed: bool = False,  # Force re-embedding on upsert
) -> None:
    """Load records into a vector database."""
```

**Load Statistics:**
```python
{
    "total_records": 100,
    "loaded_records": 98,
    "failed_records": 2,
    "skipped_records": 0,  # duplicates
    "total_tokens": 0,     # for API models
    "total_time": 15.3,
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    "database": "chromadb://localhost:8000/test_collection"
}
```

## Functional Requirements

### FR-006: Load Command - Initial Mode
- **Description:** Load records into a new or existing collection (default mode)
- **Inputs:** 
  - File path (YAML or JSON)
  - Database connection string
  - Embedding model identifier
  - Optional: parallel workers, batch size
- **Outputs:** 
  - Success message with statistics (total, loaded, failed, skipped, time)
  - Exit code 0 on success, 1 on failure
- **Business Rules:**
  - Skip records with duplicate IDs (keep first occurrence, log warning)
  - Auto-create collection if it doesn't exist
  - Verify dimension compatibility before loading
  - Display progress bar for large files (> 100 records)
  - Log summary statistics at INFO level
  - Trace entire load operation with statistics

### FR-007: Streaming File Processing
- **Description:** Process files incrementally without loading entire dataset into memory
- **Inputs:** 
  - File path (any size)
- **Outputs:** 
  - Records yielded one at a time or in small batches
- **Business Rules:**
  - Use generator pattern (yield records)
  - Memory usage must remain constant regardless of file size
  - Parse YAML/JSON incrementally (not json.load() for entire file)
  - Trace file read with file size and duration
  - Log file processing at DEBUG level

### FR-008: Batch Embedding
- **Description:** Generate embeddings in batches for efficiency
- **Inputs:** 
  - List of texts
  - Batch size (default: 32)
- **Outputs:** 
  - List of embeddings (same order as inputs)
- **Business Rules:**
  - Split texts into batches of specified size
  - Generate embeddings for each batch
  - Flatten results back to single list
  - Trace each batch with: batch_number, batch_size, duration
  - For local models: batch processing improves throughput
  - For API models: batch processing reduces API calls

### FR-017: Parallel Processing
- **Description:** Process multiple records concurrently using asyncio
- **Inputs:** 
  - Number of parallel workers (default: 1, max: 16)
- **Outputs:** 
  - Faster processing time (near-linear speedup for I/O-bound operations)
- **Business Rules:**
  - Use asyncio.gather() or asyncio.TaskGroup for parallel execution
  - Each worker processes a subset of records
  - Workers share the same embedding provider and database connection
  - Trace parallel execution with: worker_id, records_processed, duration
  - Log parallel processing at INFO level: "Using N parallel workers"
  - Handle worker failures gracefully (don't crash entire load)

### FR-018: Custom Batch Size
- **Description:** Allow users to configure embedding batch size
- **Inputs:** 
  - Batch size (default: 32, min: 1, max: 256)
- **Outputs:** 
  - Embeddings generated in specified batch size
- **Business Rules:**
  - Validate batch size is within allowed range
  - Smaller batches: lower memory usage, more API calls
  - Larger batches: higher memory usage, fewer API calls
  - Log batch size at INFO level
  - Trace each batch operation

### FR-019: Memory Efficiency (Streaming Mode)
- **Description:** Ensure constant memory usage for large files
- **Inputs:** 
  - File of any size (tested up to 10K records)
- **Outputs:** 
  - Peak memory < 500MB regardless of file size
- **Business Rules:**
  - Use generators/iterators (not lists) for file reading
  - Process records in small batches (don't accumulate in memory)
  - Release embeddings after database insert
  - Monitor memory usage in tests
  - Log memory-efficient processing at DEBUG level

### FR-020: Duplicate ID Handling
- **Description:** Detect and skip records with duplicate IDs
- **Inputs:** 
  - Records with potentially duplicate IDs
- **Outputs:** 
  - First occurrence kept, subsequent duplicates skipped
  - Warning logged for each duplicate
- **Business Rules:**
  - Track seen IDs in a set (memory-efficient)
  - Skip duplicate without generating embedding (save compute)
  - Log warning: "Duplicate ID skipped: <id>"
  - Include skipped count in summary statistics
  - Trace duplicate detection

## Acceptance Tests

> **Acceptance tests are mandatory: 100% must pass.**
> A user story is NOT considered implemented until **every single acceptance test below passes**.
> The implementing agent MUST loop (fix code → run tests → check results → repeat) until all acceptance tests pass with zero failures. Do not stop or declare the story "done" while any test is failing.
> Tests MUST be validated through the project's CI/CD chain (generally `make test` or equivalent Makefile target). No other method of running or validating tests is acceptable: do not run test files directly, do not use ad hoc commands. Use the Makefile.

### Test Data

| Data | Description | Source | Status |
|------|-------------|--------|--------|
| test_data_100.yaml | 100 records for standard testing | auto-generated (pytest fixture) | ready |
| test_data_10k.yaml | 10,000 records for streaming test | auto-generated (pytest fixture) | ready |
| test_data_duplicates.yaml | 10 records with 2 duplicates | auto-generated (pytest fixture) | ready |
| ChromaDB instance | Local ChromaDB for testing | docker-compose or persistent | ready |

### Happy Path Tests

### E2E-001: Initial Dataset Load with Local Embedding Model
- **Category:** happy
- **Scenario:** SC-001
- **Requirements:** FR-006, FR-007, FR-008
- **Preconditions:**
  - ChromaDB at localhost:8000
  - test_data_100.yaml with 100 records
  - sentence-transformers/all-MiniLM-L6-v2 available
- **Steps:**
  - Given clean ChromaDB instance
  - When: `rag-tester load --file test_data_100.yaml --database chromadb://localhost:8000/test_collection --embedding sentence-transformers/all-MiniLM-L6-v2 --parallel 4`
  - Then exit code 0
  - And stdout contains: "Successfully loaded 100 records", "Failed records: 0", "Skipped records: 0", "Total time: X seconds"
  - And collection exists with 100 documents, dimension 384
  - And trace file has spans: file_read, embedding_batch (multiple), database_insert (multiple), load_summary
  - And log file contains: "Using 4 parallel workers", "Collection created: test_collection", "Load complete: 100/100 records"
- **Cleanup:** Delete collection
- **Priority:** Critical

### E2E-010: Parallel Workers
- **Category:** happy
- **Scenario:** SC-001
- **Requirements:** FR-017
- **Preconditions:**
  - test_data_100.yaml with 100 records
- **Steps:**
  - Given clean collection
  - When: `rag-tester load --file test_data_100.yaml --database chromadb://localhost:8000/parallel_test --embedding sentence-transformers/all-MiniLM-L6-v2 --parallel 4`
  - Then exit code 0
  - And trace shows concurrent operations (4 workers with overlapping time ranges)
  - And total time < 50% of sequential load (--parallel 1)
  - And log contains: "Using 4 parallel workers"
- **Cleanup:** Delete collection
- **Priority:** High

### E2E-011: Custom Batch Size
- **Category:** happy
- **Scenario:** SC-001
- **Requirements:** FR-018
- **Preconditions:**
  - test_data_100.yaml with 100 records
- **Steps:**
  - Given clean collection
  - When: `rag-tester load --file test_data_100.yaml --database chromadb://localhost:8000/batch_test --embedding sentence-transformers/all-MiniLM-L6-v2 --batch-size 32`
  - Then exit code 0
  - And trace shows 4 embedding batches (32+32+32+4)
  - And each batch span has attribute: batch_size=32 (except last with 4)
  - And log contains: "Batch size: 32"
- **Cleanup:** Delete collection
- **Priority:** Medium

### E2E-012: Streaming Mode (Large File)
- **Category:** happy
- **Scenario:** SC-001
- **Requirements:** FR-019
- **Preconditions:**
  - test_data_10k.yaml with 10,000 records
- **Steps:**
  - Given clean collection
  - When: `rag-tester load --file test_data_10k.yaml --database chromadb://localhost:8000/streaming_test --embedding sentence-transformers/all-MiniLM-L6-v2 --parallel 4`
  - Then exit code 0
  - And peak memory usage < 500MB (monitored during test)
  - And all 10,000 records loaded successfully
  - And trace shows streaming file read (not single large read)
  - And progress bar displayed during load
- **Cleanup:** Delete collection
- **Priority:** High

### E2E-013: Duplicate IDs
- **Category:** happy
- **Scenario:** SC-001
- **Requirements:** FR-020
- **Preconditions:**
  - test_data_duplicates.yaml with 10 records, IDs: doc1-doc8, doc5 (duplicate), doc3 (duplicate)
- **Steps:**
  - Given clean collection
  - When: `rag-tester load --file test_data_duplicates.yaml --database chromadb://localhost:8000/dup_test --embedding sentence-transformers/all-MiniLM-L6-v2`
  - Then exit code 0
  - And stdout contains: "Successfully loaded 8 records", "Skipped records: 2"
  - And collection has exactly 8 documents (first occurrences of doc5 and doc3 kept)
  - And log contains: "Duplicate ID skipped: doc5", "Duplicate ID skipped: doc3"
  - And trace has duplicate_detection spans
- **Cleanup:** Delete collection
- **Priority:** High

### E2E-064: Load Latency (Local Model)
- **Category:** happy (performance baseline)
- **Scenario:** SC-001
- **Requirements:** NFR-001
- **Preconditions:**
  - test_data_100.yaml with 100 records
- **Steps:**
  - Given clean collection
  - When: `rag-tester load --file test_data_100.yaml --database chromadb://localhost:8000/perf_test --embedding sentence-transformers/all-MiniLM-L6-v2 --parallel 1`
  - Then exit code 0
  - And total time < 30 seconds
  - And stdout shows time taken
- **Cleanup:** Delete collection
- **Priority:** Medium

### Edge Case and Error Tests

### E2E-081: Empty Input File
- **Category:** failure
- **Scenario:** File validation
- **Requirements:** FR-006
- **Preconditions:**
  - empty.yaml with no records
- **Steps:**
  - Given file path "empty.yaml"
  - When: `rag-tester load --file empty.yaml --database chromadb://localhost:8000/test --embedding sentence-transformers/all-MiniLM-L6-v2`
  - Then exit code 1
  - And stderr contains: "Error: Input file is empty or has no records"
  - And no collection created
- **Cleanup:** None
- **Priority:** High

### E2E-082: Single Record Load
- **Category:** edge
- **Scenario:** Minimal data
- **Requirements:** FR-006
- **Preconditions:**
  - single.yaml with 1 record
- **Steps:**
  - Given file with 1 record
  - When: `rag-tester load --file single.yaml --database chromadb://localhost:8000/single_test --embedding sentence-transformers/all-MiniLM-L6-v2`
  - Then exit code 0
  - And stdout contains: "Successfully loaded 1 record"
  - And collection has exactly 1 document
- **Cleanup:** Delete collection
- **Priority:** Medium

### E2E-083: Very Long Text (10K chars)
- **Category:** edge
- **Scenario:** Large text handling
- **Requirements:** FR-006, FR-008
- **Preconditions:**
  - long_text.yaml with 1 record having 10,000 character text
- **Steps:**
  - Given file with very long text record
  - When: `rag-tester load --file long_text.yaml --database chromadb://localhost:8000/long_test --embedding sentence-transformers/all-MiniLM-L6-v2`
  - Then exit code 0
  - And record loaded successfully
  - And embedding dimension is 384
  - And text round-trips correctly
- **Cleanup:** Delete collection
- **Priority:** Medium

### E2E-084: Unicode and Emoji
- **Category:** edge
- **Scenario:** Special characters
- **Requirements:** FR-006
- **Preconditions:**
  - unicode.yaml with text: "Hello 世界 🌍 مرحبا"
- **Steps:**
  - Given file with Unicode and emoji
  - When: `rag-tester load --file unicode.yaml --database chromadb://localhost:8000/unicode_test --embedding sentence-transformers/all-MiniLM-L6-v2`
  - Then exit code 0
  - And record loaded successfully
  - And text round-trips correctly (exact match)
- **Cleanup:** Delete collection
- **Priority:** High

### E2E-LOAD-001: Invalid File Format
- **Category:** failure
- **Scenario:** File validation
- **Requirements:** FR-006
- **Preconditions:**
  - malformed.yaml with invalid YAML syntax
- **Steps:**
  - Given file with malformed YAML
  - When: `rag-tester load --file malformed.yaml --database chromadb://localhost:8000/test --embedding sentence-transformers/all-MiniLM-L6-v2`
  - Then exit code 1
  - And stderr contains: "Error: Invalid file format. Failed to parse YAML: <error details>"
  - And no collection created
- **Cleanup:** None
- **Priority:** Critical

### E2E-LOAD-002: Missing Required Fields
- **Category:** failure
- **Scenario:** Record validation
- **Requirements:** FR-006
- **Preconditions:**
  - invalid_records.yaml with record missing "text" field
- **Steps:**
  - Given file with invalid record: {id: "doc1"}
  - When: `rag-tester load --file invalid_records.yaml --database chromadb://localhost:8000/test --embedding sentence-transformers/all-MiniLM-L6-v2`
  - Then exit code 1
  - And stderr contains: "Error: Missing required field 'text' in record 'doc1'"
  - And no records loaded
- **Cleanup:** None
- **Priority:** Critical

### E2E-LOAD-003: Database Unreachable
- **Category:** failure
- **Scenario:** Connection error
- **Requirements:** FR-006
- **Preconditions:**
  - ChromaDB server down or unreachable
  - test_data_100.yaml
- **Steps:**
  - Given ChromaDB at wrong port
  - When: `rag-tester load --file test_data_100.yaml --database chromadb://localhost:9999/test --embedding sentence-transformers/all-MiniLM-L6-v2`
  - Then exit code 1
  - And stderr contains: "Error: Database connection failed: <error details>"
  - And trace shows 5 retry attempts with exponential backoff
  - And log contains retry messages
- **Cleanup:** None
- **Priority:** Critical

### E2E-LOAD-004: Dimension Mismatch
- **Category:** failure
- **Scenario:** Validation error
- **Requirements:** FR-006
- **Preconditions:**
  - Existing collection with dimension 768
  - test_data_100.yaml
- **Steps:**
  - Given collection "existing_768" with dimension 768
  - When: `rag-tester load --file test_data_100.yaml --database chromadb://localhost:8000/existing_768 --embedding sentence-transformers/all-MiniLM-L6-v2`
  - Then exit code 1
  - And stderr contains: "Error: Dimension mismatch: model=384, database=768"
  - And no new records loaded
- **Cleanup:** Delete collection
- **Priority:** Critical

### E2E-LOAD-005: Invalid Batch Size
- **Category:** failure
- **Scenario:** Argument validation
- **Requirements:** FR-018
- **Preconditions:**
  - test_data_100.yaml
- **Steps:**
  - Given batch size 0 (invalid)
  - When: `rag-tester load --file test_data_100.yaml --database chromadb://localhost:8000/test --embedding sentence-transformers/all-MiniLM-L6-v2 --batch-size 0`
  - Then exit code 1
  - And stderr contains: "Error: Batch size must be between 1 and 256"
- **Cleanup:** None
- **Priority:** High

### E2E-LOAD-006: Invalid Parallel Workers
- **Category:** failure
- **Scenario:** Argument validation
- **Requirements:** FR-017
- **Preconditions:**
  - test_data_100.yaml
- **Steps:**
  - Given parallel workers 0 (invalid)
  - When: `rag-tester load --file test_data_100.yaml --database chromadb://localhost:8000/test --embedding sentence-transformers/all-MiniLM-L6-v2 --parallel 0`
  - Then exit code 1
  - And stderr contains: "Error: Parallel workers must be between 1 and 16"
- **Cleanup:** None
- **Priority:** High

### E2E-LOAD-007: File Not Found
- **Category:** failure
- **Scenario:** File validation
- **Requirements:** FR-006
- **Preconditions:**
  - File "nonexistent.yaml" does not exist
- **Steps:**
  - Given non-existent file path
  - When: `rag-tester load --file nonexistent.yaml --database chromadb://localhost:8000/test --embedding sentence-transformers/all-MiniLM-L6-v2`
  - Then exit code 1
  - And stderr contains: "Error: File not found: nonexistent.yaml"
- **Cleanup:** None
- **Priority:** Critical

### E2E-LOAD-008: Progress Bar Display
- **Category:** edge
- **Scenario:** User feedback
- **Requirements:** FR-006
- **Preconditions:**
  - test_data_100.yaml with 100 records
- **Steps:**
  - Given file with 100+ records
  - When: `rag-tester load --file test_data_100.yaml --database chromadb://localhost:8000/progress_test --embedding sentence-transformers/all-MiniLM-L6-v2`
  - Then progress bar is displayed during load
  - And progress updates show: "Loading: 25/100 (25%)", "Loading: 50/100 (50%)", etc.
  - And progress bar disappears on completion
- **Cleanup:** Delete collection
- **Priority:** Medium

## Constraints

### Files Not to Touch
- `src/rag_tester/providers/` (already implemented in US-002)
- `src/rag_tester/config.py` (from US-001)
- `src/rag_tester/logging_config.py` (from US-001)
- `src/rag_tester/tracing.py` (from US-001)

### Dependencies Not to Add
- No new dependencies required (all needed packages already in pyproject.toml)

### Patterns to Avoid
- Do NOT load entire file into memory (use streaming)
- Do NOT use threading (use asyncio for concurrency)
- Do NOT create new database connections per worker (share connections)
- Do NOT accumulate all embeddings before inserting (insert in batches)

### Scope Boundary
- This story does NOT implement upsert or flush modes (that's US-007)
- This story does NOT implement test, bulk-test, or compare commands (US-004, US-005, US-006)
- This story does NOT implement API embedding providers (that's US-008)
- This story does NOT implement other database backends (that's US-009)
- This story ONLY implements: load command with initial mode, streaming, parallel processing, batch optimization

## Non Regression

### Existing Tests That Must Pass
- All tests from US-001 (infrastructure)
- All tests from US-002 (providers)

### Behaviors That Must Not Change
- Configuration system from US-001
- Logging and tracing from US-001
- Provider interfaces from US-002

### API Contracts to Preserve
- EmbeddingProvider interface from US-002
- VectorDatabase interface from US-002
