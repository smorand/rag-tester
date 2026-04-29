# US-007: Load Modes - Upsert & Flush

> Parent Spec: specs/2026-04-29_18:26:42-rag-testing-framework.md
> Status: ready
> Priority: 7
> Depends On: US-003
> Complexity: M

## Objective

Extend the `load` command with upsert and flush modes for advanced data management. Upsert mode enables updating existing records and adding new ones in a single operation, with optional re-embedding. Flush mode provides a clean slate by deleting all existing data before loading new records. These modes are essential for iterative development, data updates, and testing different datasets.

## Technical Context

### Stack
- **Language:** Python 3.13+
- **CLI Framework:** Typer (command extension)
- **Testing:** pytest, pytest-asyncio, pytest-cov

### Relevant File Structure
```
rag-tester/
├── src/rag_tester/
│   ├── commands/
│   │   └── load.py               # Load command (UPDATE: add upsert/flush modes)
│   ├── core/
│   │   └── loader.py             # Load logic (UPDATE: add mode strategies)
│   └── providers/
│       └── databases/
│           └── base.py           # VectorDatabase ABC (UPDATE: add delete methods)
├── tests/
│   ├── test_commands/
│   │   └── test_load_modes.py    # Load modes tests (NEW)
│   └── test_core/
│       └── test_loader_modes.py  # Loader modes tests (NEW)
└── pyproject.toml                # No new dependencies
```

### Existing Patterns
- **Config:** Use Settings from US-001
- **Logging:** Use module-level loggers from US-001
- **Tracing:** Use tracer from US-001 for all operations
- **Providers:** Use EmbeddingProvider and VectorDatabase from US-002
- **Load Logic:** Extend loader from US-003

### Data Model (excerpt)

**Load Command with Modes:**
```python
@app.command()
async def load(
    file: str,
    database: str,
    embedding: str,
    mode: str = "initial",        # NEW: initial, upsert, flush
    force_reembed: bool = False,  # NEW: force re-embedding on upsert
    parallel: int = 1,
    batch_size: int = 32,
) -> None:
    """Load records into a vector database."""
```

**Load Statistics (Extended):**
```python
{
    "total_records": 100,
    "loaded_records": 90,      # new records added
    "updated_records": 10,     # existing records updated (upsert only)
    "deleted_records": 50,     # records deleted (flush only)
    "failed_records": 0,
    "skipped_records": 0,
    "total_tokens": 0,
    "total_time": 15.3,
    "mode": "upsert",
    "force_reembed": true
}
```

## Functional Requirements

### FR-011: Upsert Mode
- **Description:** Update existing records and add new ones in a single operation
- **Inputs:** 
  - File with records (some IDs exist, some are new)
  - Database connection string
  - Embedding model identifier
  - Optional: force_reembed flag
- **Outputs:** 
  - Success message with statistics (updated, added, failed)
  - Exit code 0 on success, 1 on failure
- **Business Rules:**
  - For each record:
    - If ID exists in database: update text and embedding (if force_reembed=true) or just text (if force_reembed=false)
    - If ID does not exist: insert as new record
  - By default (force_reembed=false): reuse existing embeddings for unchanged text
  - With force_reembed=true: regenerate embeddings for all records (even if text unchanged)
  - Display progress bar for large files
  - Log statistics: "Records updated: X", "Records added: Y"
  - Trace upsert operation with: updated_count, added_count, duration

### FR-012: Flush Mode
- **Description:** Delete all existing data and load new records
- **Inputs:** 
  - File with new records
  - Database connection string
  - Embedding model identifier
- **Outputs:** 
  - Success message with statistics (deleted, loaded)
  - Exit code 0 on success, 1 on failure
- **Business Rules:**
  - Step 1: Delete all records from collection/table
  - Step 2: Load new records (same as initial mode)
  - Operation should be atomic where possible (transaction support)
  - If delete fails, do not proceed with load
  - Display progress bar for load phase
  - Log statistics: "Records deleted: X", "Records loaded: Y"
  - Trace flush operation with: deleted_count, loaded_count, duration

### FR-013: Force Re-embedding Flag
- **Description:** Force regeneration of embeddings during upsert
- **Inputs:** 
  - force_reembed flag (default: false)
- **Outputs:** 
  - Embeddings regenerated for all records (if true)
- **Business Rules:**
  - Only applicable in upsert mode (ignored in initial and flush modes)
  - If false: reuse existing embeddings for records with unchanged text
  - If true: regenerate embeddings for all records, even if text unchanged
  - Useful when switching embedding models or testing different embeddings
  - Log re-embedding decision at INFO level
  - Trace re-embedding operations

### FR-014: Database Delete Operations
- **Description:** Extend VectorDatabase ABC with delete methods
- **Inputs:** 
  - Collection/table name
  - Optional: specific record IDs
- **Outputs:** 
  - Confirmation of deletion
- **Business Rules:**
  - Add to VectorDatabase ABC:
    - `async def delete_all(collection: str) -> int` - delete all records, return count
    - `async def delete_by_ids(collection: str, ids: list[str]) -> int` - delete specific records, return count
  - Implement for ChromaDB (both HTTP and persistent modes)
  - Trace delete operations with: collection, count, duration
  - Log deletions at INFO level

## Acceptance Tests

> **Acceptance tests are mandatory: 100% must pass.**
> A user story is NOT considered implemented until **every single acceptance test below passes**.
> The implementing agent MUST loop (fix code → run tests → check results → repeat) until all acceptance tests pass with zero failures. Do not stop or declare the story "done" while any test is failing.
> Tests MUST be validated through the project's CI/CD chain (generally `make test` or equivalent Makefile target). No other method of running or validating tests is acceptable: do not run test files directly, do not use ad hoc commands. Use the Makefile.

### Test Data

| Data | Description | Source | Status |
|------|-------------|--------|--------|
| initial_data.yaml | 100 records for initial load | auto-generated (pytest fixture) | ready |
| updates.yaml | 10 existing + 10 new IDs | auto-generated (pytest fixture) | ready |
| new_data.yaml | 50 completely new records | auto-generated (pytest fixture) | ready |
| ChromaDB instance | Local ChromaDB for testing | docker-compose or persistent | ready |

### Happy Path Tests

### E2E-006: Upsert Mode
- **Category:** happy
- **Scenario:** SC-006
- **Requirements:** FR-011, FR-013
- **Preconditions:**
  - Collection with 100 docs
  - updates.yaml with 10 existing IDs (modified text) + 10 new IDs
- **Steps:**
  - Given collection "test_collection" with 100 documents
  - When: `rag-tester load --file updates.yaml --database chromadb://localhost:8000/test_collection --embedding sentence-transformers/all-MiniLM-L6-v2 --mode upsert --force-reembed`
  - Then exit code 0
  - And stdout contains: "Records updated: 10", "Records added: 10"
  - And collection has 110 documents total
  - And updated documents have new text and new embeddings
  - And trace has spans: upsert_operation, embedding_batch, database_update, database_insert
  - And log contains: "Mode: upsert", "Force re-embed: true"
- **Cleanup:** Delete collection
- **Priority:** High

### E2E-007: Flush Mode
- **Category:** happy
- **Scenario:** SC-007
- **Requirements:** FR-012, FR-014
- **Preconditions:**
  - Collection with 110 docs
  - new_data.yaml with 50 records
- **Steps:**
  - Given collection "test_collection" with 110 documents
  - When: `rag-tester load --file new_data.yaml --database chromadb://localhost:8000/test_collection --embedding sentence-transformers/all-MiniLM-L6-v2 --mode flush`
  - Then exit code 0
  - And stdout contains: "Records deleted: 110", "Records loaded: 50"
  - And collection has exactly 50 documents (old IDs gone, only new data)
  - And trace has spans: flush_operation, database_delete_all, embedding_batch, database_insert
  - And log contains: "Mode: flush", "Deleted 110 records", "Loaded 50 records"
- **Cleanup:** Delete collection
- **Priority:** High

### E2E-028: Upsert Updates Documents
- **Category:** happy (side effect)
- **Scenario:** SC-006
- **Requirements:** FR-011
- **Preconditions:**
  - Collection with doc42: "Old text about machine learning"
  - updates.yaml with doc42: "New text about machine learning"
- **Steps:**
  - Given collection with known document
  - When upsert is executed with doc42 having new text
  - Then doc42 text is updated to "New text about machine learning"
  - And doc42 embedding is different from original (force_reembed=true)
  - And query for "machine learning" still returns doc42
- **Cleanup:** Delete collection
- **Priority:** High

### E2E-029: Flush Deletes All
- **Category:** happy (side effect)
- **Scenario:** SC-007
- **Requirements:** FR-012
- **Preconditions:**
  - Collection with 100 docs (IDs: doc1-doc100)
  - new_data.yaml with 50 docs (IDs: new1-new50)
- **Steps:**
  - Given collection with known IDs
  - When flush is executed
  - Then old IDs (doc1-doc100) are gone
  - And only new IDs (new1-new50) exist
  - And query for old documents returns no results
- **Cleanup:** Delete collection
- **Priority:** High

### E2E-057: ID Uniqueness Enforced
- **Category:** happy (data integrity)
- **Scenario:** SC-006
- **Requirements:** FR-011
- **Preconditions:**
  - Collection with doc5
  - Upsert with doc5 (different text)
- **Steps:**
  - Given collection with doc5
  - When upsert is executed with doc5
  - Then only one doc5 exists in collection
  - And doc5 has updated data
- **Cleanup:** Delete collection
- **Priority:** High

### E2E-061: Flush Atomicity
- **Category:** happy (data integrity)
- **Scenario:** SC-007
- **Requirements:** FR-012
- **Preconditions:**
  - Collection with 100 docs
  - new_data.yaml with 50 docs
- **Steps:**
  - Given collection with data
  - When flush is executed
  - Then either: all old data deleted + new data loaded (success)
  - Or: operation fails with no partial state (failure)
  - And no state where old data deleted but new data not loaded
- **Cleanup:** Delete collection
- **Priority:** High

### E2E-079: Upsert Then Test
- **Category:** happy (cross-scenario)
- **Scenario:** SC-006 + SC-002
- **Requirements:** FR-011
- **Preconditions:**
  - Collection with doc5: "Python is a language"
  - updates.yaml with doc5: "Python is a programming language"
- **Steps:**
  - Given collection with doc5
  - When upsert is executed with updated doc5
  - And test query "programming language" is executed
  - Then results include doc5 with high score
  - And doc5 text matches updated version
- **Cleanup:** Delete collection
- **Priority:** High

### E2E-080: Flush Then Test
- **Category:** happy (cross-scenario)
- **Scenario:** SC-007 + SC-002
- **Requirements:** FR-012
- **Preconditions:**
  - Collection with old data
  - new_data.yaml with doc_new: "Machine learning"
- **Steps:**
  - Given collection with old data
  - When flush is executed with new data
  - And test query "machine learning" is executed
  - Then results include doc_new (from new data)
  - And results do NOT include any old documents
- **Cleanup:** Delete collection
- **Priority:** High

### Edge Case and Error Tests

### E2E-MODE-001: Upsert Without Force Re-embed
- **Category:** edge
- **Scenario:** Optimization
- **Requirements:** FR-011, FR-013
- **Preconditions:**
  - Collection with doc5: "Python is a language"
  - updates.yaml with doc5: "Python is a language" (same text)
- **Steps:**
  - Given collection with doc5
  - When: `rag-tester load --file updates.yaml --database chromadb://localhost:8000/test_collection --embedding sentence-transformers/all-MiniLM-L6-v2 --mode upsert`
  - Then exit code 0
  - And doc5 embedding is NOT regenerated (same as before)
  - And log contains: "Reusing existing embedding for doc5 (text unchanged)"
  - And trace shows no embedding operation for doc5
- **Cleanup:** Delete collection
- **Priority:** Medium

### E2E-MODE-002: Upsert All New Records
- **Category:** edge
- **Scenario:** Edge case
- **Requirements:** FR-011
- **Preconditions:**
  - Collection with 100 docs (doc1-doc100)
  - updates.yaml with 50 new docs (new1-new50)
- **Steps:**
  - Given collection with existing data
  - When upsert is executed with all new IDs
  - Then stdout contains: "Records updated: 0", "Records added: 50"
  - And collection has 150 documents total
- **Cleanup:** Delete collection
- **Priority:** Medium

### E2E-MODE-003: Upsert All Existing Records
- **Category:** edge
- **Scenario:** Edge case
- **Requirements:** FR-011
- **Preconditions:**
  - Collection with 100 docs
  - updates.yaml with same 100 IDs (modified text)
- **Steps:**
  - Given collection with 100 documents
  - When upsert is executed with all existing IDs
  - Then stdout contains: "Records updated: 100", "Records added: 0"
  - And collection still has 100 documents
  - And all documents have updated text
- **Cleanup:** Delete collection
- **Priority:** Medium

### E2E-MODE-004: Flush Empty Collection
- **Category:** edge
- **Scenario:** Edge case
- **Requirements:** FR-012
- **Preconditions:**
  - Empty collection
  - new_data.yaml with 50 docs
- **Steps:**
  - Given empty collection
  - When flush is executed
  - Then stdout contains: "Records deleted: 0", "Records loaded: 50"
  - And collection has 50 documents
  - And no errors
- **Cleanup:** Delete collection
- **Priority:** Low

### E2E-MODE-005: Invalid Mode
- **Category:** failure
- **Scenario:** Argument validation
- **Requirements:** FR-011, FR-012
- **Preconditions:**
  - test_data.yaml
- **Steps:**
  - Given invalid mode "replace"
  - When: `rag-tester load --file test_data.yaml --database chromadb://localhost:8000/test --embedding sentence-transformers/all-MiniLM-L6-v2 --mode replace`
  - Then exit code 1
  - And stderr contains: "Error: Invalid mode. Must be one of: initial, upsert, flush"
- **Cleanup:** None
- **Priority:** High

### E2E-MODE-006: Force Re-embed in Initial Mode
- **Category:** edge
- **Scenario:** Flag ignored
- **Requirements:** FR-013
- **Preconditions:**
  - test_data.yaml
- **Steps:**
  - Given initial mode with force_reembed flag
  - When: `rag-tester load --file test_data.yaml --database chromadb://localhost:8000/test --embedding sentence-transformers/all-MiniLM-L6-v2 --mode initial --force-reembed`
  - Then exit code 0
  - And log contains warning: "force-reembed flag ignored in initial mode"
  - And all records loaded normally
- **Cleanup:** Delete collection
- **Priority:** Low

### E2E-MODE-007: Flush Delete Failure
- **Category:** failure
- **Scenario:** Database error
- **Requirements:** FR-012
- **Preconditions:**
  - Collection with 100 docs
  - Database becomes unavailable during delete (mock)
- **Steps:**
  - Given collection with data
  - When flush is executed
  - And delete operation fails
  - Then exit code 1
  - And stderr contains: "Error: Failed to delete existing records: <error>"
  - And no new records loaded (operation aborted)
  - And old data remains intact
- **Cleanup:** Delete collection
- **Priority:** High

### E2E-MODE-008: Upsert Partial Failure
- **Category:** failure
- **Scenario:** Partial success
- **Requirements:** FR-011
- **Preconditions:**
  - Collection with 100 docs
  - updates.yaml with 20 records (10 updates, 10 new)
  - 2 records fail to update (mock)
- **Steps:**
  - Given collection with data
  - When upsert is executed
  - And 2 records fail
  - Then exit code 0 (partial success)
  - And stdout contains: "Records updated: 8", "Records added: 10", "Failed records: 2"
  - And log lists failed record IDs
  - And successful records are updated/added
- **Cleanup:** Delete collection
- **Priority:** High

## Constraints

### Files Not to Touch
- `src/rag_tester/providers/embeddings/` (from US-002)
- `src/rag_tester/commands/test.py` (from US-004)
- `src/rag_tester/commands/bulk_test.py` (from US-005)
- `src/rag_tester/commands/compare.py` (from US-006)

### Dependencies Not to Add
- No new dependencies required (all needed packages already in pyproject.toml)

### Patterns to Avoid
- Do NOT delete and re-insert for upsert (use proper update operations)
- Do NOT load all existing data into memory to check for updates (query by ID)
- Do NOT proceed with load if flush delete fails (check delete success first)

### Scope Boundary
- This story does NOT implement API embedding providers (that's US-008)
- This story does NOT implement other database backends (that's US-009)
- This story does NOT implement test, bulk-test, or compare commands (already done)
- This story ONLY implements: upsert and flush modes for the load command

## Non Regression

### Existing Tests That Must Pass
- All tests from US-001 (infrastructure)
- All tests from US-002 (providers)
- All tests from US-003 (load command - initial mode)
- All tests from US-004 (test command)
- All tests from US-005 (bulk-test command)
- All tests from US-006 (compare command)

### Behaviors That Must Not Change
- Configuration system from US-001
- Logging and tracing from US-001
- Provider interfaces from US-002
- Load command initial mode from US-003
- Test command from US-004
- Bulk-test command from US-005
- Compare command from US-006

### API Contracts to Preserve
- EmbeddingProvider interface from US-002
- VectorDatabase interface from US-002 (extended with delete methods)
- Load command arguments from US-003 (extended with mode and force_reembed)
