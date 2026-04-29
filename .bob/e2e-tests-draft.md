# RAG Tester - End-to-End Test Suite (Draft)

> This is a comprehensive test plan covering all scenarios, failure modes, edge cases, and side effects.
> Tests are organized by category and linked to usage scenarios.

## Test Summary Statistics

| Category | Count | Description |
|----------|-------|-------------|
| Core User Journeys | 7 | Complete end-to-end flows |
| Feature-Specific | 24 | Per-command functionality |
| Side Effects | 8 | Observable consequences verification |
| Error Handling | 18 | Failure modes and recovery |
| Security | 6 | Authentication and validation |
| Data Integrity | 7 | Consistency and correctness |
| Performance Baseline | 5 | Response time and resource usage |
| Integration | 8 | External dependencies |
| Cross-Scenario | 4 | Scenario interactions |
| **TOTAL** | **87** | |

**Coverage Ratios:**
- Happy path tests: 24
- Failure/error tests: 31
- Edge case tests: 19
- Side effect tests: 8
- Other: 5
- **Happy:Failure ratio: 1:1.29** ✓ (meets >1:1 requirement)

---

## 1. Core User Journeys (Complete Flows)

### E2E-001: Initial Dataset Load with Local Embedding Model
- **Category:** Core Journey
- **Scenario:** SC-001 (Load initial dataset)
- **Requirements:** FR-001, FR-002, FR-003
- **Preconditions:**
  - ChromaDB server running at localhost:8000
  - Load file `test_data.yaml` exists with 100 records (ID: "doc1" to "doc100", Text: various)
  - No existing collection named "test_collection"
  - sentence-transformers library installed
- **Steps:**
  - Given a clean ChromaDB instance
  - When user runs: `rag-tester load --file test_data.yaml --database chromadb://localhost:8000/test_collection --embedding sentence-transformers/all-MiniLM-L6-v2 --parallel 4`
  - Then command exits with code 0
  - And stdout shows: "Successfully loaded 100 records"
  - And stdout shows: "Failed records: 0"
  - And stdout shows: "Total tokens consumed: 0" (local model)
  - And stdout shows: "Total time: <X> seconds"
  - And ChromaDB collection "test_collection" exists
  - And ChromaDB collection contains exactly 100 documents
  - And each document has ID matching input file
  - And each document has embedding vector of dimension 384
  - And trace file contains spans for: file_read, embedding_batch, database_insert
- **Cleanup:** Delete ChromaDB collection "test_collection"
- **Priority:** Critical

### E2E-002: Manual Query Test Against Loaded Database
- **Category:** Core Journey
- **Scenario:** SC-002 (Manual test)
- **Requirements:** FR-004, FR-005
- **Preconditions:**
  - ChromaDB collection "test_collection" loaded with 100 documents (from E2E-001)
  - Document "doc42" contains text: "Machine learning is a subset of artificial intelligence"
- **Steps:**
  - Given a loaded ChromaDB collection
  - When user runs: `rag-tester test "What is machine learning?" --database chromadb://localhost:8000/test_collection --embedding sentence-transformers/all-MiniLM-L6-v2 --top-k 3 --format table`
  - Then command exits with code 0
  - And stdout displays a table with 3 rows
  - And table contains columns: Rank, ID, Text (truncated), Score
  - And row 1 has ID "doc42" with score > 0.7
  - And stdout shows: "Tokens consumed: 0" (local model)
  - And stdout shows: "Time taken: <X> seconds"
  - And trace file contains spans for: embedding_query, database_search
- **Cleanup:** None
- **Priority:** Critical

### E2E-003: Bulk Test Suite Execution with Pass and Fail Cases
- **Category:** Core Journey
- **Scenario:** SC-003 (Bulk test)
- **Requirements:** FR-006, FR-007, FR-008
- **Preconditions:**
  - ChromaDB collection "test_collection" loaded with 100 documents
  - Test file `test_suite.yaml` exists with 10 tests:
    - 7 tests with correct expected results (should pass)
    - 2 tests with incorrect expected IDs (should fail)
    - 1 test with correct ID but threshold too high (should fail)
- **Steps:**
  - Given a loaded ChromaDB collection and valid test file
  - When user runs: `rag-tester bulk-test --file test_suite.yaml --database chromadb://localhost:8000/test_collection --embedding sentence-transformers/all-MiniLM-L6-v2 --output results.yaml --parallel 2`
  - Then command exits with code 0
  - And file `results.yaml` is created
  - And results.yaml contains summary section with:
    - total_tests: 10
    - passed: 7
    - failed: 3
    - total_tokens: 0
    - total_time: <X> seconds
  - And results.yaml contains per-test results (only failed tests by default)
  - And failed test entries show: test_id, query, expected, actual, reason
  - And trace file contains spans for: test_execution (10 times), embedding_query, database_search
- **Cleanup:** Delete results.yaml
- **Priority:** Critical

### E2E-004: Bulk Test with Verbose Output
- **Category:** Core Journey
- **Scenario:** SC-003 (Bulk test with verbose)
- **Requirements:** FR-006, FR-007, FR-008
- **Preconditions:**
  - Same as E2E-003
- **Steps:**
  - Given a loaded ChromaDB collection and valid test file
  - When user runs: `rag-tester bulk-test --file test_suite.yaml --database chromadb://localhost:8000/test_collection --embedding sentence-transformers/all-MiniLM-L6-v2 --output results_verbose.yaml --verbose`
  - Then command exits with code 0
  - And file `results_verbose.yaml` is created
  - And results_verbose.yaml contains summary section (same as E2E-003)
  - And results_verbose.yaml contains ALL 10 test results (passed and failed)
  - And each test entry shows: test_id, query, expected, actual, score, status, duration
- **Cleanup:** Delete results_verbose.yaml
- **Priority:** High

### E2E-005: Compare Results from Two Different Embedding Models
- **Category:** Core Journey
- **Scenario:** SC-004 (Compare models)
- **Requirements:** FR-009, FR-010
- **Preconditions:**
  - Two result files exist:
    - `results_model_a.yaml` (from BAAI/bge-small-en-v1.5, 8/10 passed)
    - `results_model_b.yaml` (from all-MiniLM-L6-v2, 7/10 passed)
- **Steps:**
  - Given two valid result files from different embedding models
  - When user runs: `rag-tester compare --results results_model_a.yaml results_model_b.yaml --output comparison.yaml`
  - Then command exits with code 0
  - And file `comparison.yaml` is created
  - And comparison.yaml contains:
    - model_a: {name: "BAAI/bge-small-en-v1.5", pass_rate: 0.8, avg_score: 0.85, total_tokens: 0, total_time: 5.2}
    - model_b: {name: "all-MiniLM-L6-v2", pass_rate: 0.7, avg_score: 0.82, total_tokens: 0, total_time: 4.1}
    - per_test_diff: [{test_id: "test3", model_a_status: "passed", model_b_status: "failed", ...}, ...]
- **Cleanup:** Delete comparison.yaml
- **Priority:** High

### E2E-006: Upsert Mode - Update Existing Records
- **Category:** Core Journey
- **Scenario:** SC-006 (Upsert)
- **Requirements:** FR-011, FR-012
- **Preconditions:**
  - ChromaDB collection "test_collection" loaded with 100 documents
  - Update file `updates.yaml` exists with 20 records:
    - 10 existing IDs with modified text
    - 10 new IDs
- **Steps:**
  - Given a loaded ChromaDB collection and update file
  - When user runs: `rag-tester load --file updates.yaml --database chromadb://localhost:8000/test_collection --embedding sentence-transformers/all-MiniLM-L6-v2 --mode upsert --force-reembed`
  - Then command exits with code 0
  - And stdout shows: "Records updated: 10"
  - And stdout shows: "Records added: 10"
  - And stdout shows: "Failed records: 0"
  - And ChromaDB collection contains exactly 110 documents (100 original + 10 new)
  - And updated documents have new embeddings (verify by querying and checking scores changed)
  - And trace file contains spans for: upsert_operation, embedding_batch, database_update
- **Cleanup:** None (collection will be flushed in next test)
- **Priority:** High

### E2E-007: Flush Mode - Replace Entire Database
- **Category:** Core Journey
- **Scenario:** SC-007 (Flush)
- **Requirements:** FR-013, FR-014
- **Preconditions:**
  - ChromaDB collection "test_collection" contains 110 documents (from E2E-006)
  - New load file `new_data.yaml` exists with 50 records
- **Steps:**
  - Given a loaded ChromaDB collection and new data file
  - When user runs: `rag-tester load --file new_data.yaml --database chromadb://localhost:8000/test_collection --embedding sentence-transformers/all-MiniLM-L6-v2 --mode flush`
  - Then command exits with code 0
  - And stdout shows: "Records deleted: 110"
  - And stdout shows: "Records loaded: 50"
  - And stdout shows: "Failed records: 0"
  - And ChromaDB collection contains exactly 50 documents
  - And all document IDs match new_data.yaml (old IDs are gone)
  - And trace file contains spans for: flush_operation, database_delete, database_insert
- **Cleanup:** Delete ChromaDB collection "test_collection"
- **Priority:** High

---

## 2. Feature-Specific Tests

### 2.1 Load Command Tests

#### E2E-008: Load with OpenRouter Embedding Model
- **Category:** Feature-Specific
- **Scenario:** SC-001
- **Requirements:** FR-001, FR-002, FR-015
- **Preconditions:**
  - OPENROUTER_API_KEY environment variable set
  - ChromaDB server running
  - Load file with 10 records
- **Steps:**
  - Given valid OpenRouter API key
  - When user runs: `rag-tester load --file data.yaml --database chromadb://localhost:8000/test_openrouter --embedding openai/text-embedding-3-small`
  - Then command exits with code 0
  - And stdout shows: "Successfully loaded 10 records"
  - And stdout shows: "Total tokens consumed: <N>" (N > 0)
  - And ChromaDB collection contains 10 documents with dimension 1536
  - And trace file contains spans for: openrouter_api_call with attributes (model, tokens, cost)
- **Cleanup:** Delete collection
- **Priority:** Critical

#### E2E-009: Load with Google Gemini Embedding Model
- **Category:** Feature-Specific
- **Scenario:** SC-001
- **Requirements:** FR-001, FR-002, FR-016
- **Preconditions:**
  - GEMINI_API_KEY environment variable set
  - ChromaDB server running
  - Load file with 10 records
- **Steps:**
  - Given valid Gemini API key
  - When user runs: `rag-tester load --file data.yaml --database chromadb://localhost:8000/test_gemini --embedding models/text-embedding-004`
  - Then command exits with code 0
  - And stdout shows: "Successfully loaded 10 records"
  - And stdout shows: "Total tokens consumed: <N>" (N > 0)
  - And ChromaDB collection contains 10 documents with dimension 768
  - And trace file contains spans for: gemini_api_call with attributes (model, tokens)
- **Cleanup:** Delete collection
- **Priority:** Critical

#### E2E-010: Load with Parallel Workers (4 workers)
- **Category:** Feature-Specific
- **Scenario:** SC-001
- **Requirements:** FR-001, FR-017
- **Preconditions:**
  - ChromaDB server running
  - Load file with 100 records
- **Steps:**
  - Given a load file with 100 records
  - When user runs: `rag-tester load --file data.yaml --database chromadb://localhost:8000/test_parallel --embedding sentence-transformers/all-MiniLM-L6-v2 --parallel 4`
  - Then command exits with code 0
  - And stdout shows: "Successfully loaded 100 records"
  - And trace file shows concurrent embedding operations (4 workers active)
  - And total time is < 50% of sequential load time (verify by comparing with --parallel 1)
- **Cleanup:** Delete collection
- **Priority:** High

#### E2E-011: Load with Custom Batch Size
- **Category:** Feature-Specific
- **Scenario:** SC-001
- **Requirements:** FR-001, FR-018
- **Preconditions:**
  - ChromaDB server running
  - Load file with 100 records
- **Steps:**
  - Given a load file with 100 records
  - When user runs: `rag-tester load --file data.yaml --database chromadb://localhost:8000/test_batch --embedding sentence-transformers/all-MiniLM-L6-v2 --batch-size 32`
  - Then command exits with code 0
  - And stdout shows: "Successfully loaded 100 records"
  - And trace file shows embedding operations in batches of 32 (4 batches total: 32+32+32+4)
- **Cleanup:** Delete collection
- **Priority:** Medium

#### E2E-012: Load with Streaming Mode (Large File)
- **Category:** Feature-Specific
- **Scenario:** SC-001
- **Requirements:** FR-001, FR-019
- **Preconditions:**
  - ChromaDB server running
  - Load file with 10,000 records (large file)
- **Steps:**
  - Given a large load file (10K records)
  - When user runs: `rag-tester load --file large_data.yaml --database chromadb://localhost:8000/test_streaming --embedding sentence-transformers/all-MiniLM-L6-v2`
  - Then command exits with code 0
  - And stdout shows: "Successfully loaded 10000 records"
  - And memory usage stays < 500MB throughout (verify with monitoring)
  - And trace file shows streaming file read operations (not loading entire file into memory)
- **Cleanup:** Delete collection
- **Priority:** High

#### E2E-013: Load with Duplicate IDs in Input File
- **Category:** Feature-Specific
- **Scenario:** SC-001
- **Requirements:** FR-001, FR-020
- **Preconditions:**
  - ChromaDB server running
  - Load file with 10 records, where "doc5" appears twice (different text)
- **Steps:**
  - Given a load file with duplicate ID "doc5"
  - When user runs: `rag-tester load --file data_with_dupes.yaml --database chromadb://localhost:8000/test_dupes --embedding sentence-transformers/all-MiniLM-L6-v2`
  - Then command exits with code 0
  - And stdout shows: "Successfully loaded 9 records"
  - And stdout shows: "Skipped duplicates: 1 (doc5)"
  - And ChromaDB collection contains exactly 9 documents
  - And "doc5" exists only once (first occurrence kept)
- **Cleanup:** Delete collection
- **Priority:** High

#### E2E-014: Load with PostgreSQL Backend
- **Category:** Feature-Specific
- **Scenario:** SC-001
- **Requirements:** FR-001, FR-002, FR-040
- **Preconditions:**
  - PostgreSQL with pgvector extension running
  - Database "testdb" exists
  - Load file with 50 records
- **Steps:**
  - Given PostgreSQL with pgvector
  - When user runs: `rag-tester load --file data.yaml --database postgresql://user:pass@localhost:5432/testdb/embeddings_table --embedding sentence-transformers/all-MiniLM-L6-v2`
  - Then command exits with code 0
  - And stdout shows: "Successfully loaded 50 records"
  - And PostgreSQL table "embeddings_table" exists
  - And table contains 50 rows with columns: id, text, embedding (vector type)
- **Cleanup:** Drop table
- **Priority:** Critical

#### E2E-015: Load with Milvus Backend
- **Category:** Feature-Specific
- **Scenario:** SC-001
- **Requirements:** FR-001, FR-002, FR-041
- **Preconditions:**
  - Milvus server running
  - Load file with 50 records
- **Steps:**
  - Given Milvus server
  - When user runs: `rag-tester load --file data.yaml --database milvus://localhost:19530/test_collection --embedding sentence-transformers/all-MiniLM-L6-v2`
  - Then command exits with code 0
  - And stdout shows: "Successfully loaded 50 records"
  - And Milvus collection "test_collection" exists with 50 entities
- **Cleanup:** Drop collection
- **Priority:** Critical

#### E2E-016: Load with SQLite Backend
- **Category:** Feature-Specific
- **Scenario:** SC-001
- **Requirements:** FR-001, FR-002, FR-042
- **Preconditions:**
  - SQLite with vector extension available
  - Load file with 50 records
- **Steps:**
  - Given SQLite with vector extension
  - When user runs: `rag-tester load --file data.yaml --database sqlite:///tmp/test.db/embeddings_table --embedding sentence-transformers/all-MiniLM-L6-v2`
  - Then command exits with code 0
  - And stdout shows: "Successfully loaded 50 records"
  - And SQLite database file /tmp/test.db exists
  - And table "embeddings_table" contains 50 rows
- **Cleanup:** Delete database file
- **Priority:** Critical

#### E2E-017: Load with Elasticsearch Backend
- **Category:** Feature-Specific
- **Scenario:** SC-001
- **Requirements:** FR-001, FR-002, FR-043
- **Preconditions:**
  - Elasticsearch server running
  - Load file with 50 records
- **Steps:**
  - Given Elasticsearch server
  - When user runs: `rag-tester load --file data.yaml --database elasticsearch://localhost:9200/test_index --embedding sentence-transformers/all-MiniLM-L6-v2`
  - Then command exits with code 0
  - And stdout shows: "Successfully loaded 50 records"
  - And Elasticsearch index "test_index" exists with 50 documents
- **Cleanup:** Delete index
- **Priority:** Critical

### 2.2 Test Command Tests

#### E2E-018: Manual Test with JSON Output Format
- **Category:** Feature-Specific
- **Scenario:** SC-002
- **Requirements:** FR-004, FR-021
- **Preconditions:**
  - ChromaDB collection loaded with documents
- **Steps:**
  - Given a loaded collection
  - When user runs: `rag-tester test "machine learning" --database chromadb://localhost:8000/test_collection --embedding sentence-transformers/all-MiniLM-L6-v2 --format json`
  - Then command exits with code 0
  - And stdout contains valid JSON with structure:
    ```json
    {
      "query": "machine learning",
      "results": [
        {"rank": 1, "id": "doc42", "text": "...", "score": 0.85},
        {"rank": 2, "id": "doc17", "text": "...", "score": 0.78},
        {"rank": 3, "id": "doc91", "text": "...", "score": 0.72}
      ],
      "tokens_consumed": 0,
      "time_taken": 0.15
    }
    ```
- **Cleanup:** None
- **Priority:** Medium

#### E2E-019: Manual Test with Text Output Format
- **Category:** Feature-Specific
- **Scenario:** SC-002
- **Requirements:** FR-004, FR-021
- **Preconditions:**
  - ChromaDB collection loaded with documents
- **Steps:**
  - Given a loaded collection
  - When user runs: `rag-tester test "machine learning" --database chromadb://localhost:8000/test_collection --embedding sentence-transformers/all-MiniLM-L6-v2 --format text`
  - Then command exits with code 0
  - And stdout contains plain text output:
    ```
    Query: machine learning
    
    1. [doc42] (score: 0.85)
       Machine learning is a subset of artificial intelligence...
    
    2. [doc17] (score: 0.78)
       Deep learning uses neural networks...
    
    3. [doc91] (score: 0.72)
       Supervised learning requires labeled data...
    
    Tokens consumed: 0
    Time taken: 0.15s
    ```
- **Cleanup:** None
- **Priority:** Medium

#### E2E-020: Manual Test with Custom Top-K
- **Category:** Feature-Specific
- **Scenario:** SC-002
- **Requirements:** FR-004, FR-022
- **Preconditions:**
  - ChromaDB collection loaded with 100 documents
- **Steps:**
  - Given a loaded collection
  - When user runs: `rag-tester test "machine learning" --database chromadb://localhost:8000/test_collection --embedding sentence-transformers/all-MiniLM-L6-v2 --top-k 10`
  - Then command exits with code 0
  - And stdout displays exactly 10 results
  - And results are ordered by score (descending)
- **Cleanup:** None
- **Priority:** Medium

### 2.3 Bulk-Test Command Tests

#### E2E-021: Bulk Test with Progress Indicator
- **Category:** Feature-Specific
- **Scenario:** SC-003
- **Requirements:** FR-006, FR-023
- **Preconditions:**
  - ChromaDB collection loaded
  - Test file with 50 tests
- **Steps:**
  - Given a test file with 50 tests
  - When user runs: `rag-tester bulk-test --file tests.yaml --database chromadb://localhost:8000/test_collection --embedding sentence-transformers/all-MiniLM-L6-v2 --output results.yaml`
  - Then command exits with code 0
  - And stderr shows progress updates: "Progress: 10/50 (20%)", "Progress: 20/50 (40%)", etc.
  - And progress updates appear at least every 10 tests
- **Cleanup:** Delete results.yaml
- **Priority:** Medium

#### E2E-022: Bulk Test with Parallel Execution (4 workers)
- **Category:** Feature-Specific
- **Scenario:** SC-003
- **Requirements:** FR-006, FR-024
- **Preconditions:**
  - ChromaDB collection loaded
  - Test file with 100 tests
- **Steps:**
  - Given a test file with 100 tests
  - When user runs: `rag-tester bulk-test --file tests.yaml --database chromadb://localhost:8000/test_collection --embedding sentence-transformers/all-MiniLM-L6-v2 --output results.yaml --parallel 4`
  - Then command exits with code 0
  - And trace file shows concurrent test executions (4 workers active)
  - And total time is < 50% of sequential execution (verify by comparing with --parallel 1)
- **Cleanup:** Delete results.yaml
- **Priority:** High

### 2.4 Compare Command Tests

#### E2E-023: Compare with Cost Calculation (OpenRouter Models)
- **Category:** Feature-Specific
- **Scenario:** SC-004
- **Requirements:** FR-009, FR-025
- **Preconditions:**
  - Two result files from OpenRouter models with token counts
- **Steps:**
  - Given two result files with token consumption data
  - When user runs: `rag-tester compare --results results_a.yaml results_b.yaml --output comparison.yaml`
  - Then command exits with code 0
  - And comparison.yaml contains cost estimates:
    - model_a: {total_cost: 0.0024, cost_per_test: 0.00024}
    - model_b: {total_cost: 0.0156, cost_per_test: 0.00156}
  - And cost is calculated based on known pricing (e.g., text-embedding-3-small: $0.02/Mtok)
- **Cleanup:** Delete comparison.yaml
- **Priority:** Medium

#### E2E-024: Compare with Per-Test Diff Details
- **Category:** Feature-Specific
- **Scenario:** SC-004
- **Requirements:** FR-009, FR-026
- **Preconditions:**
  - Two result files with different pass/fail outcomes
- **Steps:**
  - Given two result files where test3 passed in A but failed in B
  - When user runs: `rag-tester compare --results results_a.yaml results_b.yaml --output comparison.yaml`
  - Then command exits with code 0
  - And comparison.yaml contains per_test_diff section
  - And per_test_diff includes entry for test3:
    ```yaml
    - test_id: "test3"
      model_a_status: "passed"
      model_b_status: "failed"
      model_a_score: 0.87
      model_b_score: 0.62
      expected_threshold: 0.85
    ```
- **Cleanup:** Delete comparison.yaml
- **Priority:** High

---

## 3. Side Effect Tests

### E2E-025: Load Creates Collection if Not Exists
- **Category:** Side Effect
- **Scenario:** SC-001
- **Requirements:** FR-027
- **Preconditions:**
  - ChromaDB server running
  - Collection "new_collection" does NOT exist
- **Steps:**
  - Given ChromaDB without "new_collection"
  - When user runs: `rag-tester load --file data.yaml --database chromadb://localhost:8000/new_collection --embedding sentence-transformers/all-MiniLM-L6-v2`
  - Then command exits with code 0
  - And ChromaDB collection "new_collection" is created
  - And collection has correct dimension (384 for all-MiniLM-L6-v2)
  - And trace file contains span for: collection_create
- **Cleanup:** Delete collection
- **Priority:** Critical

### E2E-026: Load Writes Trace File
- **Category:** Side Effect
- **Scenario:** SC-001
- **Requirements:** FR-028
- **Preconditions:**
  - Trace file path configured (default: traces/rag-tester.jsonl)
  - Directory traces/ exists
- **Steps:**
  - Given trace file does not exist or is empty
  - When user runs: `rag-tester load --file data.yaml --database chromadb://localhost:8000/test_collection --embedding sentence-transformers/all-MiniLM-L6-v2`
  - Then command exits with code 0
  - And file traces/rag-tester.jsonl exists
  - And file contains JSONL entries (one per line, valid JSON)
  - And entries include spans for: load_command, file_read, embedding_batch, database_insert
  - And each span has: trace_id, span_id, parent_span_id, name, start_time, end_time, attributes
- **Cleanup:** Delete trace file
- **Priority:** High

### E2E-027: Bulk Test Writes Results File
- **Category:** Side Effect
- **Scenario:** SC-003
- **Requirements:** FR-029
- **Preconditions:**
  - Output file path specified
- **Steps:**
  - Given output file does not exist
  - When user runs: `rag-tester bulk-test --file tests.yaml --database chromadb://localhost:8000/test_collection --embedding sentence-transformers/all-MiniLM-L6-v2 --output results.yaml`
  - Then command exits with code 0
  - And file results.yaml exists
  - And file is valid YAML
  - And file contains summary and test results sections
- **Cleanup:** Delete results.yaml
- **Priority:** Critical

### E2E-028: Upsert Updates Existing Documents
- **Category:** Side Effect
- **Scenario:** SC-006
- **Requirements:** FR-030
- **Preconditions:**
  - Collection contains document "doc42" with text "Old text"
  - Update file contains "doc42" with text "New text"
- **Steps:**
  - Given existing document "doc42"
  - When user runs: `rag-tester load --file updates.yaml --database chromadb://localhost:8000/test_collection --embedding sentence-transformers/all-MiniLM-L6-v2 --mode upsert --force-reembed`
  - Then command exits with code 0
  - And document "doc42" text is "New text" (verify by querying)
  - And document "doc42" embedding is different from before (verify by comparing vectors)
  - And trace file contains span for: document_update
- **Cleanup:** None
- **Priority:** High

### E2E-029: Flush Deletes All Existing Documents
- **Category:** Side Effect
- **Scenario:** SC-007
- **Requirements:** FR-031
- **Preconditions:**
  - Collection contains 100 documents
- **Steps:**
  - Given collection with 100 documents
  - When user runs: `rag-tester load --file new_data.yaml --database chromadb://localhost:8000/test_collection --embedding sentence-transformers/all-MiniLM-L6-v2 --mode flush`
  - Then command exits with code 0
  - And collection contains only documents from new_data.yaml
  - And old document IDs are no longer present (verify by querying)
  - And trace file contains span for: collection_flush
- **Cleanup:** None
- **Priority:** High

### E2E-030: Failed Records Are Logged
- **Category:** Side Effect
- **Scenario:** SC-001
- **Requirements:** FR-032
- **Preconditions:**
  - Load file contains 10 records
  - Embedding API will fail for record "doc5" (simulate by mocking)
- **Steps:**
  - Given a load file where one record will fail
  - When user runs: `rag-tester load --file data.yaml --database chromadb://localhost:8000/test_collection --embedding sentence-transformers/all-MiniLM-L6-v2`
  - Then command exits with code 0
  - And stdout shows: "Successfully loaded 9 records"
  - And stdout shows: "Failed records: 1"
  - And stdout lists failed record ID: "doc5"
  - And trace file contains error span for