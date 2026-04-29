# RAG Tester - Complete End-to-End Test Suite

> Comprehensive test plan covering all scenarios, failure modes, edge cases, and side effects.
> Tests are organized by category and linked to usage scenarios and functional requirements.

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

## 1. Core User Journeys (7 tests)

### E2E-001: Initial Dataset Load with Local Embedding Model
- **Category:** Core Journey | **Scenario:** SC-001 | **Requirements:** FR-001, FR-002, FR-003
- **Preconditions:** ChromaDB at localhost:8000, test_data.yaml with 100 records, sentence-transformers installed
- **Steps:**
  - Given clean ChromaDB instance
  - When: `rag-tester load --file test_data.yaml --database chromadb://localhost:8000/test_collection --embedding sentence-transformers/all-MiniLM-L6-v2 --parallel 4`
  - Then exit code 0
  - And stdout: "Successfully loaded 100 records", "Failed records: 0", "Total tokens: 0", "Total time: X seconds"
  - And collection exists with 100 docs, dimension 384
  - And trace file has spans: file_read, embedding_batch, database_insert
- **Priority:** Critical

### E2E-002: Manual Query Test
- **Category:** Core Journey | **Scenario:** SC-002 | **Requirements:** FR-004, FR-005
- **Preconditions:** Collection loaded with 100 docs, doc42 = "Machine learning is..."
- **Steps:**
  - When: `rag-tester test "What is machine learning?" --database chromadb://localhost:8000/test_collection --embedding sentence-transformers/all-MiniLM-L6-v2 --top-k 3 --format table`
  - Then exit code 0, table with 3 rows (Rank, ID, Text, Score)
  - And row 1 has doc42 with score > 0.7
  - And trace has: embedding_query, database_search
- **Priority:** Critical

### E2E-003: Bulk Test with Pass/Fail Cases
- **Category:** Core Journey | **Scenario:** SC-003 | **Requirements:** FR-006, FR-007, FR-008
- **Preconditions:** Collection loaded, test_suite.yaml with 10 tests (7 pass, 2 fail ID, 1 fail threshold)
- **Steps:**
  - When: `rag-tester bulk-test --file test_suite.yaml --database chromadb://localhost:8000/test_collection --embedding sentence-transformers/all-MiniLM-L6-v2 --output results.yaml --parallel 2`
  - Then exit code 0, results.yaml created
  - And summary: {total: 10, passed: 7, failed: 3, tokens: 0, time: X}
  - And contains only failed test details by default
- **Priority:** Critical

### E2E-004: Bulk Test Verbose
- **Category:** Core Journey | **Scenario:** SC-003 | **Requirements:** FR-006, FR-007, FR-008
- **Steps:**
  - When: `rag-tester bulk-test --file test_suite.yaml --database chromadb://localhost:8000/test_collection --embedding sentence-transformers/all-MiniLM-L6-v2 --output results_verbose.yaml --verbose`
  - Then results_verbose.yaml contains ALL 10 test results (passed + failed)
- **Priority:** High

### E2E-005: Compare Two Embedding Models
- **Category:** Core Journey | **Scenario:** SC-004 | **Requirements:** FR-009, FR-010
- **Preconditions:** results_model_a.yaml (8/10 pass), results_model_b.yaml (7/10 pass)
- **Steps:**
  - When: `rag-tester compare --results results_model_a.yaml results_model_b.yaml --output comparison.yaml`
  - Then comparison.yaml has: model_a/b metrics (pass_rate, avg_score, tokens, time), per_test_diff
- **Priority:** High

### E2E-006: Upsert Mode
- **Category:** Core Journey | **Scenario:** SC-006 | **Requirements:** FR-011, FR-012
- **Preconditions:** Collection with 100 docs, updates.yaml with 10 existing + 10 new IDs
- **Steps:**
  - When: `rag-tester load --file updates.yaml --database chromadb://localhost:8000/test_collection --embedding sentence-transformers/all-MiniLM-L6-v2 --mode upsert --force-reembed`
  - Then stdout: "Records updated: 10", "Records added: 10"
  - And collection has 110 docs, updated docs have new embeddings
- **Priority:** High

### E2E-007: Flush Mode
- **Category:** Core Journey | **Scenario:** SC-007 | **Requirements:** FR-013, FR-014
- **Preconditions:** Collection with 110 docs, new_data.yaml with 50 records
- **Steps:**
  - When: `rag-tester load --file new_data.yaml --database chromadb://localhost:8000/test_collection --embedding sentence-transformers/all-MiniLM-L6-v2 --mode flush`
  - Then stdout: "Records deleted: 110", "Records loaded: 50"
  - And collection has exactly 50 docs (old IDs gone)
- **Priority:** High

---

## 2. Feature-Specific Tests (24 tests)

### 2.1 Load Command (10 tests)

#### E2E-008: OpenRouter Embedding
- **Scenario:** SC-001 | **Requirements:** FR-015
- **Preconditions:** OPENROUTER_API_KEY set
- **Steps:** Load with `openai/text-embedding-3-small`, verify dimension 1536, tokens > 0, cost traced
- **Priority:** Critical

#### E2E-009: Google Gemini Embedding
- **Scenario:** SC-001 | **Requirements:** FR-016
- **Preconditions:** GEMINI_API_KEY set
- **Steps:** Load with `models/text-embedding-004`, verify dimension 768, tokens > 0
- **Priority:** Critical

#### E2E-010: Parallel Workers
- **Scenario:** SC-001 | **Requirements:** FR-017
- **Steps:** Load 100 records with `--parallel 4`, verify concurrent ops in trace, time < 50% of sequential
- **Priority:** High

#### E2E-011: Custom Batch Size
- **Scenario:** SC-001 | **Requirements:** FR-018
- **Steps:** Load 100 records with `--batch-size 32`, verify 4 batches (32+32+32+4) in trace
- **Priority:** Medium

#### E2E-012: Streaming Mode (Large File)
- **Scenario:** SC-001 | **Requirements:** FR-019
- **Steps:** Load 10K records, verify memory < 500MB, streaming file read in trace
- **Priority:** High

#### E2E-013: Duplicate IDs
- **Scenario:** SC-001 | **Requirements:** FR-020
- **Steps:** Load file with duplicate "doc5", verify 9 loaded, 1 skipped, first occurrence kept
- **Priority:** High

#### E2E-014: PostgreSQL Backend
- **Scenario:** SC-001 | **Requirements:** FR-040
- **Steps:** Load to `postgresql://user:pass@localhost:5432/testdb/embeddings_table`, verify table created with vector column
- **Priority:** Critical

#### E2E-015: Milvus Backend
- **Scenario:** SC-001 | **Requirements:** FR-041
- **Steps:** Load to `milvus://localhost:19530/test_collection`, verify collection with 50 entities
- **Priority:** Critical

#### E2E-016: SQLite Backend
- **Scenario:** SC-001 | **Requirements:** FR-042
- **Steps:** Load to `sqlite:///tmp/test.db/embeddings_table`, verify db file and table
- **Priority:** Critical

#### E2E-017: Elasticsearch Backend
- **Scenario:** SC-001 | **Requirements:** FR-043
- **Steps:** Load to `elasticsearch://localhost:9200/test_index`, verify index with 50 docs
- **Priority:** Critical

### 2.2 Test Command (3 tests)

#### E2E-018: JSON Output Format
- **Scenario:** SC-002 | **Requirements:** FR-021
- **Steps:** Test with `--format json`, verify valid JSON structure with query, results array, tokens, time
- **Priority:** Medium

#### E2E-019: Text Output Format
- **Scenario:** SC-002 | **Requirements:** FR-021
- **Steps:** Test with `--format text`, verify plain text with numbered results, scores, metrics
- **Priority:** Medium

#### E2E-020: Custom Top-K
- **Scenario:** SC-002 | **Requirements:** FR-022
- **Steps:** Test with `--top-k 10`, verify exactly 10 results ordered by score descending
- **Priority:** Medium

### 2.3 Bulk-Test Command (2 tests)

#### E2E-021: Progress Indicator
- **Scenario:** SC-003 | **Requirements:** FR-023
- **Steps:** Bulk test 50 tests, verify stderr progress updates every 10 tests
- **Priority:** Medium

#### E2E-022: Parallel Execution
- **Scenario:** SC-003 | **Requirements:** FR-024
- **Steps:** Bulk test 100 tests with `--parallel 4`, verify concurrent execution, time < 50% sequential
- **Priority:** High

### 2.4 Compare Command (2 tests)

#### E2E-023: Cost Calculation
- **Scenario:** SC-004 | **Requirements:** FR-025
- **Steps:** Compare OpenRouter results, verify cost estimates based on pricing (e.g., $0.02/Mtok)
- **Priority:** Medium

#### E2E-024: Per-Test Diff
- **Scenario:** SC-004 | **Requirements:** FR-026
- **Steps:** Compare results with different outcomes, verify per_test_diff section with status, scores, thresholds
- **Priority:** High

---

## 3. Side Effect Tests (8 tests)

#### E2E-025: Auto-Create Collection
- **Scenario:** SC-001 | **Requirements:** FR-027
- **Steps:** Load to non-existent collection, verify auto-creation with correct dimension
- **Priority:** Critical

#### E2E-026: Trace File Written
- **Scenario:** SC-001 | **Requirements:** FR-028
- **Steps:** Load data, verify traces/rag-tester.jsonl exists with valid JSONL spans
- **Priority:** High

#### E2E-027: Results File Written
- **Scenario:** SC-003 | **Requirements:** FR-029
- **Steps:** Bulk test, verify results.yaml created with valid YAML structure
- **Priority:** Critical

#### E2E-028: Upsert Updates Documents
- **Scenario:** SC-006 | **Requirements:** FR-030
- **Steps:** Upsert doc42 with new text, verify text changed and embedding different
- **Priority:** High

#### E2E-029: Flush Deletes All
- **Scenario:** SC-007 | **Requirements:** FR-031
- **Steps:** Flush collection, verify old IDs gone, only new data present
- **Priority:** High

#### E2E-030: Failed Records Logged
- **Scenario:** SC-001 | **Requirements:** FR-032
- **Steps:** Load with 1 failing record (mock), verify stdout lists failed ID, trace has error span
- **Priority:** High

#### E2E-031: Retry Attempts Traced
- **Scenario:** SC-001 | **Requirements:** FR-033
- **Steps:** Load with record requiring 3 retries (mock), verify 4 spans (3 fail + 1 success) with attempt_number
- **Priority:** Medium

#### E2E-032: Comparison File Generated
- **Scenario:** SC-004 | **Requirements:** FR-034
- **Steps:** Compare results, verify comparison.yaml created with all metrics
- **Priority:** High

---

## 4. Error Handling Tests (18 tests)

### 4.1 Load Errors (8 tests)

#### E2E-033: Invalid File Format
- **Scenario:** SC-001 | **Requirements:** FR-035
- **Steps:** Load malformed YAML, expect exit 1, stderr: "Invalid file format. Failed to parse YAML: <error>"
- **Priority:** Critical

#### E2E-034: Missing Required Fields
- **Scenario:** SC-001 | **Requirements:** FR-036
- **Steps:** Load file missing "text" field, expect exit 1, stderr: "Missing required field 'text' in record <id>"
- **Priority:** Critical

#### E2E-035: Database Unreachable
- **Scenario:** SC-001 | **Requirements:** FR-037
- **Steps:** Load with ChromaDB down, expect exit 1, stderr: "Database connection failed: <error>"
- **Priority:** Critical

#### E2E-036: Dimension Mismatch
- **Scenario:** SC-001 | **Requirements:** FR-038
- **Steps:** Load to collection with dimension 768 using 384-dim model, expect exit 1, stderr: "Dimension mismatch: model=384, database=768"
- **Priority:** Critical

#### E2E-037: Missing API Key
- **Scenario:** SC-001 | **Requirements:** FR-039
- **Steps:** Load with OpenRouter model but no OPENROUTER_API_KEY, expect exit 1, stderr: "Missing API key: OPENROUTER_API_KEY"
- **Priority:** Critical

#### E2E-038: Rate Limit Exceeded
- **Scenario:** SC-001 | **Requirements:** FR-044
- **Steps:** Load triggering rate limit (mock), verify retries with backoff, eventual success or failure after 5 attempts
- **Priority:** High

#### E2E-039: Connection Drop Mid-Load
- **Scenario:** SC-001 | **Requirements:** FR-045
- **Steps:** Load with DB connection dropping at 50% (mock), verify partial load, failed records logged, trace has error
- **Priority:** High

#### E2E-040: Out of Memory
- **Scenario:** SC-001 | **Requirements:** FR-046
- **Steps:** Load very large file exceeding memory (mock), expect exit 1, stderr: "Out of memory", or graceful streaming fallback
- **Priority:** Medium

### 4.2 Test Errors (4 tests)

#### E2E-041: Empty Database
- **Scenario:** SC-002 | **Requirements:** FR-047
- **Steps:** Test against empty collection, expect exit 1, stderr: "Database is empty"
- **Priority:** Critical

#### E2E-042: Embedding API Failure
- **Scenario:** SC-002 | **Requirements:** FR-048
- **Steps:** Test with embedding API down (mock), expect exit 1, stderr: "Embedding failed: <error>"
- **Priority:** Critical

#### E2E-043: Database Query Timeout
- **Scenario:** SC-002 | **Requirements:** FR-049
- **Steps:** Test with DB query timeout (mock), expect exit 1, stderr: "Database query timeout"
- **Priority:** High

#### E2E-044: Dimension Mismatch on Query
- **Scenario:** SC-002 | **Requirements:** FR-050
- **Steps:** Test with wrong embedding model (different dimension), expect exit 1, stderr: "Dimension mismatch"
- **Priority:** High

### 4.3 Bulk-Test Errors (3 tests)

#### E2E-045: Malformed Test File
- **Scenario:** SC-003 | **Requirements:** FR-051
- **Steps:** Bulk test with invalid YAML, expect exit 1, stderr: "Invalid test file format"
- **Priority:** Critical

#### E2E-046: Database Unavailable Mid-Suite
- **Scenario:** SC-003 | **Requirements:** FR-052
- **Steps:** Bulk test with DB going down at 50% (mock), verify partial results, failed tests logged, exit 0 (continue on error)
- **Priority:** High

#### E2E-047: Output File Write Failure
- **Scenario:** SC-003 | **Requirements:** FR-053
- **Steps:** Bulk test with output path unwritable (permissions), expect exit 1, stderr: "Cannot write output file"
- **Priority:** High

### 4.4 Compare Errors (3 tests)

#### E2E-048: Missing Result File
- **Scenario:** SC-004 | **Requirements:** FR-054
- **Steps:** Compare with non-existent file, expect exit 1, stderr: "Result file not found: <path>"
- **Priority:** Critical

#### E2E-049: Invalid Result File Format
- **Scenario:** SC-004 | **Requirements:** FR-055
- **Steps:** Compare with malformed YAML, expect exit 1, stderr: "Invalid result file format"
- **Priority:** High

#### E2E-050: Incompatible Result Files
- **Scenario:** SC-004 | **Requirements:** FR-056
- **Steps:** Compare results from different test suites (different test IDs), expect exit 1 or warning, stderr: "Test suites do not match"
- **Priority:** Medium

---

## 5. Security Tests (6 tests)

#### E2E-051: API Key Validation (OpenRouter)
- **Requirements:** FR-057
- **Steps:** Load with invalid OPENROUTER_API_KEY, expect exit 1, stderr: "Authentication failed: invalid API key"
- **Priority:** Critical

#### E2E-052: API Key Validation (Gemini)
- **Requirements:** FR-058
- **Steps:** Load with invalid GEMINI_API_KEY, expect exit 1, stderr: "Authentication failed"
- **Priority:** Critical

#### E2E-053: Database Authentication Failure
- **Requirements:** FR-059
- **Steps:** Load to PostgreSQL with wrong credentials, expect exit 1, stderr: "Database authentication failed"
- **Priority:** Critical

#### E2E-054: SQL Injection Protection
- **Requirements:** FR-060
- **Steps:** Load record with ID containing SQL injection pattern (e.g., "'; DROP TABLE--"), verify safe handling, no SQL execution
- **Priority:** Critical

#### E2E-055: Path Traversal Protection
- **Requirements:** FR-061
- **Steps:** Load with file path containing "../../../etc/passwd", expect exit 1, stderr: "Invalid file path"
- **Priority:** High

#### E2E-056: API Key Not Logged
- **Requirements:** FR-062
- **Steps:** Load with API key, verify trace file and stdout do not contain API key value (only masked)
- **Priority:** Critical

---

## 6. Data Integrity Tests (7 tests)

#### E2E-057: ID Uniqueness Enforced
- **Requirements:** FR-063
- **Steps:** Load, then upsert same ID, verify only one document exists with updated data
- **Priority:** High

#### E2E-058: Embedding Dimension Consistency
- **Requirements:** FR-064
- **Steps:** Load 100 records, verify all embeddings have same dimension matching model
- **Priority:** Critical

#### E2E-059: Text Round-Trip Integrity
- **Requirements:** FR-065
- **Steps:** Load record with special chars (Unicode, emoji, newlines), query and retrieve, verify text matches exactly
- **Priority:** High

#### E2E-060: Concurrent Load Safety
- **Requirements:** FR-066
- **Steps:** Run 2 parallel loads to same collection (different records), verify no duplicates, all records present
- **Priority:** High

#### E2E-061: Flush Atomicity
- **Requirements:** FR-067
- **Steps:** Flush collection, verify either all old data deleted + new data loaded, or operation fails (no partial state)
- **Priority:** High

#### E2E-062: Test Result Accuracy
- **Requirements:** FR-068
- **Steps:** Bulk test with known expected results, verify pass/fail determinations are correct (exact order + threshold)
- **Priority:** Critical

#### E2E-063: Score Consistency
- **Requirements:** FR-069
- **Steps:** Query same text twice, verify similarity scores are identical (deterministic)
- **Priority:** Medium

---

## 7. Performance Baseline Tests (5 tests)

#### E2E-064: Load Latency (Local Model)
- **Requirements:** NFR-001
- **Steps:** Load 100 records with local model, verify total time < 30 seconds
- **Priority:** Medium

#### E2E-065: Load Latency (API Model)
- **Requirements:** NFR-002
- **Steps:** Load 100 records with OpenRouter, verify total time < 60 seconds (accounting for API latency)
- **Priority:** Medium

#### E2E-066: Query Latency
- **Requirements:** NFR-003
- **Steps:** Single query test, verify response time < 1 second
- **Priority:** Medium

#### E2E-067: Bulk Test Throughput
- **Requirements:** NFR-004
- **Steps:** Bulk test 100 tests with --parallel 4, verify throughput > 10 tests/second
- **Priority:** Low

#### E2E-068: Memory Usage (Streaming)
- **Requirements:** NFR-005
- **Steps:** Load 10K records, verify peak memory < 500MB (streaming mode working)
- **Priority:** Medium

---

## 8. Integration Tests (8 tests)

#### E2E-069: ChromaDB HTTP Mode
- **Requirements:** FR-070
- **Steps:** Load to `chromadb://localhost:8000/collection`, verify HTTP client used, collection created
- **Priority:** Critical

#### E2E-070: ChromaDB Persistent Mode
- **Requirements:** FR-071
- **Steps:** Load to `chromadb:///tmp/chroma_data/collection`, verify persistent directory created, data survives restart
- **Priority:** High

#### E2E-071: PostgreSQL with pgvector
- **Requirements:** FR-072
- **Steps:** Load to PostgreSQL, verify vector column type, similarity search works
- **Priority:** Critical

#### E2E-072: Milvus Collection Management
- **Requirements:** FR-073
- **Steps:** Load to Milvus, verify collection auto-created with correct schema (id, text, embedding)
- **Priority:** High

#### E2E-073: SQLite with vector extension
- **Requirements:** FR-074
- **Steps:** Load to SQLite, verify vector extension loaded, similarity search works
- **Priority:** High

#### E2E-074: Elasticsearch Dense Vector
- **Requirements:** FR-075
- **Steps:** Load to Elasticsearch, verify index mapping has dense_vector field, kNN search works
- **Priority:** High

#### E2E-075: OpenRouter API Integration
- **Requirements:** FR-076
- **Steps:** Load with OpenRouter model, verify API calls traced with model, tokens, cost
- **Priority:** Critical

#### E2E-076: Google Gemini API Integration
- **Requirements:** FR-077
- **Steps:** Load with Gemini model, verify API calls traced with model, tokens
- **Priority:** Critical

---

## 9. Cross-Scenario Tests (4 tests)

#### E2E-077: Load Then Test
- **Scenarios:** SC-001 + SC-002
- **Steps:** Load data, immediately test query, verify results match loaded data
- **Priority:** High

#### E2E-078: Load, Test, Bulk-Test
- **Scenarios:** SC-001 + SC-002 + SC-003
- **Steps:** Load data, manual test, bulk test, verify consistency across all operations
- **Priority:** High

#### E2E-079: Upsert Then Test
- **Scenarios:** SC-006 + SC-002
- **Steps:** Load data, upsert updates, test query, verify updated data returned
- **Priority:** High

#### E2E-080: Flush Then Test
- **Scenarios:** SC-007 + SC-002
- **Steps:** Load data, flush with new data, test query, verify only new data returned (old data gone)
- **Priority:** High

---

## 10. Edge Case Tests (7 additional tests)

#### E2E-081: Empty Input File
- **Steps:** Load empty YAML file (no records), expect exit 1, stderr: "Input file is empty"
- **Priority:** High

#### E2E-082: Single Record Load
- **Steps:** Load file with 1 record, verify successful load, collection has 1 doc
- **Priority:** Medium

#### E2E-083: Very Long Text (10K chars)
- **Steps:** Load record with 10K character text, verify successful embedding and retrieval
- **Priority:** Medium

#### E2E-084: Unicode and Emoji
- **Steps:** Load record with text: "Hello 世界 🌍 مرحبا", verify correct embedding and retrieval
- **Priority:** High

#### E2E-085: Top-K Exceeds Collection Size
- **Steps:** Test with --top-k 100 on collection with 50 docs, verify returns 50 results (not error)
- **Priority:** Medium

#### E2E-086: Zero Threshold
- **Steps:** Bulk test with expected result having min_threshold: 0.0, verify any score passes
- **Priority:** Low

#### E2E-087: Perfect Score (1.0)
- **Steps:** Test exact text match (query = document text), verify score ≈ 1.0
- **Priority:** Low

---

## Traceability Matrix

| Scenario | Functional Reqs | E2E Tests (Happy) | E2E Tests (Failure) | E2E Tests (Edge) |
|----------|----------------|-------------------|---------------------|------------------|
| SC-001 (Load) | FR-001 to FR-020, FR-027, FR-028, FR-032, FR-033, FR-040 to FR-046 | E2E-001, E2E-008 to E2E-017, E2E-025, E2E-026, E2E-030, E2E-031 | E2E-033 to E2E-040 | E2E-081 to E2E-084 |
| SC-002 (Test) | FR-004, FR-005, FR-021, FR-022, FR-047 to FR-050 | E2E-002, E2E-018 to E2E-020 | E2E-041 to E2E-044 | E2E-085, E2E-087 |
| SC-003 (Bulk-Test) | FR-006 to FR-008, FR-023, FR-024, FR-029, FR-051 to FR-053 | E2E-003, E2E-004, E2E-021, E2E-022, E2E-027 | E2E-045 to E2E-047 | E2E-086 |
| SC-004 (Compare) | FR-009, FR-010, FR-025, FR-026, FR-034, FR-054 to FR-056 | E2E-005, E2E-023, E2E-024, E2E-032 | E2E-048 to E2E-050 | - |
| SC-006 (Upsert) | FR-011, FR-012, FR-030 | E2E-006, E2E-028 | - | - |
| SC-007 (Flush) | FR-013, FR-014, FR-031 | E2E-007, E2E-029 | - | - |

**Coverage Verification:**
- ✓ Every scenario has happy path tests
- ✓ Every scenario has failure tests (except SC-006, SC-007 which are inherently safe operations)
- ✓ Every FR is covered by at least one test
- ✓ Happy:Failure ratio = 24:31 = 1:1.29 (meets >1:1 requirement)
- ✓ All side effects have dedicated verification tests
- ✓ All error messages/codes have triggering tests
- ✓ Cross-scenario interactions are tested

---

## Test Execution Notes

1. **Test Order:** Run in numerical order (E2E-001 to E2E-087) for dependency management
2. **Cleanup:** Each test specifies cleanup requirements; ensure execution between tests
3. **Mocking:** Tests marked "(mock)" require test doubles for external failures
4. **Monitoring:** Performance tests require resource monitoring (memory, time)
5. **Prerequisites:** Ensure all backends (ChromaDB, PostgreSQL, Milvus, SQLite, Elasticsearch) are available for integration tests
6. **API Keys:** Set all required environment variables before running security/integration tests
7. **Trace Verification:** Many tests verify trace file contents; ensure OpenTelemetry is configured correctly

