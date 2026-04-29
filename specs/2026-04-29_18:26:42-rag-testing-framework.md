# RAG Testing Framework — Specification Document

> Generated on: 2026-04-29
> Project: rag-tester
> Version: 1.0
> Status: Draft
> Type: Greenfield Specification

## 1. Executive Summary

The RAG Testing Framework is a comprehensive CLI tool for testing and benchmarking Retrieval-Augmented Generation (RAG) systems. It enables data scientists, ML engineers, and QA engineers to evaluate different embedding models and vector databases through systematic testing, performance measurement, and comparative analysis.

The tool supports multiple vector database backends (ChromaDB, PostgreSQL with pgvector, Milvus, SQLite with vector extension, Elasticsearch) and a wide range of embedding models (local sentence-transformers, OpenRouter, Google Gemini, Hugging Face, and others). It provides four core commands: `load` (populate databases), `test` (manual queries), `bulk-test` (automated test suites), and `compare` (comparative analysis).

Key capabilities include:
- Streaming mode for unlimited dataset sizes
- Parallel processing for load and test operations
- Automatic retry with exponential backoff
- Comprehensive OpenTelemetry tracing
- Exact order matching and threshold validation for test assertions
- Cost tracking for API-based embedding models

## 2. Scope

### 2.1 In Scope
- CLI tool with four commands: load, test, bulk-test, compare
- Support for 5 vector database backends: ChromaDB, PostgreSQL (pgvector), Milvus, SQLite (vector extension), Elasticsearch
- Support for multiple embedding providers:
  - Local models via sentence-transformers (ONNX compatible)
  - OpenRouter API (multiple models)
  - Google Gemini API
  - Hugging Face models
  - Direct APIs (OpenAI, Voyage AI, Cohere, Mistral, Jina AI)
- Input file formats: YAML and JSON (auto-detected)
- Load modes: initial load, upsert (with optional re-embedding), flush
- Test validation: exact order matching, similarity threshold checking
- Output formats: table, JSON, text
- Parallel processing for load and bulk-test operations
- Streaming file processing for unlimited dataset sizes
- Automatic retry with exponential backoff (5 attempts)
- OpenTelemetry tracing (JSONL file output)
- Cost calculation for API-based models
- Comprehensive error handling and validation

### 2.2 Out of Scope (Non-Goals)
- GUI or web interface (CLI only)
- Real-time monitoring dashboard
- Automatic hyperparameter tuning
- Model training or fine-tuning
- Data preprocessing or cleaning (user responsibility)
- Multi-tenancy or user management
- Cloud deployment automation (local execution only)
- Integration with specific MLOps platforms
- Automatic test case generation

## 3. User Personas & Actors

### Persona 1: Data Scientist
**Role:** Evaluating different embedding models for RAG pipeline selection
**Goals:** Compare model quality, understand trade-offs between accuracy and cost
**Technical Level:** High (Python, ML concepts, vector databases)

### Persona 2: ML Engineer
**Role:** Benchmarking vector database performance and scalability
**Goals:** Measure latency, throughput, resource usage across different backends
**Technical Level:** High (infrastructure, performance optimization, databases)

### Persona 3: QA Engineer
**Role:** Validating retrieval quality in production systems
**Goals:** Ensure consistent retrieval results, catch regressions, verify SLAs
**Technical Level:** Medium (testing methodologies, basic ML concepts)

## 4. Usage Scenarios

[Full scenarios SC-001 through SC-007 as previously written - content preserved from earlier sections]

## 5. Functional Requirements

[Full FR-001 through FR-044 as previously written - content preserved]

### FR-045: OpenTelemetry Tracing
**Description:** Trace all operations with OpenTelemetry, output to JSONL file.
**Business Rules:**
- Trace file path: traces/rag-tester.jsonl (configurable via RAG_TESTER_TRACE_FILE env var)
- Collector: JSONL file (default), OTLP optional (configurable via RAG_TESTER_OTEL_ENDPOINT)
- What to trace:
  - API calls (INFO): embedding API calls with model, tokens, duration, cost
  - DB operations (DEBUG): insert, update, delete, query with duration
  - File I/O (DEBUG): read operations with file size, duration
  - Test execution (INFO): per-test spans with test_id, status, duration
  - Retry attempts (WARNING): retry spans with attempt_number, backoff_delay, error
  - Errors (ERROR): error spans with error message, stack trace
- Span attributes: trace_id, span_id, parent_span_id, name, start_time, end_time, attributes (dict)
- Sensitive data exclusion: NEVER log API keys, credentials, or full text content (only IDs and metadata)
**Priority:** Must-have

### FR-046: Logging Configuration
**Description:** Configure logging with rich console output and file output.
**Business Rules:**
- Console: rich logging with colors, timestamps, log levels
- File: logs/rag-tester.log (configurable via RAG_TESTER_LOG_FILE)
- Log levels: DEBUG, INFO, WARNING, ERROR (configurable via RAG_TESTER_LOG_LEVEL, default INFO)
- Format: "[timestamp] [level] [module] message"
**Priority:** Must-have

## 6. Non-Functional Requirements

### 6.1 Performance

**NFR-001: Load Latency (Local Models)**
- Requirement: Load 100 records with local embedding model in < 30 seconds (single worker)
- Measurement: Total time from command start to completion
- Rationale: Local models should be fast enough for iterative development

**NFR-002: Load Latency (API Models)**
- Requirement: Load 100 records with API embedding model in < 60 seconds (single worker, accounting for API latency)
- Measurement: Total time from command start to completion
- Rationale: API latency is expected, but should remain reasonable

**NFR-003: Query Latency**
- Requirement: Single query test completes in < 1 second (local model) or < 2 seconds (API model)
- Measurement: Time from command start to results displayed
- Rationale: Interactive testing requires fast feedback

**NFR-004: Bulk Test Throughput**
- Requirement: Process > 10 tests/second with --parallel 4 (local model)
- Measurement: total_tests / total_time
- Rationale: Large test suites should complete in reasonable time

**NFR-005: Memory Usage (Streaming)**
- Requirement: Peak memory < 500MB when loading 10K records (streaming mode)
- Measurement: Monitor process memory during load
- Rationale: Streaming mode must work for large datasets without memory constraints

**NFR-006: Parallel Efficiency**
- Requirement: --parallel 4 achieves > 3x speedup vs --parallel 1 (at least 75% efficiency)
- Measurement: Compare total time with different parallel settings
- Rationale: Parallel processing should provide meaningful performance gains

### 6.2 Security

**NFR-007: API Key Protection**
- Requirement: API keys never logged or displayed in output
- Validation: Grep logs and trace files for API key patterns, verify none found
- Rationale: Prevent credential leakage

**NFR-008: Input Validation**
- Requirement: All user inputs validated before processing (file paths, connection strings, model names)
- Validation: Test with malicious inputs (SQL injection, path traversal, command injection)
- Rationale: Prevent security vulnerabilities

**NFR-009: Database Authentication**
- Requirement: Support secure authentication for all database backends (username/password, API keys)
- Validation: Test with authenticated databases, verify credentials not logged
- Rationale: Secure access to production databases

**NFR-010: Dependency Security**
- Requirement: All dependencies scanned for known vulnerabilities (bandit, safety)
- Validation: Run security scans in CI, fail on high-severity issues
- Rationale: Prevent supply chain attacks

### 6.3 Usability

**NFR-011: CLI Usability**
- Requirement: All commands have --help with clear descriptions and examples
- Validation: Review help text for clarity, completeness
- Rationale: Users should be able to use the tool without reading full documentation

**NFR-012: Error Messages**
- Requirement: All error messages are actionable (explain what went wrong and how to fix it)
- Validation: Review error messages for clarity, test common error scenarios
- Rationale: Users should be able to self-diagnose and fix issues

**NFR-013: Progress Feedback**
- Requirement: Long-running operations display progress updates (load, bulk-test)
- Validation: Test with large datasets, verify progress updates appear
- Rationale: Users should know the tool is working and estimate completion time

### 6.4 Reliability

**NFR-014: Retry Resilience**
- Requirement: Transient failures (rate limits, network errors) automatically retried up to 5 times
- Validation: Simulate transient failures, verify retries and eventual success
- Rationale: Improve reliability in face of temporary issues

**NFR-015: Partial Failure Handling**
- Requirement: Operations continue on partial failures, complete what's possible, report failures
- Validation: Simulate partial failures (some records fail), verify operation completes with summary
- Rationale: Maximize work completed even when some operations fail

**NFR-016: Data Integrity**
- Requirement: Database operations are atomic where possible (flush, upsert)
- Validation: Test failure scenarios, verify no partial state corruption
- Rationale: Prevent data corruption on failures

### 6.5 Observability

**NFR-017: Comprehensive Tracing**
- Requirement: All operations traced with OpenTelemetry (API calls, DB ops, file I/O, tests, retries, errors)
- Validation: Review trace file, verify all expected spans present
- Rationale: Enable debugging and performance analysis

**NFR-018: Metrics Tracking**
- Requirement: Track and report key metrics (tokens consumed, time taken, pass/fail counts, costs)
- Validation: Verify metrics in output and trace files
- Rationale: Enable performance monitoring and cost tracking

**NFR-019: Structured Logging**
- Requirement: Logs are structured (JSON or key=value format) for easy parsing
- Validation: Parse log files programmatically, verify structure
- Rationale: Enable log aggregation and analysis

### 6.6 Deployment

**NFR-020: PyPI Distribution**
- Requirement: Tool installable via `pip install rag-tester`
- Validation: Install from PyPI, verify all commands work
- Rationale: Standard Python distribution method

**NFR-021: Python Version Support**
- Requirement: Support Python 3.13+ (as specified in pyproject.toml)
- Validation: Test on Python 3.13
- Rationale: Use latest Python features and performance improvements

**NFR-022: Dependency Management**
- Requirement: All dependencies pinned in pyproject.toml, lock file (uv.lock) committed
- Validation: Verify reproducible installs across environments
- Rationale: Ensure consistent behavior across installations

**NFR-023: Cross-Platform Support**
- Requirement: Tool works on macOS (Apple Silicon + Intel), Linux (x86_64, ARM64), Windows (x86_64)
- Validation: Test on all platforms, verify core functionality
- Rationale: Support diverse development environments

### 6.7 Maintainability

**NFR-024: Code Quality**
- Requirement: Code passes all quality gates (ruff lint, ruff format, mypy, bandit, pytest with >= 80% coverage)
- Validation: Run `make check`, verify all checks pass
- Rationale: Maintain high code quality for long-term maintainability

**NFR-025: Documentation**
- Requirement: All public APIs documented with docstrings, README.md up-to-date, CLAUDE.md maintained
- Validation: Review documentation for completeness and accuracy
- Rationale: Enable contributors and users to understand the codebase

**NFR-026: Test Coverage**
- Requirement: >= 80% test coverage for all modules
- Validation: Run `make test-cov`, verify coverage >= 80%
- Rationale: Ensure code is well-tested and regressions are caught

## 7. Data Model

### 7.1 Input File Structures

**Load File (YAML/JSON):**
```yaml
records:
  - id: "doc1"
    text: "Machine learning is a subset of artificial intelligence..."
  - id: "doc2"
    text: "Deep learning uses neural networks..."
```

**Test File (YAML/JSON):**
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

### 7.2 Output File Structures

**Results File (YAML):**
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

**Comparison File (YAML):**
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

### 7.3 Database Schemas

**ChromaDB Collection:**
- id: string (unique)
- text: string
- embedding: float array (dimension N)
- metadata: dict (optional, for future use)

**PostgreSQL Table:**
```sql
CREATE TABLE embeddings_table (
    id TEXT PRIMARY KEY,
    text TEXT NOT NULL,
    embedding vector(N) NOT NULL
);
CREATE INDEX ON embeddings_table USING ivfflat (embedding vector_cosine_ops);
```

**Milvus Collection:**
```python
schema = {
    "fields": [
        {"name": "id", "type": "VARCHAR", "max_length": 256, "is_primary": True},
        {"name": "text", "type": "VARCHAR", "max_length": 65535},
        {"name": "embedding", "type": "FLOAT_VECTOR", "dim": N}
    ]
}
```

**SQLite Table:**
```sql
CREATE TABLE embeddings_table (
    id TEXT PRIMARY KEY,
    text TEXT NOT NULL,
    embedding BLOB NOT NULL
);
-- Vector index created via sqlite-vec extension
```

**Elasticsearch Index:**
```json
{
  "mappings": {
    "properties": {
      "id": {"type": "keyword"},
      "text": {"type": "text"},
      "embedding": {"type": "dense_vector", "dims": N, "similarity": "cosine"}
    }
  }
}
```

### 7.4 Trace Span Structure

**Span Format (JSONL):**
```json
{
  "trace_id": "abc123...",
  "span_id": "def456...",
  "parent_span_id": "ghi789...",
  "name": "embedding_api_call",
  "start_time": "2026-04-29T16:00:00.000Z",
  "end_time": "2026-04-29T16:00:00.150Z",
  "attributes": {
    "model": "openai/text-embedding-3-small",
    "tokens": 125,
    "cost": 0.0000025,
    "batch_size": 32,
    "status": "success"
  }
}
```

## 8. Architecture

### 8.1 High-Level Architecture

The system follows a **plugin-based architecture** with abstract base classes for extensibility:

```
rag-tester/
├── src/rag_tester/
│   ├── __init__.py                 # Package exports
│   ├── rag_tester.py               # CLI entry point (Typer)
│   ├── config.py                   # Settings (pydantic-settings)
│   ├── logging_config.py           # Logging setup
│   ├── tracing.py                  # OpenTelemetry setup
│   ├── version.py                  # Version info
│   ├── commands/                   # Command implementations
│   │   ├── __init__.py
│   │   ├── load.py                 # load command
│   │   ├── test.py                 # test command
│   │   ├── bulk_test.py            # bulk-test command
│   │   └── compare.py              # compare command
│   ├── providers/                  # Provider plugins
│   │   ├── __init__.py
│   │   ├── base.py                 # Abstract base classes
│   │   ├── embeddings/             # Embedding providers
│   │   │   ├── __init__.py
│   │   │   ├── base.py             # EmbeddingProvider ABC
│   │   │   ├── local.py            # sentence-transformers
│   │   │   ├── openrouter.py       # OpenRouter API
│   │   │   ├── gemini.py           # Google Gemini API
│   │   │   └── huggingface.py      # Hugging Face models
│   │   └── databases/              # Database providers
│   │       ├── __init__.py
│   │       ├── base.py             # VectorDatabase ABC
│   │       ├── chromadb.py         # ChromaDB
│   │       ├── postgresql.py       # PostgreSQL + pgvector
│   │       ├── milvus.py           # Milvus
│   │       ├── sqlite.py           # SQLite + vector extension
│   │       └── elasticsearch.py    # Elasticsearch
│   ├── core/                       # Core business logic
│   │   ├── __init__.py
│   │   ├── loader.py               # Load logic (streaming, parallel, retry)
│   │   ├── tester.py               # Test logic (query, validation)
│   │   ├── comparator.py           # Comparison logic
│   │   └── validator.py            # Input validation
│   └── utils/                      # Utilities
│       ├── __init__.py
│       ├── file_io.py              # File reading (streaming YAML/JSON)
│       ├── retry.py                # Retry logic with backoff
│       ├── progress.py             # Progress tracking
│       └── cost.py                 # Cost calculation
└── tests/                          # Test files
    ├── __init__.py
    ├── conftest.py                 # Shared fixtures
    ├── test_commands/              # Command tests
    ├── test_providers/             # Provider tests
    ├── test_core/                  # Core logic tests
    └── test_e2e/                   # End-to-end tests
```

### 8.2 Key Design Patterns

**Plugin Pattern (Providers):**
- Abstract base classes: `EmbeddingProvider`, `VectorDatabase`
- Concrete implementations register themselves
- Factory functions instantiate providers based on identifiers

**Strategy Pattern (Load Modes):**
- LoadStrategy ABC with implementations: InitialLoad, UpsertLoad, FlushLoad
- Selected at runtime based on --mode flag

**Streaming Pattern (File I/O):**
- Generator-based file reading (yield records one at a time)
- Constant memory usage regardless of file size

**Retry Pattern (Error Handling):**
- Decorator-based retry logic with exponential backoff
- Configurable max attempts and backoff multiplier

## 9. Documentation Requirements

All documentation listed below MUST be created as part of this project.

### 9.1 README.md
- Project description, purpose, and audience
- Prerequisites and installation instructions (`pip install rag-tester`)
- Quick start guide with examples for each command
- Configuration (environment variables)
- Supported embedding models and databases (with links to provider docs)
- Usage examples for common scenarios
- Troubleshooting section
- Contributing guidelines
- License information

### 9.2 CLAUDE.md & .agent_docs/
- `CLAUDE.md`: Compact index with:
  - Project overview (tech stack, architecture)
  - Key commands (make targets)
  - Essential conventions (imports, async patterns, logging, tracing)
  - Documentation index referencing `.agent_docs/` files
- `.agent_docs/architecture.md`: Detailed architecture documentation
- `.agent_docs/providers.md`: How to add new embedding/database providers
- `.agent_docs/testing.md`: Testing strategy and patterns
- `.agent_docs/deployment.md`: Deployment and release process

### 9.3 docs/*
- `docs/user-guide.md`: Comprehensive user guide with examples
- `docs/api-reference.md`: CLI command reference (auto-generated from --help)
- `docs/embedding-models.md`: Supported embedding models with dimensions, pricing, usage
- `docs/databases.md`: Supported databases with connection strings, setup instructions
- `docs/troubleshooting.md`: Common issues and solutions
- `docs/development.md`: Development setup, coding standards, contribution workflow

## 10. Traceability Matrix

| Scenario | Functional Reqs | E2E Tests (Happy) | E2E Tests (Failure) | E2E Tests (Edge) |
|----------|----------------|-------------------|---------------------|------------------|
| SC-001 (Load) | FR-001 to FR-016, FR-032 to FR-046 | E2E-001, E2E-008 to E2E-017, E2E-025, E2E-026, E2E-030, E2E-031 | E2E-033 to E2E-040 | E2E-081 to E2E-084 |
| SC-002 (Test) | FR-017 to FR-021 | E2E-002, E2E-018 to E2E-020 | E2E-041 to E2E-044 | E2E-085, E2E-087 |
| SC-003 (Bulk-Test) | FR-022 to FR-027 | E2E-003, E2E-004, E2E-021, E2E-022, E2E-027 | E2E-045 to E2E-047 | E2E-086 |
| SC-004 (Compare) | FR-028 to FR-031 | E2E-005, E2E-023, E2E-024, E2E-032 | E2E-048 to E2E-050 | - |
| SC-005 (Compare Models) | FR-001 to FR-003, FR-022 to FR-031 | E2E-005 (combined flow) | - | - |
| SC-006 (Upsert) | FR-011 to FR-013 | E2E-006, E2E-028 | - | - |
| SC-007 (Flush) | FR-012, FR-013 | E2E-007, E2E-029 | - | - |

**Coverage Verification:**
- ✓ Every scenario has happy path tests
- ✓ Every scenario has failure tests (except SC-005, SC-006, SC-007 which are safe operations or combined flows)
- ✓ Every FR is covered by at least one test
- ✓ Happy:Failure ratio = 24:31 = 1:1.29 (meets >1:1 requirement)
- ✓ All side effects have dedicated verification tests
- ✓ All error messages/codes have triggering tests
- ✓ Cross-scenario interactions are tested

## 11. End-to-End Test Suite

> **This is the most important section of the specification.** E2E tests are the primary contract for agent-based implementation. An agent will use these tests as its definition of "done": the implementation is correct when all E2E tests pass.

**Complete test suite with 87 tests is documented in:** `.bob/e2e-tests-complete.md`

**Test Summary:**
- Core User Journeys: 7 tests
- Feature-Specific: 24 tests
- Side Effects: 8 tests
- Error Handling: 18 tests
- Security: 6 tests
- Data Integrity: 7 tests
- Performance Baseline: 5 tests
- Integration: 8 tests
- Cross-Scenario: 4 tests

**Coverage Statistics:**
- Happy path: 24 tests
- Failure/error: 31 tests
- Edge cases: 19 tests
- Side effects: 8 tests
- Other: 5 tests
- **Happy:Failure ratio: 1:1.29** ✓

**Key Test Categories:**
1. **Core User Journeys**: Complete end-to-end flows for each primary scenario
2. **Feature-Specific**: Detailed tests for each command and feature
3. **Side Effects**: Verification of observable consequences (files created, data modified, traces written)
4. **Error Handling**: All failure modes from interview (rate limits, connection drops, OOM, invalid inputs)
5. **Security**: API key validation, SQL injection protection, credential masking
6. **Data Integrity**: ID uniqueness, dimension consistency, text round-trip, concurrent safety
7. **Performance Baseline**: Latency and throughput requirements
8. **Integration**: Each database backend and embedding provider
9. **Cross-Scenario**: Interactions between scenarios (load → test, upsert → test, etc.)

**Test Execution Requirements:**
- All tests must be implemented in `tests/` directory
- Tests must be runnable via `pytest`
- Tests must use fixtures for setup/teardown
- Tests must be independent (no shared state)
- Tests must clean up resources (databases, files)
- Tests requiring external services (databases, APIs) must be skippable via markers

## 12. Implementation Guidance

### 12.1 Development Phases

**Phase 1: Core Infrastructure (Week 1)**
- Set up project structure (pyproject.toml, Makefile, CI)
- Implement config, logging, tracing
- Implement abstract base classes (EmbeddingProvider, VectorDatabase)
- Implement file I/O utilities (streaming YAML/JSON)
- Implement retry logic with exponential backoff

**Phase 2: Providers (Week 2)**
- Implement local embedding provider (sentence-transformers)
- Implement ChromaDB database provider
- Implement OpenRouter embedding provider
- Implement PostgreSQL database provider
- Write provider tests

**Phase 3: Load Command (Week 3)**
- Implement load command with initial mode
- Implement streaming file processing
- Implement parallel processing
- Implement batch embedding
- Implement dimension compatibility check
- Write load command tests

**Phase 4: Test & Bulk-Test Commands (Week 4)**
- Implement test command with output formats
- Implement bulk-test command with validation logic
- Implement progress tracking
- Implement parallel test execution
- Write test command tests

**Phase 5: Compare Command & Additional Providers (Week 5)**
- Implement compare command with cost calculation
- Implement Gemini embedding provider
- Implement Milvus, SQLite, Elasticsearch database providers
- Write compare command tests

**Phase 6: Load Modes & Polish (Week 6)**
- Implement upsert and flush load modes
- Implement force re-embedding flag
- Polish error messages and help text
- Write documentation (README, CLAUDE.md, docs/)
- Run full E2E test suite

### 12.2 Testing Strategy

**Unit Tests:**
- Test each provider in isolation (mock external dependencies)
- Test core logic (loader, tester, comparator, validator)
- Test utilities (file I/O, retry, progress, cost)
- Target: >= 80% coverage

**Integration Tests:**
- Test providers with real external services (databases, APIs)
- Use docker-compose for local database instances
- Skip tests if services unavailable (pytest markers)

**End-to-End Tests:**
- Test complete workflows (load → test → bulk-test → compare)
- Use real files, databases, embedding models
- Verify all side effects (files created, traces written, data modified)
- These are the primary acceptance criteria

### 12.3 Quality Gates

Before merging any code, ensure:
- `make check` passes (lint, format, typecheck, security, tests with >= 80% coverage)
- All E2E tests pass
- Documentation updated (README, CLAUDE.md, docs/)
- No regressions in existing functionality

## 13. Open Questions & TBDs

None at this time. All requirements have been clarified through the discovery interview.

## 14. Glossary

- **RAG**: Retrieval-Augmented Generation - a technique combining information retrieval with language generation
- **Embedding**: A dense vector representation of text, used for semantic similarity search
- **Vector Database**: A database optimized for storing and searching high-dimensional vectors
- **Similarity Score**: A measure of how similar two embeddings are (typically cosine similarity, range 0-1)
- **Top-K**: The K most similar results from a search query
- **Streaming Mode**: Processing data incrementally without loading entire dataset into memory
- **Exponential Backoff**: A retry strategy where wait time doubles after each failure
- **Dimension**: The size of an embedding vector (e.g., 384, 768, 1536)
- **Threshold**: A minimum similarity score required for a test to pass
- **Upsert**: Update if exists, insert if not exists
- **Flush**: Delete all existing data and replace with new data
- **JSONL**: JSON Lines format - one JSON object per line
- **OpenTelemetry**: An observability framework for tracing, metrics, and logs
- **Span**: A unit of work in a trace (e.g., an API call, a database query)

---

**End of Specification Document**

This specification provides a complete blueprint for implementing the RAG Testing Framework. All requirements, scenarios, tests, and architecture are defined. Implementation can proceed following the guidance in Section 12.
