# US-002: Local Embeddings + ChromaDB Foundation

> Parent Spec: specs/2026-04-29_18:26:42-rag-testing-framework.md
> Status: ready
> Priority: 2
> Depends On: US-001
> Complexity: L

## Objective

Establish the plugin architecture for embedding providers and vector databases by implementing abstract base classes and their first concrete implementations: local sentence-transformers embeddings and ChromaDB. This story creates the foundation for the RAG pipeline, enabling basic data loading and querying with a single embedding model and database backend. After this story, the system can load documents, generate embeddings locally, store them in ChromaDB, and perform similarity searches.

## Technical Context

### Stack
- **Language:** Python 3.13+
- **Embeddings:** sentence-transformers (ONNX compatible local models)
- **Database:** ChromaDB (HTTP and persistent modes)
- **Async:** httpx for async HTTP, aiofiles for async file I/O
- **Testing:** pytest, pytest-asyncio, pytest-cov

### Relevant File Structure
```
rag-tester/
├── src/rag_tester/
│   ├── providers/
│   │   ├── __init__.py           # Provider exports (NEW)
│   │   ├── base.py               # Abstract base classes (NEW)
│   │   ├── embeddings/
│   │   │   ├── __init__.py       # Embedding provider exports (NEW)
│   │   │   ├── base.py           # EmbeddingProvider ABC (NEW)
│   │   │   └── local.py          # LocalEmbeddingProvider (NEW)
│   │   └── databases/
│   │       ├── __init__.py       # Database provider exports (NEW)
│   │       ├── base.py           # VectorDatabase ABC (NEW)
│   │       └── chromadb.py       # ChromaDBProvider (NEW)
│   └── utils/
│       └── file_io.py            # File reading utilities (NEW)
├── tests/
│   ├── test_providers/
│   │   ├── test_embeddings/
│   │   │   └── test_local.py     # Local embedding tests (NEW)
│   │   └── test_databases/
│   │       └── test_chromadb.py  # ChromaDB tests (NEW)
│   └── test_utils/
│       └── test_file_io.py       # File I/O tests (NEW)
└── pyproject.toml                # Add: chromadb, sentence-transformers
```

### Existing Patterns
- **Config:** Use Settings from US-001 for configuration
- **Logging:** Use module-level loggers from US-001
- **Tracing:** Use tracer from US-001 for all operations
- **Retry:** Use retry decorator from US-001 for transient failures

### Data Model (excerpt)

**EmbeddingProvider ABC:**
```python
from abc import ABC, abstractmethod

class EmbeddingProvider(ABC):
    @abstractmethod
    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of texts."""
        pass
    
    @abstractmethod
    def get_dimension(self) -> int:
        """Return the embedding dimension."""
        pass
    
    @abstractmethod
    def get_model_name(self) -> str:
        """Return the model identifier."""
        pass
```

**VectorDatabase ABC:**
```python
from abc import ABC, abstractmethod

class VectorDatabase(ABC):
    @abstractmethod
    async def create_collection(self, name: str, dimension: int) -> None:
        """Create a collection/table for embeddings."""
        pass
    
    @abstractmethod
    async def insert(self, collection: str, records: list[dict]) -> None:
        """Insert records with embeddings."""
        pass
    
    @abstractmethod
    async def query(self, collection: str, query_embedding: list[float], top_k: int) -> list[dict]:
        """Query for similar embeddings."""
        pass
    
    @abstractmethod
    async def collection_exists(self, name: str) -> bool:
        """Check if collection exists."""
        pass
```

**Record Format:**
```python
{
    "id": "doc1",
    "text": "Machine learning is...",
    "embedding": [0.1, 0.2, ..., 0.384]  # dimension depends on model
}
```

**Input File Format (YAML):**
```yaml
records:
  - id: "doc1"
    text: "Machine learning is a subset of artificial intelligence..."
  - id: "doc2"
    text: "Deep learning uses neural networks..."
```

## Functional Requirements

### FR-001: Local Embedding Provider
- **Description:** Generate embeddings using local sentence-transformers models (ONNX compatible)
- **Inputs:** 
  - Model name (e.g., "sentence-transformers/all-MiniLM-L6-v2")
  - List of texts to embed
- **Outputs:** 
  - List of embedding vectors (list of floats)
  - Each vector has dimension matching the model (e.g., 384 for all-MiniLM-L6-v2)
- **Business Rules:**
  - Model is loaded once and cached for reuse
  - Embeddings are generated synchronously (sentence-transformers is CPU-bound)
  - Model dimension is determined from model metadata
  - Trace each embedding operation with: model name, number of texts, duration
  - Log model loading at INFO level
  - Handle model loading errors gracefully (raise ModelLoadError)

### FR-002: ChromaDB Provider (HTTP Mode)
- **Description:** Store and query embeddings using ChromaDB HTTP API
- **Inputs:** 
  - Connection string: "chromadb://host:port/collection_name"
  - Records with id, text, embedding
  - Query embedding and top_k
- **Outputs:** 
  - Insert: confirmation of records stored
  - Query: list of records with id, text, similarity score (sorted by score descending)
- **Business Rules:**
  - Parse connection string to extract host, port, collection name
  - Use httpx.AsyncClient for HTTP requests
  - Auto-create collection if it doesn't exist (with correct dimension)
  - Verify dimension compatibility before insert (raise DimensionMismatchError if mismatch)
  - Trace each database operation with: operation type, collection name, record count, duration
  - Retry transient failures (connection errors, timeouts) using retry decorator from US-001
  - Log collection creation at INFO level

### FR-003: ChromaDB Provider (Persistent Mode)
- **Description:** Store and query embeddings using ChromaDB persistent storage
- **Inputs:** 
  - Connection string: "chromadb:///path/to/data/collection_name"
  - Records with id, text, embedding
  - Query embedding and top_k
- **Outputs:** 
  - Same as HTTP mode
- **Business Rules:**
  - Parse connection string to extract path and collection name
  - Use chromadb.PersistentClient with the specified path
  - Auto-create directory if it doesn't exist
  - Same dimension verification and tracing as HTTP mode
  - Log persistent storage path at INFO level

### FR-004: File Reading (YAML)
- **Description:** Read records from YAML files with streaming support
- **Inputs:** 
  - File path (absolute or relative)
- **Outputs:** 
  - Generator yielding records one at a time: {"id": str, "text": str}
- **Business Rules:**
  - Use aiofiles for async file reading
  - Parse YAML incrementally (streaming mode for large files)
  - Validate each record has required fields: "id" and "text"
  - Raise ValidationError if required fields missing
  - Trace file read operation with: file path, file size, duration
  - Log file reading at DEBUG level

### FR-005: File Reading (JSON)
- **Description:** Read records from JSON files with streaming support
- **Inputs:** 
  - File path (absolute or relative)
- **Outputs:** 
  - Generator yielding records one at a time: {"id": str, "text": str}
- **Business Rules:**
  - Auto-detect JSON format (same as YAML but with JSON parser)
  - Use aiofiles for async file reading
  - Parse JSON incrementally (streaming mode for large files)
  - Same validation and tracing as YAML
  - Log file reading at DEBUG level

## Acceptance Tests

> **Acceptance tests are mandatory: 100% must pass.**
> A user story is NOT considered implemented until **every single acceptance test below passes**.
> The implementing agent MUST loop (fix code → run tests → check results → repeat) until all acceptance tests pass with zero failures. Do not stop or declare the story "done" while any test is failing.
> Tests MUST be validated through the project's CI/CD chain (generally `make test` or equivalent Makefile target). No other method of running or validating tests is acceptable: do not run test files directly, do not use ad hoc commands. Use the Makefile.

### Test Data

| Data | Description | Source | Status |
|------|-------------|--------|--------|
| test_data.yaml | 100 records with id and text fields | auto-generated (pytest fixture) | ready |
| test_data.json | Same 100 records in JSON format | auto-generated (pytest fixture) | ready |
| ChromaDB instance | Local ChromaDB server or persistent storage | docker-compose or persistent client | ready |
| sentence-transformers model | all-MiniLM-L6-v2 (dimension 384) | auto-downloaded on first use | ready |

### Happy Path Tests

### E2E-001: Initial Dataset Load with Local Embedding Model
- **Category:** happy
- **Scenario:** SC-001
- **Requirements:** FR-001, FR-002, FR-004
- **Preconditions:**
  - ChromaDB at localhost:8000 (or persistent mode)
  - test_data.yaml with 100 records
  - sentence-transformers/all-MiniLM-L6-v2 model available
- **Steps:**
  - Given clean ChromaDB instance
  - When records are loaded from test_data.yaml
  - And embeddings are generated using LocalEmbeddingProvider("sentence-transformers/all-MiniLM-L6-v2")
  - And records are inserted into ChromaDB collection "test_collection"
  - Then all 100 records are successfully inserted
  - And collection exists with 100 documents
  - And each document has dimension 384
  - And trace file has spans: file_read, embedding_batch, database_insert
  - And log file contains: "Model loaded: sentence-transformers/all-MiniLM-L6-v2", "Collection created: test_collection", "Inserted 100 records"
- **Cleanup:** Delete collection
- **Priority:** Critical

### E2E-002: Manual Query Test
- **Category:** happy
- **Scenario:** SC-002
- **Requirements:** FR-001, FR-002
- **Preconditions:**
  - Collection loaded with 100 docs
  - doc42 has text: "Machine learning is a subset of artificial intelligence that enables computers to learn from data."
- **Steps:**
  - Given collection "test_collection" with 100 documents
  - When query text is "What is machine learning?"
  - And query embedding is generated using LocalEmbeddingProvider
  - And ChromaDB is queried with top_k=3
  - Then 3 results are returned
  - And result 1 has id="doc42" with score > 0.7
  - And results are sorted by score descending
  - And trace has spans: embedding_query, database_search
- **Cleanup:** None (uses existing collection)
- **Priority:** Critical

### E2E-025: Auto-Create Collection
- **Category:** happy (side effect)
- **Scenario:** SC-001
- **Requirements:** FR-002
- **Preconditions:**
  - ChromaDB instance running
  - Collection "new_collection" does not exist
- **Steps:**
  - Given ChromaDB provider initialized with "chromadb://localhost:8000/new_collection"
  - When insert is called with 10 records (dimension 384)
  - Then collection "new_collection" is auto-created with dimension 384
  - And all 10 records are inserted
  - And log contains: "Collection created: new_collection (dimension: 384)"
- **Cleanup:** Delete collection
- **Priority:** Critical

### E2E-069: ChromaDB HTTP Mode
- **Category:** happy (integration)
- **Scenario:** SC-001
- **Requirements:** FR-002
- **Preconditions:**
  - ChromaDB HTTP server at localhost:8000
- **Steps:**
  - Given connection string "chromadb://localhost:8000/http_test_collection"
  - When ChromaDBProvider is initialized
  - Then HTTP client is used (verify via httpx.AsyncClient)
  - And collection is created via HTTP API
  - And insert/query operations work correctly
- **Cleanup:** Delete collection
- **Priority:** Critical

### E2E-070: ChromaDB Persistent Mode
- **Category:** happy (integration)
- **Scenario:** SC-001
- **Requirements:** FR-003
- **Preconditions:**
  - Writable directory /tmp/chroma_test
- **Steps:**
  - Given connection string "chromadb:///tmp/chroma_test/persistent_collection"
  - When ChromaDBProvider is initialized
  - Then persistent client is used (verify via chromadb.PersistentClient)
  - And directory /tmp/chroma_test is created
  - And collection is created in persistent storage
  - And data survives provider restart (re-initialize and query)
- **Cleanup:** Delete /tmp/chroma_test directory
- **Priority:** High

### E2E-077: Load Then Test
- **Category:** happy (cross-scenario)
- **Scenario:** SC-001 + SC-002
- **Requirements:** FR-001, FR-002, FR-004
- **Preconditions:**
  - Clean ChromaDB instance
  - test_data.yaml with known records
- **Steps:**
  - Given test_data.yaml with record id="doc5", text="Python is a programming language"
  - When records are loaded into collection "integration_test"
  - And query "programming language" is executed
  - Then results include doc5 with high similarity score (> 0.7)
  - And results match the loaded data
- **Cleanup:** Delete collection
- **Priority:** High

### Edge Case and Error Tests

### E2E-PROV-001: Empty Input File
- **Category:** failure
- **Scenario:** File validation
- **Requirements:** FR-004
- **Preconditions:**
  - empty.yaml file with no records
- **Steps:**
  - Given file path "empty.yaml"
  - When file_io.read_yaml() is called
  - Then a ValidationError is raised with message "Input file is empty or has no records"
  - And no records are yielded
- **Cleanup:** None
- **Priority:** High

### E2E-PROV-002: Missing Required Fields
- **Category:** failure
- **Scenario:** Record validation
- **Requirements:** FR-004
- **Preconditions:**
  - invalid.yaml with record missing "text" field: {id: "doc1"}
- **Steps:**
  - Given file path "invalid.yaml"
  - When file_io.read_yaml() is called
  - Then a ValidationError is raised with message "Missing required field 'text' in record 'doc1'"
  - And processing stops at the invalid record
- **Cleanup:** None
- **Priority:** Critical

### E2E-PROV-003: Dimension Mismatch
- **Category:** failure
- **Scenario:** Database validation
- **Requirements:** FR-002
- **Preconditions:**
  - Collection "existing_collection" with dimension 768
  - Records with dimension 384 (from all-MiniLM-L6-v2)
- **Steps:**
  - Given ChromaDB provider connected to "existing_collection" (dimension 768)
  - When insert is called with records having dimension 384
  - Then a DimensionMismatchError is raised with message "Dimension mismatch: model=384, database=768"
  - And no records are inserted
  - And log contains error message
- **Cleanup:** Delete collection
- **Priority:** Critical

### E2E-PROV-004: Database Unreachable
- **Category:** failure
- **Scenario:** Connection error
- **Requirements:** FR-002
- **Preconditions:**
  - ChromaDB server is down or unreachable
- **Steps:**
  - Given connection string "chromadb://localhost:9999/test_collection" (wrong port)
  - When ChromaDBProvider is initialized
  - And insert is called
  - Then a ConnectionError is raised after retry attempts (5 attempts with exponential backoff)
  - And trace contains 5 retry attempt spans
  - And log contains: "Retry attempt 1 failed", ..., "Max retry attempts exceeded"
- **Cleanup:** None
- **Priority:** Critical

### E2E-PROV-005: Model Loading Error
- **Category:** failure
- **Scenario:** Embedding provider error
- **Requirements:** FR-001
- **Preconditions:**
  - Invalid model name "invalid/model-name"
- **Steps:**
  - Given model name "invalid/model-name"
  - When LocalEmbeddingProvider is initialized
  - Then a ModelLoadError is raised with message "Failed to load model: invalid/model-name"
  - And log contains error message
- **Cleanup:** None
- **Priority:** Critical

### E2E-PROV-006: Single Record Load
- **Category:** edge
- **Scenario:** Minimal data
- **Requirements:** FR-001, FR-002, FR-004
- **Preconditions:**
  - single.yaml with 1 record
- **Steps:**
  - Given file with 1 record: {id: "doc1", text: "Test"}
  - When record is loaded and embedded
  - And inserted into ChromaDB
  - Then collection has exactly 1 document
  - And document can be queried successfully
- **Cleanup:** Delete collection
- **Priority:** Medium

### E2E-PROV-007: Very Long Text (10K chars)
- **Category:** edge
- **Scenario:** Large text handling
- **Requirements:** FR-001, FR-002
- **Preconditions:**
  - Record with 10,000 character text
- **Steps:**
  - Given record with text of 10,000 characters
  - When embedding is generated
  - And record is inserted into ChromaDB
  - Then embedding is generated successfully (dimension 384)
  - And record can be retrieved and queried
  - And text round-trips correctly (matches original)
- **Cleanup:** Delete collection
- **Priority:** Medium

### E2E-PROV-008: Unicode and Emoji
- **Category:** edge
- **Scenario:** Special character handling
- **Requirements:** FR-001, FR-002
- **Preconditions:**
  - Record with text: "Hello 世界 🌍 مرحبا"
- **Steps:**
  - Given record with Unicode and emoji text
  - When embedding is generated
  - And record is inserted into ChromaDB
  - Then embedding is generated successfully
  - And text round-trips correctly (exact match including Unicode and emoji)
  - And query with similar text returns the record
- **Cleanup:** Delete collection
- **Priority:** High

### E2E-PROV-009: JSON File Format
- **Category:** edge
- **Scenario:** Alternative file format
- **Requirements:** FR-005
- **Preconditions:**
  - test_data.json with 50 records
- **Steps:**
  - Given file path "test_data.json"
  - When file_io.read_json() is called
  - Then 50 records are yielded
  - And each record has id and text fields
  - And records can be embedded and inserted successfully
- **Cleanup:** None
- **Priority:** High

### E2E-PROV-010: Persistent Storage Directory Creation
- **Category:** edge
- **Scenario:** Directory auto-creation
- **Requirements:** FR-003
- **Preconditions:**
  - Directory /tmp/new_chroma_dir does not exist
- **Steps:**
  - Given connection string "chromadb:///tmp/new_chroma_dir/collection"
  - When ChromaDBProvider is initialized
  - Then directory /tmp/new_chroma_dir is created
  - And collection is created successfully
- **Cleanup:** Delete directory
- **Priority:** Medium

## Constraints

### Files Not to Touch
- `src/rag_tester/rag_tester.py` (CLI entry point - will be implemented in later stories)
- `src/rag_tester/commands/` (not created yet)
- `src/rag_tester/core/` (not created yet)

### Dependencies Not to Add
- Add to pyproject.toml:
  - chromadb>=0.4.0
  - sentence-transformers>=2.2.0
  - aiofiles>=23.0.0
- Do NOT add: langchain, llama-index, or any other high-level RAG frameworks

### Patterns to Avoid
- Do NOT use synchronous I/O in async functions (use aiofiles, httpx.AsyncClient)
- Do NOT load entire files into memory (use streaming/generators)
- Do NOT create database connections in __init__ (use async context managers or explicit connect/disconnect)
- Do NOT hardcode model paths or connection strings (use configuration)

### Scope Boundary
- This story does NOT implement CLI commands (load, test, bulk-test, compare)
- This story does NOT implement parallel processing (that's US-003)
- This story does NOT implement batch embedding optimization (that's US-003)
- This story does NOT implement other embedding providers (OpenRouter, Gemini - that's US-008)
- This story does NOT implement other database backends (PostgreSQL, Milvus, etc. - that's US-009)
- This story ONLY implements: abstract base classes, local embeddings, ChromaDB, file I/O

## Non Regression

### Existing Tests That Must Pass
- All tests from US-001 (config, logging, tracing, retry)
- `tests/test_version.py`
- `tests/test_cli.py`
- `tests/test_config.py`

### Behaviors That Must Not Change
- Configuration system from US-001
- Logging and tracing from US-001
- Retry logic from US-001

### API Contracts to Preserve
- Settings class interface from US-001
- Logger interface from US-001
- Tracer interface from US-001
