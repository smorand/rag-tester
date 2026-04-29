# US-009: Additional Database Backends

> Parent Spec: specs/2026-04-29_18:26:42-rag-testing-framework.md
> Status: ready
> Priority: 9
> Depends On: US-003
> Complexity: L

## Objective

Implement support for additional vector database backends: PostgreSQL (with pgvector), Milvus, SQLite (with vector extension), and Elasticsearch. This story completes the multi-backend architecture, enabling users to choose the database that best fits their infrastructure, performance requirements, and deployment constraints. Each backend implements the VectorDatabase interface from US-002, ensuring consistent behavior across all databases.

## Technical Context

### Stack
- **Language:** Python 3.13+
- **PostgreSQL:** psycopg3 (async), pgvector extension
- **Milvus:** pymilvus (async client)
- **SQLite:** aiosqlite, sqlite-vec extension
- **Elasticsearch:** elasticsearch-py (async client)
- **Testing:** pytest, pytest-asyncio, pytest-cov

### Relevant File Structure
```
rag-tester/
├── src/rag_tester/
│   └── providers/
│       └── databases/
│           ├── __init__.py       # Database provider exports (UPDATE)
│           ├── postgresql.py     # PostgreSQL + pgvector (NEW)
│           ├── milvus.py         # Milvus (NEW)
│           ├── sqlite.py         # SQLite + vector extension (NEW)
│           └── elasticsearch.py  # Elasticsearch (NEW)
├── tests/
│   └── test_providers/
│       └── test_databases/
│           ├── test_postgresql.py  # PostgreSQL tests (NEW)
│           ├── test_milvus.py      # Milvus tests (NEW)
│           ├── test_sqlite.py      # SQLite tests (NEW)
│           └── test_elasticsearch.py # Elasticsearch tests (NEW)
└── pyproject.toml                # Add: psycopg[binary], pymilvus, aiosqlite, elasticsearch
```

### Existing Patterns
- **Config:** Use Settings from US-001
- **Logging:** Use module-level loggers from US-001
- **Tracing:** Use tracer from US-001 for all operations
- **Retry:** Use retry decorator from US-001 for transient failures
- **Base Class:** Implement VectorDatabase from US-002

### Data Model (excerpt)

**Connection Strings:**
- PostgreSQL: `postgresql://user:pass@localhost:5432/dbname/table_name`
- Milvus: `milvus://localhost:19530/collection_name`
- SQLite: `sqlite:///path/to/db.db/table_name`
- Elasticsearch: `elasticsearch://localhost:9200/index_name`

**PostgreSQL Schema:**
```sql
CREATE TABLE embeddings_table (
    id TEXT PRIMARY KEY,
    text TEXT NOT NULL,
    embedding vector(N) NOT NULL
);
CREATE INDEX ON embeddings_table USING ivfflat (embedding vector_cosine_ops);
```

**Milvus Schema:**
```python
{
    "fields": [
        {"name": "id", "type": "VARCHAR", "max_length": 256, "is_primary": True},
        {"name": "text", "type": "VARCHAR", "max_length": 65535},
        {"name": "embedding", "type": "FLOAT_VECTOR", "dim": N}
    ]
}
```

**SQLite Schema:**
```sql
CREATE TABLE embeddings_table (
    id TEXT PRIMARY KEY,
    text TEXT NOT NULL,
    embedding BLOB NOT NULL
);
-- Vector index via sqlite-vec extension
```

**Elasticsearch Mapping:**
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

## Functional Requirements

### FR-037: PostgreSQL Provider (pgvector)
- **Description:** Store and query embeddings using PostgreSQL with pgvector extension
- **Inputs:** 
  - Connection string: "postgresql://user:pass@host:port/dbname/table_name"
  - Records with id, text, embedding
  - Query embedding and top_k
- **Outputs:** 
  - Insert: confirmation of records stored
  - Query: list of records with id, text, similarity score
- **Business Rules:**
  - Parse connection string to extract host, port, dbname, table name, credentials
  - Use psycopg3 AsyncConnection for database operations
  - Auto-create table if it doesn't exist (with vector column)
  - Create IVFFlat index for similarity search
  - Use cosine similarity for queries: `ORDER BY embedding <=> query_embedding`
  - Verify dimension compatibility before insert
  - Trace each database operation with: operation type, table name, record count, duration
  - Retry transient failures (connection errors, deadlocks)
  - Log table creation at INFO level

### FR-038: Milvus Provider
- **Description:** Store and query embeddings using Milvus vector database
- **Inputs:** 
  - Connection string: "milvus://host:port/collection_name"
  - Records with id, text, embedding
  - Query embedding and top_k
- **Outputs:** 
  - Insert: confirmation of records stored
  - Query: list of records with id, text, similarity score
- **Business Rules:**
  - Parse connection string to extract host, port, collection name
  - Use pymilvus MilvusClient (async)
  - Auto-create collection if it doesn't exist (with schema: id, text, embedding)
  - Create index on embedding field (IVF_FLAT with cosine metric)
  - Load collection before querying (Milvus requirement)
  - Verify dimension compatibility before insert
  - Trace each database operation
  - Retry transient failures
  - Log collection creation at INFO level

### FR-039: SQLite Provider (vector extension)
- **Description:** Store and query embeddings using SQLite with vector extension
- **Inputs:** 
  - Connection string: "sqlite:///path/to/db.db/table_name"
  - Records with id, text, embedding
  - Query embedding and top_k
- **Outputs:** 
  - Insert: confirmation of records stored
  - Query: list of records with id, text, similarity score
- **Business Rules:**
  - Parse connection string to extract path and table name
  - Use aiosqlite for async database operations
  - Load sqlite-vec extension
  - Auto-create table if it doesn't exist
  - Store embeddings as BLOB (serialized float array)
  - Use vector extension for similarity search
  - Verify dimension compatibility before insert
  - Trace each database operation
  - Log table creation at INFO level

### FR-040: Elasticsearch Provider
- **Description:** Store and query embeddings using Elasticsearch
- **Inputs:** 
  - Connection string: "elasticsearch://host:port/index_name"
  - Records with id, text, embedding
  - Query embedding and top_k
- **Outputs:** 
  - Insert: confirmation of records stored
  - Query: list of records with id, text, similarity score
- **Business Rules:**
  - Parse connection string to extract host, port, index name
  - Use elasticsearch-py AsyncElasticsearch client
  - Auto-create index if it doesn't exist (with dense_vector mapping)
  - Use kNN search for similarity queries
  - Verify dimension compatibility before insert
  - Trace each database operation
  - Retry transient failures
  - Log index creation at INFO level

### FR-041: Database Authentication
- **Description:** Support secure authentication for all database backends
- **Inputs:** 
  - Credentials in connection string or environment variables
- **Outputs:** 
  - Authenticated database connection
- **Business Rules:**
  - PostgreSQL: username/password in connection string
  - Milvus: optional token authentication
  - SQLite: file system permissions (no authentication)
  - Elasticsearch: optional basic auth or API key
  - Never log credentials (mask in logs and traces)
  - Raise AuthenticationError on auth failures
  - Log authentication method at DEBUG level

### FR-042: SQL Injection Protection
- **Description:** Prevent SQL injection attacks in PostgreSQL and SQLite
- **Inputs:** 
  - User-provided IDs and text
- **Outputs:** 
  - Safe database operations
- **Business Rules:**
  - Use parameterized queries (never string concatenation)
  - Validate table/collection names (alphanumeric + underscore only)
  - Escape special characters in text
  - Log validation at DEBUG level

### FR-043: Connection Pooling
- **Description:** Reuse database connections for efficiency
- **Inputs:** 
  - Database connection string
- **Outputs:** 
  - Connection pool
- **Business Rules:**
  - PostgreSQL: use psycopg3 connection pool
  - Milvus: reuse MilvusClient instance
  - SQLite: single connection (no pooling needed)
  - Elasticsearch: reuse AsyncElasticsearch client
  - Close connections on cleanup
  - Log connection pool creation at DEBUG level

### FR-044: Dimension Compatibility Check
- **Description:** Verify embedding dimension matches database schema
- **Inputs:** 
  - Embedding dimension from model
  - Database schema dimension
- **Outputs:** 
  - Validation result
- **Business Rules:**
  - Check dimension before first insert
  - For existing collections/tables: query schema to get dimension
  - For new collections/tables: use model dimension
  - Raise DimensionMismatchError if mismatch
  - Log dimension check at DEBUG level

## Acceptance Tests

> **Acceptance tests are mandatory: 100% must pass.**
> A user story is NOT considered implemented until **every single acceptance test below passes**.
> The implementing agent MUST loop (fix code → run tests → check results → repeat) until all acceptance tests pass with zero failures. Do not stop or declare the story "done" while any test is failing.
> Tests MUST be validated through the project's CI/CD chain (generally `make test` or equivalent Makefile target). No other method of running or validating tests is acceptable: do not run test files directly, do not use ad hoc commands. Use the Makefile.

### Test Data

| Data | Description | Source | Status |
|------|-------------|--------|--------|
| test_data_50.yaml | 50 records for backend testing | auto-generated (pytest fixture) | ready |
| PostgreSQL instance | Local PostgreSQL with pgvector | docker-compose | ready |
| Milvus instance | Local Milvus standalone | docker-compose | ready |
| SQLite database | Local SQLite file | auto-created | ready |
| Elasticsearch instance | Local Elasticsearch | docker-compose | ready |

### Happy Path Tests

### E2E-014: PostgreSQL Backend
- **Category:** happy
- **Scenario:** SC-001
- **Requirements:** FR-037, FR-041, FR-042
- **Preconditions:**
  - PostgreSQL at localhost:5432 with pgvector extension
  - test_data_50.yaml with 50 records
- **Steps:**
  - Given clean PostgreSQL database
  - When: `rag-tester load --file test_data_50.yaml --database postgresql://user:pass@localhost:5432/testdb/embeddings_table --embedding sentence-transformers/all-MiniLM-L6-v2`
  - Then exit code 0
  - And table "embeddings_table" is created with vector column (dimension 384)
  - And all 50 records inserted
  - And IVFFlat index created on embedding column
  - And query returns correct results with cosine similarity
  - And trace has spans: table_create, database_insert, index_create
- **Cleanup:** Drop table
- **Priority:** Critical

### E2E-015: Milvus Backend
- **Category:** happy
- **Scenario:** SC-001
- **Requirements:** FR-038
- **Preconditions:**
  - Milvus at localhost:19530
  - test_data_50.yaml with 50 records
- **Steps:**
  - Given clean Milvus instance
  - When: `rag-tester load --file test_data_50.yaml --database milvus://localhost:19530/test_collection --embedding sentence-transformers/all-MiniLM-L6-v2`
  - Then exit code 0
  - And collection "test_collection" is created with schema (id, text, embedding)
  - And all 50 records inserted
  - And collection has 50 entities
  - And query returns correct results
  - And trace has spans: collection_create, database_insert, index_create
- **Cleanup:** Drop collection
- **Priority:** Critical

### E2E-016: SQLite Backend
- **Category:** happy
- **Scenario:** SC-001
- **Requirements:** FR-039
- **Preconditions:**
  - sqlite-vec extension available
  - test_data_50.yaml with 50 records
- **Steps:**
  - Given clean SQLite database file
  - When: `rag-tester load --file test_data_50.yaml --database sqlite:///tmp/test.db/embeddings_table --embedding sentence-transformers/all-MiniLM-L6-v2`
  - Then exit code 0
  - And database file /tmp/test.db is created
  - And table "embeddings_table" is created
  - And all 50 records inserted
  - And query returns correct results using vector extension
  - And trace has spans: table_create, database_insert
- **Cleanup:** Delete database file
- **Priority:** Critical

### E2E-017: Elasticsearch Backend
- **Category:** happy
- **Scenario:** SC-001
- **Requirements:** FR-040
- **Preconditions:**
  - Elasticsearch at localhost:9200
  - test_data_50.yaml with 50 records
- **Steps:**
  - Given clean Elasticsearch instance
  - When: `rag-tester load --file test_data_50.yaml --database elasticsearch://localhost:9200/test_index --embedding sentence-transformers/all-MiniLM-L6-v2`
  - Then exit code 0
  - And index "test_index" is created with dense_vector mapping (dimension 384)
  - And all 50 records indexed
  - And kNN search returns correct results
  - And trace has spans: index_create, database_insert
- **Cleanup:** Delete index
- **Priority:** Critical

### E2E-053: Database Authentication Failure
- **Category:** failure
- **Scenario:** Authentication error
- **Requirements:** FR-041
- **Preconditions:**
  - PostgreSQL with authentication enabled
  - Wrong credentials
- **Steps:**
  - Given PostgreSQL with wrong credentials
  - When: `rag-tester load --file test_data_50.yaml --database postgresql://wrong:pass@localhost:5432/testdb/table --embedding sentence-transformers/all-MiniLM-L6-v2`
  - Then exit code 1
  - And stderr contains: "Error: Database authentication failed: invalid credentials"
  - And no table created
- **Cleanup:** None
- **Priority:** Critical

### E2E-054: SQL Injection Protection
- **Category:** failure (security)
- **Scenario:** Security validation
- **Requirements:** FR-042
- **Preconditions:**
  - PostgreSQL instance
  - Record with ID containing SQL injection: "'; DROP TABLE--"
- **Steps:**
  - Given record with malicious ID
  - When load is executed
  - Then record is safely inserted (ID is escaped)
  - And no SQL injection occurs (table not dropped)
  - And log contains: "Validated record ID: safe"
- **Cleanup:** Drop table
- **Priority:** Critical

### E2E-055: Path Traversal Protection
- **Category:** failure (security)
- **Scenario:** Security validation
- **Requirements:** FR-039
- **Preconditions:**
  - SQLite connection string with path traversal: "sqlite:///../../../etc/passwd/table"
- **Steps:**
  - Given malicious path
  - When: `rag-tester load --file test_data_50.yaml --database sqlite:///../../../etc/passwd/table --embedding sentence-transformers/all-MiniLM-L6-v2`
  - Then exit code 1
  - And stderr contains: "Error: Invalid database path: path traversal detected"
  - And no file created outside working directory
- **Cleanup:** None
- **Priority:** High

### E2E-071: PostgreSQL with pgvector
- **Category:** happy (integration)
- **Scenario:** SC-001
- **Requirements:** FR-037
- **Preconditions:**
  - PostgreSQL with pgvector extension installed
- **Steps:**
  - Given PostgreSQL with pgvector
  - When PostgreSQLProvider is used
  - Then vector column type is created correctly
  - And similarity search works with cosine distance
  - And IVFFlat index improves query performance
- **Cleanup:** Drop table
- **Priority:** Critical

### E2E-072: Milvus Collection Management
- **Category:** happy (integration)
- **Scenario:** SC-001
- **Requirements:** FR-038
- **Preconditions:**
  - Milvus instance running
- **Steps:**
  - Given Milvus instance
  - When MilvusProvider is used
  - Then collection is auto-created with correct schema
  - And schema has fields: id (VARCHAR), text (VARCHAR), embedding (FLOAT_VECTOR)
  - And index is created on embedding field
  - And collection is loaded before querying
- **Cleanup:** Drop collection
- **Priority:** High

### E2E-073: SQLite with vector extension
- **Category:** happy (integration)
- **Scenario:** SC-001
- **Requirements:** FR-039
- **Preconditions:**
  - sqlite-vec extension available
- **Steps:**
  - Given SQLite with vector extension
  - When SQLiteProvider is used
  - Then vector extension is loaded successfully
  - And embeddings are stored as BLOB
  - And similarity search works correctly
- **Cleanup:** Delete database file
- **Priority:** High

### E2E-074: Elasticsearch Dense Vector
- **Category:** happy (integration)
- **Scenario:** SC-001
- **Requirements:** FR-040
- **Preconditions:**
  - Elasticsearch instance running
- **Steps:**
  - Given Elasticsearch instance
  - When ElasticsearchProvider is used
  - Then index mapping has dense_vector field
  - And dense_vector has correct dimension and similarity metric
  - And kNN search returns accurate results
- **Cleanup:** Delete index
- **Priority:** High

### Edge Case and Error Tests

### E2E-DB-001: PostgreSQL Connection Pool
- **Category:** edge
- **Scenario:** Connection management
- **Requirements:** FR-043
- **Preconditions:**
  - PostgreSQL instance
  - Multiple concurrent operations
- **Steps:**
  - Given PostgreSQL provider with connection pool
  - When multiple operations are executed concurrently
  - Then connections are reused from pool
  - And no connection leaks occur
  - And log contains: "Connection pool created: size=10"
- **Cleanup:** Drop table
- **Priority:** Medium

### E2E-DB-002: Milvus Collection Already Exists
- **Category:** edge
- **Scenario:** Existing collection
- **Requirements:** FR-038
- **Preconditions:**
  - Milvus collection already exists with dimension 384
- **Steps:**
  - Given existing collection with dimension 384
  - When load is executed with same dimension
  - Then no error occurs
  - And records are inserted into existing collection
  - And log contains: "Using existing collection: test_collection"
- **Cleanup:** Drop collection
- **Priority:** Medium

### E2E-DB-003: SQLite File Permissions
- **Category:** failure
- **Scenario:** File I/O error
- **Requirements:** FR-039
- **Preconditions:**
  - SQLite path is unwritable (permissions)
- **Steps:**
  - Given unwritable path
  - When: `rag-tester load --file test_data_50.yaml --database sqlite:///root/test.db/table --embedding sentence-transformers/all-MiniLM-L6-v2`
  - Then exit code 1
  - And stderr contains: "Error: Cannot create database file: permission denied"
- **Cleanup:** None
- **Priority:** High

### E2E-DB-004: Elasticsearch Index Mapping Conflict
- **Category:** failure
- **Scenario:** Schema mismatch
- **Requirements:** FR-040, FR-044
- **Preconditions:**
  - Elasticsearch index exists with dimension 768
  - Model has dimension 384
- **Steps:**
  - Given existing index with dimension 768
  - When load is executed with dimension 384 model
  - Then exit code 1
  - And stderr contains: "Error: Dimension mismatch: model=384, index=768"
- **Cleanup:** Delete index
- **Priority:** Critical

### E2E-DB-005: PostgreSQL Table Name Validation
- **Category:** failure (security)
- **Scenario:** Input validation
- **Requirements:** FR-042
- **Preconditions:**
  - Invalid table name with special characters: "table'; DROP--"
- **Steps:**
  - Given invalid table name
  - When: `rag-tester load --file test_data_50.yaml --database postgresql://user:pass@localhost:5432/testdb/table'; DROP-- --embedding sentence-transformers/all-MiniLM-L6-v2`
  - Then exit code 1
  - And stderr contains: "Error: Invalid table name: must be alphanumeric with underscores only"
- **Cleanup:** None
- **Priority:** Critical

### E2E-DB-006: Milvus Connection Timeout
- **Category:** failure
- **Scenario:** Network error
- **Requirements:** FR-038
- **Preconditions:**
  - Milvus at wrong port (unreachable)
- **Steps:**
  - Given unreachable Milvus instance
  - When: `rag-tester load --file test_data_50.yaml --database milvus://localhost:9999/test --embedding sentence-transformers/all-MiniLM-L6-v2`
  - Then exit code 1
  - And stderr contains: "Error: Milvus connection failed: timeout"
  - And trace shows retry attempts
- **Cleanup:** None
- **Priority:** High

## Constraints

### Files Not to Touch
- `src/rag_tester/providers/embeddings/` (from US-002, US-008)
- `src/rag_tester/providers/databases/chromadb.py` (from US-002)
- `src/rag_tester/commands/` (from US-003-007)

### Dependencies Not to Add
- Add to pyproject.toml:
  - psycopg[binary]>=3.1.0
  - pymilvus>=2.3.0
  - aiosqlite>=0.19.0
  - elasticsearch>=8.11.0
- Do NOT add: langchain, llama-index, or any other high-level frameworks

### Patterns to Avoid
- Do NOT use string concatenation for SQL queries (use parameterized queries)
- Do NOT store credentials in code (use environment variables or connection strings)
- Do NOT create new connections per operation (use connection pooling)
- Do NOT ignore dimension mismatches (validate before insert)

### Scope Boundary
- This story does NOT implement embedding providers (already done in US-002, US-008)
- This story does NOT implement commands (already done in US-003-007)
- This story does NOT implement ChromaDB (already done in US-002)
- This story ONLY implements: PostgreSQL, Milvus, SQLite, Elasticsearch database providers

## Non Regression

### Existing Tests That Must Pass
- All tests from US-001 (infrastructure)
- All tests from US-002 (providers - local embeddings, ChromaDB)
- All tests from US-003 (load command)
- All tests from US-004 (test command)
- All tests from US-005 (bulk-test command)
- All tests from US-006 (compare command)
- All tests from US-007 (load modes)
- All tests from US-008 (API embedding providers)

### Behaviors That Must Not Change
- Configuration system from US-001
- Logging and tracing from US-001
- VectorDatabase interface from US-002
- All commands from US-003-007
- All embedding providers from US-002, US-008

### API Contracts to Preserve
- VectorDatabase interface from US-002 (implemented by all backends)
- Load command arguments from US-003 (no changes)
