# RAG Tester

CLI framework for loading data into vector databases, running retrieval tests
against multiple backends, and comparing the quality of different embedding
models on the same workload.

Built with **Python 3.13+**, **Typer**, **OpenTelemetry**, **Ruff**, **mypy**,
and **pytest**. The package is async-first end to end.

## Features

- **5 vector database backends** with a uniform interface
  - ChromaDB (HTTP and persistent file modes)
  - PostgreSQL with pgvector
  - Milvus
  - SQLite with optional sqlite-vec extension
  - Elasticsearch with dense_vector kNN
- **3 embedding providers**
  - Local (sentence-transformers / HuggingFace)
  - Google Gemini API
  - OpenRouter API (OpenAI, Cohere, Voyage models)
- **1 LLM provider for end-to-end RAG**
  - OpenRouter (routes to GPT, Claude, Gemini, Llama, etc. with one key)
- **5 CLI commands**
  - `load`, stream YAML/JSON records into a backend with batching, upsert and flush modes
  - `test`, run a single query with table, JSON, or text output (retrieval only)
  - `bulk-test`, run a YAML test suite in parallel with pass/fail validation
  - `compare`, compare multiple bulk-test result files to rank models
  - `answer`, full RAG round-trip: retrieval + LLM generation with cited sources
- **Production-grade observability**
  - OpenTelemetry tracing with JSONL export and PII-aware sanitization
  - Rich-based logging with file rotation
  - Built-in retry with exponential backoff
  - Cost calculation for paid embedding APIs

## Installation

The project uses **uv** for dependency management. Install uv if you do not
have it (see https://docs.astral.sh/uv/), then:

```bash
make sync           # creates .venv and installs all dependencies
```

## Quick start

```bash
# 1. Run a self-contained ChromaDB load with the local embedding model
rag-tester load \
    --source examples/records.yaml \
    --target "chromadb:///tmp/chroma_data/my_collection" \
    --embedding sentence-transformers/all-MiniLM-L6-v2

# 2. Issue a single query
rag-tester test \
    --query "What is retrieval augmented generation?" \
    --database "chromadb:///tmp/chroma_data/my_collection" \
    --embedding sentence-transformers/all-MiniLM-L6-v2 \
    --top-k 5

# 3. Run a YAML test suite
rag-tester bulk-test \
    --source tests/suite.yaml \
    --database "chromadb:///tmp/chroma_data/my_collection" \
    --embedding sentence-transformers/all-MiniLM-L6-v2 \
    --output results.yaml

# 4. Compare multiple result files
rag-tester compare results-modelA.yaml results-modelB.yaml --output comparison.yaml

# 5. Answer a question using retrieved context (full RAG)
export OPENROUTER_API_KEY=sk-or-...
rag-tester answer "What is retrieval augmented generation?" \
    --database "chromadb:///tmp/chroma_data/my_collection" \
    --embedding sentence-transformers/all-MiniLM-L6-v2 \
    --top-k 5 \
    --llm-model openai/gpt-4o-mini
```

Full help is available on every subcommand: `rag-tester <command> --help`.

## Connection-string formats

| Backend | URI scheme |
|---------|------------|
| ChromaDB (HTTP) | `chromadb://host:port/collection` |
| ChromaDB (persistent) | `chromadb:///abs/path/to/data/collection` |
| PostgreSQL | `postgresql://user:pass@host:port/dbname/table_name` |
| Milvus | `milvus://host:port/collection_name` |
| SQLite | `sqlite:///abs/path/to/db.sqlite/table_name` |
| Elasticsearch | `elasticsearch://host:port/index_name` |

Table, collection, and index names must be alphanumeric (plus underscore, plus
hyphen for Elasticsearch). Hostile inputs are rejected at parse time.

## Environment variables

API keys are read from environment variables (or a `.env` file at the project
root). Both unprefixed and `RAG_TESTER_`-prefixed names are accepted.

| Variable | Purpose |
|----------|---------|
| `GEMINI_API_KEY` (or `RAG_TESTER_GEMINI_API_KEY`) | Required by the Gemini embedding provider |
| `OPENROUTER_API_KEY` (or `RAG_TESTER_OPENROUTER_API_KEY`) | Required by the OpenRouter embedding provider AND the OpenRouter LLM provider used by `answer` |
| `RAG_TESTER_LLM_MODEL` | Default model used by `answer` when `--llm-model` is omitted (default: `openai/gpt-4o-mini`) |
| `RAG_TESTER_LOG_LEVEL` | `DEBUG`, `INFO`, `WARNING`, `ERROR` (default: `INFO`) |
| `RAG_TESTER_LOG_FILE` | Path to the rotating log file (default: `logs/rag-tester.log`) |
| `RAG_TESTER_TRACE_FILE` | Path to the JSONL trace file (default: `traces/rag-tester.jsonl`) |
| `RAG_TESTER_OTEL_ENDPOINT` | Optional OTLP collector endpoint |

API keys are stored as `pydantic.SecretStr`, so they never appear in `repr`
or default logging output.

## Development

```bash
make sync             # install all dependencies
make check            # run the full quality gate before committing
make test             # run tests without coverage
make test-cov         # run tests with coverage (>= 80% required)
make test-e2e         # run E2E tests (requires running ChromaDB and friends)
make lint             # ruff lint
make lint-fix         # ruff lint with autofix
make format           # ruff format
make typecheck        # mypy strict
make security         # bandit
make docker-build     # build the runtime image, version injected from git describe
```

`make check` is the contract: lint, format-check, typecheck, security, and
test-cov must all pass before any commit.

## Architecture

```
rag-tester
+-- commands/         CLI entry points (Typer)
|   +-- load          load records into a backend
|   +-- test          single-query retrieval (R only)
|   +-- bulk-test     YAML test-suite runner
|   +-- compare       cross-model result comparator
|   \-- answer        full RAG: retrieval + LLM generation
+-- core/
|   +-- loader        streaming YAML/JSON loader, dimension/duplicate guards
|   +-- tester        single-query execution, output formatters
|   +-- comparator    multi-result comparison and ranking
|   \-- validator     record schema validation
+-- providers/
|   +-- databases/    plugin: VectorDatabase + 5 implementations + factory
|   +-- embeddings/   plugin: EmbeddingProvider + 3 implementations + factory
|   \-- llm/          plugin: LLMProvider + 1 implementation (OpenRouter) + factory
\-- utils/            retry, file I/O, cost calculation, progress
```

New backends register themselves in
`src/rag_tester/providers/databases/__init__._REGISTRY`; new embedding
providers in `src/rag_tester/providers/embeddings/__init__._REGISTRY`.

## License

MIT, see `LICENSE` (if present) or the `license` field in `pyproject.toml`.

## Author

Sebastien MORAND
