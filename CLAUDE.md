# RAG Tester

## Overview
A Python project for testing and evaluating Retrieval-Augmented Generation (RAG) systems.

**Tech Stack:** Python 3.13+, Typer (CLI), OpenTelemetry (tracing), Ruff (linting/formatting), mypy (type checking), pytest (testing)

**Supported Databases:**
- ChromaDB (HTTP and persistent modes)
- PostgreSQL with pgvector extension
- Milvus vector database
- SQLite with vector extension
- Elasticsearch with dense_vector

**Supported Embedding Providers:**
- Local models (sentence-transformers via HuggingFace)
- Gemini API (requires GEMINI_API_KEY)
- OpenRouter API (requires OPENROUTER_API_KEY)

## Key Commands
```bash
make sync          # Install dependencies
make run           # Run the application
make check         # Full quality gate (lint, format, typecheck, security, tests+coverage)
make docker-build  # Build Docker image
make help          # Show all available commands
```

## Project Structure
```
rag-tester/
├── src/rag_tester/
│   ├── __init__.py           # Package exports
│   ├── rag_tester.py         # CLI entry point with Typer
│   ├── config.py             # Settings with pydantic-settings
│   ├── logging_config.py     # Logging setup (rich + file output)
│   ├── tracing.py            # OpenTelemetry tracing
│   ├── version.py            # Version info (injected at build)
│   └── py.typed              # PEP 561 marker
├── tests/                    # Test files
│   ├── __init__.py
│   └── conftest.py           # Shared fixtures
├── Makefile                  # Build automation
├── pyproject.toml            # Project configuration
├── Dockerfile                # Multi-stage build
└── docker-compose.yml        # Service orchestration
```

## Conventions
- **Entry point:** `rag-tester` command (defined in pyproject.toml [project.scripts])
- **Imports:** Use `from rag_tester.module import ...` (package-based layout)
- **Async-first:** All I/O operations use async libraries (httpx, aiofiles, etc.)
- **Logging:** Use `logger = logging.getLogger(__name__)` per module
- **Tracing:** All external calls traced with OpenTelemetry
- **Config:** Environment variables prefixed with `RAG_TESTER_`

## Quality Gate
Run `make check` before every commit. It runs:
- `make lint` - Ruff linting
- `make format-check` - Ruff format verification
- `make typecheck` - mypy strict type checking
- `make security` - bandit security scan
- `make test-cov` - pytest with >= 80% coverage

## Auto-Evaluation Checklist
Before considering any task complete:
- [ ] `make check` passes
- [ ] No sync blocking calls in async code
- [ ] All external calls traced with OpenTelemetry
- [ ] No forbidden practices (bare except, print, mutable defaults, .format(), assert)
- [ ] Config via Settings class, not os.environ
- [ ] Dependencies injected, not created inline
- [ ] Test coverage >= 80%

## Coding Standards
This project follows the `python` skill. Reload it for full coding standards reference.

## Documentation Index
- `.agent_docs/python.md` : Python coding standards (logger %, async-first, etc.)
- `.agent_docs/makefile.md` : Makefile documentation
- `README.md` : User-facing usage guide, CLI reference, environment variables
- `.github/workflows/ci.yml` : CI pipeline (`make check` + Docker build on every push and PR)

## Provider Factories
- `rag_tester.providers.databases.get_database_provider(connection_string)` : URI-scheme dispatch (chromadb/postgresql/milvus/sqlite/elasticsearch)
- `rag_tester.providers.embeddings.get_embedding_provider(name, model_name)` : provider-name dispatch (local/gemini/openrouter)

Both factories use a `_REGISTRY` dict; new backends register themselves there.