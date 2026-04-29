# End-to-End Tests

This directory contains comprehensive end-to-end tests for the RAG Tester CLI application.

## Test Organization

Tests are organized following the test plan in `.bob/e2e-tests-complete.md`:

- **test_core_journeys.py**: Core user workflows (E2E-001 to E2E-007)
- **test_error_handling.py**: Error scenarios and failure modes (E2E-033 to E2E-081)

## Running E2E Tests

### Prerequisites

1. **ChromaDB**: Running instance at `localhost:8000` (or set `CHROMADB_URL` env var)
   ```bash
   docker run -p 8000:8000 chromadb/chroma
   ```

2. **API Keys** (optional, for API provider tests):
   ```bash
   export OPENROUTER_API_KEY="your-key"
   export GEMINI_API_KEY="your-key"
   ```

3. **Install dependencies**:
   ```bash
   uv sync --all-extras
   ```

### Run All E2E Tests

```bash
# Run all E2E tests
pytest tests/e2e/ -v

# Run only critical tests
pytest tests/e2e/ -v -m critical

# Run with coverage
pytest tests/e2e/ --cov=src/rag_tester --cov-report=html
```

### Run Specific Test Categories

```bash
# Core journeys only
pytest tests/e2e/test_core_journeys.py -v

# Error handling only
pytest tests/e2e/test_error_handling.py -v

# Run a specific test
pytest tests/e2e/test_core_journeys.py::TestCoreJourneys::test_e2e_001_initial_load_local_embedding -v
```

## Test Markers

Tests use pytest markers for categorization:

- `@pytest.mark.e2e`: All E2E tests
- `@pytest.mark.critical`: Critical path tests (must pass)
- `@pytest.mark.high`: High priority tests
- `@pytest.mark.medium`: Medium priority tests
- `@pytest.mark.low`: Low priority tests

## Test Fixtures

Common fixtures are defined in `conftest.py`:

- `temp_dir`: Temporary directory for test files
- `sample_data_file`: Sample data with 10 records
- `sample_test_suite`: Sample test suite for bulk testing
- `large_data_file`: Large dataset (1000 records) for performance testing
- `invalid_yaml_file`: Malformed YAML for error testing
- `missing_fields_file`: Data with missing required fields
- `duplicate_ids_file`: Data with duplicate IDs
- `chromadb_url`: ChromaDB connection URL
- `embedding_model`: Default embedding model for tests

## Test Coverage

Current test coverage:

| Category | Tests | Status |
|----------|-------|--------|
| Core Journeys | 7 | ✅ Implemented |
| Error Handling | 11 | ✅ Implemented |
| Feature-Specific | 0 | 🚧 Planned |
| Security | 0 | 🚧 Planned |
| Performance | 0 | 🚧 Planned |
| Integration | 0 | 🚧 Planned |

## Skipped Tests

Some tests are marked with `@pytest.mark.skip` because they require:
- Pre-loaded database state from previous tests
- External services (PostgreSQL, Milvus, Elasticsearch)
- API keys for cloud providers

These will be enabled as the implementation progresses.

## Adding New Tests

1. Follow the naming convention: `test_e2e_XXX_description`
2. Add appropriate markers (`@pytest.mark.e2e`, `@pytest.mark.critical`, etc.)
3. Document the test purpose in the docstring
4. Use fixtures from `conftest.py` for common setup
5. Verify exit codes, stdout, stderr, and side effects
6. Update this README with new test categories

## Continuous Integration

E2E tests are run in CI on:
- Every pull request
- Merge to main branch
- Nightly builds (full suite including slow tests)

See `.github/workflows/` for CI configuration.
