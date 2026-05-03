"""Pytest configuration and fixtures for E2E tests."""

import os
import tempfile
from collections.abc import Generator
from pathlib import Path

import pytest
import yaml

from rag_tester.providers.databases.chromadb import ChromaDBProvider
from rag_tester.providers.embeddings.local import LocalEmbeddingProvider


@pytest.fixture
def temp_dir() -> Generator[Path]:
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_data_file(temp_dir: Path) -> Path:
    """Create a sample data file with 10 test records."""
    data = {
        "records": [
            {
                "id": f"doc{i}",
                "text": f"This is test document {i} about machine learning and AI.",
                "metadata": {"source": "test", "index": i},
            }
            for i in range(1, 11)
        ]
    }
    file_path = temp_dir / "test_data.yaml"
    file_path.write_text(yaml.dump(data))
    return file_path


@pytest.fixture
def sample_data_50(temp_dir: Path) -> Path:
    """Create a sample data file with 50 test records."""
    data = {
        "records": [
            {
                "id": f"doc{i}",
                "text": f"This is test document {i} about machine learning and AI.",
                "metadata": {"source": "test", "index": i},
            }
            for i in range(1, 51)
        ]
    }
    file_path = temp_dir / "test_data_50.yaml"
    with open(file_path, "w", encoding="utf-8") as f:
        yaml.dump(data, f)
    return file_path


@pytest.fixture
def sample_test_suite(temp_dir: Path) -> Path:
    """Create a sample test suite file."""
    tests = {
        "tests": [
            {
                "id": "test1",
                "query": "What is machine learning?",
                "expected_results": [
                    {"id": "doc1", "min_threshold": 0.7},
                    {"id": "doc2", "min_threshold": 0.5},
                ],
            },
            {
                "id": "test2",
                "query": "Tell me about AI",
                "expected_results": [
                    {"id": "doc3", "min_threshold": 0.6},
                ],
            },
        ]
    }
    file_path = temp_dir / "test_suite.yaml"
    file_path.write_text(yaml.dump(tests))
    return file_path


@pytest.fixture
def chromadb_server() -> tuple[str, int]:
    """Get ChromaDB server host and port."""
    url = os.getenv("CHROMADB_URL", "chromadb://localhost:8001")
    # Parse URL to extract host and port
    # Format: chromadb://host:port/...
    if url.startswith("chromadb://"):
        url = url[len("chromadb://") :]
    parts = url.split("/")[0].split(":")
    host = parts[0] if len(parts) > 0 else "localhost"
    port = int(parts[1]) if len(parts) > 1 else 8001
    return (host, port)


@pytest.fixture
def chromadb_url() -> str:
    """Get ChromaDB URL from environment or use default."""
    return os.getenv("CHROMADB_URL", "chromadb://localhost:8001")


@pytest.fixture
def postgresql_url() -> str:
    """Get PostgreSQL URL from environment or use default."""
    return os.getenv("POSTGRESQL_URL", "postgresql://postgres:postgres@localhost:5432/testdb")


@pytest.fixture
def embedding_model() -> str:
    """Get default embedding model for tests."""
    return "sentence-transformers/all-MiniLM-L6-v2"


@pytest.fixture
def openrouter_api_key() -> str | None:
    """Get OpenRouter API key from environment."""
    return os.getenv("OPENROUTER_API_KEY")


@pytest.fixture
def gemini_api_key() -> str | None:
    """Get Gemini API key from environment."""
    return os.getenv("GEMINI_API_KEY")


@pytest.fixture(autouse=True)
def cleanup_traces(temp_dir: Path) -> Generator[None]:
    """Clean up trace files after each test."""
    yield
    # Cleanup is handled by temp_dir fixture


@pytest.fixture
def mock_chromadb_unavailable(monkeypatch):
    """Mock ChromaDB being unavailable."""

    def mock_connect(*args, **kwargs):
        raise ConnectionError("Database connection failed: Connection refused")

    # This would need to be implemented based on actual ChromaDB client
    # monkeypatch.setattr("chromadb.Client", mock_connect)
    return mock_connect


@pytest.fixture
def large_data_file(temp_dir: Path) -> Path:
    """Create a large data file with 1000 records for performance testing."""
    data = {
        "records": [
            {
                "id": f"doc{i}",
                "text": f"This is test document {i} with some longer content about various topics including machine learning, artificial intelligence, data science, and natural language processing.",
                "metadata": {"source": "test", "index": i},
            }
            for i in range(1, 1001)
        ]
    }
    file_path = temp_dir / "large_test_data.yaml"
    file_path.write_text(yaml.dump(data))
    return file_path


@pytest.fixture
def invalid_yaml_file(temp_dir: Path) -> Path:
    """Create an invalid YAML file for error testing."""
    file_path = temp_dir / "invalid.yaml"
    file_path.write_text("invalid: yaml: content:\n  - broken\n    indentation")
    return file_path


@pytest.fixture
def missing_fields_file(temp_dir: Path) -> Path:
    """Create a data file with missing required fields."""
    data = {
        "records": [
            {"id": "doc1", "text": "Valid record"},
            {"id": "doc2"},  # Missing 'text' field
            {"text": "Missing id"},  # Missing 'id' field
        ]
    }
    file_path = temp_dir / "missing_fields.yaml"
    file_path.write_text(yaml.dump(data))
    return file_path


@pytest.fixture
def duplicate_ids_file(temp_dir: Path) -> Path:
    """Create a data file with duplicate IDs."""
    data = {
        "records": [
            {"id": "doc1", "text": "First occurrence"},
            {"id": "doc2", "text": "Unique record"},
            {"id": "doc1", "text": "Duplicate occurrence"},
        ]
    }
    file_path = temp_dir / "duplicate_ids.yaml"
    file_path.write_text(yaml.dump(data))
    return file_path


@pytest.fixture
async def loaded_collection(chromadb_server):
    """Create a test collection with known documents for bulk-test testing."""
    host, port = chromadb_server
    collection_name = "test_collection"
    connection_string = f"chromadb://{host}:{port}/{collection_name}"

    db = ChromaDBProvider(connection_string=connection_string)
    embedding_provider = LocalEmbeddingProvider(model_name="sentence-transformers/all-MiniLM-L6-v2")

    dimension = embedding_provider.get_dimension()

    # Create collection
    await db.create_collection(collection_name, dimension)

    # Create known test documents
    texts = [
        "Machine learning is a subset of artificial intelligence.",
        "Deep learning uses neural networks with multiple layers.",
        "Neural networks are inspired by biological neural networks.",
    ]

    records = []
    embeddings = await embedding_provider.embed_texts(texts)

    for i, (text, embedding) in enumerate(zip(texts, embeddings, strict=True), start=1):
        records.append(
            {
                "id": f"doc{i}",
                "text": text,
                "embedding": embedding,
                "metadata": {},
            }
        )

    # Insert records
    await db.insert(collection_name, records)

    # Return connection string
    yield connection_string

    # Cleanup
    await db.delete_collection(collection_name)
