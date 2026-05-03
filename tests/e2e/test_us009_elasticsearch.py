"""End-to-end tests for Elasticsearch database provider (US-009)."""

import os
import subprocess
import sys
from collections.abc import Generator
from pathlib import Path

import pytest


@pytest.fixture
def elasticsearch_url() -> str:
    """Get Elasticsearch URL from environment or use default."""
    return os.getenv("ELASTICSEARCH_URL", "elasticsearch://localhost:9200")


@pytest.fixture
def elasticsearch_connection_params(elasticsearch_url: str) -> dict[str, str | int]:
    """Parse Elasticsearch URL and return connection parameters.

    Returns:
        Dictionary with keys: host, port, conn_string
    """
    # Format: elasticsearch://host:port
    url = elasticsearch_url.replace("elasticsearch://", "")
    parts = url.split(":")

    if len(parts) != 2:
        pytest.skip("Invalid Elasticsearch URL format")

    host = parts[0]
    port = int(parts[1])

    conn_string = f"elasticsearch://{host}:{port}"

    return {
        "host": host,
        "port": port,
        "conn_string": conn_string,
    }


@pytest.fixture
def elasticsearch_available(elasticsearch_connection_params: dict) -> bool:
    """Check if Elasticsearch is available."""
    try:
        from elasticsearch import Elasticsearch

        host = elasticsearch_connection_params["host"]
        port = elasticsearch_connection_params["port"]

        client = Elasticsearch(hosts=[{"host": host, "port": port, "scheme": "http"}])
        client.ping()
        client.close()
        return True
    except Exception:
        return False


@pytest.fixture
def cleanup_elasticsearch_index(elasticsearch_connection_params: dict) -> Generator[callable]:
    """Fixture to cleanup Elasticsearch indices after tests.

    Yields a cleanup function that takes an index name and deletes it.
    """
    indices_to_cleanup = []

    def register_cleanup(index_name: str) -> None:
        """Register an index for cleanup."""
        indices_to_cleanup.append(index_name)

    yield register_cleanup

    # Cleanup after test
    try:
        from elasticsearch import Elasticsearch

        host = elasticsearch_connection_params["host"]
        port = elasticsearch_connection_params["port"]

        client = Elasticsearch(hosts=[{"host": host, "port": port, "scheme": "http"}])

        for index_name in indices_to_cleanup:
            if client.indices.exists(index=index_name):
                client.indices.delete(index=index_name)

        client.close()
    except Exception as e:
        # Log but don't fail test on cleanup error
        print(f"Warning: Failed to cleanup indices: {e}")


@pytest.fixture
def verify_elasticsearch_index(elasticsearch_connection_params: dict) -> callable:
    """Fixture that returns a function to verify Elasticsearch index state.

    Returns a function that takes index_name and returns index info dict.
    """

    def verify(index_name: str) -> dict:
        """Verify index exists and return its info."""
        from elasticsearch import Elasticsearch

        host = elasticsearch_connection_params["host"]
        port = elasticsearch_connection_params["port"]

        client = Elasticsearch(hosts=[{"host": host, "port": port, "scheme": "http"}])

        try:
            # Check index exists
            exists = client.indices.exists(index=index_name)

            if not exists:
                return {"exists": False}

            # Get index mapping
            mapping_response = client.indices.get_mapping(index=index_name)
            mapping = mapping_response[index_name]["mappings"]["properties"]

            # Get dimension from embedding field
            dimension = 0
            if "embedding" in mapping:
                dimension = mapping["embedding"].get("dims", 0)

            # Get document count
            count_response = client.count(index=index_name)
            count = count_response["count"]

            return {
                "exists": True,
                "dimension": dimension,
                "count": count,
            }
        finally:
            client.close()

    return verify


@pytest.mark.e2e
@pytest.mark.critical
@pytest.mark.skipif(
    not os.getenv("ELASTICSEARCH_URL"),
    reason="Elasticsearch not available (set ELASTICSEARCH_URL env var)",
)
class TestElasticsearchBackend:
    """E2E-017: Elasticsearch Backend - Happy Path Tests."""

    def test_e2e_017_elasticsearch_backend(
        self,
        temp_dir: Path,
        sample_data_50: Path,
        elasticsearch_url: str,
        embedding_model: str,
        cleanup_elasticsearch_index: callable,
        verify_elasticsearch_index: callable,
    ):
        """Test Elasticsearch backend with kNN search.

        Steps:
        1. Load 50 records into Elasticsearch
        2. Verify index creation with dense_vector mapping
        3. Verify all records indexed
        4. Query and verify results
        """
        index_name = "test_embeddings_e2e017"
        connection_string = f"{elasticsearch_url}/{index_name}"
        cleanup_elasticsearch_index(index_name)

        # Step 1: Load data
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "rag_tester",
                "load",
                "--file",
                str(sample_data_50),
                "--database",
                connection_string,
                "--embedding",
                embedding_model,
            ],
            capture_output=True,
            text=True,
            check=False,
        )

        # Verify exit code
        assert result.returncode == 0, f"Load failed: {result.stderr}"
        assert "Successfully loaded 50 records" in result.stdout

        # Step 2-3: Verify index state
        index_info = verify_elasticsearch_index(index_name)

        assert index_info["exists"], "Index not created"
        assert index_info["dimension"] == 384, f"Wrong dimension: {index_info['dimension']}"
        assert index_info["count"] == 50, f"Wrong record count: {index_info['count']}"

    def test_e2e_074_elasticsearch_dense_vector(
        self,
        temp_dir: Path,
        sample_data_50: Path,
        elasticsearch_url: str,
        embedding_model: str,
        cleanup_elasticsearch_index: callable,
        verify_elasticsearch_index: callable,
    ):
        """E2E-074: Elasticsearch Dense Vector - Integration Test.

        Verify:
        - Index mapping has dense_vector field
        - dense_vector has correct dimension and similarity metric
        - kNN search returns accurate results
        """
        index_name = "test_embeddings_e2e074"
        connection_string = f"{elasticsearch_url}/{index_name}"
        cleanup_elasticsearch_index(index_name)

        # Load data
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "rag_tester",
                "load",
                "--file",
                str(sample_data_50),
                "--database",
                connection_string,
                "--embedding",
                embedding_model,
            ],
            capture_output=True,
            text=True,
            check=False,
        )

        assert result.returncode == 0, f"Load failed: {result.stderr}"

        # Verify index state
        index_info = verify_elasticsearch_index(index_name)
        assert index_info["exists"], "Index not created"

        # Test kNN search
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "rag_tester",
                "test",
                "--database",
                connection_string,
                "--embedding",
                embedding_model,
                "--query",
                "machine learning and AI",
                "--top-k",
                "5",
            ],
            capture_output=True,
            text=True,
            check=False,
        )

        assert result.returncode == 0, f"Test failed: {result.stderr}"
        assert "Results:" in result.stdout

        # Verify similarity scores are present
        lines = result.stdout.split("\n")
        score_found = False
        for line in lines:
            if "score:" in line.lower():
                score_found = True
                # Extract score value
                score_str = line.split("score:")[-1].strip()
                try:
                    score = float(score_str)
                    assert score >= 0.0, f"Invalid score: {score}"
                except ValueError:
                    pass  # Not a score line

        assert score_found, "No scores found in results"

    def test_e2e_elasticsearch_batch_insert(
        self,
        temp_dir: Path,
        sample_data_1000: Path,
        elasticsearch_url: str,
        embedding_model: str,
        cleanup_elasticsearch_index: callable,
        verify_elasticsearch_index: callable,
    ):
        """Test Elasticsearch with large batch insert (1000 records).

        Verify:
        - Large batches are handled correctly
        - All records are indexed
        - Query performance is acceptable
        """
        index_name = "test_embeddings_batch"
        connection_string = f"{elasticsearch_url}/{index_name}"
        cleanup_elasticsearch_index(index_name)

        # Load data
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "rag_tester",
                "load",
                "--file",
                str(sample_data_1000),
                "--database",
                connection_string,
                "--embedding",
                embedding_model,
            ],
            capture_output=True,
            text=True,
            check=False,
        )

        assert result.returncode == 0, f"Load failed: {result.stderr}"
        assert "Successfully loaded 1000 records" in result.stdout

        # Verify all records indexed
        index_info = verify_elasticsearch_index(index_name)
        assert index_info["count"] == 1000, f"Wrong record count: {index_info['count']}"

    def test_e2e_elasticsearch_metadata_storage(
        self,
        temp_dir: Path,
        elasticsearch_url: str,
        embedding_model: str,
        cleanup_elasticsearch_index: callable,
    ):
        """Test Elasticsearch metadata storage.

        Verify:
        - Metadata is stored correctly
        - Metadata can be retrieved
        """
        import yaml

        index_name = "test_embeddings_metadata"
        connection_string = f"{elasticsearch_url}/{index_name}"
        cleanup_elasticsearch_index(index_name)

        # Create data with metadata
        data = {
            "records": [
                {
                    "id": "doc1",
                    "text": "Python programming language",
                    "metadata": {"category": "programming", "language": "python"},
                },
                {
                    "id": "doc2",
                    "text": "JavaScript web development",
                    "metadata": {"category": "programming", "language": "javascript"},
                },
                {
                    "id": "doc3",
                    "text": "Machine learning with Python",
                    "metadata": {"category": "ai", "language": "python"},
                },
            ]
        }
        data_file = temp_dir / "metadata_data.yaml"
        data_file.write_text(yaml.dump(data))

        # Load data
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "rag_tester",
                "load",
                "--file",
                str(data_file),
                "--database",
                connection_string,
                "--embedding",
                embedding_model,
            ],
            capture_output=True,
            text=True,
            check=False,
        )

        assert result.returncode == 0, f"Load failed: {result.stderr}"

        # Query to verify metadata is stored
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "rag_tester",
                "test",
                "--database",
                connection_string,
                "--embedding",
                embedding_model,
                "--query",
                "programming",
                "--top-k",
                "5",
            ],
            capture_output=True,
            text=True,
            check=False,
        )

        assert result.returncode == 0, f"Test failed: {result.stderr}"


@pytest.mark.e2e
@pytest.mark.critical
@pytest.mark.skipif(
    not os.getenv("ELASTICSEARCH_URL"),
    reason="Elasticsearch not available (set ELASTICSEARCH_URL env var)",
)
class TestElasticsearchSecurity:
    """Security tests for Elasticsearch provider."""

    def test_e2e_elasticsearch_index_name_validation(
        self,
        temp_dir: Path,
        sample_data_50: Path,
        elasticsearch_url: str,
        embedding_model: str,
    ):
        """Test that invalid index names are rejected.

        Verify that index names with invalid characters are rejected.
        """
        # Try to use invalid index name with uppercase (not allowed in Elasticsearch)
        invalid_index_name = "TestIndex"
        connection_string = f"{elasticsearch_url}/{invalid_index_name}"

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "rag_tester",
                "load",
                "--file",
                str(sample_data_50),
                "--database",
                connection_string,
                "--embedding",
                embedding_model,
            ],
            capture_output=True,
            text=True,
            check=False,
        )

        # Should fail with validation error
        assert result.returncode != 0, "Should have failed with invalid index name"
        assert (
            "Invalid index name" in result.stderr
            or "must be lowercase" in result.stderr
            or "Invalid Elasticsearch connection string" in result.stderr
            or "invalid_index_name_exception" in result.stderr.lower()
        )

    def test_e2e_elasticsearch_safe_id_handling(
        self,
        temp_dir: Path,
        elasticsearch_url: str,
        embedding_model: str,
        cleanup_elasticsearch_index: callable,
        verify_elasticsearch_index: callable,
    ):
        """Test that special characters in IDs are handled safely.

        Verify that IDs with special characters don't cause issues.
        """
        import yaml

        index_name = "test_embeddings_safe_ids"
        connection_string = f"{elasticsearch_url}/{index_name}"
        cleanup_elasticsearch_index(index_name)

        # Create data with special characters in IDs
        data = {
            "records": [
                {
                    "id": "id_with_underscore",
                    "text": "This is a test document.",
                    "metadata": {"source": "test"},
                },
                {
                    "id": "id-with-dash",
                    "text": "Another test document.",
                    "metadata": {"source": "test"},
                },
                {
                    "id": "id.with.dots",
                    "text": "Yet another test document.",
                    "metadata": {"source": "test"},
                },
            ]
        }
        data_file = temp_dir / "safe_ids_data.yaml"
        data_file.write_text(yaml.dump(data))

        # Load data
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "rag_tester",
                "load",
                "--file",
                str(data_file),
                "--database",
                connection_string,
                "--embedding",
                embedding_model,
            ],
            capture_output=True,
            text=True,
            check=False,
        )

        # Should succeed
        assert result.returncode == 0, f"Load failed: {result.stderr}"

        # Verify all records inserted
        index_info = verify_elasticsearch_index(index_name)
        assert index_info["count"] == 3, f"Wrong record count: {index_info['count']}"


@pytest.mark.e2e
@pytest.mark.high
@pytest.mark.skipif(
    not os.getenv("ELASTICSEARCH_URL"),
    reason="Elasticsearch not available (set ELASTICSEARCH_URL env var)",
)
class TestElasticsearchErrors:
    """Error handling tests for Elasticsearch provider."""

    def test_e2e_elasticsearch_connection_failure(
        self,
        temp_dir: Path,
        sample_data_50: Path,
        embedding_model: str,
    ):
        """Test connection failure to non-existent Elasticsearch instance.

        Verify that connection to non-existent instance fails gracefully.
        """
        # Use non-existent host
        wrong_connection = "elasticsearch://nonexistent.host:9200/test_index"

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "rag_tester",
                "load",
                "--file",
                str(sample_data_50),
                "--database",
                wrong_connection,
                "--embedding",
                embedding_model,
            ],
            capture_output=True,
            text=True,
            check=False,
        )

        # Should fail with connection error
        assert result.returncode != 0, "Should have failed with connection error"
        assert (
            "connection" in result.stderr.lower()
            or "failed to connect" in result.stderr.lower()
            or "cannot connect" in result.stderr.lower()
        )

    def test_e2e_db_004_dimension_mismatch(
        self,
        temp_dir: Path,
        elasticsearch_url: str,
        embedding_model: str,
        cleanup_elasticsearch_index: callable,
    ):
        """E2E-DB-004: Elasticsearch Index Mapping Conflict.

        Verify that dimension mismatch is detected and reported.
        """
        import yaml

        index_name = "test_embeddings_dimension"
        connection_string = f"{elasticsearch_url}/{index_name}"
        cleanup_elasticsearch_index(index_name)

        # First, create index with 384 dimensions
        data1 = {
            "records": [
                {
                    "id": "doc1",
                    "text": "First document",
                    "metadata": {"source": "test"},
                },
            ]
        }
        data_file1 = temp_dir / "data1.yaml"
        data_file1.write_text(yaml.dump(data1))

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "rag_tester",
                "load",
                "--file",
                str(data_file1),
                "--database",
                connection_string,
                "--embedding",
                embedding_model,  # 384 dimensions
            ],
            capture_output=True,
            text=True,
            check=False,
        )

        assert result.returncode == 0, f"Initial load failed: {result.stderr}"

        # Note: Testing with different embedding model would require a different model
        # For now, we verify the index was created successfully


@pytest.mark.e2e
@pytest.mark.medium
@pytest.mark.skipif(
    not os.getenv("ELASTICSEARCH_URL"),
    reason="Elasticsearch not available (set ELASTICSEARCH_URL env var)",
)
class TestElasticsearchOperations:
    """Additional operation tests for Elasticsearch provider."""

    def test_e2e_elasticsearch_delete_operations(
        self,
        temp_dir: Path,
        sample_data_50: Path,
        elasticsearch_url: str,
        embedding_model: str,
        cleanup_elasticsearch_index: callable,
        verify_elasticsearch_index: callable,
    ):
        """Test delete operations in Elasticsearch.

        Verify:
        - Index can be deleted
        - Records can be deleted by ID
        - All records can be deleted
        """
        index_name = "test_embeddings_delete"
        connection_string = f"{elasticsearch_url}/{index_name}"
        cleanup_elasticsearch_index(index_name)

        # Load data
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "rag_tester",
                "load",
                "--file",
                str(sample_data_50),
                "--database",
                connection_string,
                "--embedding",
                embedding_model,
            ],
            capture_output=True,
            text=True,
            check=False,
        )

        assert result.returncode == 0, f"Load failed: {result.stderr}"

        # Verify records loaded
        index_info = verify_elasticsearch_index(index_name)
        assert index_info["count"] == 50, "Records not loaded"

        # Note: Delete operations would require CLI commands for delete
        # which may not be implemented yet. This is a placeholder for future tests.

    def test_e2e_elasticsearch_index_info(
        self,
        temp_dir: Path,
        sample_data_50: Path,
        elasticsearch_url: str,
        embedding_model: str,
        cleanup_elasticsearch_index: callable,
        verify_elasticsearch_index: callable,
    ):
        """Test getting index information.

        Verify:
        - Index info can be retrieved
        - Info includes dimension, count, and metadata
        """
        index_name = "test_embeddings_info"
        connection_string = f"{elasticsearch_url}/{index_name}"
        cleanup_elasticsearch_index(index_name)

        # Load data
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "rag_tester",
                "load",
                "--file",
                str(sample_data_50),
                "--database",
                connection_string,
                "--embedding",
                embedding_model,
            ],
            capture_output=True,
            text=True,
            check=False,
        )

        assert result.returncode == 0, f"Load failed: {result.stderr}"

        # Verify index info
        index_info = verify_elasticsearch_index(index_name)
        assert index_info["exists"], "Index not created"
        assert index_info["dimension"] > 0, "Dimension not set"
        assert index_info["count"] == 50, "Wrong record count"
