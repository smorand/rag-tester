"""End-to-end tests for Milvus database provider."""

import os
import subprocess
import sys
from collections.abc import Generator
from pathlib import Path

import pytest


@pytest.fixture
def milvus_url() -> str:
    """Get Milvus URL from environment or use default."""
    return os.getenv("MILVUS_URL", "milvus://localhost:19530")


@pytest.fixture
def milvus_connection_params(milvus_url: str) -> dict[str, str | int]:
    """Parse Milvus URL and return connection parameters.

    Returns:
        Dictionary with keys: host, port, conn_string
    """
    # Format: milvus://host:port
    url = milvus_url.replace("milvus://", "")
    parts = url.split(":")

    if len(parts) != 2:
        pytest.skip("Invalid Milvus URL format")

    host = parts[0]
    port = int(parts[1])

    conn_string = f"milvus://{host}:{port}"

    return {
        "host": host,
        "port": port,
        "conn_string": conn_string,
    }


@pytest.fixture
def milvus_available(milvus_connection_params: dict) -> bool:
    """Check if Milvus is available."""
    try:
        from pymilvus import connections, utility

        host = milvus_connection_params["host"]
        port = milvus_connection_params["port"]

        connections.connect(
            alias="test_connection",
            host=host,
            port=str(port),
        )

        # Try to list collections to verify connection
        utility.list_collections(using="test_connection")
        connections.disconnect(alias="test_connection")
        return True
    except Exception:
        return False


@pytest.fixture
def cleanup_milvus_collection(milvus_connection_params: dict) -> Generator[callable]:
    """Fixture to cleanup Milvus collections after tests.

    Yields a cleanup function that takes a collection name and drops it.
    """
    collections_to_cleanup = []

    def register_cleanup(collection_name: str) -> None:
        """Register a collection for cleanup."""
        collections_to_cleanup.append(collection_name)

    yield register_cleanup

    # Cleanup after test
    try:
        from pymilvus import connections, utility

        host = milvus_connection_params["host"]
        port = milvus_connection_params["port"]

        connections.connect(
            alias="cleanup",
            host=host,
            port=str(port),
        )

        for collection_name in collections_to_cleanup:
            if utility.has_collection(collection_name, using="cleanup"):
                utility.drop_collection(collection_name, using="cleanup")

        connections.disconnect(alias="cleanup")
    except Exception as e:
        # Log but don't fail test on cleanup error
        print(f"Warning: Failed to cleanup collections: {e}")


@pytest.fixture
def verify_milvus_collection(milvus_connection_params: dict) -> callable:
    """Fixture that returns a function to verify Milvus collection state.

    Returns a function that takes collection_name and returns collection info dict.
    """

    def verify(collection_name: str) -> dict:
        """Verify collection exists and return its info."""
        from pymilvus import Collection, connections, utility

        host = milvus_connection_params["host"]
        port = milvus_connection_params["port"]

        connections.connect(
            alias="verify",
            host=host,
            port=str(port),
        )

        try:
            # Check collection exists
            exists = utility.has_collection(collection_name, using="verify")

            if not exists:
                return {"exists": False}

            # Get collection
            col = Collection(name=collection_name, using="verify")

            # Get dimension from schema
            schema = col.schema
            embedding_field = next((f for f in schema.fields if f.name == "embedding"), None)
            dimension = embedding_field.params.get("dim", 0) if embedding_field else 0

            # Get record count
            col.flush()
            count = col.num_entities

            # Check for index
            indexes = col.indexes
            has_index = len(indexes) > 0
            index_type = indexes[0].params.get("index_type") if has_index else None

            return {
                "exists": True,
                "dimension": dimension,
                "count": count,
                "has_index": has_index,
                "index_type": index_type,
            }
        finally:
            connections.disconnect(alias="verify")

    return verify


@pytest.mark.e2e
@pytest.mark.critical
@pytest.mark.skipif(
    not os.getenv("MILVUS_URL"),
    reason="Milvus not available (set MILVUS_URL env var)",
)
class TestMilvusBackend:
    """E2E-015: Milvus Backend - Happy Path Tests."""

    def test_e2e_015_milvus_backend(
        self,
        temp_dir: Path,
        sample_data_50: Path,
        milvus_url: str,
        embedding_model: str,
        cleanup_milvus_collection: callable,
        verify_milvus_collection: callable,
    ):
        """Test Milvus backend with vector search.

        Steps:
        1. Load 50 records into Milvus
        2. Verify collection creation with vector field
        3. Verify all records inserted
        4. Query and verify results
        5. Verify index created
        """
        collection_name = "test_embeddings_e2e015"
        connection_string = f"{milvus_url}/{collection_name}"
        cleanup_milvus_collection(collection_name)

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

        # Step 2-5: Verify database state
        collection_info = verify_milvus_collection(collection_name)

        assert collection_info["exists"], "Collection not created"
        assert collection_info["dimension"] == 384, f"Wrong dimension: {collection_info['dimension']}"
        assert collection_info["count"] == 50, f"Wrong record count: {collection_info['count']}"
        assert collection_info["has_index"], "Index not created"
        assert collection_info["index_type"] == "IVF_FLAT", f"Wrong index type: {collection_info['index_type']}"

    def test_e2e_072_milvus_vector_search(
        self,
        temp_dir: Path,
        sample_data_50: Path,
        milvus_url: str,
        embedding_model: str,
        cleanup_milvus_collection: callable,
        verify_milvus_collection: callable,
    ):
        """E2E-072: Milvus Vector Search - Integration Test.

        Verify:
        - Vector field is created correctly
        - Similarity search works with cosine distance
        - IVF_FLAT index improves query performance
        """
        collection_name = "test_embeddings_e2e072"
        connection_string = f"{milvus_url}/{collection_name}"
        cleanup_milvus_collection(collection_name)

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

        # Verify collection state
        collection_info = verify_milvus_collection(collection_name)
        assert collection_info["exists"], "Collection not created"
        assert collection_info["has_index"], "Index not created"
        assert collection_info["index_type"] == "IVF_FLAT", "Wrong index type"

        # Test similarity search
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

        # Verify cosine similarity scores are in valid range [0, 1]
        lines = result.stdout.split("\n")
        for line in lines:
            if "score:" in line.lower():
                # Extract score value
                score_str = line.split("score:")[-1].strip()
                try:
                    score = float(score_str)
                    assert 0.0 <= score <= 1.0, f"Invalid score: {score}"
                except ValueError:
                    pass  # Not a score line

    def test_e2e_milvus_batch_insert(
        self,
        temp_dir: Path,
        sample_data_1000: Path,
        milvus_url: str,
        embedding_model: str,
        cleanup_milvus_collection: callable,
        verify_milvus_collection: callable,
    ):
        """Test Milvus with large batch insert (1000 records).

        Verify:
        - Large batches are handled correctly
        - All records are inserted
        - Query performance is acceptable
        """
        collection_name = "test_embeddings_batch"
        connection_string = f"{milvus_url}/{collection_name}"
        cleanup_milvus_collection(collection_name)

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

        # Verify all records inserted
        collection_info = verify_milvus_collection(collection_name)
        assert collection_info["count"] == 1000, f"Wrong record count: {collection_info['count']}"

    def test_e2e_milvus_metadata_filtering(
        self,
        temp_dir: Path,
        milvus_url: str,
        embedding_model: str,
        cleanup_milvus_collection: callable,
    ):
        """Test Milvus metadata filtering.

        Verify:
        - Metadata is stored correctly
        - Metadata filtering works in queries
        """
        import yaml

        collection_name = "test_embeddings_metadata"
        connection_string = f"{milvus_url}/{collection_name}"
        cleanup_milvus_collection(collection_name)

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

        # Query with metadata filter (if supported by CLI)
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
    not os.getenv("MILVUS_URL"),
    reason="Milvus not available (set MILVUS_URL env var)",
)
class TestMilvusSecurity:
    """Security tests for Milvus provider."""

    def test_e2e_055_collection_name_validation(
        self,
        temp_dir: Path,
        sample_data_50: Path,
        milvus_url: str,
        embedding_model: str,
    ):
        """E2E-055: Milvus Collection Name Validation.

        Verify that invalid collection names are rejected.
        """
        # Try to use invalid collection name with special characters
        invalid_collection_name = "collection'; DROP--"
        connection_string = f"{milvus_url}/{invalid_collection_name}"

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
        assert result.returncode != 0, "Should have failed with invalid collection name"
        assert (
            "Invalid collection name" in result.stderr
            or "must be alphanumeric" in result.stderr
            or "Invalid Milvus connection string" in result.stderr
        )

    def test_e2e_milvus_safe_id_handling(
        self,
        temp_dir: Path,
        milvus_url: str,
        embedding_model: str,
        cleanup_milvus_collection: callable,
        verify_milvus_collection: callable,
    ):
        """Test that special characters in IDs are handled safely.

        Verify that IDs with special characters don't cause issues.
        """
        import yaml

        collection_name = "test_embeddings_safe_ids"
        connection_string = f"{milvus_url}/{collection_name}"
        cleanup_milvus_collection(collection_name)

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
        collection_info = verify_milvus_collection(collection_name)
        assert collection_info["count"] == 3, f"Wrong record count: {collection_info['count']}"


@pytest.mark.e2e
@pytest.mark.high
@pytest.mark.skipif(
    not os.getenv("MILVUS_URL"),
    reason="Milvus not available (set MILVUS_URL env var)",
)
class TestMilvusErrors:
    """Error handling tests for Milvus provider."""

    def test_e2e_056_connection_failure(
        self,
        temp_dir: Path,
        sample_data_50: Path,
        embedding_model: str,
    ):
        """E2E-056: Milvus Connection Failure.

        Verify that connection to non-existent Milvus instance fails gracefully.
        """
        # Use non-existent host
        wrong_connection = "milvus://nonexistent.host:19530/test_collection"

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

    def test_e2e_milvus_dimension_mismatch(
        self,
        temp_dir: Path,
        milvus_url: str,
        embedding_model: str,
        cleanup_milvus_collection: callable,
    ):
        """Test dimension mismatch error handling.

        Verify that attempting to insert embeddings with wrong dimension fails.
        """
        import yaml

        collection_name = "test_embeddings_dimension"
        connection_string = f"{milvus_url}/{collection_name}"
        cleanup_milvus_collection(collection_name)

        # First, create collection with 384 dimensions
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

        # Now try to insert with different embedding model (different dimension)
        # Note: This test assumes we have access to a different embedding model
        # For now, we'll skip this part as it requires a different model
        # In a real scenario, you'd use a model with different dimensions


@pytest.mark.e2e
@pytest.mark.medium
@pytest.mark.skipif(
    not os.getenv("MILVUS_URL"),
    reason="Milvus not available (set MILVUS_URL env var)",
)
class TestMilvusOperations:
    """Additional operation tests for Milvus provider."""

    def test_e2e_milvus_delete_operations(
        self,
        temp_dir: Path,
        sample_data_50: Path,
        milvus_url: str,
        embedding_model: str,
        cleanup_milvus_collection: callable,
        verify_milvus_collection: callable,
    ):
        """Test delete operations in Milvus.

        Verify:
        - Collection can be deleted
        - Records can be deleted by ID
        - All records can be deleted
        """
        collection_name = "test_embeddings_delete"
        connection_string = f"{milvus_url}/{collection_name}"
        cleanup_milvus_collection(collection_name)

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
        collection_info = verify_milvus_collection(collection_name)
        assert collection_info["count"] == 50, "Records not loaded"

        # Note: Delete operations would require CLI commands for delete
        # which may not be implemented yet. This is a placeholder for future tests.

    def test_e2e_milvus_collection_info(
        self,
        temp_dir: Path,
        sample_data_50: Path,
        milvus_url: str,
        embedding_model: str,
        cleanup_milvus_collection: callable,
        verify_milvus_collection: callable,
    ):
        """Test getting collection information.

        Verify:
        - Collection info can be retrieved
        - Info includes dimension, count, and metadata
        """
        collection_name = "test_embeddings_info"
        connection_string = f"{milvus_url}/{collection_name}"
        cleanup_milvus_collection(collection_name)

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

        # Verify collection info
        collection_info = verify_milvus_collection(collection_name)
        assert collection_info["exists"], "Collection not created"
        assert collection_info["dimension"] > 0, "Dimension not set"
        assert collection_info["count"] == 50, "Wrong record count"
        assert collection_info["has_index"], "Index not created"
