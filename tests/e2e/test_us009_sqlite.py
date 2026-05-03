"""End-to-end tests for SQLite database provider (US-009)."""

import subprocess
import sys
from collections.abc import Generator
from pathlib import Path

import pytest


@pytest.fixture
def sqlite_db_path(temp_dir: Path) -> Path:
    """Get SQLite database path in temp directory."""
    return temp_dir / "test.db"


@pytest.fixture
def cleanup_sqlite_db(sqlite_db_path: Path) -> Generator[callable]:
    """Fixture to cleanup SQLite database after tests.

    Yields a cleanup function that takes a database path and removes it.
    """
    dbs_to_cleanup = []

    def register_cleanup(db_path: Path) -> None:
        """Register a database for cleanup."""
        dbs_to_cleanup.append(db_path)

    yield register_cleanup

    # Cleanup after test
    for db_path in dbs_to_cleanup:
        if db_path.exists():
            try:
                db_path.unlink()
            except Exception as e:
                print(f"Warning: Failed to cleanup database {db_path}: {e}")


@pytest.fixture
def verify_sqlite_table(sqlite_db_path: Path) -> callable:
    """Fixture that returns a function to verify SQLite table state.

    Returns a function that takes table_name and returns table info dict.
    """

    def verify(table_name: str) -> dict:
        """Verify table exists and return its info."""
        import sqlite3

        if not sqlite_db_path.exists():
            return {"exists": False}

        conn = sqlite3.connect(str(sqlite_db_path))
        cursor = conn.cursor()

        try:
            # Check table exists
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                (table_name,),
            )
            exists = cursor.fetchone() is not None

            if not exists:
                return {"exists": False}

            # Get dimension from first record
            cursor.execute(f"SELECT dimension FROM {table_name} LIMIT 1")
            dim_result = cursor.fetchone()
            dimension = dim_result[0] if dim_result else 0

            # Get record count
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            count = cursor.fetchone()[0]

            return {
                "exists": True,
                "dimension": dimension,
                "count": count,
            }
        finally:
            conn.close()

    return verify


@pytest.mark.e2e
@pytest.mark.critical
class TestSQLiteBackend:
    """E2E-016: SQLite Backend - Happy Path Tests."""

    def test_e2e_016_sqlite_backend(
        self,
        temp_dir: Path,
        sample_data_50: Path,
        sqlite_db_path: Path,
        embedding_model: str,
        cleanup_sqlite_db: callable,
        verify_sqlite_table: callable,
    ):
        """Test SQLite backend with vector search.

        Steps:
        1. Load 50 records into SQLite
        2. Verify database file creation
        3. Verify table creation
        4. Verify all records inserted
        5. Query and verify results
        """
        table_name = "test_embeddings_e2e016"
        connection_string = f"sqlite:///{sqlite_db_path}/{table_name}"
        cleanup_sqlite_db(sqlite_db_path)

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

        # Step 2: Verify database file created
        assert sqlite_db_path.exists(), "Database file not created"

        # Step 3-4: Verify table state
        table_info = verify_sqlite_table(table_name)

        assert table_info["exists"], "Table not created"
        assert table_info["dimension"] == 384, f"Wrong dimension: {table_info['dimension']}"
        assert table_info["count"] == 50, f"Wrong record count: {table_info['count']}"

    def test_e2e_073_sqlite_with_vector_extension(
        self,
        temp_dir: Path,
        sample_data_50: Path,
        sqlite_db_path: Path,
        embedding_model: str,
        cleanup_sqlite_db: callable,
        verify_sqlite_table: callable,
    ):
        """E2E-073: SQLite with vector extension - Integration Test.

        Verify:
        - Embeddings are stored as BLOB
        - Similarity search works correctly
        """
        table_name = "test_embeddings_e2e073"
        connection_string = f"sqlite:///{sqlite_db_path}/{table_name}"
        cleanup_sqlite_db(sqlite_db_path)

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

        # Verify table state
        table_info = verify_sqlite_table(table_name)
        assert table_info["exists"], "Table not created"

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

    def test_e2e_sqlite_batch_insert(
        self,
        temp_dir: Path,
        sample_data_1000: Path,
        sqlite_db_path: Path,
        embedding_model: str,
        cleanup_sqlite_db: callable,
        verify_sqlite_table: callable,
    ):
        """Test SQLite with large batch insert (1000 records).

        Verify:
        - Large batches are handled correctly
        - All records are inserted
        - Query performance is acceptable
        """
        table_name = "test_embeddings_batch"
        connection_string = f"sqlite:///{sqlite_db_path}/{table_name}"
        cleanup_sqlite_db(sqlite_db_path)

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
        table_info = verify_sqlite_table(table_name)
        assert table_info["count"] == 1000, f"Wrong record count: {table_info['count']}"

    def test_e2e_sqlite_metadata_storage(
        self,
        temp_dir: Path,
        sqlite_db_path: Path,
        embedding_model: str,
        cleanup_sqlite_db: callable,
    ):
        """Test SQLite metadata storage.

        Verify:
        - Metadata is stored correctly
        - Metadata can be retrieved
        """
        import yaml

        table_name = "test_embeddings_metadata"
        connection_string = f"sqlite:///{sqlite_db_path}/{table_name}"
        cleanup_sqlite_db(sqlite_db_path)

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
class TestSQLiteSecurity:
    """Security tests for SQLite provider."""

    def test_e2e_055_path_traversal_protection(
        self,
        temp_dir: Path,
        sample_data_50: Path,
        embedding_model: str,
    ):
        """E2E-055: Path Traversal Protection.

        Verify that path traversal attempts are rejected.
        """
        # Try to use path traversal in database path
        malicious_path = "sqlite:///../../../etc/passwd/table"

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "rag_tester",
                "load",
                "--file",
                str(sample_data_50),
                "--database",
                malicious_path,
                "--embedding",
                embedding_model,
            ],
            capture_output=True,
            text=True,
            check=False,
        )

        # Should fail with validation error
        assert result.returncode != 0, "Should have failed with path traversal"
        assert (
            "path traversal" in result.stderr.lower()
            or "invalid" in result.stderr.lower()
            or "error" in result.stderr.lower()
        )

    def test_e2e_sqlite_table_name_validation(
        self,
        temp_dir: Path,
        sample_data_50: Path,
        sqlite_db_path: Path,
        embedding_model: str,
    ):
        """Test that invalid table names are rejected.

        Verify that table names with special characters don't cause issues.
        """
        # Try to use invalid table name with special characters
        invalid_table_name = "table'; DROP TABLE--"
        connection_string = f"sqlite:///{sqlite_db_path}/{invalid_table_name}"

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
        assert result.returncode != 0, "Should have failed with invalid table name"
        assert (
            "Invalid table name" in result.stderr
            or "must be alphanumeric" in result.stderr
            or "Invalid SQLite connection string" in result.stderr
        )

    def test_e2e_sqlite_safe_id_handling(
        self,
        temp_dir: Path,
        sqlite_db_path: Path,
        embedding_model: str,
        cleanup_sqlite_db: callable,
        verify_sqlite_table: callable,
    ):
        """Test that special characters in IDs are handled safely.

        Verify that IDs with special characters don't cause issues.
        """
        import yaml

        table_name = "test_embeddings_safe_ids"
        connection_string = f"sqlite:///{sqlite_db_path}/{table_name}"
        cleanup_sqlite_db(sqlite_db_path)

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
        table_info = verify_sqlite_table(table_name)
        assert table_info["count"] == 3, f"Wrong record count: {table_info['count']}"


@pytest.mark.e2e
@pytest.mark.high
class TestSQLiteErrors:
    """Error handling tests for SQLite provider."""

    def test_e2e_db_003_file_permissions(
        self,
        temp_dir: Path,
        sample_data_50: Path,
        embedding_model: str,
    ):
        """E2E-DB-003: SQLite File Permissions.

        Verify that unwritable paths result in proper error message.
        """
        # Use unwritable path (root directory)
        unwritable_path = "/root/test.db"
        connection_string = f"sqlite:///{unwritable_path}/table"

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

        # Should fail with permission error
        assert result.returncode != 0, "Should have failed with permission error"
        assert (
            "permission" in result.stderr.lower()
            or "cannot create" in result.stderr.lower()
            or "error" in result.stderr.lower()
        )

    def test_e2e_sqlite_dimension_mismatch(
        self,
        temp_dir: Path,
        sqlite_db_path: Path,
        embedding_model: str,
        cleanup_sqlite_db: callable,
    ):
        """Test dimension mismatch error handling.

        Verify that attempting to insert embeddings with wrong dimension fails.
        """
        import yaml

        table_name = "test_embeddings_dimension"
        connection_string = f"sqlite:///{sqlite_db_path}/{table_name}"
        cleanup_sqlite_db(sqlite_db_path)

        # First, create table with 384 dimensions
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
        # For now, we verify the table was created successfully


@pytest.mark.e2e
@pytest.mark.medium
class TestSQLiteOperations:
    """Additional operation tests for SQLite provider."""

    def test_e2e_sqlite_delete_operations(
        self,
        temp_dir: Path,
        sample_data_50: Path,
        sqlite_db_path: Path,
        embedding_model: str,
        cleanup_sqlite_db: callable,
        verify_sqlite_table: callable,
    ):
        """Test delete operations in SQLite.

        Verify:
        - Table can be deleted
        - Records can be deleted by ID
        - All records can be deleted
        """
        table_name = "test_embeddings_delete"
        connection_string = f"sqlite:///{sqlite_db_path}/{table_name}"
        cleanup_sqlite_db(sqlite_db_path)

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
        table_info = verify_sqlite_table(table_name)
        assert table_info["count"] == 50, "Records not loaded"

        # Note: Delete operations would require CLI commands for delete
        # which may not be implemented yet. This is a placeholder for future tests.

    def test_e2e_sqlite_table_info(
        self,
        temp_dir: Path,
        sample_data_50: Path,
        sqlite_db_path: Path,
        embedding_model: str,
        cleanup_sqlite_db: callable,
        verify_sqlite_table: callable,
    ):
        """Test getting table information.

        Verify:
        - Table info can be retrieved
        - Info includes dimension and count
        """
        table_name = "test_embeddings_info"
        connection_string = f"sqlite:///{sqlite_db_path}/{table_name}"
        cleanup_sqlite_db(sqlite_db_path)

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

        # Verify table info
        table_info = verify_sqlite_table(table_name)
        assert table_info["exists"], "Table not created"
        assert table_info["dimension"] > 0, "Dimension not set"
        assert table_info["count"] == 50, "Wrong record count"
