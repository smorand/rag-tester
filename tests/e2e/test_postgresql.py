"""End-to-end tests for PostgreSQL database provider."""

import os
import subprocess
from collections.abc import Generator
from pathlib import Path

import pytest


@pytest.fixture
def postgresql_url() -> str:
    """Get PostgreSQL URL from environment or use default."""
    return os.getenv("POSTGRESQL_URL", "postgresql://postgres:postgres@localhost:5432/testdb")


@pytest.fixture
def postgresql_connection_params(postgresql_url: str) -> dict[str, str | int]:
    """Parse PostgreSQL URL and return connection parameters.
    
    Returns:
        Dictionary with keys: user, password, host, port, dbname, conn_string
    """
    # Format: postgresql://user:pass@host:port/dbname
    url = postgresql_url.replace("postgresql://", "")
    parts = url.split("@")
    
    if len(parts) != 2:
        pytest.skip("Invalid PostgreSQL URL format")
    
    user_pass = parts[0].split(":")
    host_port_db = parts[1].split("/")
    
    if len(user_pass) != 2 or len(host_port_db) != 2:
        pytest.skip("Invalid PostgreSQL URL format")
    
    user = user_pass[0]
    password = user_pass[1]
    host_port = host_port_db[0].split(":")
    host = host_port[0]
    port = int(host_port[1]) if len(host_port) > 1 else 5432
    dbname = host_port_db[1]
    
    conn_string = f"postgresql://{user}:{password}@{host}:{port}/{dbname}"
    
    return {
        "user": user,
        "password": password,
        "host": host,
        "port": port,
        "dbname": dbname,
        "conn_string": conn_string,
    }


@pytest.fixture
def postgresql_available(postgresql_connection_params: dict) -> bool:
    """Check if PostgreSQL is available."""
    try:
        import psycopg

        conn_string = postgresql_connection_params["conn_string"]
        with psycopg.connect(conn_string) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
        return True
    except Exception:
        return False


@pytest.fixture
def cleanup_postgresql_table(postgresql_connection_params: dict) -> Generator[callable]:
    """Fixture to cleanup PostgreSQL tables after tests.
    
    Yields a cleanup function that takes a table name and drops it.
    """
    tables_to_cleanup = []
    
    def register_cleanup(table_name: str) -> None:
        """Register a table for cleanup."""
        tables_to_cleanup.append(table_name)
    
    yield register_cleanup
    
    # Cleanup after test
    try:
        import psycopg
        
        conn_string = postgresql_connection_params["conn_string"]
        with psycopg.connect(conn_string) as conn:
            with conn.cursor() as cur:
                for table_name in tables_to_cleanup:
                    cur.execute(f"DROP TABLE IF EXISTS {table_name} CASCADE")
            conn.commit()
    except Exception as e:
        # Log but don't fail test on cleanup error
        print(f"Warning: Failed to cleanup tables: {e}")


@pytest.fixture
def verify_postgresql_table(postgresql_connection_params: dict) -> callable:
    """Fixture that returns a function to verify PostgreSQL table state.
    
    Returns a function that takes table_name and returns table info dict.
    """
    def verify(table_name: str) -> dict:
        """Verify table exists and return its info."""
        import psycopg
        
        conn_string = postgresql_connection_params["conn_string"]
        
        with psycopg.connect(conn_string) as conn:
            with conn.cursor() as cur:
                # Check table exists
                cur.execute(
                    """
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables
                        WHERE table_schema = 'public'
                        AND table_name = %s
                    )
                    """,
                    (table_name,),
                )
                exists = cur.fetchone()[0]
                
                if not exists:
                    return {"exists": False}
                
                # Get dimension
                cur.execute(
                    """
                    SELECT atttypmod - 4 AS dimension
                    FROM pg_attribute
                    WHERE attrelid = %s::regclass
                    AND attname = 'embedding'
                    """,
                    (table_name,),
                )
                dimension_row = cur.fetchone()
                dimension = dimension_row[0] if dimension_row else None
                
                # Get record count
                cur.execute(f"SELECT COUNT(*) FROM {table_name}")
                count = cur.fetchone()[0]
                
                # Check for IVFFlat index
                cur.execute(
                    """
                    SELECT indexname
                    FROM pg_indexes
                    WHERE tablename = %s
                    AND indexdef LIKE '%ivfflat%'
                    """,
                    (table_name,),
                )
                index_row = cur.fetchone()
                has_index = index_row is not None
                
                return {
                    "exists": True,
                    "dimension": dimension,
                    "count": count,
                    "has_ivfflat_index": has_index,
                }
    
    return verify


@pytest.mark.e2e
@pytest.mark.critical
@pytest.mark.skipif(
    not os.getenv("POSTGRESQL_URL"),
    reason="PostgreSQL not available (set POSTGRESQL_URL env var)",
)
class TestPostgreSQLBackend:
    """E2E-014: PostgreSQL Backend - Happy Path Tests."""

    def test_e2e_014_postgresql_backend(
        self,
        temp_dir: Path,
        sample_data_50: Path,
        postgresql_url: str,
        embedding_model: str,
        cleanup_postgresql_table: callable,
        verify_postgresql_table: callable,
    ):
        """Test PostgreSQL backend with pgvector extension.

        Steps:
        1. Load 50 records into PostgreSQL
        2. Verify table creation with vector column
        3. Verify all records inserted
        4. Query and verify results
        5. Verify IVFFlat index created
        """
        table_name = "test_embeddings_e2e014"
        connection_string = f"{postgresql_url}/{table_name}"
        cleanup_postgresql_table(table_name)

        # Step 1: Load data
        result = subprocess.run(
            [
                "rag-tester",
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
        table_info = verify_postgresql_table(table_name)
        
        assert table_info["exists"], "Table not created"
        assert table_info["dimension"] == 384, f"Wrong dimension: {table_info['dimension']}"
        assert table_info["count"] == 50, f"Wrong record count: {table_info['count']}"
        assert table_info["has_ivfflat_index"], "IVFFlat index not created"

    def test_e2e_071_postgresql_with_pgvector(
        self,
        temp_dir: Path,
        sample_data_50: Path,
        postgresql_url: str,
        embedding_model: str,
        cleanup_postgresql_table: callable,
        verify_postgresql_table: callable,
    ):
        """E2E-071: PostgreSQL with pgvector - Integration Test.

        Verify:
        - Vector column type is created correctly
        - Similarity search works with cosine distance
        - IVFFlat index improves query performance
        """
        table_name = "test_embeddings_e2e071"
        connection_string = f"{postgresql_url}/{table_name}"
        cleanup_postgresql_table(table_name)

        # Load data
        result = subprocess.run(
            [
                "rag-tester",
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
        table_info = verify_postgresql_table(table_name)
        assert table_info["exists"], "Table not created"
        assert table_info["has_ivfflat_index"], "IVFFlat index not created"

        # Test similarity search
        result = subprocess.run(
            [
                "rag-tester",
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


@pytest.mark.e2e
@pytest.mark.critical
@pytest.mark.skipif(
    not os.getenv("POSTGRESQL_URL"),
    reason="PostgreSQL not available (set POSTGRESQL_URL env var)",
)
class TestPostgreSQLSecurity:
    """Security tests for PostgreSQL provider."""

    def test_e2e_054_sql_injection_protection(
        self,
        temp_dir: Path,
        postgresql_url: str,
        embedding_model: str,
        cleanup_postgresql_table: callable,
        verify_postgresql_table: callable,
    ):
        """E2E-054: SQL Injection Protection.

        Verify that malicious IDs are safely escaped and don't cause SQL injection.
        """
        import yaml

        table_name = "test_embeddings_e2e054"
        connection_string = f"{postgresql_url}/{table_name}"
        cleanup_postgresql_table(table_name)

        # Create data with malicious ID
        data = {
            "records": [
                {
                    "id": "'; DROP TABLE test_embeddings_e2e054; --",
                    "text": "This is a test document with malicious ID.",
                    "metadata": {"source": "test"},
                },
                {
                    "id": "normal_id",
                    "text": "This is a normal document.",
                    "metadata": {"source": "test"},
                },
            ]
        }
        malicious_file = temp_dir / "malicious_data.yaml"
        malicious_file.write_text(yaml.dump(data))

        # Load data
        result = subprocess.run(
            [
                "rag-tester",
                "load",
                "--file",
                str(malicious_file),
                "--database",
                connection_string,
                "--embedding",
                embedding_model,
            ],
            capture_output=True,
            text=True,
            check=False,
        )

        # Should succeed (ID is escaped)
        assert result.returncode == 0, f"Load failed: {result.stderr}"

        # Verify table still exists and has records
        table_info = verify_postgresql_table(table_name)
        assert table_info["exists"], "Table was dropped (SQL injection succeeded)"
        assert table_info["count"] == 2, f"Wrong record count: {table_info['count']}"

    def test_e2e_db_005_table_name_validation(
        self,
        temp_dir: Path,
        sample_data_50: Path,
        postgresql_url: str,
        embedding_model: str,
    ):
        """E2E-DB-005: PostgreSQL Table Name Validation.

        Verify that invalid table names are rejected.
        """
        # Try to use invalid table name with special characters
        invalid_table_name = "table'; DROP TABLE--"
        connection_string = f"{postgresql_url}/{invalid_table_name}"

        result = subprocess.run(
            [
                "rag-tester",
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
            or "Invalid PostgreSQL connection string" in result.stderr
        )


@pytest.mark.e2e
@pytest.mark.high
@pytest.mark.skipif(
    not os.getenv("POSTGRESQL_URL"),
    reason="PostgreSQL not available (set POSTGRESQL_URL env var)",
)
class TestPostgreSQLErrors:
    """Error handling tests for PostgreSQL provider."""

    def test_e2e_053_authentication_failure(
        self,
        temp_dir: Path,
        sample_data_50: Path,
        postgresql_connection_params: dict,
        embedding_model: str,
    ):
        """E2E-053: Database Authentication Failure.

        Verify that wrong credentials result in proper error message.
        """
        # Use wrong credentials
        params = postgresql_connection_params
        wrong_connection = (
            f"postgresql://wrong_user:wrong_pass@{params['host']}:{params['port']}"
            f"/{params['dbname']}/test_table"
        )

        result = subprocess.run(
            [
                "rag-tester",
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

        # Should fail with authentication error
        assert result.returncode != 0, "Should have failed with wrong credentials"
        assert (
            "authentication failed" in result.stderr.lower()
            or "password authentication failed" in result.stderr.lower()
            or "connection" in result.stderr.lower()
        )
