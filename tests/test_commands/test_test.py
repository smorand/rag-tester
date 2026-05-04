"""Unit tests for the test command."""

from unittest.mock import AsyncMock

import pytest
from typer.testing import CliRunner

from rag_tester.core.tester import TestError, ValidationError
from rag_tester.providers.databases.base import DatabaseError
from rag_tester.providers.embeddings.base import EmbeddingError
from rag_tester.rag_tester import app

runner = CliRunner()


@pytest.fixture
def mock_tester(mocker):
    """Mock the Tester class."""
    mock_class = mocker.patch("rag_tester.commands.test.Tester")
    mock_instance = mock_class.return_value

    # test_query is async, must be an AsyncMock
    mock_instance.test_query = AsyncMock(
        return_value={
            "query": "test query",
            "results": [
                {"rank": 1, "id": "doc1", "text": "First result", "score": 0.95},
                {"rank": 2, "id": "doc2", "text": "Second result", "score": 0.87},
            ],
            "tokens": 0,
            "time": 0.123,
        }
    )

    # Mock format_results to return formatted string
    mock_instance.format_results.return_value = "Formatted output"

    return mock_instance


@pytest.fixture
def mock_providers(mocker):
    """Mock embedding and database providers."""
    mock_embedding = mocker.patch("rag_tester.commands.test.LocalEmbeddingProvider")
    mock_db = mocker.patch("rag_tester.commands.test.ChromaDBProvider")
    return mock_embedding, mock_db


class TestTestCommand:
    """Test the test command."""

    def test_command_success_table_format(self, mock_tester, mock_providers):
        """Test successful query with table format."""
        result = runner.invoke(
            app,
            [
                "test",
                "What is machine learning?",
                "--database",
                "chromadb://localhost:8000/test_collection",
                "--embedding",
                "sentence-transformers/all-MiniLM-L6-v2",
            ],
        )

        assert result.exit_code == 0
        assert "Formatted output" in result.stdout
        mock_tester.test_query.assert_called_once()
        mock_tester.format_results.assert_called_once()

    def test_command_success_json_format(self, mock_tester, mock_providers):
        """Test successful query with JSON format."""
        result = runner.invoke(
            app,
            [
                "test",
                "test query",
                "--database",
                "chromadb://localhost:8000/test_collection",
                "--embedding",
                "sentence-transformers/all-MiniLM-L6-v2",
                "--format",
                "json",
            ],
        )

        assert result.exit_code == 0
        mock_tester.test_query.assert_called_once_with(
            query="test query",
            top_k=5,
            output_format="json",
        )

    def test_command_success_custom_top_k(self, mock_tester, mock_providers):
        """Test successful query with custom top-k."""
        result = runner.invoke(
            app,
            [
                "test",
                "test query",
                "--database",
                "chromadb://localhost:8000/test_collection",
                "--embedding",
                "sentence-transformers/all-MiniLM-L6-v2",
                "--top-k",
                "10",
            ],
        )

        assert result.exit_code == 0
        mock_tester.test_query.assert_called_once_with(
            query="test query",
            top_k=10,
            output_format="table",
        )

    def test_command_invalid_database_format(self, mock_providers):
        """Test command with invalid database connection string."""
        result = runner.invoke(
            app,
            [
                "test",
                "test query",
                "--database",
                "invalid://connection",
                "--embedding",
                "sentence-transformers/all-MiniLM-L6-v2",
            ],
        )

        assert result.exit_code == 1
        assert "Unsupported database" in result.stderr

    def test_command_invalid_database_format_missing_port(self, mock_providers):
        """Test command with database string missing port."""
        result = runner.invoke(
            app,
            [
                "test",
                "test query",
                "--database",
                "chromadb://localhost/test_collection",
                "--embedding",
                "sentence-transformers/all-MiniLM-L6-v2",
            ],
        )

        assert result.exit_code == 1
        assert "Invalid database connection string" in result.stderr

    def test_command_invalid_port_number(self, mock_providers):
        """Test command with invalid port number."""
        result = runner.invoke(
            app,
            [
                "test",
                "test query",
                "--database",
                "chromadb://localhost:abc/test_collection",
                "--embedding",
                "sentence-transformers/all-MiniLM-L6-v2",
            ],
        )

        assert result.exit_code == 1
        assert "Invalid port number" in result.stderr

    def test_command_embedding_load_failure(self, mocker, mock_providers):
        """Test command when embedding model fails to load."""
        mock_embedding, _ = mock_providers
        mock_embedding.side_effect = Exception("Model not found")

        result = runner.invoke(
            app,
            [
                "test",
                "test query",
                "--database",
                "chromadb://localhost:8000/test_collection",
                "--embedding",
                "invalid/model",
            ],
        )

        assert result.exit_code == 1
        assert "Failed to load embedding model" in result.stderr

    def test_command_database_connection_failure(self, mocker, mock_providers):
        """Test command when database connection fails."""
        _, mock_db = mock_providers
        mock_db.side_effect = Exception("Connection refused")

        result = runner.invoke(
            app,
            [
                "test",
                "test query",
                "--database",
                "chromadb://localhost:8000/test_collection",
                "--embedding",
                "sentence-transformers/all-MiniLM-L6-v2",
            ],
        )

        assert result.exit_code == 1
        assert "Database connection failed" in result.stderr

    def test_command_validation_error(self, mocker, mock_tester, mock_providers):
        """Test command when validation error occurs."""
        mock_tester.test_query.side_effect = ValidationError("Query cannot be empty")

        result = runner.invoke(
            app,
            [
                "test",
                "",
                "--database",
                "chromadb://localhost:8000/test_collection",
                "--embedding",
                "sentence-transformers/all-MiniLM-L6-v2",
            ],
        )

        assert result.exit_code == 1
        assert "Query cannot be empty" in result.stderr

    def test_command_test_error(self, mocker, mock_tester, mock_providers):
        """Test command when test error occurs."""
        mock_tester.test_query.side_effect = TestError("Database is empty")

        result = runner.invoke(
            app,
            [
                "test",
                "test query",
                "--database",
                "chromadb://localhost:8000/test_collection",
                "--embedding",
                "sentence-transformers/all-MiniLM-L6-v2",
            ],
        )

        assert result.exit_code == 1
        assert "Database is empty" in result.stderr

    def test_command_database_error(self, mocker, mock_tester, mock_providers):
        """Test command when database error occurs."""
        mock_tester.test_query.side_effect = DatabaseError("Query failed")

        result = runner.invoke(
            app,
            [
                "test",
                "test query",
                "--database",
                "chromadb://localhost:8000/test_collection",
                "--embedding",
                "sentence-transformers/all-MiniLM-L6-v2",
            ],
        )

        assert result.exit_code == 1
        assert "Database operation failed" in result.stderr

    def test_command_embedding_error(self, mocker, mock_tester, mock_providers):
        """Test command when embedding error occurs."""
        mock_tester.test_query.side_effect = EmbeddingError("Embedding failed")

        result = runner.invoke(
            app,
            [
                "test",
                "test query",
                "--database",
                "chromadb://localhost:8000/test_collection",
                "--embedding",
                "sentence-transformers/all-MiniLM-L6-v2",
            ],
        )

        assert result.exit_code == 1
        assert "Embedding generation failed" in result.stderr

    def test_command_unexpected_error(self, mocker, mock_tester, mock_providers):
        """Test command when unexpected error occurs."""
        mock_tester.test_query.side_effect = RuntimeError("Unexpected error")

        result = runner.invoke(
            app,
            [
                "test",
                "test query",
                "--database",
                "chromadb://localhost:8000/test_collection",
                "--embedding",
                "sentence-transformers/all-MiniLM-L6-v2",
            ],
        )

        assert result.exit_code == 1
        assert "Unexpected error" in result.stderr
