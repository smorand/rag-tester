"""Error handling E2E tests.

Tests error scenarios and failure modes:
- E2E-033 to E2E-040: Load command errors
- E2E-041 to E2E-044: Test command errors
- E2E-045 to E2E-047: Bulk-test command errors
- E2E-048 to E2E-050: Compare command errors
"""

import subprocess
from pathlib import Path

import pytest


class TestErrorHandling:
    """Error handling and failure mode tests."""

    @pytest.mark.e2e
    @pytest.mark.critical
    def test_e2e_033_invalid_file_format(
        self,
        invalid_yaml_file: Path,
        chromadb_url: str,
        embedding_model: str,
    ):
        """E2E-033: Invalid File Format.

        Verifies:
        - Malformed YAML is detected
        - Exit code is 1
        - Error message is clear
        """
        result = subprocess.run(
            [
                "rag-tester",
                "load",
                "--file",
                str(invalid_yaml_file),
                "--database",
                f"{chromadb_url}/test_collection",
                "--embedding",
                embedding_model,
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )

        assert result.returncode == 1, "Expected failure for invalid YAML"
        assert "Invalid file format" in result.stderr or "Failed to parse" in result.stderr

    @pytest.mark.e2e
    @pytest.mark.critical
    def test_e2e_034_missing_required_fields(
        self,
        missing_fields_file: Path,
        chromadb_url: str,
        embedding_model: str,
    ):
        """E2E-034: Missing Required Fields.

        Verifies:
        - Missing 'text' or 'id' fields are detected
        - Exit code is 1
        - Error message identifies the problem
        """
        result = subprocess.run(
            [
                "rag-tester",
                "load",
                "--file",
                str(missing_fields_file),
                "--database",
                f"{chromadb_url}/test_collection",
                "--embedding",
                embedding_model,
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )

        assert result.returncode == 1, "Expected failure for missing fields"
        assert "Missing required field" in result.stderr or "required" in result.stderr.lower()

    @pytest.mark.e2e
    @pytest.mark.critical
    def test_e2e_035_database_unreachable(
        self,
        sample_data_file: Path,
        embedding_model: str,
    ):
        """E2E-035: Database Unreachable.

        Verifies:
        - Connection failure is detected
        - Exit code is 1
        - Error message indicates connection problem
        """
        # Use invalid port to simulate unreachable database
        result = subprocess.run(
            [
                "rag-tester",
                "load",
                "--file",
                str(sample_data_file),
                "--database",
                "chromadb://localhost:9999/test_collection",
                "--embedding",
                embedding_model,
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )

        assert result.returncode == 1, "Expected failure for unreachable database"
        assert "connection" in result.stderr.lower() or "failed" in result.stderr.lower()

    @pytest.mark.e2e
    @pytest.mark.critical
    @pytest.mark.skip(reason="Requires API key validation")
    def test_e2e_037_missing_api_key(
        self,
        sample_data_file: Path,
        chromadb_url: str,
        monkeypatch,
    ):
        """E2E-037: Missing API Key.

        Verifies:
        - Missing API key is detected
        - Exit code is 1
        - Error message identifies missing key
        """
        # Remove API key from environment
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

        result = subprocess.run(
            [
                "rag-tester",
                "load",
                "--file",
                str(sample_data_file),
                "--database",
                f"{chromadb_url}/test_collection",
                "--embedding",
                "openai/text-embedding-3-small",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )

        assert result.returncode == 1, "Expected failure for missing API key"
        assert "API key" in result.stderr or "OPENROUTER_API_KEY" in result.stderr

    @pytest.mark.e2e
    @pytest.mark.high
    def test_e2e_013_duplicate_ids(
        self,
        duplicate_ids_file: Path,
        chromadb_url: str,
        embedding_model: str,
    ):
        """E2E-013: Duplicate IDs.

        Verifies:
        - Duplicate IDs are detected
        - Only first occurrence is kept
        - Warning or info message is shown
        """
        result = subprocess.run(
            [
                "rag-tester",
                "load",
                "--file",
                str(duplicate_ids_file),
                "--database",
                f"{chromadb_url}/test_e2e_013",
                "--embedding",
                embedding_model,
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )

        # Should succeed but with warnings
        assert result.returncode == 0, f"Command failed: {result.stderr}"
        assert "2 records" in result.stdout or "duplicate" in result.stdout.lower()

    @pytest.mark.e2e
    @pytest.mark.critical
    def test_e2e_041_empty_database(
        self,
        chromadb_url: str,
        embedding_model: str,
    ):
        """E2E-041: Empty Database.

        Verifies:
        - Query against empty collection fails gracefully
        - Exit code is 1
        - Error message is clear
        """
        result = subprocess.run(
            [
                "rag-tester",
                "test",
                "What is machine learning?",
                "--database",
                f"{chromadb_url}/empty_collection",
                "--embedding",
                embedding_model,
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )

        assert result.returncode == 1, "Expected failure for empty database"
        assert "empty" in result.stderr.lower() or "no documents" in result.stderr.lower()

    @pytest.mark.e2e
    @pytest.mark.critical
    def test_e2e_045_malformed_test_file(
        self,
        invalid_yaml_file: Path,
        chromadb_url: str,
        embedding_model: str,
        temp_dir: Path,
    ):
        """E2E-045: Malformed Test File.

        Verifies:
        - Invalid test suite YAML is detected
        - Exit code is 1
        - Error message is clear
        """
        result = subprocess.run(
            [
                "rag-tester",
                "bulk-test",
                "--file",
                str(invalid_yaml_file),
                "--database",
                f"{chromadb_url}/test_collection",
                "--embedding",
                embedding_model,
                "--output",
                str(temp_dir / "results.yaml"),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )

        assert result.returncode == 1, "Expected failure for malformed test file"
        assert "Invalid" in result.stderr or "parse" in result.stderr.lower()

    @pytest.mark.e2e
    @pytest.mark.critical
    def test_e2e_048_missing_result_file(
        self,
        temp_dir: Path,
    ):
        """E2E-048: Missing Result File.

        Verifies:
        - Non-existent result file is detected
        - Exit code is 1
        - Error message identifies missing file
        """
        result = subprocess.run(
            [
                "rag-tester",
                "compare",
                "--results",
                str(temp_dir / "nonexistent1.yaml"),
                str(temp_dir / "nonexistent2.yaml"),
                "--output",
                str(temp_dir / "comparison.yaml"),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )

        assert result.returncode == 1, "Expected failure for missing result file"
        assert "not found" in result.stderr.lower() or "does not exist" in result.stderr.lower()

    @pytest.mark.e2e
    @pytest.mark.high
    def test_e2e_081_empty_input_file(
        self,
        temp_dir: Path,
        chromadb_url: str,
        embedding_model: str,
    ):
        """E2E-081: Empty Input File.

        Verifies:
        - Empty YAML file is detected
        - Exit code is 1
        - Error message is clear
        """
        empty_file = temp_dir / "empty.yaml"
        empty_file.write_text("")

        result = subprocess.run(
            [
                "rag-tester",
                "load",
                "--file",
                str(empty_file),
                "--database",
                f"{chromadb_url}/test_collection",
                "--embedding",
                embedding_model,
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )

        assert result.returncode == 1, "Expected failure for empty file"
        assert "empty" in result.stderr.lower() or "no records" in result.stderr.lower()
