"""Tests for compare command."""

from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from rag_tester.rag_tester import app

runner = CliRunner()


@pytest.fixture
def temp_result_files(tmp_path: Path) -> tuple[Path, Path]:
    """Create temporary result files for testing."""
    result_a = tmp_path / "results_a.yaml"
    result_b = tmp_path / "results_b.yaml"

    data_a = {
        "summary": {
            "total_tests": 10,
            "passed": 8,
            "failed": 2,
            "total_tokens": 0,
            "total_time": 5.2,
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            "database": "chromadb://localhost:8000/collection_a",
        },
        "tests": [
            {
                "test_id": "test1",
                "status": "passed",
                "actual": [{"id": "doc1", "score": 0.9}],
            },
            {
                "test_id": "test2",
                "status": "failed",
                "actual": [{"id": "doc2", "score": 0.7}],
            },
        ],
    }

    data_b = {
        "summary": {
            "total_tests": 10,
            "passed": 7,
            "failed": 3,
            "total_tokens": 0,
            "total_time": 6.1,
            "embedding_model": "sentence-transformers/all-mpnet-base-v2",
            "database": "chromadb://localhost:8000/collection_b",
        },
        "tests": [
            {
                "test_id": "test1",
                "status": "passed",
                "actual": [{"id": "doc1", "score": 0.88}],
            },
            {
                "test_id": "test2",
                "status": "passed",
                "actual": [{"id": "doc2", "score": 0.86}],
            },
        ],
    }

    with open(result_a, "w") as f:
        yaml.dump(data_a, f)

    with open(result_b, "w") as f:
        yaml.dump(data_b, f)

    return result_a, result_b


class TestCompareCommand:
    """Tests for compare command."""

    def test_compare_basic(self, temp_result_files: tuple[Path, Path], tmp_path: Path) -> None:
        """Test basic compare command execution."""
        result_a, result_b = temp_result_files
        output = tmp_path / "comparison.yaml"

        result = runner.invoke(
            app,
            [
                "compare",
                str(result_a),
                str(result_b),
                "--output",
                str(output),
            ],
        )

        assert result.exit_code == 0
        assert output.exists()

        # Verify output file structure
        with open(output) as f:
            comparison = yaml.safe_load(f)

        assert "model_a" in comparison
        assert "model_b" in comparison
        assert "per_test_diff" in comparison

    def test_compare_missing_result_file(self, temp_result_files: tuple[Path, Path], tmp_path: Path) -> None:
        """Test compare with missing result file."""
        result_a, _ = temp_result_files
        output = tmp_path / "comparison.yaml"

        result = runner.invoke(
            app,
            [
                "compare",
                str(result_a),
                "nonexistent.yaml",
                "--output",
                str(output),
            ],
        )

        assert result.exit_code == 1
        assert "Result file not found" in result.stdout

    def test_compare_single_file(self, temp_result_files: tuple[Path, Path], tmp_path: Path) -> None:
        """Test compare with only one result file."""
        result_a, _ = temp_result_files
        output = tmp_path / "comparison.yaml"

        result = runner.invoke(
            app,
            [
                "compare",
                str(result_a),
                "--output",
                str(output),
            ],
        )

        assert result.exit_code == 1
        assert "At least 2 result files required" in result.stdout

    def test_compare_invalid_yaml(self, tmp_path: Path) -> None:
        """Test compare with invalid YAML file."""
        invalid_file = tmp_path / "invalid.yaml"
        invalid_file.write_text("{ invalid yaml content")

        valid_file = tmp_path / "valid.yaml"
        valid_file.write_text("summary:\n  total_tests: 1\ntests: []")

        output = tmp_path / "comparison.yaml"

        result = runner.invoke(
            app,
            [
                "compare",
                str(invalid_file),
                str(valid_file),
                "--output",
                str(output),
            ],
        )

        assert result.exit_code == 1
        assert "Invalid result file format" in result.stdout

    def test_compare_missing_summary(self, tmp_path: Path) -> None:
        """Test compare with result file missing summary section."""
        invalid_file = tmp_path / "no_summary.yaml"
        invalid_file.write_text("tests: []")

        valid_file = tmp_path / "valid.yaml"
        valid_file.write_text("summary:\n  total_tests: 1\ntests: []")

        output = tmp_path / "comparison.yaml"

        result = runner.invoke(
            app,
            [
                "compare",
                str(invalid_file),
                str(valid_file),
                "--output",
                str(output),
            ],
        )

        assert result.exit_code == 1
        assert "Missing 'summary' section" in result.stdout

    def test_compare_incompatible_test_suites(self, tmp_path: Path) -> None:
        """Test compare with incompatible test suites."""
        file_a = tmp_path / "results_a.yaml"
        file_b = tmp_path / "results_b.yaml"

        data_a = {
            "summary": {"total_tests": 10},
            "tests": [],
        }

        data_b = {
            "summary": {"total_tests": 15},
            "tests": [],
        }

        with open(file_a, "w") as f:
            yaml.dump(data_a, f)

        with open(file_b, "w") as f:
            yaml.dump(data_b, f)

        output = tmp_path / "comparison.yaml"

        result = runner.invoke(
            app,
            [
                "compare",
                str(file_a),
                str(file_b),
                "--output",
                str(output),
            ],
        )

        assert result.exit_code == 1
        assert "Test suites do not match" in result.stdout

    def test_compare_three_models(self, tmp_path: Path) -> None:
        """Test compare with three models."""
        files = []
        for i in range(3):
            file = tmp_path / f"results_{i}.yaml"
            data = {
                "summary": {
                    "total_tests": 5,
                    "passed": 4,
                    "failed": 1,
                    "total_tokens": 0,
                    "total_time": 2.0,
                    "embedding_model": f"model_{i}",
                    "database": "chromadb://localhost:8000/test",
                },
                "tests": [],
            }
            with open(file, "w") as f:
                yaml.dump(data, f)
            files.append(file)

        output = tmp_path / "comparison.yaml"

        result = runner.invoke(
            app,
            [
                "compare",
                str(files[0]),
                str(files[1]),
                str(files[2]),
                "--output",
                str(output),
            ],
        )

        assert result.exit_code == 0
        assert output.exists()

        # Verify output has three models
        with open(output) as f:
            comparison = yaml.safe_load(f)

        assert "model_a" in comparison
        assert "model_b" in comparison
        assert "model_c" in comparison

    def test_compare_output_directory_not_exists(self, temp_result_files: tuple[Path, Path], tmp_path: Path) -> None:
        """Test compare with output directory that doesn't exist."""
        result_a, result_b = temp_result_files
        output = tmp_path / "nonexistent" / "comparison.yaml"

        result = runner.invoke(
            app,
            [
                "compare",
                str(result_a),
                str(result_b),
                "--output",
                str(output),
            ],
        )

        assert result.exit_code == 1
        assert "Output directory does not exist" in result.stdout

    def test_compare_with_cost_calculation(self, tmp_path: Path) -> None:
        """Test compare includes cost calculation for API models."""
        file_a = tmp_path / "results_api.yaml"
        file_b = tmp_path / "results_local.yaml"

        data_a = {
            "summary": {
                "total_tests": 10,
                "passed": 8,
                "failed": 2,
                "total_tokens": 15000,
                "total_time": 8.1,
                "embedding_model": "openai/text-embedding-3-small",
                "database": "chromadb://localhost:8000/test",
            },
            "tests": [],
        }

        data_b = {
            "summary": {
                "total_tests": 10,
                "passed": 7,
                "failed": 3,
                "total_tokens": 0,
                "total_time": 5.2,
                "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
                "database": "chromadb://localhost:8000/test",
            },
            "tests": [],
        }

        with open(file_a, "w") as f:
            yaml.dump(data_a, f)

        with open(file_b, "w") as f:
            yaml.dump(data_b, f)

        output = tmp_path / "comparison.yaml"

        result = runner.invoke(
            app,
            [
                "compare",
                str(file_a),
                str(file_b),
                "--output",
                str(output),
            ],
        )

        assert result.exit_code == 0

        # Verify cost is calculated
        with open(output) as f:
            comparison = yaml.safe_load(f)

        # API model should have cost
        assert comparison["model_a"]["total_cost"] == 0.0003
        assert comparison["model_a"]["cost_per_test"] == 0.00003

        # Local model should have zero cost
        assert comparison["model_b"]["total_cost"] == 0.0
        assert comparison["model_b"]["cost_per_test"] == 0.0
