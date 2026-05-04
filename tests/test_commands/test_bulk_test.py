"""Unit tests for bulk-test command."""

import json
from pathlib import Path

import pytest
import yaml

from rag_tester.commands._bulk_test_execution import validate_results as _validate_results
from rag_tester.commands._bulk_test_parsing import (
    parse_database_connection as _parse_database_connection,
)
from rag_tester.commands._bulk_test_parsing import (
    parse_test_file as _parse_test_file,
)
from rag_tester.commands._bulk_test_parsing import (
    validate_output_path as _validate_output_path,
)
from rag_tester.commands._bulk_test_parsing import (
    validate_parallel_workers as _validate_parallel_workers,
)
from rag_tester.commands._bulk_test_parsing import (
    validate_test_case as _validate_test_case,
)
from rag_tester.core.validator import ValidationError


class TestValidateTestCase:
    """Tests for _validate_test_case function."""

    def test_valid_test_case(self) -> None:
        """Test validation of a valid test case."""
        test = {
            "test_id": "test1",
            "query": "What is machine learning?",
            "expected": [
                {"id": "doc1", "text": "ML is...", "min_threshold": 0.85},
                {"id": "doc2", "text": "Machine learning..."},
            ],
        }
        # Should not raise
        _validate_test_case(test, 0)

    def test_missing_test_id(self) -> None:
        """Test validation fails when test_id is missing."""
        test = {
            "query": "What is ML?",
            "expected": [{"id": "doc1", "text": "ML is..."}],
        }
        with pytest.raises(ValidationError, match="Missing required field 'test_id'"):
            _validate_test_case(test, 0)

    def test_missing_query(self) -> None:
        """Test validation fails when query is missing."""
        test = {
            "test_id": "test1",
            "expected": [{"id": "doc1", "text": "ML is..."}],
        }
        with pytest.raises(ValidationError, match="Missing required field 'query'"):
            _validate_test_case(test, 0)

    def test_missing_expected(self) -> None:
        """Test validation fails when expected is missing."""
        test = {
            "test_id": "test1",
            "query": "What is ML?",
        }
        with pytest.raises(ValidationError, match="Missing required field 'expected'"):
            _validate_test_case(test, 0)

    def test_expected_not_list(self) -> None:
        """Test validation fails when expected is not a list."""
        test = {
            "test_id": "test1",
            "query": "What is ML?",
            "expected": "not a list",
        }
        with pytest.raises(ValidationError, match="'expected' must be a list"):
            _validate_test_case(test, 0)

    def test_expected_empty(self) -> None:
        """Test validation fails when expected is empty."""
        test = {
            "test_id": "test1",
            "query": "What is ML?",
            "expected": [],
        }
        with pytest.raises(ValidationError, match="'expected' cannot be empty"):
            _validate_test_case(test, 0)

    def test_expected_result_missing_id(self) -> None:
        """Test validation fails when expected result is missing id."""
        test = {
            "test_id": "test1",
            "query": "What is ML?",
            "expected": [{"text": "ML is..."}],
        }
        with pytest.raises(ValidationError, match="Missing 'id' in expected result"):
            _validate_test_case(test, 0)

    def test_expected_result_missing_text(self) -> None:
        """Test validation fails when expected result is missing text."""
        test = {
            "test_id": "test1",
            "query": "What is ML?",
            "expected": [{"id": "doc1"}],
        }
        with pytest.raises(ValidationError, match="Missing 'text' in expected result"):
            _validate_test_case(test, 0)

    def test_invalid_threshold_type(self) -> None:
        """Test validation fails when min_threshold is not a number."""
        test = {
            "test_id": "test1",
            "query": "What is ML?",
            "expected": [{"id": "doc1", "text": "ML is...", "min_threshold": "not a number"}],
        }
        with pytest.raises(ValidationError, match="'min_threshold' must be a number"):
            _validate_test_case(test, 0)

    def test_threshold_out_of_range(self) -> None:
        """Test validation fails when min_threshold is out of range."""
        test = {
            "test_id": "test1",
            "query": "What is ML?",
            "expected": [{"id": "doc1", "text": "ML is...", "min_threshold": 1.5}],
        }
        with pytest.raises(ValidationError, match="'min_threshold' must be between 0 and 1"):
            _validate_test_case(test, 0)


class TestValidateParallelWorkers:
    """Tests for _validate_parallel_workers function."""

    def test_valid_parallel_workers(self) -> None:
        """Test validation of valid parallel workers."""
        _validate_parallel_workers(1)
        _validate_parallel_workers(8)
        _validate_parallel_workers(16)

    def test_parallel_workers_too_low(self) -> None:
        """Test validation fails when parallel workers is too low."""
        with pytest.raises(ValidationError, match="Parallel workers must be between 1 and 16"):
            _validate_parallel_workers(0)

    def test_parallel_workers_too_high(self) -> None:
        """Test validation fails when parallel workers is too high."""
        with pytest.raises(ValidationError, match="Parallel workers must be between 1 and 16"):
            _validate_parallel_workers(17)

    def test_parallel_workers_not_int(self) -> None:
        """Test validation fails when parallel workers is not an integer."""
        with pytest.raises(ValidationError, match="Parallel workers must be an integer"):
            _validate_parallel_workers("4")  # type: ignore


class TestValidateOutputPath:
    """Tests for _validate_output_path function."""

    def test_valid_output_path(self, tmp_path: Path) -> None:
        """Test validation of valid output path."""
        output = tmp_path / "results.yaml"
        _validate_output_path(str(output))

    def test_output_directory_not_exists(self) -> None:
        """Test validation fails when output directory doesn't exist."""
        with pytest.raises(ValidationError, match="Output directory does not exist"):
            _validate_output_path("/nonexistent/directory/results.yaml")

    def test_output_parent_not_directory(self, tmp_path: Path) -> None:
        """Test validation fails when output parent is not a directory."""
        file_path = tmp_path / "file.txt"
        file_path.touch()
        output = file_path / "results.yaml"
        with pytest.raises(ValidationError, match="Output parent path is not a directory"):
            _validate_output_path(str(output))


class TestValidateResults:
    """Tests for _validate_results function."""

    def test_exact_order_match_passes(self) -> None:
        """Test validation passes when order matches exactly."""
        expected = [
            {"id": "doc1", "text": "Text 1"},
            {"id": "doc3", "text": "Text 3"},
        ]
        actual = [
            {"id": "doc1", "text": "Text 1", "score": 0.9},
            {"id": "doc2", "text": "Text 2", "score": 0.85},
            {"id": "doc3", "text": "Text 3", "score": 0.8},
        ]
        result = _validate_results(expected, actual)
        assert result["status"] == "passed"

    def test_wrong_order_fails(self) -> None:
        """Test validation fails when order is wrong."""
        expected = [
            {"id": "doc1", "text": "Text 1"},
            {"id": "doc3", "text": "Text 3"},
        ]
        actual = [
            {"id": "doc3", "text": "Text 3", "score": 0.9},
            {"id": "doc1", "text": "Text 1", "score": 0.85},
        ]
        result = _validate_results(expected, actual)
        assert result["status"] == "failed"
        assert "Expected order not matched" in result["reason"]

    def test_missing_id_fails(self) -> None:
        """Test validation fails when expected ID is missing."""
        expected = [
            {"id": "doc1", "text": "Text 1"},
            {"id": "doc3", "text": "Text 3"},
        ]
        actual = [
            {"id": "doc1", "text": "Text 1", "score": 0.9},
            {"id": "doc2", "text": "Text 2", "score": 0.85},
        ]
        result = _validate_results(expected, actual)
        assert result["status"] == "failed"
        assert "Expected ID 'doc3' not found" in result["reason"]

    def test_threshold_met_passes(self) -> None:
        """Test validation passes when threshold is met."""
        expected = [
            {"id": "doc1", "text": "Text 1", "min_threshold": 0.85},
        ]
        actual = [
            {"id": "doc1", "text": "Text 1", "score": 0.87},
        ]
        result = _validate_results(expected, actual)
        assert result["status"] == "passed"

    def test_threshold_not_met_fails(self) -> None:
        """Test validation fails when threshold is not met."""
        expected = [
            {"id": "doc1", "text": "Text 1", "min_threshold": 0.85},
        ]
        actual = [
            {"id": "doc1", "text": "Text 1", "score": 0.78},
        ]
        result = _validate_results(expected, actual)
        assert result["status"] == "failed"
        assert "Threshold not met" in result["reason"]
        assert "expected >= 0.85, got 0.78" in result["reason"]

    def test_zero_threshold_passes(self) -> None:
        """Test validation passes with zero threshold."""
        expected = [
            {"id": "doc1", "text": "Text 1", "min_threshold": 0.0},
        ]
        actual = [
            {"id": "doc1", "text": "Text 1", "score": 0.01},
        ]
        result = _validate_results(expected, actual)
        assert result["status"] == "passed"

    def test_no_threshold_always_passes(self) -> None:
        """Test validation passes when no threshold is specified."""
        expected = [
            {"id": "doc1", "text": "Text 1"},
        ]
        actual = [
            {"id": "doc1", "text": "Text 1", "score": 0.01},
        ]
        result = _validate_results(expected, actual)
        assert result["status"] == "passed"


class TestParseTestFile:
    """Tests for _parse_test_file function."""

    def test_parse_yaml_file(self, tmp_path: Path) -> None:
        """Test parsing a valid YAML test file."""
        test_file = tmp_path / "tests.yaml"
        test_data = {
            "tests": [
                {
                    "test_id": "test1",
                    "query": "What is ML?",
                    "expected": [{"id": "doc1", "text": "ML is..."}],
                }
            ]
        }
        with open(test_file, "w") as f:
            yaml.dump(test_data, f)

        tests = _parse_test_file(str(test_file))
        assert len(tests) == 1
        assert tests[0]["test_id"] == "test1"

    def test_parse_json_file(self, tmp_path: Path) -> None:
        """Test parsing a valid JSON test file."""
        test_file = tmp_path / "tests.json"
        test_data = {
            "tests": [
                {
                    "test_id": "test1",
                    "query": "What is ML?",
                    "expected": [{"id": "doc1", "text": "ML is..."}],
                }
            ]
        }
        with open(test_file, "w") as f:
            json.dump(test_data, f)

        tests = _parse_test_file(str(test_file))
        assert len(tests) == 1
        assert tests[0]["test_id"] == "test1"

    def test_file_not_found(self) -> None:
        """Test error when file doesn't exist."""
        with pytest.raises(ValidationError, match="File not found"):
            _parse_test_file("/nonexistent/file.yaml")

    def test_invalid_yaml(self, tmp_path: Path) -> None:
        """Test error when YAML is malformed."""
        test_file = tmp_path / "invalid.yaml"
        with open(test_file, "w") as f:
            f.write("invalid: yaml: content:")

        with pytest.raises(ValidationError, match="Failed to parse YAML"):
            _parse_test_file(str(test_file))

    def test_invalid_json(self, tmp_path: Path) -> None:
        """Test error when JSON is malformed."""
        test_file = tmp_path / "invalid.json"
        with open(test_file, "w") as f:
            f.write("{invalid json")

        with pytest.raises(ValidationError, match="Failed to parse JSON"):
            _parse_test_file(str(test_file))

    def test_missing_tests_key(self, tmp_path: Path) -> None:
        """Test error when 'tests' key is missing."""
        test_file = tmp_path / "tests.yaml"
        with open(test_file, "w") as f:
            yaml.dump({"data": []}, f)

        with pytest.raises(ValidationError, match="must contain a 'tests' key"):
            _parse_test_file(str(test_file))

    def test_tests_not_list(self, tmp_path: Path) -> None:
        """Test error when 'tests' is not a list."""
        test_file = tmp_path / "tests.yaml"
        with open(test_file, "w") as f:
            yaml.dump({"tests": "not a list"}, f)

        with pytest.raises(ValidationError, match="'tests' must be a list"):
            _parse_test_file(str(test_file))


class TestParseDatabaseConnection:
    """Tests for _parse_database_connection function."""

    def test_valid_connection_string(self) -> None:
        """Test parsing a valid connection string."""
        result = _parse_database_connection("chromadb://localhost:8000/my_collection")
        assert result is not None
        assert result["host"] == "localhost"
        assert result["port"] == 8000
        assert result["collection"] == "my_collection"

    def test_invalid_protocol(self) -> None:
        """Test error when protocol is not chromadb."""
        result = _parse_database_connection("postgres://localhost:5432/db")
        assert result is None

    def test_missing_collection(self) -> None:
        """Test error when collection is missing."""
        result = _parse_database_connection("chromadb://localhost:8000")
        assert result is None

    def test_missing_port(self) -> None:
        """Test error when port is missing."""
        result = _parse_database_connection("chromadb://localhost/collection")
        assert result is None

    def test_invalid_port(self) -> None:
        """Test error when port is not a number."""
        result = _parse_database_connection("chromadb://localhost:abc/collection")
        assert result is None
