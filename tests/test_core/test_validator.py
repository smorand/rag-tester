"""Tests for core.validator module."""

import pytest
from pathlib import Path
from rag_tester.core.validator import (
    ValidationError,
    validate_batch_size,
    validate_file_path,
    validate_load_mode,
    validate_parallel_workers,
    validate_record,
)


class TestValidateFilePath:
    """Tests for validate_file_path function."""

    def test_valid_yaml_file(self, tmp_path: Path) -> None:
        """Test validation of valid YAML file."""
        file_path = tmp_path / "test.yaml"
        file_path.write_text("test: data")
        
        result = validate_file_path(str(file_path))
        assert result == file_path

    def test_valid_yml_file(self, tmp_path: Path) -> None:
        """Test validation of valid YML file."""
        file_path = tmp_path / "test.yml"
        file_path.write_text("test: data")
        
        result = validate_file_path(str(file_path))
        assert result == file_path

    def test_valid_json_file(self, tmp_path: Path) -> None:
        """Test validation of valid JSON file."""
        file_path = tmp_path / "test.json"
        file_path.write_text('{"test": "data"}')
        
        result = validate_file_path(str(file_path))
        assert result == file_path

    def test_file_not_found(self) -> None:
        """Test validation fails for non-existent file."""
        with pytest.raises(ValidationError, match="File not found"):
            validate_file_path("/nonexistent/file.yaml")

    def test_path_is_directory(self, tmp_path: Path) -> None:
        """Test validation fails for directory."""
        with pytest.raises(ValidationError, match="Path is not a file"):
            validate_file_path(str(tmp_path))

    def test_unsupported_file_format(self, tmp_path: Path) -> None:
        """Test validation fails for unsupported file format."""
        file_path = tmp_path / "test.txt"
        file_path.write_text("test data")
        
        with pytest.raises(ValidationError, match="Unsupported file format"):
            validate_file_path(str(file_path))


class TestValidateBatchSize:
    """Tests for validate_batch_size function."""

    def test_valid_batch_size_min(self) -> None:
        """Test validation of minimum batch size."""
        validate_batch_size(1)  # Should not raise

    def test_valid_batch_size_max(self) -> None:
        """Test validation of maximum batch size."""
        validate_batch_size(256)  # Should not raise

    def test_valid_batch_size_middle(self) -> None:
        """Test validation of middle-range batch size."""
        validate_batch_size(32)  # Should not raise

    def test_batch_size_too_small(self) -> None:
        """Test validation fails for batch size < 1."""
        with pytest.raises(ValidationError, match="Batch size must be between 1 and 256"):
            validate_batch_size(0)

    def test_batch_size_too_large(self) -> None:
        """Test validation fails for batch size > 256."""
        with pytest.raises(ValidationError, match="Batch size must be between 1 and 256"):
            validate_batch_size(257)


class TestValidateParallelWorkers:
    """Tests for validate_parallel_workers function."""

    def test_valid_parallel_workers_min(self) -> None:
        """Test validation of minimum parallel workers."""
        validate_parallel_workers(1)  # Should not raise

    def test_valid_parallel_workers_max(self) -> None:
        """Test validation of maximum parallel workers."""
        validate_parallel_workers(16)  # Should not raise

    def test_valid_parallel_workers_middle(self) -> None:
        """Test validation of middle-range parallel workers."""
        validate_parallel_workers(4)  # Should not raise

    def test_parallel_workers_too_small(self) -> None:
        """Test validation fails for parallel workers < 1."""
        with pytest.raises(ValidationError, match="Parallel workers must be between 1 and 16"):
            validate_parallel_workers(0)

    def test_parallel_workers_too_large(self) -> None:
        """Test validation fails for parallel workers > 16."""
        with pytest.raises(ValidationError, match="Parallel workers must be between 1 and 16"):
            validate_parallel_workers(17)


class TestValidateRecord:
    """Tests for validate_record function."""

    def test_valid_record(self) -> None:
        """Test validation of valid record."""
        record = {"id": "doc1", "text": "Sample text"}
        validate_record(record, 0)  # Should not raise

    def test_valid_record_with_metadata(self) -> None:
        """Test validation of valid record with metadata."""
        record = {
            "id": "doc1",
            "text": "Sample text",
            "metadata": {"key": "value"},
        }
        validate_record(record, 0)  # Should not raise

    def test_record_not_dict(self) -> None:
        """Test validation fails for non-dict record."""
        with pytest.raises(ValidationError, match="not a dictionary"):
            validate_record("not a dict", 0)  # type: ignore[arg-type]

    def test_record_missing_id(self) -> None:
        """Test validation fails for record missing id."""
        record = {"text": "Sample text"}
        with pytest.raises(ValidationError, match="Missing required field 'id'"):
            validate_record(record, 0)

    def test_record_missing_text(self) -> None:
        """Test validation fails for record missing text."""
        record = {"id": "doc1"}
        with pytest.raises(ValidationError, match="Missing required field 'text' in record 'doc1'"):
            validate_record(record, 0)

    def test_record_id_not_string(self) -> None:
        """Test validation fails for non-string id."""
        record = {"id": 123, "text": "Sample text"}
        with pytest.raises(ValidationError, match="Field 'id' must be a string"):
            validate_record(record, 0)  # type: ignore[dict-item]

    def test_record_text_not_string(self) -> None:
        """Test validation fails for non-string text."""
        record = {"id": "doc1", "text": 123}
        with pytest.raises(ValidationError, match="Field 'text' must be a string"):
            validate_record(record, 0)  # type: ignore[dict-item]


class TestValidateLoadMode:
    """Tests for validate_load_mode function."""

    def test_valid_mode_initial(self) -> None:
        """Test validation of 'initial' mode."""
        validate_load_mode("initial")  # Should not raise

    def test_valid_mode_upsert(self) -> None:
        """Test validation of 'upsert' mode."""
        validate_load_mode("upsert")  # Should not raise

    def test_valid_mode_flush(self) -> None:
        """Test validation of 'flush' mode."""
        validate_load_mode("flush")  # Should not raise

    def test_invalid_mode(self) -> None:
        """Test validation fails for invalid mode."""
        with pytest.raises(ValidationError, match="Invalid load mode"):
            validate_load_mode("invalid")
