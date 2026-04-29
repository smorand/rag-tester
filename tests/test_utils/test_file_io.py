"""Tests for file I/O utilities."""

import json
from pathlib import Path

import pytest
import yaml

from rag_tester.utils.file_io import ValidationError, read_json, read_yaml


@pytest.fixture
def temp_yaml_file(tmp_path: Path) -> Path:
    """Create a temporary YAML file with test data."""
    file_path = tmp_path / "test_data.yaml"
    data = {
        "records": [
            {"id": "doc1", "text": "First document"},
            {"id": "doc2", "text": "Second document"},
            {"id": "doc3", "text": "Third document"},
        ]
    }
    with open(file_path, "w") as f:
        yaml.dump(data, f)
    return file_path


@pytest.fixture
def temp_json_file(tmp_path: Path) -> Path:
    """Create a temporary JSON file with test data."""
    file_path = tmp_path / "test_data.json"
    data = {
        "records": [
            {"id": "doc1", "text": "First document"},
            {"id": "doc2", "text": "Second document"},
            {"id": "doc3", "text": "Third document"},
        ]
    }
    with open(file_path, "w") as f:
        json.dump(data, f)
    return file_path


@pytest.fixture
def empty_yaml_file(tmp_path: Path) -> Path:
    """Create an empty YAML file."""
    file_path = tmp_path / "empty.yaml"
    file_path.write_text("")
    return file_path


@pytest.fixture
def invalid_yaml_file(tmp_path: Path) -> Path:
    """Create a YAML file with missing required fields."""
    file_path = tmp_path / "invalid.yaml"
    data = {
        "records": [
            {"id": "doc1"},  # Missing 'text' field
        ]
    }
    with open(file_path, "w") as f:
        yaml.dump(data, f)
    return file_path


@pytest.fixture
def yaml_file_no_records(tmp_path: Path) -> Path:
    """Create a YAML file with no records key."""
    file_path = tmp_path / "no_records.yaml"
    data = {"other_key": "value"}
    with open(file_path, "w") as f:
        yaml.dump(data, f)
    return file_path


class TestReadYaml:
    """Tests for read_yaml function."""

    async def test_read_yaml_success(self, temp_yaml_file: Path) -> None:
        """Test reading a valid YAML file."""
        records = []
        async for record in read_yaml(temp_yaml_file):
            records.append(record)

        assert len(records) == 3
        assert records[0]["id"] == "doc1"
        assert records[0]["text"] == "First document"
        assert records[1]["id"] == "doc2"
        assert records[2]["id"] == "doc3"

    async def test_read_yaml_empty_file(self, empty_yaml_file: Path) -> None:
        """Test reading an empty YAML file raises ValidationError."""
        with pytest.raises(ValidationError, match="Input file is empty or has no records"):
            async for _ in read_yaml(empty_yaml_file):
                pass

    async def test_read_yaml_missing_text_field(self, invalid_yaml_file: Path) -> None:
        """Test reading YAML with missing text field raises ValidationError."""
        with pytest.raises(ValidationError, match="Missing required field 'text' in record 'doc1'"):
            async for _ in read_yaml(invalid_yaml_file):
                pass

    async def test_read_yaml_no_records_key(self, yaml_file_no_records: Path) -> None:
        """Test reading YAML without records key raises ValidationError."""
        with pytest.raises(ValidationError, match="Input file is empty or has no records"):
            async for _ in read_yaml(yaml_file_no_records):
                pass

    async def test_read_yaml_file_not_found(self, tmp_path: Path) -> None:
        """Test reading non-existent file raises FileNotFoundError."""
        non_existent = tmp_path / "does_not_exist.yaml"
        with pytest.raises(FileNotFoundError):
            async for _ in read_yaml(non_existent):
                pass

    async def test_read_yaml_with_unicode(self, tmp_path: Path) -> None:
        """Test reading YAML with Unicode and emoji."""
        file_path = tmp_path / "unicode.yaml"
        data = {
            "records": [
                {"id": "doc1", "text": "Hello 世界 🌍 مرحبا"},
            ]
        }
        with open(file_path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, allow_unicode=True)

        records = []
        async for record in read_yaml(file_path):
            records.append(record)

        assert len(records) == 1
        assert records[0]["text"] == "Hello 世界 🌍 مرحبا"

    async def test_read_yaml_with_long_text(self, tmp_path: Path) -> None:
        """Test reading YAML with very long text (10K chars)."""
        file_path = tmp_path / "long_text.yaml"
        long_text = "a" * 10000
        data = {
            "records": [
                {"id": "doc1", "text": long_text},
            ]
        }
        with open(file_path, "w") as f:
            yaml.dump(data, f)

        records = []
        async for record in read_yaml(file_path):
            records.append(record)

        assert len(records) == 1
        assert len(records[0]["text"]) == 10000
        assert records[0]["text"] == long_text


class TestReadJson:
    """Tests for read_json function."""

    async def test_read_json_success(self, temp_json_file: Path) -> None:
        """Test reading a valid JSON file."""
        records = []
        async for record in read_json(temp_json_file):
            records.append(record)

        assert len(records) == 3
        assert records[0]["id"] == "doc1"
        assert records[0]["text"] == "First document"
        assert records[1]["id"] == "doc2"
        assert records[2]["id"] == "doc3"

    async def test_read_json_empty_file(self, tmp_path: Path) -> None:
        """Test reading an empty JSON file raises ValidationError."""
        file_path = tmp_path / "empty.json"
        file_path.write_text("")

        with pytest.raises(ValidationError):
            async for _ in read_json(file_path):
                pass

    async def test_read_json_missing_text_field(self, tmp_path: Path) -> None:
        """Test reading JSON with missing text field raises ValidationError."""
        file_path = tmp_path / "invalid.json"
        data = {
            "records": [
                {"id": "doc1"},  # Missing 'text' field
            ]
        }
        with open(file_path, "w") as f:
            json.dump(data, f)

        with pytest.raises(ValidationError, match="Missing required field 'text' in record 'doc1'"):
            async for _ in read_json(file_path):
                pass

    async def test_read_json_no_records_key(self, tmp_path: Path) -> None:
        """Test reading JSON without records key raises ValidationError."""
        file_path = tmp_path / "no_records.json"
        data = {"other_key": "value"}
        with open(file_path, "w") as f:
            json.dump(data, f)

        with pytest.raises(ValidationError, match="Input file is empty or has no records"):
            async for _ in read_json(file_path):
                pass

    async def test_read_json_file_not_found(self, tmp_path: Path) -> None:
        """Test reading non-existent file raises FileNotFoundError."""
        non_existent = tmp_path / "does_not_exist.json"
        with pytest.raises(FileNotFoundError):
            async for _ in read_json(non_existent):
                pass

    async def test_read_json_with_unicode(self, tmp_path: Path) -> None:
        """Test reading JSON with Unicode and emoji."""
        file_path = tmp_path / "unicode.json"
        data = {
            "records": [
                {"id": "doc1", "text": "Hello 世界 🌍 مرحبا"},
            ]
        }
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

        records = []
        async for record in read_json(file_path):
            records.append(record)

        assert len(records) == 1
        assert records[0]["text"] == "Hello 世界 🌍 مرحبا"

    async def test_read_json_with_long_text(self, tmp_path: Path) -> None:
        """Test reading JSON with very long text (10K chars)."""
        file_path = tmp_path / "long_text.json"
        long_text = "a" * 10000
        data = {
            "records": [
                {"id": "doc1", "text": long_text},
            ]
        }
        with open(file_path, "w") as f:
            json.dump(data, f)

        records = []
        async for record in read_json(file_path):
            records.append(record)

        assert len(records) == 1
        assert len(records[0]["text"]) == 10000
        assert records[0]["text"] == long_text

    async def test_read_json_single_record(self, tmp_path: Path) -> None:
        """Test reading JSON with a single record."""
        file_path = tmp_path / "single.json"
        data = {
            "records": [
                {"id": "doc1", "text": "Test"},
            ]
        }
        with open(file_path, "w") as f:
            json.dump(data, f)

        records = []
        async for record in read_json(file_path):
            records.append(record)

        assert len(records) == 1
        assert records[0]["id"] == "doc1"
        assert records[0]["text"] == "Test"
