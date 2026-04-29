"""E2E tests for US-003: Load Command - Streaming & Parallel Processing.

These tests validate the complete load command functionality including:
- Initial dataset loading
- Parallel processing
- Custom batch sizes
- Streaming mode for large files
- Duplicate ID handling
- Edge cases and error handling
"""

import subprocess
import time
from pathlib import Path

import pytest
import yaml


@pytest.fixture
def test_data_100(tmp_path: Path) -> Path:
    """Generate test data with 100 records."""
    records = [
        {"id": f"doc{i:03d}", "text": f"This is document number {i} with some sample text for testing."}
        for i in range(1, 101)
    ]

    file_path = tmp_path / "test_data_100.yaml"
    with open(file_path, "w", encoding="utf-8") as f:
        yaml.dump(records, f)

    return file_path


@pytest.fixture
def test_data_10k(tmp_path: Path) -> Path:
    """Generate test data with 10,000 records for streaming test."""
    records = [{"id": f"doc{i:05d}", "text": f"Document {i}: " + "Sample text content. " * 10} for i in range(1, 10001)]

    file_path = tmp_path / "test_data_10k.yaml"
    with open(file_path, "w", encoding="utf-8") as f:
        yaml.dump(records, f)

    return file_path


@pytest.fixture
def test_data_duplicates(tmp_path: Path) -> Path:
    """Generate test data with duplicate IDs."""
    records = [
        {"id": "doc1", "text": "First document"},
        {"id": "doc2", "text": "Second document"},
        {"id": "doc3", "text": "Third document"},
        {"id": "doc4", "text": "Fourth document"},
        {"id": "doc5", "text": "Fifth document"},
        {"id": "doc6", "text": "Sixth document"},
        {"id": "doc7", "text": "Seventh document"},
        {"id": "doc8", "text": "Eighth document"},
        {"id": "doc5", "text": "Duplicate fifth document"},  # Duplicate
        {"id": "doc3", "text": "Duplicate third document"},  # Duplicate
    ]

    file_path = tmp_path / "test_data_duplicates.yaml"
    with open(file_path, "w", encoding="utf-8") as f:
        yaml.dump(records, f)

    return file_path


@pytest.fixture
def empty_file(tmp_path: Path) -> Path:
    """Generate empty YAML file."""
    file_path = tmp_path / "empty.yaml"
    file_path.write_text("")
    return file_path


@pytest.fixture
def single_record_file(tmp_path: Path) -> Path:
    """Generate file with single record."""
    records = [{"id": "doc1", "text": "Single document"}]

    file_path = tmp_path / "single.yaml"
    with open(file_path, "w", encoding="utf-8") as f:
        yaml.dump(records, f)

    return file_path


@pytest.fixture
def long_text_file(tmp_path: Path) -> Path:
    """Generate file with very long text (10K chars)."""
    long_text = "A" * 10000
    records = [{"id": "doc1", "text": long_text}]

    file_path = tmp_path / "long_text.yaml"
    with open(file_path, "w", encoding="utf-8") as f:
        yaml.dump(records, f)

    return file_path


@pytest.fixture
def unicode_file(tmp_path: Path) -> Path:
    """Generate file with Unicode and emoji."""
    records = [{"id": "doc1", "text": "Hello 世界 🌍 مرحبا"}]

    file_path = tmp_path / "unicode.yaml"
    with open(file_path, "w", encoding="utf-8") as f:
        yaml.dump(records, f)

    return file_path


@pytest.fixture
def malformed_yaml_file(tmp_path: Path) -> Path:
    """Generate malformed YAML file."""
    file_path = tmp_path / "malformed.yaml"
    file_path.write_text("invalid: yaml: content:")
    return file_path


@pytest.fixture
def invalid_records_file(tmp_path: Path) -> Path:
    """Generate file with invalid records (missing text field)."""
    records = [
        {"id": "doc1", "text": "Valid document"},
        {"id": "doc2"},  # Missing text field
    ]

    file_path = tmp_path / "invalid_records.yaml"
    with open(file_path, "w", encoding="utf-8") as f:
        yaml.dump(records, f)

    return file_path


class TestE2E001InitialDatasetLoad:
    """E2E-001: Initial Dataset Load with Local Embedding Model."""

    @pytest.mark.e2e
    def test_load_100_documents(self, test_data_100: Path) -> None:
        """Test loading 100 documents with local embedding model."""
        collection_name = f"test_collection_{int(time.time())}"

        # Run load command
        result = subprocess.run(
            [
                "rag-tester",
                "load",
                "--file",
                str(test_data_100),
                "--database",
                f"chromadb://localhost:8000/{collection_name}",
                "--embedding",
                "sentence-transformers/all-MiniLM-L6-v2",
                "--parallel",
                "4",
            ],
            capture_output=True,
            text=True,
        )

        # Verify exit code
        assert result.returncode == 0, f"Command failed: {result.stderr}"

        # Verify output
        assert "Successfully loaded: 100 records" in result.stdout
        assert "Failed records: 0" in result.stdout
        assert "Skipped records: 0" in result.stdout
        assert "Total time:" in result.stdout

        # TODO: Verify collection exists and has correct data
        # This requires ChromaDB client access


class TestE2E010ParallelWorkers:
    """E2E-010: Parallel Workers."""

    @pytest.mark.e2e
    def test_parallel_processing(self, test_data_100: Path) -> None:
        """Test parallel processing with 4 workers."""
        collection_name = f"parallel_test_{int(time.time())}"

        result = subprocess.run(
            [
                "rag-tester",
                "load",
                "--file",
                str(test_data_100),
                "--database",
                f"chromadb://localhost:8000/{collection_name}",
                "--embedding",
                "sentence-transformers/all-MiniLM-L6-v2",
                "--parallel",
                "4",
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0
        assert "Successfully loaded: 100 records" in result.stdout


class TestE2E011CustomBatchSize:
    """E2E-011: Custom Batch Size."""

    @pytest.mark.e2e
    def test_custom_batch_size(self, test_data_100: Path) -> None:
        """Test custom batch size of 32."""
        collection_name = f"batch_test_{int(time.time())}"

        result = subprocess.run(
            [
                "rag-tester",
                "load",
                "--file",
                str(test_data_100),
                "--database",
                f"chromadb://localhost:8000/{collection_name}",
                "--embedding",
                "sentence-transformers/all-MiniLM-L6-v2",
                "--batch-size",
                "32",
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0
        assert "Successfully loaded: 100 records" in result.stdout


class TestE2E012StreamingMode:
    """E2E-012: Streaming Mode (Large File)."""

    @pytest.mark.e2e
    @pytest.mark.slow
    def test_streaming_large_file(self, test_data_10k: Path) -> None:
        """Test streaming mode with 10K records."""
        collection_name = f"streaming_test_{int(time.time())}"

        result = subprocess.run(
            [
                "rag-tester",
                "load",
                "--file",
                str(test_data_10k),
                "--database",
                f"chromadb://localhost:8000/{collection_name}",
                "--embedding",
                "sentence-transformers/all-MiniLM-L6-v2",
                "--parallel",
                "4",
            ],
            capture_output=True,
            text=True,
            timeout=300,  # 5 minutes timeout
        )

        assert result.returncode == 0
        assert "Successfully loaded: 10000 records" in result.stdout


class TestE2E013DuplicateIDs:
    """E2E-013: Duplicate IDs."""

    @pytest.mark.e2e
    def test_duplicate_ids(self, test_data_duplicates: Path) -> None:
        """Test handling of duplicate IDs."""
        collection_name = f"dup_test_{int(time.time())}"

        result = subprocess.run(
            [
                "rag-tester",
                "load",
                "--file",
                str(test_data_duplicates),
                "--database",
                f"chromadb://localhost:8000/{collection_name}",
                "--embedding",
                "sentence-transformers/all-MiniLM-L6-v2",
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0
        assert "Successfully loaded: 8 records" in result.stdout
        assert "Skipped records: 2" in result.stdout


class TestE2E064LoadLatency:
    """E2E-064: Load Latency (Local Model)."""

    @pytest.mark.e2e
    def test_load_latency(self, test_data_100: Path) -> None:
        """Test load latency is under 30 seconds."""
        collection_name = f"perf_test_{int(time.time())}"

        start_time = time.time()
        result = subprocess.run(
            [
                "rag-tester",
                "load",
                "--file",
                str(test_data_100),
                "--database",
                f"chromadb://localhost:8000/{collection_name}",
                "--embedding",
                "sentence-transformers/all-MiniLM-L6-v2",
                "--parallel",
                "1",
            ],
            capture_output=True,
            text=True,
        )
        elapsed_time = time.time() - start_time

        assert result.returncode == 0
        assert elapsed_time < 30, f"Load took {elapsed_time:.2f}s, expected < 30s"


class TestE2E081EmptyInputFile:
    """E2E-081: Empty Input File."""

    @pytest.mark.e2e
    def test_empty_file(self, empty_file: Path) -> None:
        """Test error handling for empty file."""
        result = subprocess.run(
            [
                "rag-tester",
                "load",
                "--file",
                str(empty_file),
                "--database",
                "chromadb://localhost:8000/test",
                "--embedding",
                "sentence-transformers/all-MiniLM-L6-v2",
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 1
        assert "Error:" in result.stderr
        assert "empty" in result.stderr.lower()


class TestE2E082SingleRecordLoad:
    """E2E-082: Single Record Load."""

    @pytest.mark.e2e
    def test_single_record(self, single_record_file: Path) -> None:
        """Test loading single record."""
        collection_name = f"single_test_{int(time.time())}"

        result = subprocess.run(
            [
                "rag-tester",
                "load",
                "--file",
                str(single_record_file),
                "--database",
                f"chromadb://localhost:8000/{collection_name}",
                "--embedding",
                "sentence-transformers/all-MiniLM-L6-v2",
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0
        assert "Successfully loaded: 1 record" in result.stdout


class TestE2E083VeryLongText:
    """E2E-083: Very Long Text (10K chars)."""

    @pytest.mark.e2e
    def test_long_text(self, long_text_file: Path) -> None:
        """Test handling of very long text."""
        collection_name = f"long_test_{int(time.time())}"

        result = subprocess.run(
            [
                "rag-tester",
                "load",
                "--file",
                str(long_text_file),
                "--database",
                f"chromadb://localhost:8000/{collection_name}",
                "--embedding",
                "sentence-transformers/all-MiniLM-L6-v2",
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0
        assert "Successfully loaded: 1 record" in result.stdout


class TestE2E084UnicodeAndEmoji:
    """E2E-084: Unicode and Emoji."""

    @pytest.mark.e2e
    def test_unicode_and_emoji(self, unicode_file: Path) -> None:
        """Test handling of Unicode and emoji."""
        collection_name = f"unicode_test_{int(time.time())}"

        result = subprocess.run(
            [
                "rag-tester",
                "load",
                "--file",
                str(unicode_file),
                "--database",
                f"chromadb://localhost:8000/{collection_name}",
                "--embedding",
                "sentence-transformers/all-MiniLM-L6-v2",
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0
        assert "Successfully loaded: 1 record" in result.stdout


class TestE2ELOAD001InvalidFileFormat:
    """E2E-LOAD-001: Invalid File Format."""

    @pytest.mark.e2e
    def test_malformed_yaml(self, malformed_yaml_file: Path) -> None:
        """Test error handling for malformed YAML."""
        result = subprocess.run(
            [
                "rag-tester",
                "load",
                "--file",
                str(malformed_yaml_file),
                "--database",
                "chromadb://localhost:8000/test",
                "--embedding",
                "sentence-transformers/all-MiniLM-L6-v2",
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 1
        assert "Error:" in result.stderr


class TestE2ELOAD002MissingRequiredFields:
    """E2E-LOAD-002: Missing Required Fields."""

    @pytest.mark.e2e
    def test_missing_text_field(self, invalid_records_file: Path) -> None:
        """Test error handling for missing text field."""
        collection_name = f"invalid_test_{int(time.time())}"

        result = subprocess.run(
            [
                "rag-tester",
                "load",
                "--file",
                str(invalid_records_file),
                "--database",
                f"chromadb://localhost:8000/{collection_name}",
                "--embedding",
                "sentence-transformers/all-MiniLM-L6-v2",
            ],
            capture_output=True,
            text=True,
        )

        # Should succeed but with 1 failed record
        assert result.returncode == 0
        assert "Successfully loaded: 1 record" in result.stdout
        assert "Failed records: 1" in result.stdout


class TestE2ELOAD003DatabaseUnreachable:
    """E2E-LOAD-003: Database Unreachable."""

    @pytest.mark.e2e
    def test_database_unreachable(self, test_data_100: Path) -> None:
        """Test error handling for unreachable database."""
        result = subprocess.run(
            [
                "rag-tester",
                "load",
                "--file",
                str(test_data_100),
                "--database",
                "chromadb://localhost:9999/test",  # Wrong port
                "--embedding",
                "sentence-transformers/all-MiniLM-L6-v2",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )

        assert result.returncode == 1
        assert "Error:" in result.stderr


class TestE2ELOAD005InvalidBatchSize:
    """E2E-LOAD-005: Invalid Batch Size."""

    @pytest.mark.e2e
    def test_invalid_batch_size(self, test_data_100: Path) -> None:
        """Test error handling for invalid batch size."""
        result = subprocess.run(
            [
                "rag-tester",
                "load",
                "--file",
                str(test_data_100),
                "--database",
                "chromadb://localhost:8000/test",
                "--embedding",
                "sentence-transformers/all-MiniLM-L6-v2",
                "--batch-size",
                "0",
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 1
        assert "Error:" in result.stderr
        assert "Batch size" in result.stderr


class TestE2ELOAD006InvalidParallelWorkers:
    """E2E-LOAD-006: Invalid Parallel Workers."""

    @pytest.mark.e2e
    def test_invalid_parallel_workers(self, test_data_100: Path) -> None:
        """Test error handling for invalid parallel workers."""
        result = subprocess.run(
            [
                "rag-tester",
                "load",
                "--file",
                str(test_data_100),
                "--database",
                "chromadb://localhost:8000/test",
                "--embedding",
                "sentence-transformers/all-MiniLM-L6-v2",
                "--parallel",
                "0",
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 1
        assert "Error:" in result.stderr
        assert "Parallel workers" in result.stderr


class TestE2ELOAD007FileNotFound:
    """E2E-LOAD-007: File Not Found."""

    @pytest.mark.e2e
    def test_file_not_found(self) -> None:
        """Test error handling for non-existent file."""
        result = subprocess.run(
            [
                "rag-tester",
                "load",
                "--file",
                "/nonexistent/file.yaml",
                "--database",
                "chromadb://localhost:8000/test",
                "--embedding",
                "sentence-transformers/all-MiniLM-L6-v2",
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 1
        assert "Error:" in result.stderr
        assert "not found" in result.stderr.lower()
