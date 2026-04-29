"""Tests for commands.load module."""

import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from rag_tester.commands.load import _load_async


class TestLoadCommand:
    """Tests for load command."""

    @pytest.mark.asyncio
    async def test_load_with_invalid_file(self) -> None:
        """Test load command with non-existent file."""
        exit_code = await _load_async(
            file="/nonexistent/file.yaml",
            database="chromadb://localhost:8000/test",
            embedding="sentence-transformers/all-MiniLM-L6-v2",
            mode="initial",
            parallel=1,
            batch_size=32,
            force_reembed=False,
        )
        
        assert exit_code == 1

    @pytest.mark.asyncio
    async def test_load_with_invalid_mode(self, tmp_path: Path) -> None:
        """Test load command with unsupported mode."""
        # Create a valid test file
        test_file = tmp_path / "test.yaml"
        test_file.write_text("- id: doc1\n  text: test")
        
        exit_code = await _load_async(
            file=str(test_file),
            database="chromadb://localhost:8000/test",
            embedding="sentence-transformers/all-MiniLM-L6-v2",
            mode="upsert",  # Not yet implemented
            parallel=1,
            batch_size=32,
            force_reembed=False,
        )
        
        assert exit_code == 1

    @pytest.mark.asyncio
    async def test_load_with_invalid_batch_size(self, tmp_path: Path) -> None:
        """Test load command with invalid batch size."""
        test_file = tmp_path / "test.yaml"
        test_file.write_text("- id: doc1\n  text: test")
        
        exit_code = await _load_async(
            file=str(test_file),
            database="chromadb://localhost:8000/test",
            embedding="sentence-transformers/all-MiniLM-L6-v2",
            mode="initial",
            parallel=1,
            batch_size=0,  # Invalid
            force_reembed=False,
        )
        
        assert exit_code == 1

    @pytest.mark.asyncio
    async def test_load_with_invalid_parallel_workers(self, tmp_path: Path) -> None:
        """Test load command with invalid parallel workers."""
        test_file = tmp_path / "test.yaml"
        test_file.write_text("- id: doc1\n  text: test")
        
        exit_code = await _load_async(
            file=str(test_file),
            database="chromadb://localhost:8000/test",
            embedding="sentence-transformers/all-MiniLM-L6-v2",
            mode="initial",
            parallel=0,  # Invalid
            batch_size=32,
            force_reembed=False,
        )
        
        assert exit_code == 1

    @pytest.mark.asyncio
    async def test_load_with_invalid_database_format(self, tmp_path: Path) -> None:
        """Test load command with invalid database connection string."""
        test_file = tmp_path / "test.yaml"
        test_file.write_text("- id: doc1\n  text: test")
        
        exit_code = await _load_async(
            file=str(test_file),
            database="invalid://format",
            embedding="sentence-transformers/all-MiniLM-L6-v2",
            mode="initial",
            parallel=1,
            batch_size=32,
            force_reembed=False,
        )
        
        assert exit_code == 1

    @pytest.mark.asyncio
    async def test_load_with_missing_collection_name(self, tmp_path: Path) -> None:
        """Test load command with missing collection name in database string."""
        test_file = tmp_path / "test.yaml"
        test_file.write_text("- id: doc1\n  text: test")
        
        exit_code = await _load_async(
            file=str(test_file),
            database="chromadb://localhost:8000",  # Missing collection
            embedding="sentence-transformers/all-MiniLM-L6-v2",
            mode="initial",
            parallel=1,
            batch_size=32,
            force_reembed=False,
        )
        
        assert exit_code == 1

    @pytest.mark.asyncio
    async def test_load_with_invalid_port(self, tmp_path: Path) -> None:
        """Test load command with invalid port number."""
        test_file = tmp_path / "test.yaml"
        test_file.write_text("- id: doc1\n  text: test")
        
        exit_code = await _load_async(
            file=str(test_file),
            database="chromadb://localhost:invalid/test",
            embedding="sentence-transformers/all-MiniLM-L6-v2",
            mode="initial",
            parallel=1,
            batch_size=32,
            force_reembed=False,
        )
        
        assert exit_code == 1
