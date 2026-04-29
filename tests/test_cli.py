"""Tests for CLI module."""

from typer.testing import CliRunner

from rag_tester.rag_tester import app

runner = CliRunner()


def test_version_command() -> None:
    """Test the version command."""
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "rag-tester version" in result.stdout
    assert "dev" in result.stdout


def test_help_command() -> None:
    """Test the help command."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "RAG Tester" in result.stdout


def test_verbose_flag() -> None:
    """Test the verbose flag."""
    result = runner.invoke(app, ["--verbose", "version"])
    assert result.exit_code == 0


def test_quiet_flag() -> None:
    """Test the quiet flag."""
    result = runner.invoke(app, ["--quiet", "version"])
    assert result.exit_code == 0
