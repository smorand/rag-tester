"""Tests for logging configuration."""

import logging
from pathlib import Path

from rich.logging import RichHandler

from rag_tester.config import Settings
from rag_tester.logging_config import setup_logging


class TestLoggingConfiguration:
    """Tests for logging setup."""

    def test_log_directory_auto_created(self, tmp_path: Path) -> None:
        """Test that log directory is automatically created.

        Corresponds to: E2E-INFRA-006
        """
        log_file = tmp_path / "new_logs" / "test.log"
        settings = Settings(log_file=str(log_file))

        # Directory should not exist yet
        assert not log_file.parent.exists()

        # Setup logging
        setup_logging(settings)

        # Write a log message
        logger = logging.getLogger("test_logger")
        logger.info("Test message")

        # Directory and file should now exist
        assert log_file.parent.exists()
        assert log_file.exists()

        # Verify log content
        content = log_file.read_text()
        assert "Test message" in content

    def test_log_file_written(self, tmp_path: Path) -> None:
        """Test that log messages are written to file."""
        log_file = tmp_path / "test.log"
        settings = Settings(log_file=str(log_file), log_level="INFO")

        setup_logging(settings)

        logger = logging.getLogger("test_module")
        logger.info("Info message")
        logger.warning("Warning message")
        logger.error("Error message")

        # Read log file
        content = log_file.read_text()

        # Verify all messages are present
        assert "Info message" in content
        assert "Warning message" in content
        assert "Error message" in content
        assert "[INFO]" in content
        assert "[WARNING]" in content
        assert "[ERROR]" in content
        assert "[test_module]" in content

    def test_log_level_filtering(self, tmp_path: Path) -> None:
        """Test that log level filtering works correctly."""
        log_file = tmp_path / "test.log"
        settings = Settings(log_file=str(log_file), log_level="WARNING")

        setup_logging(settings)

        logger = logging.getLogger("test_module")
        logger.debug("Debug message")
        logger.info("Info message")
        logger.warning("Warning message")
        logger.error("Error message")

        # Read log file
        content = log_file.read_text()

        # Only WARNING and ERROR should be present
        assert "Debug message" not in content
        assert "Info message" not in content
        assert "Warning message" in content
        assert "Error message" in content

    def test_log_format(self, tmp_path: Path) -> None:
        """Test that log format matches expected pattern."""
        log_file = tmp_path / "test.log"
        settings = Settings(log_file=str(log_file), log_level="INFO")

        setup_logging(settings)

        logger = logging.getLogger("test_module")
        logger.info("Test message")

        # Read log file
        content = log_file.read_text()

        # Verify format: [timestamp] [level] [module] message
        assert "[INFO]" in content
        assert "[test_module]" in content
        assert "Test message" in content

        # Check timestamp format (should contain date and time)
        lines = content.strip().split("\n")
        for line in lines:
            if "Test message" in line:
                # Should start with [YYYY-MM-DD HH:MM:SS]
                assert line.startswith("[")
                assert "]" in line

    def test_multiple_loggers(self, tmp_path: Path) -> None:
        """Test that multiple loggers work correctly."""
        log_file = tmp_path / "test.log"
        settings = Settings(log_file=str(log_file), log_level="INFO")

        setup_logging(settings)

        logger1 = logging.getLogger("module1")
        logger2 = logging.getLogger("module2")

        logger1.info("Message from module1")
        logger2.info("Message from module2")

        # Read log file
        content = log_file.read_text()

        # Both messages should be present with correct module names
        assert "Message from module1" in content
        assert "[module1]" in content
        assert "Message from module2" in content
        assert "[module2]" in content

    def test_log_rotation_configuration(self, tmp_path: Path) -> None:
        """Test that log rotation is configured correctly."""
        log_file = tmp_path / "test.log"
        settings = Settings(log_file=str(log_file), log_level="INFO")

        setup_logging(settings)

        # Get the file handler
        root_logger = logging.getLogger()
        file_handlers = [h for h in root_logger.handlers if hasattr(h, "maxBytes")]

        assert len(file_handlers) > 0
        file_handler = file_handlers[0]

        # Verify rotation settings
        assert file_handler.maxBytes == 10 * 1024 * 1024  # 10MB
        assert file_handler.backupCount == 5

    def test_rich_handler_configured(self, tmp_path: Path) -> None:
        """Test that rich console handler is configured."""
        log_file = tmp_path / "test.log"
        settings = Settings(log_file=str(log_file), log_level="INFO")

        setup_logging(settings)

        # Get root logger
        root_logger = logging.getLogger()

        # Should have at least 2 handlers (console + file)
        assert len(root_logger.handlers) >= 2

        # Check for RichHandler
        rich_handlers = [h for h in root_logger.handlers if isinstance(h, RichHandler)]
        assert len(rich_handlers) > 0

    def test_logging_setup_message(self, tmp_path: Path) -> None:
        """Test that logging setup writes an initial message."""
        log_file = tmp_path / "test.log"
        settings = Settings(log_file=str(log_file), log_level="INFO")

        setup_logging(settings)

        # Read log file
        content = log_file.read_text()

        # Should contain setup message
        assert "Logging configured" in content
        assert "INFO" in content
        assert str(log_file) in content
