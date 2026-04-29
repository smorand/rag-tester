"""Logging configuration with rich console output and file logging.

Provides structured logging with:
- Rich console handler with colors and formatting
- Rotating file handler (max 10MB, 5 backups)
- Module-level loggers via logging.getLogger(__name__)
"""

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from rich.logging import RichHandler

from rag_tester.config import Settings


def setup_logging(settings: Settings) -> None:
    """Configure logging with rich console and file output.

    Args:
        settings: Application settings containing log_level and log_file
    """
    # Create log directory if it doesn't exist
    log_path = Path(settings.log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(settings.log_level.upper())

    # Remove existing handlers to avoid duplicates
    root_logger.handlers.clear()

    # Console handler with rich formatting
    console_handler = RichHandler(
        rich_tracebacks=True,
        tracebacks_show_locals=True,
        show_time=True,
        show_level=True,
        show_path=True,
    )
    console_handler.setLevel(settings.log_level.upper())
    console_formatter = logging.Formatter(
        "%(message)s",
        datefmt="[%X]",
    )
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    # File handler with rotation
    file_handler = RotatingFileHandler(
        settings.log_file,
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setLevel(settings.log_level.upper())
    file_formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)

    # Log initial message
    logger = logging.getLogger(__name__)
    logger.info("Logging configured: level=%s, file=%s", settings.log_level, settings.log_file)
