"""Progress tracking utilities for RAG Tester.

This module provides progress bar functionality using rich.progress.
"""

import logging

from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

logger = logging.getLogger(__name__)


class ProgressTracker:
    """Progress tracker with rich progress bar.

    Displays progress for long-running operations with:
    - Spinner animation
    - Task description
    - Progress bar
    - Completed/total count
    - Time elapsed and remaining
    """

    def __init__(self, description: str, total: int, show_progress: bool = True) -> None:
        """Initialize progress tracker.

        Args:
            description: Description of the task being tracked
            total: Total number of items to process
            show_progress: Whether to show progress bar (False for small datasets)
        """
        self.description = description
        self.total = total
        self.show_progress = show_progress and total > 100  # Only show for large datasets
        self._progress: Progress | None = None
        self._task_id: TaskID | None = None

    def __enter__(self) -> "ProgressTracker":
        """Start progress tracking."""
        if self.show_progress:
            self._progress = Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                TextColumn("•"),
                TimeElapsedColumn(),
                TextColumn("•"),
                TimeRemainingColumn(),
            )
            self._progress.start()
            self._task_id = self._progress.add_task(
                self.description,
                total=self.total,
            )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:  # type: ignore[no-untyped-def]
        """Stop progress tracking."""
        if self._progress is not None:
            self._progress.stop()

    def update(self, advance: int = 1) -> None:
        """Update progress by advancing the counter.

        Args:
            advance: Number of items completed (default: 1)
        """
        if self._progress is not None and self._task_id is not None:
            self._progress.update(self._task_id, advance=advance)
