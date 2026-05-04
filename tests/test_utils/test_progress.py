"""Tests for the ProgressTracker context manager."""

from rag_tester.utils.progress import ProgressTracker


class TestProgressTracker:
    def test_small_dataset_skips_display(self) -> None:
        with ProgressTracker("loading", total=10) as p:
            assert p.show_progress is False
            p.update(1)
            p.update(advance=5)
        # No-op when display is skipped: nothing to assert beyond non-failure

    def test_large_dataset_displays(self) -> None:
        with ProgressTracker("loading", total=200) as p:
            assert p.show_progress is True
            assert p._progress is not None
            assert p._task_id is not None
            p.update(advance=50)
            p.update()
        # After exit, _progress.stop() must have been called (live=False).
        assert p._progress is not None

    def test_show_progress_false_overrides(self) -> None:
        with ProgressTracker("loading", total=1000, show_progress=False) as p:
            assert p.show_progress is False
            p.update(advance=500)

    def test_update_before_enter_is_safe(self) -> None:
        p = ProgressTracker("loading", total=10)
        # Calling update before __enter__ should be a no-op, not raise.
        p.update(advance=1)

    def test_exit_with_exception_propagates(self) -> None:
        try:
            with ProgressTracker("loading", total=200):
                raise RuntimeError("boom")
        except RuntimeError as e:
            assert str(e) == "boom"
