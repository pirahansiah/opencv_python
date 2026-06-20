"""Tests for ProgressTracker."""

from __future__ import annotations

from progress_bar import ProgressTracker


def test_tracker_starts_at_zero() -> None:
    tracker = ProgressTracker(100)
    assert tracker.current == 0
    assert tracker.percentage == 0.0


def test_tracker_updates_correctly() -> None:
    tracker = ProgressTracker(10)
    for _ in range(5):
        tracker.update()
    assert tracker.current == 5
    assert tracker.percentage == 50.0


def test_tracker_full_progress() -> None:
    tracker = ProgressTracker(10)
    for _ in range(10):
        tracker.update()
    assert tracker.current == 10
    assert tracker.percentage == 100.0


def test_tracker_zero_total() -> None:
    tracker = ProgressTracker(0)
    tracker.update()
    assert tracker.current == 0
    assert tracker.percentage == 0.0


def test_tracker_reset() -> None:
    tracker = ProgressTracker(10)
    for _ in range(5):
        tracker.update()
    tracker.reset()
    assert tracker.current == 0
    assert tracker.percentage == 0.0


def test_tracker_output(capsys: object) -> None:
    tracker = ProgressTracker(100, report_interval=50.0)
    for _ in range(51):
        tracker.update()
    captured = capsys.readouterr()  # type: ignore[attr-defined]
    assert "Progress:" in captured.out
