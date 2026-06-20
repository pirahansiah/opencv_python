"""Progress tracking utility for batch processing."""

from __future__ import annotations


class ProgressTracker:
    """Track and display progress for batch operations."""

    def __init__(self, total: int, report_interval: float = 10.0) -> None:
        self.total = total
        self.current = 0
        self.report_interval = report_interval
        self._last_reported = 0.0

    def update(self) -> None:
        """Increment counter and print progress at intervals."""
        if self.total == 0:
            return
        self.current += 1
        progress = (self.current / self.total) * 100.0
        if progress >= self._last_reported + self.report_interval:
            print(f"Progress: {progress:.1f}%")
            self._last_reported += self.report_interval

    @property
    def percentage(self) -> float:
        """Return current progress as a percentage."""
        return (self.current / self.total * 100.0) if self.total > 0 else 0.0

    def reset(self) -> None:
        """Reset the tracker."""
        self.current = 0
        self._last_reported = 0.0


def main() -> None:
    total = 100
    tracker = ProgressTracker(total)
    for _ in range(total):
        tracker.update()
    print(f"Done. {tracker.percentage:.1f}%")


if __name__ == "__main__":
    main()
