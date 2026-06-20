"""List specific files in specific folder using recursive glob."""

from __future__ import annotations

import argparse
from pathlib import Path


def list_files_recursive(root: Path, pattern: str = "**/*.jpg") -> list[Path]:
    """Recursively list files matching a pattern under root."""
    return sorted(root.glob(pattern))


def print_grouped_files(root: Path, files: list[Path]) -> None:
    """Print files grouped by their immediate parent directory."""
    prev_parent: str = ""
    for file in files:
        parent = str(file.parent)
        if parent != prev_parent:
            print("*" * 80)
            print(f"Directory: {parent}")
            prev_parent = parent
        print(file)


def main() -> None:
    parser = argparse.ArgumentParser(description="List files recursively")
    parser.add_argument("root", type=Path, help="Root directory to search")
    parser.add_argument(
        "--pattern", default="**/*.jpg", help="Glob pattern (default: **/*.jpg)"
    )
    args = parser.parse_args()
    files = list_files_recursive(args.root, args.pattern)
    print_grouped_files(args.root, files)


if __name__ == "__main__":
    main()
