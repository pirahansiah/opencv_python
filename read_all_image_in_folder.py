"""Batch read and display all images in a folder."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2


def read_images_from_folder(
    folder: Path, pattern: str = "*.png", delay: int = 100
) -> list[cv2.typing.MatLike]:
    """Read all matching images from a folder.

    Args:
        folder: Directory to scan.
        pattern: Glob pattern for image files.
        delay: Milliseconds to display each image.

    Returns:
        List of loaded images.
    """
    images: list[cv2.typing.MatLike] = []
    for path in sorted(folder.glob(pattern)):
        img = cv2.imread(str(path))
        if img is not None:
            cv2.imshow("image", img)
            cv2.waitKey(delay)
            images.append(img)
    cv2.destroyAllWindows()
    return images


def main() -> None:
    parser = argparse.ArgumentParser(description="Read and display all images in a folder")
    parser.add_argument("folder", type=Path, help="Path to image folder")
    parser.add_argument("--pattern", default="*.png", help="Glob pattern (default: *.png)")
    parser.add_argument("--delay", type=int, default=100, help="Display delay in ms (default: 100)")
    args = parser.parse_args()
    images = read_images_from_folder(args.folder, args.pattern, args.delay)
    print(f"Loaded {len(images)} images")


if __name__ == "__main__":
    main()
