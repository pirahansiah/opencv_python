"""Convert OpenCV Mat to NumPy array with thresholding."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def mat_to_numpy(
    image: np.ndarray, threshold: int = 130
) -> tuple[np.ndarray, np.ndarray]:
    """Convert image to binary using a threshold value.

    Args:
        image: Input BGR or grayscale image.
        threshold: Threshold value (0-255).

    Returns:
        Tuple of (binary_image, reshaped_image).
    """
    height, width = image.shape[:2]
    array = np.asarray(image, np.uint8)
    binary = np.where(array < threshold, 0, 255).astype(np.uint8)
    reshaped = binary.reshape(height, width)
    return binary, reshaped


def main() -> None:
    import sys

    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("test.jpg")
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"Error: Could not read {path}")
        return
    binary, reshaped = mat_to_numpy(img)
    cv2.imshow("binary", binary)
    cv2.waitKey(1000)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
