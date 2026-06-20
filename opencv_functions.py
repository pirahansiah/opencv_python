"""Reusable OpenCV utility functions for image display, saving, cartoon effects, and face detection."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def show_image_plt(image: np.ndarray, title: str = "ORIGINAL") -> None:
    """Display image using matplotlib (converts BGR to RGB)."""
    from matplotlib import pyplot as plt

    rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(rgb_img)
    plt.title(title)
    plt.show(block=False)


def show_image_opencv(image: np.ndarray, window_name: str = "image", delay: int = 1000) -> None:
    """Display image using OpenCV window."""
    cv2.imshow(window_name, image)
    cv2.waitKey(delay)


def save_image_opencv(
    filename: str | Path, image: np.ndarray, default_name: str = "output.jpg"
) -> bool:
    """Save image to disk. Returns True on success."""
    path = Path(filename) if filename else Path(default_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    return cv2.imwrite(str(path), image)


def cartoon_image(image: np.ndarray, num_down: int = 5, num_bilateral: int = 9) -> np.ndarray:
    """Apply cartoon effect to an image.

    Args:
        image: Input BGR image.
        num_down: Number of downsampling steps.
        num_bilateral: Number of bilateral filtering steps.

    Returns:
        Cartoon-style image.
    """
    img_color = image.copy()
    for _ in range(num_down):
        img_color = cv2.pyrDown(img_color)
    for _ in range(num_bilateral):
        img_color = cv2.bilateralFilter(img_color, d=9, sigmaColor=9, sigmaSpace=7)
    for _ in range(num_down):
        img_color = cv2.pyrUp(img_color)

    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.medianBlur(img_gray, 7)
    img_edge = cv2.adaptiveThreshold(
        img_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, blockSize=9, C=2
    )
    img_edge = cv2.cvtColor(img_edge, cv2.COLOR_GRAY2RGB)
    return cv2.bitwise_and(img_color, img_edge)


def face_detection(
    img: np.ndarray,
    cascade_dir: Path | str = "haarcascades",
) -> np.ndarray:
    """Detect faces and eyes, draw bounding boxes.

    Args:
        img: Input BGR image.
        cascade_dir: Directory containing Haar cascade XML files.

    Returns:
        Image with drawn rectangles.
    """
    cascade_path = Path(cascade_dir)
    face_cascade = cv2.CascadeClassifier(str(cascade_path / "haarcascade_frontalface_default.xml"))
    eye_cascade = cv2.CascadeClassifier(str(cascade_path / "haarcascade_eye.xml"))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for x, y, w, h in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y : y + h, x : x + w]
        roi_color = img[y : y + h, x : x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for ex, ey, ew, eh in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
    return img


def main() -> None:
    print("OpenCV functions module")
    print(cv2.__version__)
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        _, frame = cap.read()
        show_image_opencv(frame, "original")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
        cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
