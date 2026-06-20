"""VS Code integration examples: capture webcam, apply cartoon effect, save."""

from __future__ import annotations

import cv2

from opencv_functions import cartoon_image, save_image_opencv


def main() -> None:
    print("farshid pirahansiah")
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        _, frame = cap.read()
        frame = cartoon_image(frame)
        save_image_opencv("cartoon_output.jpg", frame)
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
