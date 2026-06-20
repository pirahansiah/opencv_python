"""SAM-2 style image and video segmentation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np


@dataclass
class SegmentResult:
    """Segmentation output for a single image."""

    masks: list[np.ndarray] = field(default_factory=list)
    polygons: list[list[np.ndarray]] = field(default_factory=list)
    areas: list[float] = field(default_factory=list)
    image_shape: tuple[int, int, int] = (0, 0, 0)


class ImageSegmenter:
    """Image segmentation using ultralytics SAM or OpenCV DNN.

    Args:
        model_type: Model type ('sam2_n', 'sam2_s', 'sam2_b', 'sam2_l') or ONNX path.
        device: Device string ('cpu', 'cuda', 'mps').
    """

    def __init__(self, model_type: str = "sam2_n", device: str = "cpu") -> None:
        self._model_type = model_type
        self._device = device
        self._model = None

    def _load_model(self) -> None:
        if self._model is not None:
            return
        try:
            from ultralytics import SAM
            self._model = SAM(self._model_type)
        except ImportError as e:
            raise ImportError("Install ultralytics: pip install ultralytics") from e

    def segment(
        self,
        source: str | Path | np.ndarray,
        points: list[tuple[int, int]] | None = None,
        labels: list[int] | None = None,
    ) -> SegmentResult:
        """Segment objects in an image.

        Args:
            source: Image path or BGR numpy array.
            points: Prompt points for SAM (x, y). If None, auto-segment.
            labels: Binary labels for points (1=foreground, 0=background).

        Returns:
            SegmentResult with masks, polygons, and areas.
        """
        self._load_model()

        img = source if isinstance(source, np.ndarray) else cv2.imread(str(source))
        if img is None:
            raise ValueError(f"Cannot load image: {source}")

        kwargs: dict = {}
        if points is not None:
            pts = np.array(points, dtype=np.float32)
            kwargs["points"] = pts
            if labels is not None:
                kwargs["labels"] = np.array(labels, dtype=np.int32)

        results = self._model.predict(str(source) if not isinstance(source, np.ndarray) else img, **kwargs)

        masks: list[np.ndarray] = []
        polygons: list[list[np.ndarray]] = []
        areas: list[float] = []

        for r in results:
            if r.masks is None:
                continue
            for mask_tensor in r.masks.data:
                mask = mask_tensor.cpu().numpy().astype(np.uint8)
                masks.append(mask)
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                polys = [c.squeeze() for c in contours if len(c.squeeze().shape) == 2]
                polygons.append(polys)
                areas.append(float(mask.sum()))

        return SegmentResult(masks=masks, polygons=polygons, areas=areas, image_shape=img.shape)

    def segment_interactive(
        self,
        source: str | Path | np.ndarray,
    ) -> SegmentResult:
        """Interactive segmentation — click points to select objects.

        Opens a window where left-click adds foreground points,
        right-click adds background points. Press Enter to confirm,
        Escape to cancel.

        Args:
            source: Image path or BGR numpy array.

        Returns:
            SegmentResult after user interaction.
        """
        img = source if isinstance(source, np.ndarray) else cv2.imread(str(source))
        if img is None:
            raise ValueError(f"Cannot load image: {source}")

        display = img.copy()
        points: list[tuple[int, int]] = []
        labels: list[int] = []
        window = "Interactive Segmentation (LMB=FG, RMB=BG, Enter=OK, Esc=Cancel)"

        def _mouse_cb(event: int, x: int, y: int, flags: int, param: object) -> None:
            nonlocal display
            if event == cv2.EVENT_LBUTTONDOWN:
                points.append((x, y))
                labels.append(1)
                cv2.circle(display, (x, y), 5, (0, 255, 0), -1)
            elif event == cv2.EVENT_RBUTTONDOWN:
                points.append((x, y))
                labels.append(0)
                cv2.circle(display, (x, y), 5, (0, 0, 255), -1)
            cv2.imshow(window, display)

        cv2.namedWindow(window)
        cv2.setMouseCallback(window, _mouse_cb)
        cv2.imshow(window, display)

        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == 13:  # Enter
                break
            elif key == 27:  # Escape
                cv2.destroyWindow(window)
                return SegmentResult()

        cv2.destroyWindow(window)

        if not points:
            return SegmentResult()

        return self.segment(source, points=points, labels=labels)

    def segment_video(
        self,
        source: str | Path | int = 0,
        output_path: str | Path | None = None,
        show: bool = True,
    ) -> dict[str, float]:
        """Segment objects frame-by-frame in a video.

        Args:
            source: Video path, RTSP URL, or camera index.
            output_path: Save annotated output video.
            show: Display in a window.

        Returns:
            Dict with total_frames, avg_fps.
        """
        cap = cv2.VideoCapture(str(source) if not isinstance(source, int) else source)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {source}")

        writer: cv2.VideoWriter | None = None
        if output_path is not None:
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            writer = cv2.VideoWriter(
                str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
            )

        import time

        total_frames = 0
        total_time = 0.0
        colors = np.random.randint(0, 255, (100, 3), dtype=np.uint8).tolist()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            t0 = time.perf_counter()
            result = self.segment(frame)
            total_time += time.perf_counter() - t0
            total_frames += 1

            overlay = self.visualize(frame, result, colors)
            if show:
                cv2.imshow("Segmentation", overlay)
            if writer is not None:
                writer.write(overlay)
            if show and cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        if writer is not None:
            writer.release()
        if show:
            cv2.destroyAllWindows()

        avg_fps = total_frames / max(total_time, 1e-9)
        return {"total_frames": float(total_frames), "avg_fps": avg_fps}

    @staticmethod
    def visualize(
        image: np.ndarray,
        result: SegmentResult,
        colors: list[tuple[int, int, int]] | None = None,
        alpha: float = 0.4,
    ) -> np.ndarray:
        """Overlay segmentation masks on the image.

        Args:
            image: Input BGR image.
            result: SegmentResult with masks.
            colors: List of BGR colors per mask.
            alpha: Overlay transparency.

        Returns:
            Annotated image.
        """
        if not result.masks:
            return image.copy()

        if colors is None:
            colors = np.random.randint(0, 255, (len(result.masks), 3), dtype=np.uint8).tolist()

        overlay = image.copy()
        for i, mask in enumerate(result.masks):
            color = colors[i % len(colors)]
            color_img = np.zeros_like(image, dtype=np.uint8)
            color_img[:] = color

            mask_bool = mask.astype(bool) if mask.max() <= 1 else mask.astype(bool)
            if mask_bool.shape != image.shape[:2]:
                mask_bool = cv2.resize(
                    mask_bool.astype(np.uint8), (image.shape[1], image.shape[0])
                ).astype(bool)

            overlay[mask_bool] = cv2.addWeighted(
                image[mask_bool], 1 - alpha, color_img[mask_bool], alpha, 0
            )
            contours, _ = cv2.findContours(
                mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(overlay, contours, -1, color, 2)

        return overlay

    @staticmethod
    def extract_polygons(result: SegmentResult) -> list[list[np.ndarray]]:
        """Extract polygon contours from segmentation result.

        Args:
            result: SegmentResult with polygons.

        Returns:
            List of polygon arrays per mask.
        """
        return result.polygons


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Image segmentation with SAM")
    parser.add_argument("source", help="Image or video path")
    parser.add_argument("--model", default="sam2_n", help="SAM model type")
    parser.add_argument("--device", default="cpu", help="Device (cpu/cuda/mps)")
    parser.add_argument("--output", default=None, help="Save output")
    parser.add_argument("--interactive", action="store_true", help="Interactive point selection")
    args = parser.parse_args()

    segmenter = ImageSegmenter(args.model, args.device)
    source = int(args.source) if args.source.isdigit() else args.source

    if isinstance(source, int) or any(args.source.lower().endswith(ext) for ext in (".mp4", ".avi", ".mov")):
        stats = segmenter.segment_video(source, args.output)
        print(f"Stats: {stats}")
    elif args.interactive:
        result = segmenter.segment_interactive(args.source)
        img = cv2.imread(args.source)
        overlay = ImageSegmenter.visualize(img, result)
        if args.output:
            cv2.imwrite(args.output, overlay)
            print(f"Saved to {args.output}")
        else:
            cv2.imshow("Segmentation", overlay)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        print(f"Masks: {len(result.masks)}, Polygons: {sum(len(p) for p in result.polygons)}")
    else:
        result = segmenter.segment(args.source)
        img = cv2.imread(args.source)
        overlay = ImageSegmenter.visualize(img, result)
        if args.output:
            cv2.imwrite(args.output, overlay)
            print(f"Saved to {args.output}")
        print(f"Masks: {len(result.masks)}, Areas: {result.areas}")


if __name__ == "__main__":
    main()
