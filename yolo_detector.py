"""Modern YOLOv11 object detection with ultralytics."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np


@dataclass
class Detection:
    """Single detection result."""

    class_id: int
    class_name: str
    confidence: float
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2


@dataclass
class DetectionResult:
    """Container for detection results on a single image."""

    detections: list[Detection] = field(default_factory=list)
    fps: float = 0.0
    inference_ms: float = 0.0
    image_shape: tuple[int, int, int] = (0, 0, 0)


class YOLODetector:
    """YOLOv11 object detection wrapper.

    Args:
        model_path: Path to YOLO model (.pt or .onnx). Downloads yolo11n.pt if None.
        confidence: Detection confidence threshold.
        device: Device string ('cpu', 'cuda', 'mps').
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        confidence: float = 0.25,
        device: str = "cpu",
    ) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise ImportError("Install ultralytics: pip install ultralytics") from e

        if model_path is None:
            self._model = YOLO("yolo11n.pt")
        else:
            self._model = YOLO(str(model_path))
        self._confidence = confidence
        self._device = device
        self._prev_time: float = 0.0
        self._fps_history: list[float] = []

    @property
    def class_names(self) -> dict[int, str]:
        return self._model.names

    def detect(
        self,
        source: str | Path | np.ndarray,
        confidence: float | None = None,
        max_detections: int = 300,
    ) -> DetectionResult:
        """Run detection on a single image.

        Args:
            source: Image path, URL, or numpy array (BGR).
            confidence: Override default confidence threshold.
            max_detections: Maximum detections per image.

        Returns:
            DetectionResult with bounding boxes, classes, and timing.
        """
        conf = confidence if confidence is not None else self._confidence
        t0 = time.perf_counter()
        results = self._model.predict(
            source=str(source) if isinstance(source, (str, Path)) else source,
            conf=conf,
            max_det=max_detections,
            device=self._device,
            verbose=False,
        )
        inference_ms = (time.perf_counter() - t0) * 1000.0

        detections: list[Detection] = []
        img_shape = (0, 0, 0)
        for r in results:
            img_shape = r.orig_shape + (3,) if len(r.orig_shape) == 2 else r.orig_shape
            if r.boxes is None:
                continue
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int).tolist()
                detections.append(
                    Detection(
                        class_id=int(box.cls.item()),
                        class_name=r.names[int(box.cls.item())],
                        confidence=float(box.conf.item()),
                        bbox=(x1, y1, x2, y2),
                    )
                )

        now = time.perf_counter()
        fps = 1.0 / max(now - self._prev_time, 1e-9) if self._prev_time > 0 else 0.0
        self._prev_time = now
        self._fps_history.append(fps)

        return DetectionResult(
            detections=detections, fps=fps, inference_ms=inference_ms, image_shape=img_shape
        )

    def detect_batch(
        self,
        sources: list[str | Path | np.ndarray],
        confidence: float | None = None,
    ) -> list[DetectionResult]:
        """Run detection on a batch of images.

        Args:
            sources: List of image paths or arrays.
            confidence: Override default confidence threshold.

        Returns:
            List of DetectionResult per image.
        """
        return [self.detect(s, confidence) for s in sources]

    def detect_video(
        self,
        source: str | Path | int = 0,
        output_path: str | Path | None = None,
        confidence: float | None = None,
        show: bool = False,
    ) -> dict[str, float]:
        """Run detection on a video file or stream.

        Args:
            source: Video path, RTSP URL, or camera index.
            output_path: Save annotated video to this path. None = don't save.
            confidence: Override default confidence threshold.
            show: Display frames in a window.

        Returns:
            Dict with total_frames, avg_fps, total_time_s.
        """
        conf = confidence if confidence is not None else self._confidence
        cap = cv2.VideoCapture(str(source) if not isinstance(source, int) else source)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {source}")

        writer: cv2.VideoWriter | None = None
        if output_path is not None:
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

        total_frames = 0
        total_time = 0.0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            t0 = time.perf_counter()
            results = self._model.predict(frame, conf=conf, device=self._device, verbose=False)
            dt = time.perf_counter() - t0
            total_time += dt
            total_frames += 1

            for r in results:
                annotated = r.plot()
                if show:
                    cv2.imshow("YOLO Detection", annotated)
                if writer is not None:
                    writer.write(annotated)

            if show and cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        if writer is not None:
            writer.release()
        if show:
            cv2.destroyAllWindows()

        avg_fps = total_frames / max(total_time, 1e-9)
        return {
            "total_frames": float(total_frames),
            "avg_fps": avg_fps,
            "total_time_s": total_time,
        }

    def detect_webcam(
        self,
        camera_index: int = 0,
        confidence: float | None = None,
        output_path: str | Path | None = None,
    ) -> dict[str, float]:
        """Real-time detection from webcam.

        Args:
            camera_index: Camera device index.
            confidence: Override default confidence threshold.
            output_path: Optional path to save the annotated stream.

        Returns:
            Dict with session stats.
        """
        return self.detect_video(
            source=camera_index,
            output_path=output_path,
            confidence=confidence,
            show=True,
        )

    def export_onnx(
        self,
        output_path: str | Path = "yolo11n.onnx",
        input_size: tuple[int, int] = (640, 640),
        simplify: bool = True,
    ) -> Path:
        """Export model to ONNX format.

        Args:
            output_path: Output ONNX file path.
            input_size: Input image size (H, W).
            simplify: Run onnx-simplifier after export.

        Returns:
            Path to exported ONNX file.
        """
        out = Path(output_path)
        self._model.export(format="onnx", imgsz=input_size, simplify=simplify)
        return out

    def average_fps(self) -> float:
        """Return average FPS across all detections in this session."""
        return sum(self._fps_history) / len(self._fps_history) if self._fps_history else 0.0

    def draw_detections(
        self,
        image: np.ndarray,
        result: DetectionResult,
        color: tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2,
    ) -> np.ndarray:
        """Draw detection boxes and labels on an image.

        Args:
            image: Input BGR image.
            result: DetectionResult to draw.
            color: BGR color for boxes.
            thickness: Line thickness.

        Returns:
            Annotated image copy.
        """
        annotated = image.copy()
        for det in result.detections:
            x1, y1, x2, y2 = det.bbox
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)
            label = f"{det.class_name} {det.confidence:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated, (x1, y1 - th - 10), (x1 + tw, y1), color, -1)
            cv2.putText(
                annotated, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1
            )
        return annotated


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="YOLOv11 object detection")
    parser.add_argument("source", help="Image, video path, or camera index (0)")
    parser.add_argument("--model", default=None, help="YOLO model path (default: yolo11n.pt)")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--device", default="cpu", help="Device (cpu/cuda/mps)")
    parser.add_argument("--output", default=None, help="Save annotated output")
    parser.add_argument("--export-onnx", default=None, help="Export model to ONNX")
    args = parser.parse_args()

    if args.export_onnx:
        det = YOLODetector(args.model, args.conf, args.device)
        path = det.export_onnx(args.export_onnx)
        print(f"Exported to {path}")
        return

    det = YOLODetector(args.model, args.conf, args.device)

    source = int(args.source) if args.source.isdigit() else args.source
    if isinstance(source, int):
        stats = det.detect_webcam(source, args.conf, args.output)
    elif any(args.source.lower().endswith(ext) for ext in (".mp4", ".avi", ".mov", ".mkv")):
        stats = det.detect_video(args.source, args.output, args.conf, show=True)
    else:
        result = det.detect(args.source)
        img = cv2.imread(args.source) if Path(args.source).exists() else None
        if img is not None:
            annotated = det.draw_detections(img, result)
            if args.output:
                cv2.imwrite(args.output, annotated)
                print(f"Saved to {args.output}")
            else:
                cv2.imshow("YOLO", annotated)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
        stats = {"fps": result.fps, "inference_ms": result.inference_ms, "detections": len(result.detections)}

    print(f"Stats: {stats}")


if __name__ == "__main__":
    main()
