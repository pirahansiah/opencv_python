"""Advanced video analysis: motion detection, object tracking, optical flow."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np


@dataclass
class AnalysisResult:
    """Result container for video analysis."""

    total_frames: int = 0
    avg_fps: float = 0.0
    total_time_s: float = 0.0
    motion_frames: int = 0
    tracked_objects: list[tuple[int, int, int, int]] = field(default_factory=list)


class VideoAnalyzer:
    """Video analysis toolkit with background subtraction, tracking, and optical flow.

    Args:
        source: Video path, RTSP URL, or camera index.
    """

    def __init__(self, source: str | Path | int) -> None:
        if isinstance(source, int):
            self._cap = cv2.VideoCapture(source)
        else:
            self._cap = cv2.VideoCapture(str(source))
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {source}")

        self._width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0

    def __del__(self) -> None:
        if hasattr(self, "_cap") and self._cap.isOpened():
            self._cap.release()

    def analyze(
        self,
        max_frames: int = 0,
        show: bool = False,
        output_path: str | Path | None = None,
    ) -> AnalysisResult:
        """Basic analysis: frame count, FPS, timing.

        Args:
            max_frames: Stop after N frames. 0 = process all.
            show: Display frames.
            output_path: Save processed video.

        Returns:
            AnalysisResult with stats.
        """
        writer: cv2.VideoWriter | None = None
        if output_path is not None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(output_path), fourcc, self._fps, (self._width, self._height))

        total_frames = 0
        total_time = 0.0

        while True:
            if 0 < max_frames <= total_frames:
                break
            ret, frame = self._cap.read()
            if not ret:
                break

            t0 = time.perf_counter()
            total_time += time.perf_counter() - t0
            total_frames += 1

            if show:
                cv2.imshow("Analysis", frame)
            if writer is not None:
                writer.write(frame)
            if show and cv2.waitKey(1) & 0xFF == ord("q"):
                break

        self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        if writer is not None:
            writer.release()
        if show:
            cv2.destroyAllWindows()

        avg_fps = total_frames / max(total_time, 1e-9)
        return AnalysisResult(
            total_frames=total_frames, avg_fps=avg_fps, total_time_s=total_time
        )

    def detect_motion(
        self,
        method: str = "mog2",
        history: int = 500,
        threshold: int = 50,
        min_area: int = 500,
        max_frames: int = 0,
        show: bool = False,
        output_path: str | Path | None = None,
    ) -> AnalysisResult:
        """Detect motion using background subtraction.

        Args:
            method: 'mog2' or 'knn'.
            history: Background history length.
            threshold: Motion threshold.
            min_area: Minimum contour area to count as motion.
            max_frames: Stop after N frames. 0 = process all.
            show: Display frames with contours.
            output_path: Save annotated video.

        Returns:
            AnalysisResult with motion frame count.
        """
        if method == "mog2":
            bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=history, varThreshold=threshold)
        elif method == "knn":
            bg_subtractor = cv2.createBackgroundSubtractorKNN(history=history, dist2Threshold=threshold)
        else:
            raise ValueError(f"Unknown method: {method}. Use 'mog2' or 'knn'.")

        writer: cv2.VideoWriter | None = None
        if output_path is not None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(output_path), fourcc, self._fps, (self._width, self._height))

        total_frames = 0
        motion_frames = 0
        total_time = 0.0

        while True:
            if 0 < max_frames <= total_frames:
                break
            ret, frame = self._cap.read()
            if not ret:
                break

            t0 = time.perf_counter()
            fg_mask = bg_subtractor.apply(frame)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)

            contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            has_motion = False

            for c in contours:
                if cv2.contourArea(c) >= min_area:
                    has_motion = True
                    x, y, w, h = cv2.boundingRect(c)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, "Motion", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            if has_motion:
                motion_frames += 1

            total_time += time.perf_counter() - t0
            total_frames += 1

            if show:
                combined = np.hstack([frame, cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR)])
                cv2.imshow("Motion Detection", combined)
            if writer is not None:
                writer.write(frame)
            if show and cv2.waitKey(1) & 0xFF == ord("q"):
                break

        self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        if writer is not None:
            writer.release()
        if show:
            cv2.destroyAllWindows()

        avg_fps = total_frames / max(total_time, 1e-9)
        return AnalysisResult(
            total_frames=total_frames,
            avg_fps=avg_fps,
            total_time_s=total_time,
            motion_frames=motion_frames,
        )

    def track_objects(
        self,
        tracker_type: str = "csrt",
        bbox: tuple[int, int, int, int] | None = None,
        max_frames: int = 0,
        show: bool = True,
        output_path: str | Path | None = None,
    ) -> AnalysisResult:
        """Track objects using OpenCV trackers.

        Args:
            tracker_type: 'csrt', 'kcf', or 'mosse'.
            bbox: Initial bounding box (x, y, w, h). If None, user selects via ROI.
            max_frames: Stop after N frames. 0 = process all.
            show: Display tracking window.
            output_path: Save tracked video.

        Returns:
            AnalysisResult with tracked objects.
        """
        ret, frame = self._cap.read()
        if not ret:
            raise RuntimeError("Cannot read first frame for tracking")

        if bbox is None:
            bbox_cv2 = cv2.selectROI("Select Object", frame, fromCenter=False, showCrosshair=True)
            cv2.destroyWindow("Select Object")
            bbox = bbox_cv2

        tracker = self._create_tracker(tracker_type)
        tracker.init(frame, bbox)

        writer: cv2.VideoWriter | None = None
        if output_path is not None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(output_path), fourcc, self._fps, (self._width, self._height))

        total_frames = 0
        total_time = 0.0
        tracked_boxes: list[tuple[int, int, int, int]] = []

        while True:
            if 0 < max_frames <= total_frames:
                break
            ret, frame = self._cap.read()
            if not ret:
                break

            t0 = time.perf_counter()
            success, new_bbox = tracker.update(frame)
            total_time += time.perf_counter() - t0
            total_frames += 1

            if success:
                x, y, w, h = [int(v) for v in new_bbox]
                tracked_boxes.append((x, y, w, h))
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, tracker_type.upper(), (x, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "LOST", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

            if show:
                cv2.imshow("Tracking", frame)
            if writer is not None:
                writer.write(frame)
            if show and cv2.waitKey(1) & 0xFF == ord("q"):
                break

        self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        if writer is not None:
            writer.release()
        if show:
            cv2.destroyAllWindows()

        avg_fps = total_frames / max(total_time, 1e-9)
        return AnalysisResult(
            total_frames=total_frames,
            avg_fps=avg_fps,
            total_time_s=total_time,
            tracked_objects=tracked_boxes,
        )

    def compute_optical_flow(
        self,
        method: str = "farneback",
        max_frames: int = 0,
        show: bool = True,
        output_path: str | Path | None = None,
    ) -> AnalysisResult:
        """Compute and visualize dense optical flow.

        Args:
            method: 'farneback' (dense) or 'lucas_kanade' (sparse).
            max_frames: Stop after N frames. 0 = process all.
            show: Display flow visualization.
            output_path: Save flow video.

        Returns:
            AnalysisResult with stats.
        """
        ret, prev_frame = self._cap.read()
        if not ret:
            raise RuntimeError("Cannot read first frame for optical flow")
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

        writer: cv2.VideoWriter | None = None
        if output_path is not None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(output_path), fourcc, self._fps, (self._width, self._height))

        total_frames = 0
        total_time = 0.0

        while True:
            if 0 < max_frames <= total_frames:
                break
            ret, frame = self._cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            t0 = time.perf_counter()

            if method == "farneback":
                flow = cv2.calcOpticalFlowFarneback(
                    prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
                )
                hsv = np.zeros_like(frame)
                hsv[..., 1] = 255
                magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                hsv[..., 0] = angle * 180 / np.pi / 2
                hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
                flow_vis = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            elif method == "lucas_kanade":
                feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
                p0 = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)
                if p0 is not None:
                    lk_params = dict(winSize=(15, 15), maxLevel=2,
                                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
                    p1, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, p0, None, **lk_params)
                    flow_vis = frame.copy()
                    if p1 is not None:
                        good_new = p1[status == 1]
                        good_old = p0[status == 1]
                        for new, old in zip(good_new, good_old):
                            x_new, y_new = new.ravel().astype(int)
                            x_old, y_old = old.ravel().astype(int)
                            cv2.line(flow_vis, (x_old, y_old), (x_new, y_new), (0, 255, 0), 2)
                            cv2.circle(flow_vis, (x_new, y_new), 4, (0, 0, 255), -1)
                else:
                    flow_vis = frame.copy()
            else:
                raise ValueError(f"Unknown method: {method}. Use 'farneback' or 'lucas_kanade'.")

            total_time += time.perf_counter() - t0
            total_frames += 1
            prev_gray = gray

            if show:
                cv2.imshow("Optical Flow", flow_vis)
            if writer is not None:
                writer.write(flow_vis)
            if show and cv2.waitKey(1) & 0xFF == ord("q"):
                break

        self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        if writer is not None:
            writer.release()
        if show:
            cv2.destroyAllWindows()

        avg_fps = total_frames / max(total_time, 1e-9)
        return AnalysisResult(
            total_frames=total_frames, avg_fps=avg_fps, total_time_s=total_time
        )

    @staticmethod
    def _create_tracker(tracker_type: str) -> cv2.Tracker:
        """Create an OpenCV tracker by name."""
        trackers = {
            "csrt": cv2.TrackerCSRT_create,
            "kcf": cv2.TrackerKCF_create,
            "mosse": cv2.TrackerMOSSE_create,
        }
        factory = trackers.get(tracker_type.lower())
        if factory is None:
            raise ValueError(f"Unknown tracker: {tracker_type}. Use {list(trackers.keys())}")
        return factory()

    @property
    def video_info(self) -> dict[str, float]:
        """Return video metadata."""
        return {
            "width": self._width,
            "height": self._height,
            "fps": self._fps,
            "frame_count": self._cap.get(cv2.CAP_PROP_FRAME_COUNT),
        }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Video analyzer")
    parser.add_argument("source", help="Video path or camera index (0)")
    parser.add_argument("--mode", choices=["analyze", "motion", "track", "flow"], default="analyze")
    parser.add_argument("--tracker", default="csrt", help="Tracker type for track mode")
    parser.add_argument("--motion-method", default="mog2", help="Background subtraction method")
    parser.add_argument("--flow-method", default="farneback", help="Optical flow method")
    parser.add_argument("--output", default=None, help="Save output video")
    parser.add_argument("--max-frames", type=int, default=0, help="Max frames to process")
    args = parser.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source
    analyzer = VideoAnalyzer(source)
    print(f"Video info: {analyzer.video_info}")

    if args.mode == "analyze":
        result = analyzer.analyze(args.max_frames, show=True, output_path=args.output)
    elif args.mode == "motion":
        result = analyzer.detect_motion(
            method=args.motion_method, max_frames=args.max_frames, show=True, output_path=args.output
        )
    elif args.mode == "track":
        result = analyzer.track_objects(
            tracker_type=args.tracker, max_frames=args.max_frames, show=True, output_path=args.output
        )
    elif args.mode == "flow":
        result = analyzer.compute_optical_flow(
            method=args.flow_method, max_frames=args.max_frames, show=True, output_path=args.output
        )
    else:
        raise ValueError(f"Unknown mode: {args.mode}")

    print(f"Result: frames={result.total_frames}, fps={result.avg_fps:.1f}, time={result.total_time_s:.2f}s")


if __name__ == "__main__":
    main()
