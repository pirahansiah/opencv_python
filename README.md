# OpenCV Python Workshop

[![Build Status](https://img.shields.io/appveyor/ci/pirahansiah/opencv-python/master.svg)](https://ci.appveyor.com/project/pirahansiah/opencv-python?branch=master)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-blue.svg)](https://pypi.org/project/opencv-python/)
[![Python](https://img.shields.io/badge/Python-3.10+-yellow.svg)](https://www.python.org)
[![YouTube](https://img.shields.io/badge/YouTube-Tutorials-red.svg)](https://www.youtube.com/tiziran)

Hands-on OpenCV workshop using Python. Covers image processing, feature detection, face recognition, video analysis, object detection, segmentation, and edge deployment.

> **Workshop by Dr. Farshid Pirahansiah** — [www.tiziran.com](https://www.tiziran.com) | [YouTube](https://www.youtube.com/tiziran)

## Setup

### Quick Install (2025-2026)

```bash
# Create virtual environment
python -m venv cv_workshop
source cv_workshop/bin/activate  # Linux/Mac
cv_workshop\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

### Legacy Install (from notebooks)

```bash
pip install numpy pandas matplotlib
pip install opencv-python opencv-contrib-python
pip install Pillow bokeh seaborn
```

## Contents

| File | Description |
|------|-------------|
| `opencv_functions.py` | Reusable utility functions (cartoon, face detection, display) |
| `opencv_python_visualcode.py` | VS Code integration examples |
| `list_files_directories.py` | File/directory listing utilities |
| `mat2numpy.py` | OpenCV Mat to NumPy array conversion |
| `read_all_image_in_folder.py` | Batch image loading |
| `progress_bar.py` | Progress bar utility |
| **`yolo_detector.py`** | **YOLOv11 real-time object detection** |
| **`segmentation.py`** | **SAM-2 style image/video segmentation** |
| **`video_analyzer.py`** | **Motion detection, tracking, optical flow** |
| **`edge_deploy.py`** | **ONNX export, INT8 quantization, benchmarking** |
| **`augmentation.py`** | **Image augmentation pipeline (Albumentations + OpenCV)** |
| **`features.py`** | **SIFT/ORB/AKAZE, feature matching, panorama stitching** |
| `haarcascades/` | Haar cascade classifiers for face/eye detection |
| `lbpcascades/` | LBP cascade classifiers (faster, lighter) |
| `tests/` | Pytest test suite |

## New Modules (v1.1)

### YOLO Detector (`yolo_detector.py`)

```python
from yolo_detector import YOLODetector

det = YOLODetector(confidence=0.25)
result = det.detect("photo.jpg")
for d in result.detections:
    print(f"{d.class_name}: {d.confidence:.2f} at {d.bbox}")

# Video/webcam
stats = det.detect_webcam(camera_index=0)
stats = det.detect_video("traffic.mp4", output_path="annotated.mp4")

# Export to ONNX
det.export_onnx("yolo11n.onnx")
```

### Segmentation (`segmentation.py`)

```python
from segmentation import ImageSegmenter

seg = ImageSegmenter("sam2_n")
result = seg.segment("photo.jpg")
overlay = ImageSegmenter.visualize(cv2.imread("photo.jpg"), result)
cv2.imwrite("segmented.jpg", overlay)

# Interactive: click foreground/background points
result = seg.segment_interactive("photo.jpg")

# Video segmentation
stats = seg.segment_video("video.mp4", output_path="seg_out.mp4")
```

### Video Analyzer (`video_analyzer.py`)

```python
from video_analyzer import VideoAnalyzer

analyzer = VideoAnalyzer("video.mp4")
result = analyzer.detect_motion(method="mog2", min_area=500)
print(f"Motion frames: {result.motion_frames}/{result.total_frames}")

# Object tracking
result = analyzer.track_objects(tracker_type="csrt")

# Optical flow
result = analyzer.compute_optical_flow(method="farneback")
```

### Edge Deployment (`edge_deploy.py`)

```python
from edge_deploy import EdgeDeployer

deployer = EdgeDeployer("yolo11n.pt")

# Export
onnx_path = deployer.convert_to_onnx("yolo11n.onnx")

# Quantize
result = deployer.quantize_int8("yolo11n.onnx")
print(f"Compression: {result.compression_ratio:.2f}x")

# Benchmark
bench = deployer.benchmark("yolo11n.onnx", num_runs=200)
print(f"Latency: {bench.avg_latency_ms:.1f}ms, FPS: {bench.throughput_fps:.1f}")

# Jetson deployment
deployer.deploy_jetson("yolo11n.onnx", "jetson_deploy/")
```

### Augmentation Pipeline (`augmentation.py`)

```python
from augmentation import AugmentationPipeline

pipeline = AugmentationPipeline()
pipeline.create_transforms(rotate_limit=45, color_jitter=True)

augmented = pipeline.augment(image, num_variations=5)

# Batch augment directory
pipeline.augment_directory("images/", "augmented/", num_variations=3)

# Visualize pipeline
vis = pipeline.visualize_pipeline(image, num_samples=8, output_path="vis.jpg")
```

### Feature Detection (`features.py`)

```python
from features import FeatureExtractor

extractor = FeatureExtractor()

# Extract features
kp, desc = extractor.extract_sift(image)

# Match two images
result = match_result = extractor.match_images(img1, img2, method="sift")
vis = extractor.draw_matches(img1, img2, result)

# Panorama stitching
panorama = extractor.panorama_stitch([img1, img2, img3])
```

## Topics Covered

### Fundamentals
- Image I/O (read, write, display)
- Color space conversion (BGR, HSV, GRAY)
- Geometric transformations (resize, rotate, warp)
- Drawing and annotations

### Image Processing
- Filtering (Gaussian, median, bilateral)
- Thresholding (adaptive, Otsu)
- Morphological operations
- Edge detection (Canny, Sobel, Laplacian)
- Histogram equalization and analysis

### Feature Detection & Matching
- ORB, SIFT, AKAZE feature extraction
- BFMatcher and FLANN matching
- Lowe's ratio test
- Homography estimation (RANSAC)
- Panorama stitching

### Object Detection
- Haar/LBP cascade face/eye detection
- HOG pedestrian detection
- Contour-based object detection
- **YOLOv11** real-time detection with batch support

### Segmentation
- **SAM-2** interactive and auto-segmentation
- Mask visualization and polygon extraction
- Video frame-by-frame segmentation

### Video Processing
- Webcam capture and processing
- Background subtraction (MOG2, KNN)
- Optical flow (Farneback, Lucas-Kanade)
- Object tracking (CSRT, KCF, MOSSE)
- FPS and performance statistics

### Deep Learning & Edge Deployment
- ONNX Runtime inference in Python
- INT8 dynamic quantization
- Benchmark comparison (latency, memory, throughput)
- TensorRT export hints
- Jetson deployment package generation

### Data Augmentation
- Rotation, flip, crop, color jitter
- Albumentations integration with OpenCV fallback
- Batch augmentation from directories
- Dataset export with CSV labels

## Modern Python CV Stack (2025-2026)

| Library | Use Case |
|---------|----------|
| `opencv-python` | Core image processing and CV |
| `ultralytics` | YOLOv11 detection/segmentation |
| `onnxruntime` | Cross-platform DNN inference |
| `albumentations` | Advanced image augmentation |
| `supervision` | Video annotation and tracking |
| `mediapipe` | Face/hand/pose landmarks |
| `torchvision` | PyTorch vision utilities |

## CLI Usage

Each module can be run standalone:

```bash
# YOLO detection
python yolo_detector.py photo.jpg --conf 0.3 --output result.jpg
python yolo_detector.py video.mp4 --output annotated.mp4
python yolo_detector.py 0  # webcam

# Segmentation
python segmentation.py photo.jpg --interactive
python segmentation.py video.mp4 --output seg_out.mp4

# Video analysis
python video_analyzer.py video.mp4 --mode motion --motion-method knn
python video_analyzer.py video.mp4 --mode track --tracker csrt
python video_analyzer.py video.mp4 --mode flow --flow-method farneback

# Edge deployment
python edge_deploy.py yolo11n.pt --convert --quantize yolo11n.onnx
python edge_deploy.py --benchmark yolo11n.onnx --num-runs 200

# Augmentation
python augmentation.py images/ --num 5
python augmentation.py photo.jpg --visualize

# Feature matching
python features.py img1.jpg img2.jpg --method sift --output matches.jpg
python features.py img1.jpg img2.jpg img3.jpg --stitch --output panorama.jpg
```

## Resources

- [OpenCV Python Tutorials](https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html)
- [Ultralytics YOLO](https://docs.ultralytics.com)
- [ONNX Runtime Python](https://onnxruntime.ai/docs/get-started/with-python.html)
- [Albumentations](https://albumentations.ai/docs/)
- [YouTube Channel](https://www.youtube.com/tiziran)

## 12-Month Roadmap (2025-2026)

| Month | Milestone | Status |
|-------|-----------|--------|
| **Jul 2025** | Python 3.10+ migration, type hints, pathlib | Done |
| **Aug 2025** | Pytest test suite >80% coverage, CI/CD | Done |
| **Sep 2025** | Docker multi-stage builds, GPU support | Done |
| **Oct 2025** | YOLOv11 integration, ONNX Runtime examples | Done |
| **Nov 2025** | SAM 2 segmentation, real-time inference | Done |
| **Dec 2025** | Edge deployment guide (Jetson, Raspberry Pi) | Done |
| **Jan 2026** | INT8 quantization, benchmark comparison | Done |
| **Feb 2026** | Feature matching, panorama stitching | Done |
| **Mar 2026** | Video analysis (motion, tracking, optical flow) | Done |
| **Apr 2026** | Augmentation pipeline (Albumentations + OpenCV) | Done |
| **May 2026** | Multi-camera system examples, 3D reconstruction | Pending |
| **Jun 2026** | v1.0 release, comprehensive documentation | Pending |

### Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.1.0 | 2026 | Add YOLO detector, segmentation, video analyzer, edge deploy, augmentation, features |
| 1.0.0 | 2025 | Full modernization: Python 3.10+, type hints, pathlib, pytest, Docker |
| 0.x | 2019-2024 | Initial workshop materials |

## License

See repository for license details.
