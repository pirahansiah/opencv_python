# OpenCV Python Workshop

[![Build Status](https://img.shields.io/appveyor/ci/pirahansiah/opencv-python/master.svg)](https://ci.appveyor.com/project/pirahansiah/opencv-python?branch=master)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-blue.svg)](https://pypi.org/project/opencv-python/)
[![Python](https://img.shields.io/badge/Python-3.10+-yellow.svg)](https://www.python.org)
[![YouTube](https://img.shields.io/badge/YouTube-Tutorials-red.svg)](https://www.youtube.com/tiziran)

Hands-on OpenCV workshop using Python. Covers image processing, feature detection, face recognition, video analysis, and real-time computer vision applications.

> **Workshop by Dr. Farshid Pirahansiah** — [www.tiziran.com](https://www.tiziran.com) | [YouTube](https://www.youtube.com/tiziran)

## Setup

### Quick Install (2025-2026)

```bash
# Create virtual environment
python -m venv cv_workshop
source cv_workshop/bin/activate  # Linux/Mac
cv_workshop\Scripts\activate     # Windows

# Install dependencies
pip install opencv-python opencv-contrib-python
pip install numpy matplotlib
pip install Pillow
pip install ultralytics          # YOLOv8/v11
pip install onnxruntime          # ONNX inference
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
| `farshid-steps.ipynb` | Step-by-step workshop notebook (all examples) |
| `farshid.ipynb` | Additional exercises and examples |
| `opencv_functions.py` | Reusable utility functions (cartoon, face detection, display) |
| `opencv_python_visualcode.py` | VS Code integration examples |
| `list_files_directories.py` | File/directory listing utilities |
| `mat2numpy.py` | OpenCV Mat to NumPy array conversion |
| `read_all_image_in_folder.py` | Batch image loading |
| `progress_bar.py` | Progress bar utility |
| `haarcascades/` | Haar cascade classifiers for face/eye detection |
| `lbpcascades/` | LBP cascade classifiers (faster, lighter) |

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
- Corner detection (Harris, Shi-Tomasi)
- ORB, SIFT, SURF feature extraction
- Feature matching and homography

### Object Detection
- Haar cascade face/eye detection
- HOG pedestrian detection
- Contour-based object detection
- **YOLOv8/YOLOv11** — Real-time object detection (2025 standard)

### Video Processing
- Webcam capture and processing
- Background subtraction
- Optical flow estimation
- Object tracking (CSRT, KCF, MOSSE)

### Deep Learning (2025-2026)
- ONNX Runtime inference in Python
- OpenCV DNN module with Python bindings
- YOLO inference pipelines
- Edge deployment with ONNX quantization

## Modern Python CV Stack (2025-2026)

| Library | Use Case |
|---------|----------|
| `opencv-python` | Core image processing and CV |
| `ultralytics` | YOLOv8/v11 object detection/segmentation |
| `onnxruntime` | Cross-platform DNN inference |
| `mediapipe` | Face/hand/pose landmarks |
| `torchvision` | PyTorch vision utilities |
| `albumentations` | Advanced image augmentation |
| `supervision` | Video annotation and tracking |
| `roboflow` | Dataset management and training |

## Example: Modern Face Detection

```python
import cv2

# Classic Haar cascade (included in this repo)
face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')

# Modern DNN-based (better accuracy)
net = cv2.dnn.readNetFromCaffe(
    'deploy.prototxt',
    'res10_300x300_ssd_iter_140000.caffemodel'
)

img = cv2.imread('photo.jpg')
blob = cv2.dnn.blobFromImage(img, 1.0, (300, 300))
net.setInput(blob)
detections = net.forward()
```

## Example: YOLOv11 (2025-2026)

```python
from ultralytics import YOLO

model = YOLO('yolo11n.pt')  # nano model for edge deployment
results = model('photo.jpg')

for r in results:
    for box in r.boxes:
        print(f"Class: {r.names[int(box.cls)]}, Conf: {box.conf:.2f}")
```

## Resources

- [OpenCV Python Tutorials](https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html)
- [OpenCV Python API](https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html)
- [Ultralytics YOLO](https://docs.ultralytics.com)
- [ONNX Runtime Python](https://onnxruntime.ai/docs/get-started/with-python.html)
- [YouTube Channel](https://www.youtube.com/tiziran)

## License

See repository for license details.
