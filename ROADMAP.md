# ROADMAP.md — OpenCV Python Workshop

## 12-Month Vision (Jul 2025 – Jun 2026)

Transform the workshop into the definitive OpenCV + Edge AI education platform with production-grade tooling, GPU-accelerated pipelines, and comprehensive deployment guides.

### Q1 (Jul–Sep 2025): Foundation

- [ ] Migrate all modules to Python 3.10+ with full type hints and pathlib
- [ ] Achieve >80% pytest coverage with GitHub Actions CI
- [ ] Add multi-stage Docker builds with NVIDIA GPU runtime support
- [ ] Modernize requirements.txt to pyproject.toml with dependency groups

### Q2 (Oct–Dec 2025): Modern Inference

- [ ] YOLOv11 integration tutorial with real-time detection demo
- [ ] ONNX Runtime examples for cross-platform DNN inference
- [ ] SAM 2 segmentation tutorial with interactive prompting
- [ ] Edge deployment guide (Jetson Orin, Raspberry Pi 5)

### Q3 (Jan–Mar 2026): Performance

- [ ] TensorRT 10 optimization with FP16/INT8 quantization examples
- [ ] Vision-Language Model examples (CLIP, LLaVA) for image understanding
- [ ] Multi-camera system examples with 3D reconstruction basics
- [ ] GStreamer pipeline integration for low-latency video

### Q4 (Apr–Jun 2026): Release

- [ ] MediaPipe advanced features (hand tracking, pose estimation)
- [ ] v1.0 release with comprehensive documentation and video tutorials
- [ ] Performance benchmark suite across hardware targets
- [ ] Community contribution guidelines and plugin architecture

## Technical Debt

- [ ] Remove hardcoded paths from notebook cells (farshid-steps.ipynb)
- [ ] Consolidate duplicate utility functions across .py files
- [ ] Replace deprecated `cv2.VideoCapture` patterns with context managers
- [ ] Add missing type stubs for OpenCV contrib modules
- [ ] Fix inconsistent naming (snake_case vs camelCase in older functions)
- [ ] Remove Python 2 compatibility shims (`from __future__` no longer needed)
- [ ] Upgrade from AppVeyor to GitHub Actions for CI
- [ ] Add pre-commit hooks with ruff + mypy

## Future Features

- [ ] Real-time streaming inference with WebSocket support
- [ ] Browser-based demo with OpenCV.js + WebAssembly
- [ ] Model Zoo with pre-trained weights for common tasks
- [ ] Interactive Jupyter/Colab notebooks with GPU runtime
- [ ] Benchmark dashboard comparing ONNX, TensorRT, and OpenVINO
- [ ] Multi-language bindings (C++, Java) for the same pipeline
- [ ] Automated hardware detection and optimal pipeline selection
