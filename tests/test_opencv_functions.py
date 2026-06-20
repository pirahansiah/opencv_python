"""Tests for opencv_functions module."""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
import pytest

from opencv_functions import (
    cartoon_image,
    face_detection,
    save_image_opencv,
)


def test_cartoon_image_returns_correct_shape(sample_bgr_image: np.ndarray) -> None:
    result = cartoon_image(sample_bgr_image)
    assert result.shape == sample_bgr_image.shape


def test_cartoon_image_with_different_params(sample_bgr_image: np.ndarray) -> None:
    result = cartoon_image(sample_bgr_image, num_down=2, num_bilateral=3)
    assert result.shape == sample_bgr_image.shape


def test_save_image_opencv_success(sample_bgr_image: np.ndarray, tmp_path: Path) -> None:
    output = tmp_path / "test.jpg"
    assert save_image_opencv(output, sample_bgr_image) is True
    assert output.exists()
    loaded = cv2.imread(str(output))
    assert loaded is not None
    assert loaded.shape == sample_bgr_image.shape


def test_save_image_opencv_creates_parent_dirs(sample_bgr_image: np.ndarray, tmp_path: Path) -> None:
    output = tmp_path / "sub" / "deep" / "test.jpg"
    assert save_image_opencv(output, sample_bgr_image) is True
    assert output.exists()


def test_save_image_opencv_default_name(sample_bgr_image: np.ndarray, tmp_path: Path) -> None:
    os.chdir(tmp_path)
    result = save_image_opencv("", sample_bgr_image)
    assert result is True
    assert (tmp_path / "output.jpg").exists()


def test_face_detection_no_faces(sample_black_image: np.ndarray) -> None:
    result = face_detection(sample_black_image)
    assert result.shape == sample_black_image.shape
