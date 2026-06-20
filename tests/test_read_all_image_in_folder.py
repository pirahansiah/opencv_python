"""Tests for read_all_image_in_folder module."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from read_all_image_in_folder import read_images_from_folder


def test_read_images_empty_folder(tmp_path: Path) -> None:
    result = read_images_from_folder(tmp_path, delay=1)
    assert result == []


def test_read_images_with_files(tmp_path: Path) -> None:
    for i in range(3):
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        cv2.imwrite(str(tmp_path / f"test_{i}.png"), img)
    result = read_images_from_folder(tmp_path, delay=1)
    assert len(result) == 3
