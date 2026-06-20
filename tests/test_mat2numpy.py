"""Tests for mat2numpy module."""

from __future__ import annotations

import numpy as np

from mat2numpy import mat_to_numpy


def test_mat_to_numpy_shapes() -> None:
    img = np.random.randint(0, 256, (50, 100, 3), dtype=np.uint8)
    binary, reshaped = mat_to_numpy(img)
    assert binary.shape == img.shape
    assert reshaped.shape == (50, 100)


def test_mat_to_numpy_threshold_values() -> None:
    img = np.full((10, 10, 3), 100, dtype=np.uint8)
    binary, _ = mat_to_numpy(img, threshold=128)
    assert np.all(binary == 0)


def test_mat_to_numpy_above_threshold() -> None:
    img = np.full((10, 10, 3), 200, dtype=np.uint8)
    binary, _ = mat_to_numpy(img, threshold=128)
    assert np.all(binary == 255)
