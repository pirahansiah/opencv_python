"""Shared fixtures for pytest."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def sample_bgr_image() -> np.ndarray:
    """Create a 100x100 BGR test image."""
    return np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)


@pytest.fixture
def sample_grayscale_image() -> np.ndarray:
    """Create a 100x100 grayscale test image."""
    return np.random.randint(0, 256, (100, 100), dtype=np.uint8)


@pytest.fixture
def sample_black_image() -> np.ndarray:
    """Create a 100x100 all-black BGR image."""
    return np.zeros((100, 100, 3), dtype=np.uint8)


@pytest.fixture
def sample_white_image() -> np.ndarray:
    """Create a 100x100 all-white BGR image."""
    return np.ones((100, 100, 3), dtype=np.uint8) * 255
