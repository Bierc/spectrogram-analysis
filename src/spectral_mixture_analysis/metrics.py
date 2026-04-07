from __future__ import annotations

import numpy as np


def l2_error(a: np.ndarray, b: np.ndarray) -> float:
    """Compute L2 norm of the difference."""
    return float(np.linalg.norm(a - b))


def mean_absolute_error(a: np.ndarray, b: np.ndarray) -> float:
    """Compute mean absolute error."""
    return float(np.mean(np.abs(a - b)))


def relative_l2_error(a: np.ndarray, b: np.ndarray, eps: float = 1e-10) -> float:
    """
    Compute relative L2 error: ||a-b|| / ||a||.
    """
    denom = np.linalg.norm(a) + eps
    return float(np.linalg.norm(a - b) / denom)

def spectral_overlap(mag_x: np.ndarray, mag_y: np.ndarray) -> float:
    """
    Compute spectral overlap between two magnitude spectrograms.

    overlap = sum(min(x, y)) / sum(x + y)
    """
    numerator = np.sum(np.minimum(mag_x, mag_y))
    denominator = np.sum(mag_x + mag_y) + 1e-12
    return float(numerator / denominator)