from __future__ import annotations

from typing import Dict

import numpy as np

from .transforms import (
    compute_stft_complex,
    magnitude_spectrogram,
)
from .metrics import l2_error, relative_l2_error, mean_absolute_error


def evaluate_mixture_linearity_stft(
    x: np.ndarray,
    y: np.ndarray,
    n_fft: int = 2048,
    hop_length: int = 512,
) -> Dict[str, float]:
    """
    Evaluate how linear the STFT magnitude representation is under signal mixing.

    Computes:
        T(x), T(y), T(x+y)
        and compares:
        |T(x+y)| vs |T(x)| + |T(y)|

    Args:
        x: First signal (already aligned)
        y: Second signal (already aligned)
        n_fft: FFT size
        hop_length: Hop size

    Returns:
        Dictionary with metrics.
    """

    if len(x) != len(y):
        raise ValueError("Signals must have the same length")

    # Mixture
    mixture = x + y

    # STFT
    X = compute_stft_complex(x, n_fft=n_fft, hop_length=hop_length)
    Y = compute_stft_complex(y, n_fft=n_fft, hop_length=hop_length)
    M = compute_stft_complex(mixture, n_fft=n_fft, hop_length=hop_length)

    # Magnitudes
    mag_x = magnitude_spectrogram(X)
    mag_y = magnitude_spectrogram(Y)
    mag_mix = magnitude_spectrogram(M)

    mag_sum = mag_x + mag_y

    # Metrics
    l2 = l2_error(mag_mix, mag_sum)
    rel_l2 = relative_l2_error(mag_mix, mag_sum)
    mae = mean_absolute_error(mag_mix, mag_sum)

    return {
        "l2": l2,
        "relative_l2": rel_l2,
        "mae": mae,
    }

from typing import List, Tuple
import pandas as pd

from .audio import load_mono_audio, match_length


def evaluate_dataset_pairs(
    pairs: List[Tuple[str, str]],
    sr: int = 22050,
) -> pd.DataFrame:
    """
    Evaluate multiple audio pairs.

    Args:
        pairs: List of (path_x, path_y)
        sr: Sample rate

    Returns:
        DataFrame with metrics per pair
    """

    results = []

    for path_x, path_y in pairs:
        x, _ = load_mono_audio(path_x, sr=sr)
        y, _ = load_mono_audio(path_y, sr=sr)

        x, y = match_length(x, y, mode="truncate")

        metrics = evaluate_mixture_linearity_stft(x, y)

        results.append({
            "file_x": str(path_x),
            "file_y": str(path_y),
            **metrics,
        })

    return pd.DataFrame(results)