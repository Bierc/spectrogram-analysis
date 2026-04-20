from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
import numpy as np

from .audio import load_mono_audio, match_length
from .metrics import (
    l2_error,
    relative_l2_error,
    mean_absolute_error,
    spectral_overlap,
)
from .transforms import compute_representation


def evaluate_mixture_linearity(
    x: np.ndarray,
    y: np.ndarray,
    sr: int,
    transform_type: str = "stft",
    include_overlap: bool = True,
    **transform_kwargs: Any,
) -> Dict[str, float]:
    """
    Evaluate how linear a spectral representation is under signal mixing.

    Computes:
        R(x), R(y), R(x+y)
    and compares:
        R(x+y) vs R(x) + R(y)

    where R(.) is the selected spectral representation.

    Args:
        x: First mono signal, already aligned.
        y: Second mono signal, already aligned.
        sr: Sampling rate.
        transform_type: Spectral representation type ('stft', 'mel', later 'nsgt').
        include_overlap: Whether to compute spectral overlap.
        **transform_kwargs: Extra arguments forwarded to the transform function.

    Returns:
        Dictionary containing error metrics and, optionally, overlap.
    """
    if len(x) != len(y):
        raise ValueError("Signals must have the same length.")

    mixture = x + y

    rep_x = compute_representation(
        audio=x,
        sr=sr,
        transform_type=transform_type,
        **transform_kwargs,
    )
    rep_y = compute_representation(
        audio=y,
        sr=sr,
        transform_type=transform_type,
        **transform_kwargs,
    )
    rep_mix = compute_representation(
        audio=mixture,
        sr=sr,
        transform_type=transform_type,
        **transform_kwargs,
    )

    rep_sum = rep_x + rep_y

    metrics = {
        "l2": l2_error(rep_mix, rep_sum),
        "relative_l2": relative_l2_error(rep_mix, rep_sum),
        "mae": mean_absolute_error(rep_mix, rep_sum),
    }

    if include_overlap:
        metrics["overlap"] = spectral_overlap(rep_x, rep_y)

    return metrics


def evaluate_mixture_linearity_stft(
    x: np.ndarray,
    y: np.ndarray,
    sr: int,
    n_fft: int = 2048,
    hop_length: int = 512,
    win_length: int | None = None,
    window: str = "hann",
    include_overlap: bool = True,
) -> Dict[str, float]:
    """
    Backward-compatible helper for STFT magnitude analysis.

    Args:
        x: First mono signal, already aligned.
        y: Second mono signal, already aligned.
        sr: Sampling rate.
        n_fft: FFT size.
        hop_length: Hop length.
        win_length: Window length.
        window: Window function.
        include_overlap: Whether to compute spectral overlap.

    Returns:
        Dictionary containing error metrics and, optionally, overlap.
    """
    return evaluate_mixture_linearity(
        x=x,
        y=y,
        sr=sr,
        transform_type="stft",
        include_overlap=include_overlap,
        output="magnitude",
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
    )


def evaluate_dataset_pairs(
    pairs: List[Tuple[str | Path, str | Path]],
    sr: int = 22050,
    transform_type: str = "stft",
    include_overlap: bool = True,
    **transform_kwargs: Any,
) -> pd.DataFrame:
    """
    Evaluate multiple audio pairs using the selected spectral representation.

    Args:
        pairs: List of (path_x, path_y) pairs.
        sr: Target sampling rate.
        transform_type: Spectral representation type ('stft', 'mel', later 'nsgt').
        include_overlap: Whether to compute spectral overlap.
        **transform_kwargs: Extra transform-specific arguments.

    Returns:
        DataFrame with one row per pair and the computed metrics.
    """
    results: List[Dict[str, Any]] = []

    for path_x, path_y in pairs:
        x, _ = load_mono_audio(path_x, sr=sr)
        y, _ = load_mono_audio(path_y, sr=sr)

        x, y = match_length(x, y, mode="truncate")

        metrics = evaluate_mixture_linearity(
            x=x,
            y=y,
            sr=sr,
            transform_type=transform_type,
            include_overlap=include_overlap,
            **transform_kwargs,
        )

        results.append(
            {
                "file_x": Path(path_x).stem,
                "file_y": Path(path_y).stem,
                "transform_type": transform_type,
                **metrics,
            }
        )

    return pd.DataFrame(results)