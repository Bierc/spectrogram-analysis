from __future__ import annotations

from pathlib import Path
from typing import Tuple

import librosa
import numpy as np


def load_mono_audio(
    path: str | Path,
    sr: int = 22050,
    normalize: bool = True,
) -> Tuple[np.ndarray, int]:
    """
    Load an audio file as mono.

    Args:
        path: Path to the audio file.
        sr: Target sample rate.
        normalize: Whether to peak-normalize the signal.

    Returns:
        A tuple (audio, sample_rate).
    """
    audio, sample_rate = librosa.load(path, sr=sr, mono=True)

    if normalize:
        peak = np.max(np.abs(audio))
        if peak > 0:
            audio = audio / peak

    return audio.astype(np.float32), sample_rate


def match_length(
    x: np.ndarray,
    y: np.ndarray,
    mode: str = "truncate",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Match the length of two signals.

    Args:
        x: First signal.
        y: Second signal.
        mode: 'truncate' cuts both to the shortest length.
              'pad' pads both to the longest length with zeros.

    Returns:
        Tuple of aligned signals.
    """
    if mode not in {"truncate", "pad"}:
        raise ValueError("mode must be either 'truncate' or 'pad'")

    len_x, len_y = len(x), len(y)

    if mode == "truncate":
        min_len = min(len_x, len_y)
        return x[:min_len], y[:min_len]

    max_len = max(len_x, len_y)
    x_pad = np.pad(x, (0, max_len - len_x))
    y_pad = np.pad(y, (0, max_len - len_y))
    return x_pad, y_pad


def mix_signals(
    x: np.ndarray,
    y: np.ndarray,
    normalize: bool = True,
) -> np.ndarray:
    """
    Mix two aligned signals by summation.

    Args:
        x: First signal.
        y: Second signal.
        normalize: Whether to peak-normalize the mixture.

    Returns:
        Mixed signal.
    """
    if len(x) != len(y):
        raise ValueError("Signals must have the same length before mixing.")

    mixture = x + y

    if normalize:
        peak = np.max(np.abs(mixture))
        if peak > 0:
            mixture = mixture / peak

    return mixture.astype(np.float32)