from __future__ import annotations

from typing import Tuple

import librosa
import numpy as np


def compute_stft_complex(
    audio: np.ndarray,
    n_fft: int = 2048,
    hop_length: int = 512,
    win_length: int | None = None,
    window: str = "hann",
) -> np.ndarray:
    """
    Compute complex STFT.

    Returns:
        Complex-valued STFT matrix.
    """
    return librosa.stft(
        y=audio,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
    )


def magnitude_spectrogram(stft_matrix: np.ndarray) -> np.ndarray:
    """Return magnitude from complex STFT."""
    return np.abs(stft_matrix)


def power_spectrogram(stft_matrix: np.ndarray) -> np.ndarray:
    """Return power spectrogram from complex STFT."""
    return np.abs(stft_matrix) ** 2


def amplitude_to_db(
    magnitude: np.ndarray,
    ref: float | callable = np.max,
    top_db: float = 80.0,
) -> np.ndarray:
    """Convert amplitude spectrogram to dB."""
    return librosa.amplitude_to_db(magnitude, ref=ref, top_db=top_db)


def compute_mel_spectrogram(
    audio: np.ndarray,
    sr: int,
    n_fft: int = 2048,
    hop_length: int = 512,
    n_mels: int = 128,
    power: float = 2.0,
) -> np.ndarray:
    """
    Compute mel spectrogram.
    """
    return librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        power=power,
    )