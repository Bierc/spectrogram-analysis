from __future__ import annotations

from typing import Any, Literal

import librosa
import numpy as np


TransformType = Literal["stft", "mel", "nsgt"]
StftOutputType = Literal["complex", "magnitude", "power"]


def compute_stft_complex(
    audio: np.ndarray,
    n_fft: int = 2048,
    hop_length: int = 512,
    win_length: int | None = None,
    window: str = "hann",
) -> np.ndarray:
    """
    Compute the complex-valued STFT of an audio signal.

    Args:
        audio: Input mono audio signal.
        n_fft: FFT window size.
        hop_length: Hop length between frames.
        win_length: Window length. If None, defaults to n_fft.
        window: Window function name.

    Returns:
        Complex STFT matrix.
    """
    return librosa.stft(
        y=audio,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
    )


def magnitude_spectrogram(stft_matrix: np.ndarray) -> np.ndarray:
    """
    Compute the magnitude spectrogram from a complex STFT matrix.
    """
    return np.abs(stft_matrix)


def power_spectrogram(stft_matrix: np.ndarray) -> np.ndarray:
    """
    Compute the power spectrogram from a complex STFT matrix.
    """
    return np.abs(stft_matrix) ** 2


def compute_stft_representation(
    audio: np.ndarray,
    output: StftOutputType = "magnitude",
    n_fft: int = 2048,
    hop_length: int = 512,
    win_length: int | None = None,
    window: str = "hann",
) -> np.ndarray:
    """
    Compute an STFT-based representation.

    Args:
        audio: Input mono audio signal.
        output: Type of STFT output. One of:
            - 'complex'
            - 'magnitude'
            - 'power'
        n_fft: FFT window size.
        hop_length: Hop length between frames.
        win_length: Window length. If None, defaults to n_fft.
        window: Window function name.

    Returns:
        STFT representation according to the selected output type.
    """
    stft_matrix = compute_stft_complex(
        audio=audio,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
    )

    if output == "complex":
        return stft_matrix
    if output == "magnitude":
        return magnitude_spectrogram(stft_matrix)
    if output == "power":
        return power_spectrogram(stft_matrix)

    raise ValueError(f"Unsupported STFT output type: {output}")


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

    Args:
        audio: Input mono audio signal.
        sr: Sampling rate.
        n_fft: FFT window size.
        hop_length: Hop length between frames.
        n_mels: Number of mel bands.
        power: Exponent for the spectrogram.
            - 1.0 -> amplitude-like mel spectrogram
            - 2.0 -> power mel spectrogram

    Returns:
        Mel spectrogram matrix.
    """
    return librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        power=power,
    )


def amplitude_to_db(
    magnitude: np.ndarray,
    ref: float | Any = np.max,
    top_db: float = 80.0,
) -> np.ndarray:
    """
    Convert an amplitude spectrogram to decibel scale.
    """
    return librosa.amplitude_to_db(magnitude, ref=ref, top_db=top_db)


def power_to_db(
    power: np.ndarray,
    ref: float | Any = np.max,
    top_db: float = 80.0,
) -> np.ndarray:
    """
    Convert a power spectrogram to decibel scale.
    """
    return librosa.power_to_db(power, ref=ref, top_db=top_db)


def representation_to_db(
    representation: np.ndarray,
    scale: Literal["amplitude", "power"] = "amplitude",
    ref: float | Any = np.max,
    top_db: float = 80.0,
) -> np.ndarray:
    """
    Convert a spectral representation to decibel scale.

    Args:
        representation: Input spectral representation.
        scale: Whether representation is amplitude or power.
        ref: Reference value for dB conversion.
        top_db: Dynamic range threshold.

    Returns:
        Representation in dB.
    """
    if scale == "amplitude":
        return amplitude_to_db(representation, ref=ref, top_db=top_db)
    if scale == "power":
        return power_to_db(representation, ref=ref, top_db=top_db)

    raise ValueError(f"Unsupported scale type: {scale}")


def compute_representation(
    audio: np.ndarray,
    sr: int,
    transform_type: TransformType = "stft",
    **kwargs: Any,
) -> np.ndarray:
    """
    Compute a generic time-frequency representation.

    Supported transform types:
        - 'stft'
        - 'mel'
        - 'nsgt' (placeholder for future implementation)

    Args:
        audio: Input mono audio signal.
        sr: Sampling rate.
        transform_type: Type of transformation.
        **kwargs: Parameters forwarded to the corresponding transform function.

    Returns:
        Spectral representation matrix.
    """
    if transform_type == "stft":
        return compute_stft_representation(audio=audio, **kwargs)

    if transform_type == "mel":
        return compute_mel_spectrogram(audio=audio, sr=sr, **kwargs)

    if transform_type == "nsgt":
        raise NotImplementedError(
            "NSGT support has not been implemented yet. "
            "Add compute_nsgt_representation() when ready."
        )

    raise ValueError(f"Unsupported transform_type: {transform_type}")