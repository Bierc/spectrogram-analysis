from __future__ import annotations

import os
import sys
from typing import Any, Literal

import librosa
import numpy as np


TransformType = Literal["stft", "mel", "nsgt"]


def _import_nsgt3() -> Any:
    # Try the environment first (user set PYTHONPATH or installed the package).
    try:
        import nsgt3
        return nsgt3
    except ImportError:
        pass

    # Fall back to the co-located variational-timbre repository.
    # Assumes both repos sit under the same parent directory.
    _here = os.path.dirname(os.path.abspath(__file__))
    _candidate = os.path.normpath(os.path.join(_here, "../../../variational-timbre"))
    if os.path.isdir(os.path.join(_candidate, "nsgt3")) and _candidate not in sys.path:
        sys.path.insert(0, _candidate)
    try:
        import nsgt3
        return nsgt3
    except ImportError:
        raise ImportError(
            "nsgt3 could not be imported. Add the variational-timbre repo to "
            "PYTHONPATH, e.g.:  export PYTHONPATH=/path/to/variational-timbre"
        )


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


def compute_nsgt_representation(
    audio: np.ndarray,
    sr: int,
    fmin: float = 30.0,
    fmax: float = 11000.0,
    bins: int = 48,
    scale: str = "oct",
) -> np.ndarray:
    """
    Compute NSGT magnitude spectrogram using the nsgt3 library.

    Args:
        audio: Input mono audio signal.
        sr: Sampling rate.
        fmin: Minimum frequency in Hz.
        fmax: Maximum frequency in Hz.
        bins: Frequency bins per octave (for 'oct'/'log') or total bins (for 'mel').
        scale: Frequency scale — one of 'oct', 'log', 'mel'.

    Returns:
        Magnitude spectrogram, shape (n_freqs, n_frames).
        Same shape convention as compute_stft_representation(output='magnitude').

    Notes:
        matrixform=True forces all frequency bands to share the same number of
        time frames, which is what allows a clean 2D numpy array output.
        reducedform=1 + real=True keeps only the positive-frequency half,
        mirroring librosa's STFT behaviour for real signals.
    """
    nsgt3 = _import_nsgt3()

    scale_map = {
        "oct": nsgt3.OctScale,
        "log": nsgt3.LogScale,
        "mel": nsgt3.MelScale,
    }
    if scale not in scale_map:
        raise ValueError(f"Unsupported NSGT scale: {scale!r}. Use 'oct', 'log', or 'mel'.")

    scl = scale_map[scale](fmin, fmax, bins)

    # Ls must be known at construction time; we take it from the signal.
    nsgt = nsgt3.NSGT(
        scl,
        sr,
        len(audio),
        real=True,
        matrixform=True,  # rectangular output — required for np.array conversion
        reducedform=1,    # positive frequencies only (real signal)
    )

    # forward() yields complex arrays, one per frequency band.
    # With matrixform=True all bands have equal length → clean (n_freqs, n_frames) array.
    coeffs = np.array(list(nsgt.forward(audio)))  # complex, (n_freqs, n_frames)
    return np.abs(coeffs)  # magnitude, float, (n_freqs, n_frames)


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
        - 'nsgt'

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
        return compute_nsgt_representation(audio=audio, sr=sr, **kwargs)

    raise ValueError(f"Unsupported transform_type: {transform_type}")