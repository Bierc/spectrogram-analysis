from __future__ import annotations

import matplotlib.pyplot as plt
import librosa.display
import numpy as np


def plot_spectrogram(
    spec_db: np.ndarray,
    sr: int,
    hop_length: int,
    title: str,
    x_axis: str = "time",
    y_axis: str = "log",
    figsize: tuple[int, int] = (10, 4),
) -> None:
    """
    Plot a dB spectrogram.
    """
    plt.figure(figsize=figsize)
    librosa.display.specshow(
        spec_db,
        sr=sr,
        hop_length=hop_length,
        x_axis=x_axis,
        y_axis=y_axis,
    )
    plt.colorbar(format="%+2.0f dB")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_difference_map(
    diff: np.ndarray,
    title: str,
    figsize: tuple[int, int] = (10, 4),
) -> None:
    """
    Plot raw difference matrix with imshow.
    """
    plt.figure(figsize=figsize)
    plt.imshow(diff, origin="lower", aspect="auto")
    plt.colorbar()
    plt.title(title)
    plt.tight_layout()
    plt.show()