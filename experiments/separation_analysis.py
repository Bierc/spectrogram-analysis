"""
experiments/separation_analysis.py

Runs spectrogram separation on every audio pair across three transforms
(STFT, mel, NSGT).  For each pair, computes:
  - reconstruction error  ||R_mix - (Sx + Sy)||
  - separation errors for both source assignments, picking the best

Results are written to results/<transform>/metrics/separation_metrics.csv.
Figures for the first FIGURES_MAX_TRACKS tracks of each pair type are saved to
results/<transform>/figures/.

Dataset pairing is fully delegated to dataset.py:
    - build_sample_index()
    - get_same_pair_across_tracks()

Run:
    python experiments/separation_analysis.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")           # non-interactive backend for saving figures
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── Project path setup ────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from spectral_mixture_analysis.audio import load_mono_audio, match_length
from spectral_mixture_analysis.transforms import compute_representation, amplitude_to_db
from spectral_mixture_analysis.metrics import l2_error
from spectral_mixture_analysis.separation import separate_spectrogram
from spectral_mixture_analysis.dataset import (
    build_sample_index,
    get_same_pair_across_tracks,
    extract_track_id,
)

# ── Configuration ─────────────────────────────────────────────────────────────

SR = 22050

TRANSFORM_CONFIGS: dict[str, dict] = {
    "stft": dict(output="magnitude", n_fft=2048, hop_length=512),
    "mel":  dict(n_fft=2048, hop_length=512),
    "nsgt": dict(fmin=30.0, fmax=11000.0, bins=48, scale="oct"),
}

# Hyperparameters for separate_spectrogram — shared across all transforms
SEPARATION_KWARGS: dict = dict(
    alpha=1.0,
    beta=1e-5,
    gamma=0.15,
    max_iter=300,
    tol=1e-3,
    n_inner_iter=10,
)

PAIR_TYPES: dict[str, tuple[str, str]] = {
    "bass_flute":    ("bass", "flute"),
    "piano_trumpet": ("piano", "trumpet"),
}

FIGURES_MAX_TRACKS = 2

# ── Paths ─────────────────────────────────────────────────────────────────────

SAMPLES_ROOT = PROJECT_ROOT / "data" / "samples"
RESULTS_ROOT = PROJECT_ROOT / "results"


def _ensure_dirs(transform: str) -> None:
    (RESULTS_ROOT / transform / "metrics").mkdir(parents=True, exist_ok=True)
    (RESULTS_ROOT / transform / "figures").mkdir(parents=True, exist_ok=True)


# ── Figure helpers ─────────────────────────────────────────────────────────────

def _to_db(rep: np.ndarray) -> np.ndarray:
    """Clip negatives then convert to dB amplitude scale for display."""
    return amplitude_to_db(np.clip(rep, 0.0, None))


def _save_separation_figure(
    rep_x: np.ndarray,
    rep_y: np.ndarray,
    rep_mix: np.ndarray,
    sx: np.ndarray,
    sy: np.ndarray,
    title: str,
    save_path: Path,
) -> None:
    specs  = [rep_x, rep_y, rep_mix, sx, sy]
    labels = ["R(x)", "R(y)", "R(mix)", "Sx (smooth)", "Sy (sparse)"]

    fig, axes = plt.subplots(1, 5, figsize=(22, 4))
    fig.suptitle(title, fontsize=10)

    for ax, spec, label in zip(axes, specs, labels):
        im = ax.imshow(_to_db(spec), origin="lower", aspect="auto", interpolation="none")
        plt.colorbar(im, ax=ax, format="%+2.0f dB")
        ax.set_title(label, fontsize=9)
        ax.set_xlabel("frames")
        ax.set_ylabel("bins")

    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.close(fig)


# ── Core pair processing ───────────────────────────────────────────────────────

def _best_assignment(
    sx: np.ndarray,
    sy: np.ndarray,
    rep_x: np.ndarray,
    rep_y: np.ndarray,
) -> tuple[float, float, float, str]:
    """
    Try both source assignments and return metrics for both plus the best.

    Direct:  Sx → x, Sy → y   err1 = L2(Sx, R_x) + L2(Sy, R_y)
    Swapped: Sx → y, Sy → x   err2 = L2(Sx, R_y) + L2(Sy, R_x)

    Returns (err1, err2, best_error, assignment_used).
    """
    err1 = l2_error(sx, rep_x) + l2_error(sy, rep_y)
    err2 = l2_error(sx, rep_y) + l2_error(sy, rep_x)

    if err1 <= err2:
        return err1, err2, err1, "direct"
    return err1, err2, err2, "swapped"


def _process_pair(
    path_x: Path,
    path_y: Path,
    transform: str,
    transform_kwargs: dict,
    pair_type: str,
    save_figure: bool,
) -> dict:
    """
    Load, align, compute representations, separate, and evaluate one audio pair.
    """
    x, sr = load_mono_audio(path_x, sr=SR)
    y, _  = load_mono_audio(path_y, sr=SR)
    x, y  = match_length(x, y, mode="truncate")

    rep_x   = compute_representation(x,   sr, transform_type=transform, **transform_kwargs)
    rep_y   = compute_representation(y,   sr, transform_type=transform, **transform_kwargs)
    rep_mix = compute_representation(x+y, sr, transform_type=transform, **transform_kwargs)

    sx, sy = separate_spectrogram(rep_mix, **SEPARATION_KWARGS)

    reconstruction_error = float(np.linalg.norm(rep_mix - sx - sy))

    err1, err2, best_error, assignment_used = _best_assignment(sx, sy, rep_x, rep_y)

    track_id = extract_track_id(path_x.name)

    if save_figure:
        title = f"{transform.upper()} | {pair_type.replace('_', ' vs ')} | track {track_id}"
        fig_path = (
            RESULTS_ROOT / transform / "figures"
            / f"separation_{pair_type}_track{track_id:02d}.png"
        )
        _save_separation_figure(rep_x, rep_y, rep_mix, sx, sy, title, fig_path)

    return {
        "transform":            transform,
        "pair_type":            pair_type,
        "track_id":             track_id,
        "file_x":               path_x.stem,
        "file_y":               path_y.stem,
        "reconstruction_error": reconstruction_error,
        "err1":                 err1,
        "err2":                 err2,
        "best_error":           best_error,
        "assignment_used":      assignment_used,
    }


# ── Main loop ─────────────────────────────────────────────────────────────────

def main() -> None:
    warnings.filterwarnings("ignore")

    index_df = build_sample_index(SAMPLES_ROOT)

    for transform, transform_kwargs in TRANSFORM_CONFIGS.items():
        _ensure_dirs(transform)
        print(f"\n[{transform.upper()}]")

        rows: list[dict] = []

        for pair_type, (inst_a, inst_b) in PAIR_TYPES.items():
            pairs = get_same_pair_across_tracks(index_df, inst_a, inst_b)
            print(f"  {pair_type}: {len(pairs)} pairs")

            for i, (path_x, path_y) in enumerate(pairs):
                row = _process_pair(
                    path_x, path_y,
                    transform, transform_kwargs,
                    pair_type,
                    save_figure=(i < FIGURES_MAX_TRACKS),
                )
                rows.append(row)
                print(
                    f"    track {row['track_id']:>2}"
                    f"  recon={row['reconstruction_error']:.2f}"
                    f"  best={row['best_error']:.2f}"
                    f"  [{row['assignment_used']}]"
                )

        df = pd.DataFrame(rows)

        # Column order matching the spec
        df = df[[
            "transform", "pair_type", "track_id",
            "file_x", "file_y",
            "reconstruction_error",
            "err1", "err2", "best_error",
            "assignment_used",
        ]]

        out_path = RESULTS_ROOT / transform / "metrics" / "separation_metrics.csv"
        df.to_csv(out_path, index=False)
        print(f"  Saved → {out_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
