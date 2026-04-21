"""
experiments/mixture_analysis.py

Evaluates mixture-linearity metrics for every audio pair across three transforms
(STFT, mel, NSGT).  Results are written to results/<transform>/metrics/mixture_metrics.csv.
Figures for the first FIGURES_MAX_TRACKS tracks of each pair type are saved to
results/<transform>/figures/.

Dataset pairing is fully delegated to dataset.py:
    - build_sample_index()
    - get_same_pair_across_tracks()

Run:
    python experiments/mixture_analysis.py
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
from spectral_mixture_analysis.analysis import evaluate_mixture_linearity
from spectral_mixture_analysis.transforms import compute_representation, amplitude_to_db
from spectral_mixture_analysis.dataset import (
    build_sample_index,
    get_same_pair_across_tracks,
    extract_track_id,
)

# ── Configuration ─────────────────────────────────────────────────────────────

SR = 22050

# Transform-specific kwargs forwarded to compute_representation / evaluate_mixture_linearity
TRANSFORM_CONFIGS: dict[str, dict] = {
    "stft": dict(output="magnitude", n_fft=2048, hop_length=512),
    "mel":  dict(n_fft=2048, hop_length=512),
    "nsgt": dict(fmin=30.0, fmax=11000.0, bins=48, scale="oct"),
}

# Pair types to evaluate — maps label → (instrument_a, instrument_b)
PAIR_TYPES: dict[str, tuple[str, str]] = {
    "bass_flute":    ("bass", "flute"),
    "piano_trumpet": ("piano", "trumpet"),
}

# Only generate figures for the first N tracks of each pair type (figures are slow)
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


def _save_mixture_figure(
    rep_x: np.ndarray,
    rep_y: np.ndarray,
    rep_mix: np.ndarray,
    title: str,
    save_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(title, fontsize=11)

    for ax, rep, label in zip(
        axes,
        [rep_x, rep_y, rep_mix],
        ["R(x)", "R(y)", "R(mix)"],
    ):
        im = ax.imshow(_to_db(rep), origin="lower", aspect="auto", interpolation="none")
        plt.colorbar(im, ax=ax, format="%+2.0f dB")
        ax.set_title(label)
        ax.set_xlabel("frames")
        ax.set_ylabel("bins")

    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.close(fig)


# ── Core pair processing ───────────────────────────────────────────────────────

def _process_pair(
    path_x: Path,
    path_y: Path,
    transform: str,
    transform_kwargs: dict,
    pair_type: str,
    save_figure: bool,
) -> dict:
    """
    Load, align, evaluate linearity metrics, and optionally save a figure for
    one audio pair.

    Uses evaluate_mixture_linearity() for metrics and compute_representation()
    only when a figure is requested (to avoid redundant computation).
    """
    x, sr = load_mono_audio(path_x, sr=SR)
    y, _  = load_mono_audio(path_y, sr=SR)
    x, y  = match_length(x, y, mode="truncate")

    # Metrics via the existing analysis API
    metrics = evaluate_mixture_linearity(
        x, y,
        sr=sr,
        transform_type=transform,
        include_overlap=True,
        **transform_kwargs,
    )

    track_id = extract_track_id(path_x.name)

    if save_figure:
        rep_x   = compute_representation(x,   sr, transform_type=transform, **transform_kwargs)
        rep_y   = compute_representation(y,   sr, transform_type=transform, **transform_kwargs)
        rep_mix = compute_representation(x+y, sr, transform_type=transform, **transform_kwargs)

        title = f"{transform.upper()} | {pair_type.replace('_', ' vs ')} | track {track_id}"
        fig_path = (
            RESULTS_ROOT / transform / "figures"
            / f"mixture_{pair_type}_track{track_id:02d}.png"
        )
        _save_mixture_figure(rep_x, rep_y, rep_mix, title, fig_path)

    return {
        "transform":   transform,
        "pair_type":   pair_type,
        "track_id":    track_id,
        "file_x":      path_x.stem,
        "file_y":      path_y.stem,
        **metrics,
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
                    f"  rel_l2={row['relative_l2']:.4f}"
                    f"  overlap={row['overlap']:.4f}"
                )

        df = pd.DataFrame(rows)

        # Column order matching the spec
        df = df[[
            "transform", "pair_type", "track_id",
            "file_x", "file_y",
            "l2", "relative_l2", "mae", "overlap",
        ]]

        out_path = RESULTS_ROOT / transform / "metrics" / "mixture_metrics.csv"
        df.to_csv(out_path, index=False)
        print(f"  Saved → {out_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
