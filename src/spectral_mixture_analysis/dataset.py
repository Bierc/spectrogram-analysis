from __future__ import annotations

from pathlib import Path
from itertools import combinations
import re
import pandas as pd


TRACK_PATTERN = re.compile(r"Track(\d+)", re.IGNORECASE)


def extract_track_id(filename: str) -> int:
    """
    Extract track id from filename, e.g. Track10_Bass.wav -> 10
    """
    match = TRACK_PATTERN.search(filename)
    if not match:
        raise ValueError(f"Could not extract track id from filename: {filename}")
    return int(match.group(1))


def build_sample_index(samples_root: str | Path) -> pd.DataFrame:
    """
    Build a table indexing all sample files.

    Expected structure:
        samples_root/
            bass/
            flute/
            piano/
            trumpet/

    Returns:
        DataFrame with columns:
            - instrument
            - track_id
            - path
            - filename
    """
    samples_root = Path(samples_root)

    rows = []
    for instrument_dir in sorted(samples_root.iterdir()):
        if not instrument_dir.is_dir():
            continue

        instrument = instrument_dir.name

        for audio_path in sorted(instrument_dir.glob("*.wav")):
            track_id = extract_track_id(audio_path.name)

            rows.append(
                {
                    "instrument": instrument,
                    "track_id": track_id,
                    "path": audio_path,
                    "filename": audio_path.name,
                }
            )

    df = pd.DataFrame(rows).sort_values(["track_id", "instrument"]).reset_index(drop=True)
    return df


def get_track_instrument_pairs(
    index_df: pd.DataFrame,
    track_id: int,
    instruments: list[str] | None = None,
) -> list[tuple[Path, Path]]:
    """
    Generate all instrument pairs for a given track_id.

    Example:
        track 3 with bass, flute, piano, trumpet
        -> (bass, flute), (bass, piano), ...

    Returns:
        List of tuples: (path_x, path_y)
    """
    df = index_df[index_df["track_id"] == track_id].copy()

    if instruments is not None:
        df = df[df["instrument"].isin(instruments)]

    df = df.sort_values("instrument")

    rows = list(df.itertuples(index=False))
    pair_paths = [(a.path, b.path) for a, b in combinations(rows, 2)]
    return pair_paths


def get_same_pair_across_tracks(
    index_df: pd.DataFrame,
    instrument_a: str,
    instrument_b: str,
) -> list[tuple[Path, Path]]:
    """
    Generate pairs for the same instrument combination across all tracks.

    Example:
        bass vs flute across all available tracks.
    """
    pairs = []

    track_ids = sorted(index_df["track_id"].unique())
    for track_id in track_ids:
        df_track = index_df[index_df["track_id"] == track_id]

        row_a = df_track[df_track["instrument"] == instrument_a]
        row_b = df_track[df_track["instrument"] == instrument_b]

        if len(row_a) == 1 and len(row_b) == 1:
            pairs.append((row_a.iloc[0]["path"], row_b.iloc[0]["path"]))

    return pairs