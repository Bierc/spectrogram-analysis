from __future__ import annotations

from pathlib import Path


def ensure_dir(path: str | Path) -> Path:
    """
    Create directory if it does not exist.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path