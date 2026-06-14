"""On-disk layout helpers for the local data lake."""

from __future__ import annotations

from pathlib import Path

from hkjc.common.config import Paths


def data_dirs(paths: Paths) -> list[Path]:
    """Return the data-lake directories (raw, processed, cache, live_odds, mlruns)."""
    return [
        paths.raw_dir,
        paths.processed_dir,
        paths.cache_dir,
        paths.live_odds_dir,
        paths.mlruns_dir,
    ]


def ensure_data_dirs(paths: Paths) -> list[Path]:
    """Create the data-lake directories if missing; return the list created/ensured."""
    dirs = data_dirs(paths)
    for directory in dirs:
        directory.mkdir(parents=True, exist_ok=True)
    return dirs
