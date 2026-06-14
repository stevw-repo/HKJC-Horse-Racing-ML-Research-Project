"""Tests for the data-lake path helpers."""

from __future__ import annotations

from pathlib import Path

from hkjc.common.config import Paths
from hkjc.data.store.paths import data_dirs, ensure_data_dirs


def test_data_dirs_derive_from_root(tmp_path: Path) -> None:
    paths = Paths(data_root=tmp_path)
    dirs = data_dirs(paths)
    assert paths.raw_dir == tmp_path / "raw"
    assert paths.duckdb_path == tmp_path / "processed" / "hkjc.duckdb"
    assert tmp_path / "live_odds" in dirs


def test_ensure_data_dirs_creates_them(tmp_path: Path) -> None:
    paths = Paths(data_root=tmp_path)
    created = ensure_data_dirs(paths)
    assert created
    assert all(directory.is_dir() for directory in created)
