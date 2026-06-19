"""Feature-store I/O: a DuckDB handle over the raw views, plus persist/load of the
``features_runner`` table (processed Parquet + a DuckDB view)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import duckdb
import polars as pl

from hkjc.common.config import AppConfig, get_config
from hkjc.data.store.writer import refresh_views

FEATURES_TABLE = "features_runner"


def connect(cfg: AppConfig | None = None) -> Any:
    """Open the project DuckDB and (re)create the raw views over current Parquet.

    Returns a live connection; the caller closes it. Raises ``FileNotFoundError`` if the
    database has not been created yet (i.e. nothing scraped).
    """
    cfg = cfg or get_config()
    if not cfg.paths.duckdb_path.is_file():
        msg = f"DuckDB not found at {cfg.paths.duckdb_path}; run the M1 scraper first."
        raise FileNotFoundError(msg)
    con = duckdb.connect(str(cfg.paths.duckdb_path))
    refresh_views(con, cfg.paths.raw_dir)
    return con


def load_view(con: Any, name: str) -> pl.DataFrame:
    """Load a DuckDB view/table into a Polars DataFrame (empty frame if absent)."""
    try:
        return con.execute(f"SELECT * FROM {name}").pl()  # type: ignore[no-any-return]
    except duckdb.Error:
        return pl.DataFrame()


def features_path(cfg: AppConfig | None = None) -> Path:
    cfg = cfg or get_config()
    return cfg.paths.processed_dir / FEATURES_TABLE / f"{FEATURES_TABLE}.parquet"


def write_features(df: pl.DataFrame, cfg: AppConfig | None = None) -> Path:
    """Persist the feature frame to processed Parquet and expose it as a DuckDB view."""
    cfg = cfg or get_config()
    path = features_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path)
    con = duckdb.connect(str(cfg.paths.duckdb_path))
    try:
        con.execute(
            f"CREATE OR REPLACE VIEW {FEATURES_TABLE} AS "
            f"SELECT * FROM read_parquet('{path.as_posix()}')"
        )
    finally:
        con.close()
    return path


def load_features(cfg: AppConfig | None = None) -> pl.DataFrame:
    """Load the persisted feature frame (raises ``FileNotFoundError`` if not built)."""
    cfg = cfg or get_config()
    path = features_path(cfg)
    if not path.is_file():
        msg = f"Features not built at {path}; run `hkjc features build` first."
        raise FileNotFoundError(msg)
    return pl.read_parquet(path)
