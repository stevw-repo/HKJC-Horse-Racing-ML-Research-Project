# storage/parquet_store.py
"""
Unified read/write helpers for all parquet stores.
All other modules call these functions — never write parquet directly.
"""
import logging
from pathlib import Path
from typing import List, Optional

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)


# ── Write ─────────────────────────────────────────────────────────────────────

def write(df: pd.DataFrame, path: Path, mode: str = "overwrite") -> None:
    """Persist DataFrame to parquet.

    mode:
        'overwrite' — replace file entirely (default).
        'append'    — read existing, concatenate, and write back.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if mode == "append" and path.exists():
        existing = read(path)
        df = pd.concat([existing, df], ignore_index=True)

    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, str(path), compression="snappy")
    logger.debug("Written %d rows → %s", len(df), path)


# ── Read ──────────────────────────────────────────────────────────────────────

def read(path: Path, filters: Optional[list] = None) -> pd.DataFrame:
    """Read a parquet file or directory of parquet files.

    filters: pyarrow predicate-pushdown filter list, e.g.
             [('race_year', '==', 2023)]
    """
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()

    if path.is_dir():
        files = sorted(path.glob("*.parquet"))
        if not files:
            return pd.DataFrame()
        frames = []
        for fp in files:
            try:
                frames.append(pq.read_table(str(fp), filters=filters).to_pandas())
            except Exception as exc:
                logger.warning("Could not read %s: %s", fp, exc)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    else:
        try:
            return pq.read_table(str(path), filters=filters).to_pandas()
        except Exception as exc:
            logger.warning("Could not read %s: %s", path, exc)
            return pd.DataFrame()


# ── Upsert ────────────────────────────────────────────────────────────────────

def upsert(df: pd.DataFrame, path: Path, key_cols: List[str]) -> None:
    """Read existing file, merge (update-or-insert) by key_cols, write back."""
    path = Path(path)
    if path.exists():
        existing = read(path)
        if not existing.empty:
            merged = (
                pd.concat([existing, df], ignore_index=True)
                .drop_duplicates(subset=key_cols, keep="last")
                .reset_index(drop=True)
            )
            write(merged, path)
            return
    write(df, path)


# ── Utilities ─────────────────────────────────────────────────────────────────

def list_dates(path: Path, date_col: str = "race_date") -> List[str]:
    """Return sorted list of unique dates present in a parquet store."""
    df = read(path)
    if df.empty or date_col not in df.columns:
        return []
    return sorted(pd.to_datetime(df[date_col]).dt.date.astype(str).unique().tolist())


def summary(path: Path) -> dict:
    """Return a dict with row count, column count, date range, and NaN rates."""
    path = Path(path)
    if not path.exists():
        return {"status": "path does not exist"}

    df = read(path)
    if df.empty:
        return {"status": "empty or no parquet files found"}

    info: dict = {
        "rows":    len(df),
        "columns": len(df.columns),
    }

    # Date range
    for col in ("race_date", "snapshot_date", "date"):
        if col in df.columns:
            try:
                dates = pd.to_datetime(df[col])
                info["date_range"] = f"{dates.min().date()} → {dates.max().date()}"
            except Exception:
                pass
            break

    # NaN rates (only columns with at least one NaN)
    nan_rates = (df.isna().mean() * 100).round(1)
    nan_rates = nan_rates[nan_rates > 0].sort_values(ascending=False)
    if not nan_rates.empty:
        info["nan_rates"] = {col: f"{rate}%" for col, rate in nan_rates.items()}

    # File list
    if path.is_dir():
        info["files"] = len(list(path.glob("*.parquet")))

    return info