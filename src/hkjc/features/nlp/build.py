"""Build the per-run NLP feature table from stored comments-on-running.

One row per (race, runner) with the lexicon-flag counts + anchor similarities for *that run's*
comment. The as-of feature builder joins this and **lags** it (a comment describes a run, so it
is a feature only for the horse's later runs). Cached to processed Parquet because the MiniLM
pass over the full comment corpus is the expensive part.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from hkjc.common.config import AppConfig, get_config
from hkjc.features.nlp.encode import NlpEncoder

KEYS = ["race_date", "venue", "race_no", "horse_id"]


def comment_features_path(cfg: AppConfig | None = None) -> Path:
    cfg = cfg or get_config()
    return cfg.paths.processed_dir / "nlp_comment_features" / "nlp_comment_features.parquet"


def _load_comments(cfg: AppConfig) -> pl.DataFrame:
    base = cfg.paths.raw_dir / "comments_on_running"
    files = [str(p) for p in base.rglob("*.parquet")] if base.exists() else []
    if not files:
        return pl.DataFrame()
    df = pl.read_parquet(files, columns=[*KEYS, "comment"])
    return df.filter(pl.col("horse_id").is_not_null() & pl.col("comment").is_not_null())


def build_comment_features(
    cfg: AppConfig | None = None, *, use_embeddings: bool = True, persist: bool = True
) -> pl.DataFrame:
    """Encode every stored comment into NLP features; return (and cache) the per-run table."""
    cfg = cfg or get_config()
    comments = _load_comments(cfg)
    encoder = NlpEncoder(use_embeddings=use_embeddings)
    if comments.is_empty():
        empty_cols = {
            name: pl.Series(name, [], dtype=pl.Float64) for name in encoder.feature_names()
        }
        return pl.DataFrame(
            schema={k: comments.schema.get(k, pl.String) for k in KEYS}
        ).with_columns(**empty_cols)
    cols = encoder.encode(comments["comment"].to_list())
    out = comments.select(KEYS).with_columns(
        *[pl.Series(name, values, dtype=pl.Float64) for name, values in cols.items()]
    )
    # Keep one comment per (race, horse) -- the running comment is one per runner already.
    out = out.unique(subset=KEYS, keep="first")
    if persist:
        path = comment_features_path(cfg)
        path.parent.mkdir(parents=True, exist_ok=True)
        out.write_parquet(path)
    return out
