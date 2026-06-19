"""Shared model dataset: the design matrix + labels + dividend/market arrays for the zoo.

Loaded once and handed to every model so the leaderboard compares like with like. The full
matrix carries numeric + integer-encoded categorical columns; numeric-only learners (logit,
NNs) slice ``numeric_indices``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from hkjc.backtest.engine import _dividend_lookup
from hkjc.common.config import AppConfig, get_config
from hkjc.features import store
from hkjc.features.design import build_design
from hkjc.models.base import FloatArray, IntArray


@dataclass(frozen=True, slots=True)
class ModelData:
    """Everything a model + the walk-forward evaluator need."""

    x_full: FloatArray
    numeric_indices: list[int]
    categorical_indices: list[int]
    feature_names: list[str]
    y: FloatArray
    placed: FloatArray
    n_places: IntArray
    win_odds: FloatArray
    market_prob: FloatArray
    win_div: FloatArray
    place_div: FloatArray
    race_id: IntArray
    season: np.typing.NDArray[np.str_]

    def numeric(self) -> FloatArray:
        return self.x_full[:, self.numeric_indices]


def load_model_data(cfg: AppConfig | None = None) -> ModelData:
    """Build the shared :class:`ModelData` from the persisted feature store."""
    cfg = cfg or get_config()
    df = store.load_features(cfg)
    key = pl.concat_str(
        [pl.col("race_date").cast(pl.String), pl.col("venue"), pl.col("race_no").cast(pl.String)],
        separator="|",
    )
    df = df.with_columns(_chg=(key != key.shift(1)).fill_null(True))
    df = df.with_columns(race_id=(pl.col("_chg").cum_sum() - 1))
    df = df.join(
        _dividend_lookup(cfg, "WIN"), on=["race_date", "venue", "race_no", "saddle"], how="left"
    )
    df = df.join(
        _dividend_lookup(cfg, "PLACE"), on=["race_date", "venue", "race_no", "saddle"], how="left"
    )
    design = build_design(df)
    return ModelData(
        x_full=design.x,
        numeric_indices=design.numeric_indices,
        categorical_indices=design.categorical_indices,
        feature_names=design.feature_names,
        y=df["won"].to_numpy().astype(np.float64),
        placed=df["placed"].to_numpy().astype(np.float64),
        n_places=df["n_places"].to_numpy().astype(np.int64),
        win_odds=df["win_odds"].to_numpy().astype(np.float64),
        market_prob=df["market_prob"].to_numpy().astype(np.float64),
        win_div=df["win_div"].to_numpy().astype(np.float64),
        place_div=df["place_div"].to_numpy().astype(np.float64),
        race_id=df["race_id"].to_numpy().astype(np.int64),
        season=df["season"].to_numpy().astype(str),
    )
