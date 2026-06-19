"""Gradient-boosted ``ProbabilityModel`` wrappers (PLAN.md §1D, §1G).

Each learner emits a per-runner **raw score** (margin / ranking score) = the PL log-strength;
WIN probabilities are the within-race softmax (inherited from :class:`ProbabilityModel`).
CatBoost takes the high-cardinality categoricals (sire/dam/jockey-era ids/import-type/...)
natively -- the signal the linear logit cannot use. LightGBM also fields a **LambdaMART**
(learning-to-rank) variant whose group is the race.

All wrappers accept the combined design matrix (numeric + integer-encoded categoricals) plus
the categorical column indices; GPU is used for XGBoost/CatBoost when available (LightGBM's
pip wheel is CPU-only on Windows), with a clean CPU fallback.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from hkjc.models.base import FloatArray, ProbabilityModel
from hkjc.models.device import gpu_available


def _group_sizes(groups: np.typing.ArrayLike) -> list[int]:
    """Run-length sizes of contiguous race groups (rows are race-contiguous)."""
    g = np.asarray(groups)
    if g.size == 0:
        return []
    change = np.empty(g.size, dtype=bool)
    change[0] = True
    change[1:] = g[1:] != g[:-1]
    idx = np.flatnonzero(change)
    return [int(v) for v in np.diff(np.append(idx, g.size))]


def _frame(x: FloatArray, cat_indices: Sequence[int], *, str_cats: bool = False) -> pd.DataFrame:
    """Wrap the float matrix as a DataFrame with categorical columns typed for the library."""
    df = pd.DataFrame(x, columns=[f"f{i}" for i in range(x.shape[1])])
    for i in cat_indices:
        codes = df[f"f{i}"].round().astype("int64")
        df[f"f{i}"] = codes.astype(str) if str_cats else codes.astype("category")
    return df


class LightGBMModel(ProbabilityModel):
    """LightGBM binary booster; raw margin = PL log-strength."""

    name = "lightgbm"

    def __init__(
        self,
        cat_indices: Sequence[int] = (),
        n_estimators: int = 400,
        learning_rate: float = 0.05,
        num_leaves: int = 63,
        min_child_samples: int = 50,
        reg_lambda: float = 1.0,
        random_state: int = 0,
    ) -> None:
        self.cat_indices = list(cat_indices)
        self.params = {
            "objective": "binary",
            "learning_rate": learning_rate,
            "num_leaves": num_leaves,
            "min_child_samples": min_child_samples,
            "lambda_l2": reg_lambda,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 1,
            "verbose": -1,
            "seed": random_state,
        }
        self.n_estimators = n_estimators
        self.booster: Any = None

    def _cat_names(self) -> list[str]:
        return [f"f{i}" for i in self.cat_indices]

    def fit(self, x: FloatArray, groups: np.typing.ArrayLike, y: FloatArray) -> LightGBMModel:
        import lightgbm as lgb

        ds = lgb.Dataset(
            _frame(x, self.cat_indices),
            label=np.asarray(y),
            categorical_feature=self._cat_names(),
            free_raw_data=False,
        )
        self.booster = lgb.train(self.params, ds, num_boost_round=self.n_estimators)
        return self

    def log_strength(self, x: FloatArray) -> FloatArray:
        assert self.booster is not None
        out = self.booster.predict(_frame(x, self.cat_indices), raw_score=True)
        return np.asarray(out, dtype=np.float64)


class LambdaMARTModel(LightGBMModel):
    """LightGBM LambdaMART (learning-to-rank); race = group, ranking score = PL strength."""

    name = "lambdamart"

    def __init__(self, cat_indices: Sequence[int] = (), **kwargs: float | int) -> None:
        super().__init__(cat_indices, **kwargs)  # type: ignore[arg-type]
        self.params["objective"] = "lambdarank"
        self.params["metric"] = "ndcg"

    def fit(self, x: FloatArray, groups: np.typing.ArrayLike, y: FloatArray) -> LambdaMARTModel:
        import lightgbm as lgb

        ds = lgb.Dataset(
            _frame(x, self.cat_indices),
            label=np.asarray(y).astype(int),
            group=_group_sizes(groups),
            categorical_feature=self._cat_names(),
            free_raw_data=False,
        )
        self.booster = lgb.train(self.params, ds, num_boost_round=self.n_estimators)
        return self


class XGBoostModel(ProbabilityModel):
    """XGBoost binary booster (GPU when available); output margin = PL log-strength."""

    name = "xgboost"

    def __init__(
        self,
        cat_indices: Sequence[int] = (),
        n_estimators: int = 400,
        learning_rate: float = 0.05,
        max_depth: int = 6,
        reg_lambda: float = 1.0,
        random_state: int = 0,
        use_gpu: bool | None = None,
    ) -> None:
        self.cat_indices = list(cat_indices)
        self.n_estimators = n_estimators
        gpu = gpu_available() if use_gpu is None else use_gpu
        self.params = {
            "objective": "binary:logistic",
            "tree_method": "hist",
            "device": "cuda" if gpu else "cpu",
            "eta": learning_rate,
            "max_depth": max_depth,
            "lambda": reg_lambda,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "seed": random_state,
        }
        self.bst: Any = None

    def _dmatrix(self, x: FloatArray, y: FloatArray | None = None) -> Any:
        import xgboost as xgb

        # XGBoost's native categoricals reject categories unseen in the training fold (common
        # across walk-forward splits), so the integer-encoded categoricals are fed as ordinal
        # numerics here; native categorical handling is left to CatBoost (PLAN.md §1G).
        return xgb.DMatrix(np.asarray(x, dtype=np.float64), label=y)

    def fit(self, x: FloatArray, groups: np.typing.ArrayLike, y: FloatArray) -> XGBoostModel:
        import xgboost as xgb

        self.bst = xgb.train(self.params, self._dmatrix(x, np.asarray(y)), self.n_estimators)
        return self

    def log_strength(self, x: FloatArray) -> FloatArray:
        out = self.bst.predict(self._dmatrix(x), output_margin=True)
        return np.asarray(out, dtype=np.float64)


class CatBoostModel(ProbabilityModel):
    """CatBoost classifier with native categoricals (GPU when available); raw value = strength."""

    name = "catboost"

    def __init__(
        self,
        cat_indices: Sequence[int] = (),
        n_estimators: int = 500,
        learning_rate: float = 0.05,
        depth: int = 6,
        l2_leaf_reg: float = 3.0,
        random_state: int = 0,
        use_gpu: bool | None = None,
    ) -> None:
        self.cat_indices = list(cat_indices)
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.depth = depth
        self.l2_leaf_reg = l2_leaf_reg
        self.random_state = random_state
        self.use_gpu = gpu_available() if use_gpu is None else use_gpu
        self.model: Any = None

    def fit(self, x: FloatArray, groups: np.typing.ArrayLike, y: FloatArray) -> CatBoostModel:
        from catboost import CatBoostClassifier, Pool

        pool = Pool(
            _frame(x, self.cat_indices, str_cats=True),
            label=np.asarray(y).astype(int),
            cat_features=self.cat_indices,
        )
        self.model = CatBoostClassifier(
            iterations=self.n_estimators,
            learning_rate=self.learning_rate,
            depth=self.depth,
            l2_leaf_reg=self.l2_leaf_reg,
            random_seed=self.random_state,
            task_type="GPU" if self.use_gpu else "CPU",
            devices="0" if self.use_gpu else None,
            verbose=False,
            allow_writing_files=False,
        )
        self.model.fit(pool)
        return self

    def log_strength(self, x: FloatArray) -> FloatArray:
        frame = _frame(x, self.cat_indices, str_cats=True)
        out = self.model.predict(frame, prediction_type="RawFormulaVal")
        return np.asarray(out, dtype=np.float64)
