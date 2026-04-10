# models/catboost_model.py
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from config import RANDOM_SEED
from models.base_model import BaseModel

logger = logging.getLogger(__name__)

_DEFAULT_CB_PARAMS = {
    "iterations":      1000,
    "learning_rate":   0.05,
    "depth":           6,
    "l2_leaf_reg":     3.0,
    "loss_function":   "Logloss",
    "eval_metric":     "AUC",
    "random_seed":     RANDOM_SEED,
    "verbose":         100,
    "early_stopping_rounds": 50,
}


class CatBoostModel(BaseModel):
    """CatBoost binary classifier."""

    def __init__(self, params: dict = None):
        self.params = {**_DEFAULT_CB_PARAMS, **(params or {})}
        self.model_ = None

    def fit(self, X_train, y_train, X_val, y_val):
        from catboost import CatBoostClassifier, Pool
        self.model_ = CatBoostClassifier(**self.params)
        train_pool  = Pool(X_train, label=y_train)
        eval_pool   = Pool(X_val,   label=y_val)
        self.model_.fit(train_pool, eval_set=eval_pool, use_best_model=True)
        logger.info("CatBoost trained — best iteration: %d",
                    self.model_.get_best_iteration())

    def predict_proba(self, X) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("Model not trained.")
        return self.model_.predict_proba(X)[:, 1]

    def save(self, path: Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: Path) -> "CatBoostModel":
        with open(Path(path), "rb") as f:
            return pickle.load(f)

    def feature_importance(self) -> pd.Series:
        if self.model_ is None:
            return pd.Series(dtype=float)
        imp   = self.model_.get_feature_importance()
        names = self.model_.feature_names_
        return pd.Series(imp, index=names).sort_values(ascending=False)