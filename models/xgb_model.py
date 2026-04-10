# models/xgb_model.py
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from config import RANDOM_SEED
from models.base_model import BaseModel

logger = logging.getLogger(__name__)

_DEFAULT_XGB_PARAMS = {
    "n_estimators":     1000,
    "learning_rate":    0.05,
    "max_depth":        6,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "reg_alpha":        0.1,
    "reg_lambda":       1.0,
    "eval_metric":      "logloss",
    "use_label_encoder":False,
    "verbosity":        0,
    "random_state":     RANDOM_SEED,
}


class XGBModel(BaseModel):
    """XGBoost binary classifier."""

    def __init__(self, params: dict = None):
        self.params = {**_DEFAULT_XGB_PARAMS, **(params or {})}
        self.model_ = None

    def fit(self, X_train, y_train, X_val, y_val):
        from xgboost import XGBClassifier
        scale_pw = float((y_train == 0).sum()) / max((y_train == 1).sum(), 1)
        self.model_ = XGBClassifier(
            **self.params,
            scale_pos_weight=scale_pw,
            early_stopping_rounds=50,
        )
        self.model_.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=100,
        )
        logger.info("XGBoost trained — best iteration: %d",
                    self.model_.best_iteration)

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
    def load(cls, path: Path) -> "XGBModel":
        with open(Path(path), "rb") as f:
            return pickle.load(f)

    def feature_importance(self) -> pd.Series:
        if self.model_ is None:
            return pd.Series(dtype=float)
        scores = self.model_.get_booster().get_fscore()
        return pd.Series(scores).sort_values(ascending=False)