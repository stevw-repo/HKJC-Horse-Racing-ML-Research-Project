# models/lgbm_model.py
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from config import DEFAULT_LGBM_PARAMS, RANDOM_SEED
from models.base_model import BaseModel

logger = logging.getLogger(__name__)


class LGBMModel(BaseModel):
    """LightGBM binary classifier wrapper for WIN or PLACE targets."""

    def __init__(self, params: dict = None):
        import lightgbm as lgb   # lazy import — keeps startup fast
        self._lgb    = lgb
        self.params  = {**DEFAULT_LGBM_PARAMS, **(params or {})}
        self.params["random_state"] = RANDOM_SEED
        self.model_  = None
        self._feature_names: list = []

    def fit(self, X_train, y_train, X_val, y_val):
        import lightgbm as lgb
        pos_weight = float((y_train == 0).sum()) / max((y_train == 1).sum(), 1)

        train_data = lgb.Dataset(X_train, label=y_train,
                                  free_raw_data=False)
        valid_data = lgb.Dataset(X_val,   label=y_val,
                                  reference=train_data,
                                  free_raw_data=False)
        params = {
            **self.params,
            "objective":        "binary",
            "metric":           ["binary_logloss", "auc"],
            "scale_pos_weight": pos_weight,
            "verbose":          -1,
        }
        callbacks = [
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=100),
        ]
        self.model_ = lgb.train(
            params,
            train_data,
            valid_sets=[valid_data],
            num_boost_round=self.params.get("n_estimators", 1000),
            callbacks=callbacks,
        )
        logger.info("LightGBM trained — best iteration: %d",
                    self.model_.best_iteration)

    def predict_proba(self, X) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("Model not trained. Call fit() first.")
        return self.model_.predict(X)

    def save(self, path: Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info("LGBMModel saved → %s", path)

    @classmethod
    def load(cls, path: Path) -> "LGBMModel":
        with open(Path(path), "rb") as f:
            return pickle.load(f)

    def feature_importance(self) -> pd.Series:
        if self.model_ is None:
            return pd.Series(dtype=float)
        imp = self.model_.feature_importance(importance_type="gain")
        names = self.model_.feature_name()
        return pd.Series(imp, index=names).sort_values(ascending=False)