# models/ensemble_model.py
import logging
import pickle
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from models.base_model import BaseModel

logger = logging.getLogger(__name__)


class EnsembleModel(BaseModel):
    """Weighted-average ensemble over a list of BaseModel instances."""

    def __init__(self, models: List[BaseModel],
                 weights: Optional[List[float]] = None):
        self.models  = models
        n            = len(models)
        if weights is None:
            self.weights = [1.0 / n] * n
        else:
            total = sum(weights)
            self.weights = [w / total for w in weights]

    def fit(self, X_train, y_train, X_val, y_val):
        """Fit each constituent model independently."""
        for i, model in enumerate(self.models):
            logger.info("Ensemble: fitting sub-model %d/%d …", i + 1, len(self.models))
            model.fit(X_train, y_train, X_val, y_val)

    def predict_proba(self, X) -> np.ndarray:
        preds = np.stack(
            [m.predict_proba(X) for m in self.models], axis=0
        )  # shape: (n_models, n_samples)
        weights = np.array(self.weights).reshape(-1, 1)
        return (preds * weights).sum(axis=0)

    def save(self, path: Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: Path) -> "EnsembleModel":
        with open(Path(path), "rb") as f:
            return pickle.load(f)

    def feature_importance(self) -> pd.Series:
        """Average feature importance across models that support it."""
        series_list = [m.feature_importance() for m in self.models
                       if not m.feature_importance().empty]
        if not series_list:
            return pd.Series(dtype=float)
        df  = pd.concat(series_list, axis=1).fillna(0)
        return df.mean(axis=1).sort_values(ascending=False)