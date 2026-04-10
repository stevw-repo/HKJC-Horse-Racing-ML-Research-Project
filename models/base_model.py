# models/base_model.py
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd


class BaseModel(ABC):
    """Abstract interface that every model must implement."""

    @abstractmethod
    def fit(self,
            X_train: np.ndarray, y_train: np.ndarray,
            X_val:   np.ndarray, y_val:   np.ndarray) -> None:
        ...

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return 1-D array of positive-class probabilities."""
        ...

    @abstractmethod
    def save(self, path: Path) -> None:
        ...

    @classmethod
    @abstractmethod
    def load(cls, path: Path) -> "BaseModel":
        ...

    def feature_importance(self) -> pd.Series:
        """Return named feature importances if available, else empty Series."""
        return pd.Series(dtype=float)