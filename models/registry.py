# models/registry.py
"""
Central registry mapping model name strings to their classes and
saved artefact path templates.
"""
from pathlib import Path
from typing import Dict, Tuple, Type

from config import MODELS_DIR
from models.base_model import BaseModel
from models.lgbm_model import LGBMModel
from models.xgb_model import XGBModel
from models.catboost_model import CatBoostModel
from models.nn_model import NNModel
from models.ensemble_model import EnsembleModel

# (ModelClass, path_template)
# {target} and {version} are substituted at runtime
_REGISTRY: Dict[str, Tuple[Type[BaseModel], str]] = {
    "lgbm":     (LGBMModel,     "{name}_{target}_{version}.pkl"),
    "xgb":      (XGBModel,      "{name}_{target}_{version}.pkl"),
    "catboost": (CatBoostModel, "{name}_{target}_{version}.pkl"),
    "nn":       (NNModel,       "{name}_{target}_{version}.pkl"),
    "ensemble": (EnsembleModel, "{name}_{target}_{version}.pkl"),
}


def get_model(name: str, target: str, version: str = "v1") -> BaseModel:
    """
    Return a model instance, loading from disk if a saved artefact exists.

    name   — one of: lgbm, xgb, catboost, nn, ensemble
    target — 'win' or 'place'
    """
    if name not in _REGISTRY:
        raise KeyError(f"Unknown model '{name}'. Registered: {list(_REGISTRY)}")

    model_cls, path_tmpl = _REGISTRY[name]
    path = MODELS_DIR / path_tmpl.format(name=name, target=target, version=version)

    if path.exists():
        try:
            return model_cls.load(path)
        except Exception as exc:
            import logging
            logging.getLogger(__name__).warning(
                "Could not load %s — returning fresh instance. (%s)", path, exc
            )

    return model_cls()


def register(name: str, model_cls: Type[BaseModel],
             path_template: str) -> None:
    """Register a custom model type."""
    _REGISTRY[name] = (model_cls, path_template)