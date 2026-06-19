"""Optuna hyperparameter search (PLAN.md §3), time-boxed.

Minimises walk-forward WIN log-loss (a smooth, fast objective) over a recent-season window so
a sweep stays cheap. Demonstrated on CatBoost / XGBoost; the search space is intentionally
small. Best params are logged so a tuned model is reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass

import structlog

from hkjc.backtest.dataset import ModelData, load_model_data
from hkjc.common.config import AppConfig, get_config
from hkjc.experiments.runner import evaluate_model
from hkjc.models.base import ProbabilityModel
from hkjc.models.gbm import CatBoostModel, XGBoostModel

log = structlog.get_logger(__name__)


@dataclass(frozen=True, slots=True)
class TuneResult:
    model: str
    best_params: dict[str, float | int]
    best_log_loss: float
    n_trials: int


def _build(model: str, params: dict[str, float | int], cats: list[int]) -> ProbabilityModel:
    if model == "catboost":
        return CatBoostModel(cat_indices=cats, **params)  # type: ignore[arg-type]
    if model == "xgboost":
        return XGBoostModel(cat_indices=cats, **params)  # type: ignore[arg-type]
    msg = f"Tuning not wired for model {model!r} (try catboost/xgboost)."
    raise ValueError(msg)


def _suggest(model: str, trial: object) -> dict[str, float | int]:
    t = trial  # optuna.Trial
    if model == "catboost":
        return {
            "depth": t.suggest_int("depth", 4, 8),  # type: ignore[attr-defined]
            "learning_rate": t.suggest_float("learning_rate", 0.02, 0.2, log=True),  # type: ignore[attr-defined]
            "l2_leaf_reg": t.suggest_float("l2_leaf_reg", 1.0, 10.0),  # type: ignore[attr-defined]
        }
    return {
        "max_depth": t.suggest_int("max_depth", 4, 9),  # type: ignore[attr-defined]
        "learning_rate": t.suggest_float("learning_rate", 0.02, 0.2, log=True),  # type: ignore[attr-defined]
        "reg_lambda": t.suggest_float("reg_lambda", 1.0, 10.0),  # type: ignore[attr-defined]
    }


def tune(
    model: str = "catboost",
    *,
    cfg: AppConfig | None = None,
    n_trials: int = 15,
    max_test_seasons: int = 4,
    seed: int = 0,
    data: ModelData | None = None,
) -> TuneResult:
    """Run an Optuna study minimising walk-forward WIN log-loss for ``model``."""
    import optuna

    cfg = cfg or get_config()
    data = data or load_model_data(cfg)
    n_seasons = len(set(data.season.tolist()))
    min_train = max(1, n_seasons - max_test_seasons)

    def objective(trial: object) -> float:
        params = _suggest(model, trial)
        result = evaluate_model(
            lambda: _build(model, params, data.categorical_indices),
            data.x_full,
            data,
            market_weight=cfg.models.market_blend_weight,
            ev_threshold=cfg.risk.ev_threshold,
            stake=cfg.risk.min_bet,
            min_train_seasons=min_train,
            seed=seed,
            cfg=cfg,
        )
        return result.win_log_loss

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials)
    log.info("tune.done", model=model, best=study.best_value, params=study.best_params)
    return TuneResult(model, dict(study.best_params), float(study.best_value), n_trials)
