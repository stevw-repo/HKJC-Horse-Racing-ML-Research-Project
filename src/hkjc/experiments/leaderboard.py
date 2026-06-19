"""Model leaderboard (PLAN.md §2 M3 exit criterion).

Runs every model in the zoo through the identical walk-forward protocol, logs each run to a
local MLflow store (with the feature-store data hash for reproducibility), and ranks them on
walk-forward ROI / Sharpe / log-loss / calibration.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial

import structlog

from hkjc.backtest.dataset import ModelData, load_model_data
from hkjc.backtest.engine import BacktestResult
from hkjc.common.config import AppConfig, get_config
from hkjc.experiments.runner import evaluate_model
from hkjc.experiments.tracking import log_run
from hkjc.features.design import data_hash
from hkjc.models.base import FloatArray, ProbabilityModel
from hkjc.models.ensemble import EnsembleModel
from hkjc.models.gbm import CatBoostModel, LambdaMARTModel, LightGBMModel, XGBoostModel
from hkjc.models.logit import ConditionalLogit
from hkjc.models.nn import FTTransformerModel, MLPModel

log = structlog.get_logger(__name__)

# name -> (factory(data) -> model, design: "numeric" | "full")
ModelSpec = tuple[Callable[[ModelData], ProbabilityModel], str]


def default_models(nn_epochs: int = 120) -> dict[str, ModelSpec]:
    """The M3 zoo. GBMs/ensemble take the full (numeric+categorical) matrix; logit/NNs the
    numeric slice."""
    return {
        "logit": (lambda d: ConditionalLogit(), "numeric"),
        "lightgbm": (lambda d: LightGBMModel(cat_indices=d.categorical_indices), "full"),
        "lambdamart": (lambda d: LambdaMARTModel(cat_indices=d.categorical_indices), "full"),
        "xgboost": (lambda d: XGBoostModel(cat_indices=d.categorical_indices), "full"),
        "catboost": (lambda d: CatBoostModel(cat_indices=d.categorical_indices), "full"),
        "mlp": (lambda d: MLPModel(epochs=nn_epochs), "numeric"),
        "ft_transformer": (lambda d: FTTransformerModel(epochs=nn_epochs), "numeric"),
        "ensemble": (
            lambda d: EnsembleModel(
                [
                    CatBoostModel(cat_indices=d.categorical_indices),
                    XGBoostModel(cat_indices=d.categorical_indices),
                    LightGBMModel(cat_indices=d.categorical_indices),
                ]
            ),
            "full",
        ),
    }


@dataclass(frozen=True, slots=True)
class LeaderboardEntry:
    """One model's walk-forward result."""

    name: str
    result: BacktestResult


def run_leaderboard(
    cfg: AppConfig | None = None,
    *,
    models: dict[str, ModelSpec] | None = None,
    market_weight: float | None = None,
    ev_threshold: float | None = None,
    max_test_seasons: int | None = None,
    nn_epochs: int = 120,
    seed: int = 0,
) -> list[LeaderboardEntry]:
    """Train + score every model; log to MLflow; return entries sorted by model-only WIN ROI."""
    cfg = cfg or get_config()
    market_weight = cfg.models.market_blend_weight if market_weight is None else market_weight
    ev_threshold = cfg.risk.ev_threshold if ev_threshold is None else ev_threshold
    specs = models or default_models(nn_epochs=nn_epochs)
    data = load_model_data(cfg)
    dh = data_hash()
    min_train = _min_train_for(data, max_test_seasons)

    entries: list[LeaderboardEntry] = []
    for name, (factory, design) in specs.items():
        x: FloatArray = data.numeric() if design == "numeric" else data.x_full
        log.info("leaderboard.fit", model=name, design=design, rows=int(x.shape[0]))
        result = evaluate_model(
            partial(factory, data),
            x,
            data,
            market_weight=market_weight,
            ev_threshold=ev_threshold,
            stake=cfg.risk.min_bet,
            min_train_seasons=min_train,
            seed=seed,
            cfg=cfg,
        )
        log_run(cfg, name, result, data_hash=dh, market_weight=market_weight)
        entries.append(LeaderboardEntry(name, result))

    entries.sort(key=lambda e: e.result.policies["model_win"].roi, reverse=True)
    return entries


def _min_train_for(data: ModelData, max_test_seasons: int | None) -> int:
    """Map a 'last N test seasons' request to a min-train-seasons cutoff (1 = full history)."""
    if max_test_seasons is None:
        return 1
    n_seasons = len(set(data.season.tolist()))
    return max(1, n_seasons - max_test_seasons)


def format_leaderboard(entries: list[LeaderboardEntry]) -> str:
    """ASCII table of the leaderboard (ranked, model-only + blended ROI, log-loss, ECE)."""
    header = (
        f"{'model':<16}{'logloss':>9}{'top1':>7}{'ECE':>7}"
        f"{'WIN ROI':>10}{'PLACE ROI':>11}{'blend ROI':>11}"
    )
    lines = [header, "-" * len(header)]
    for e in entries:
        r = e.result
        lines.append(
            f"{e.name:<16}{r.win_log_loss:>9.4f}{r.top1_hit_rate:>7.3f}{r.ece:>7.3f}"
            f"{r.policies['model_win'].roi:>+10.1%}{r.policies['model_place'].roi:>+11.1%}"
            f"{r.policies['blend_win'].roi:>+11.1%}"
        )
    return "\n".join(lines)
