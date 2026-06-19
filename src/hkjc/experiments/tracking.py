"""MLflow (local) experiment tracking (PLAN.md §3).

Logs each model run to a file-backed MLflow store under ``data/mlruns`` with the feature-store
**data hash**, so any leaderboard number is reproducible from (config + data hash + params).
MLflow is best-effort: if it errors, the run is logged as a warning and the leaderboard
continues.
"""

from __future__ import annotations

import structlog

from hkjc.backtest.engine import BacktestResult
from hkjc.common.config import AppConfig

log = structlog.get_logger(__name__)

EXPERIMENT = "hkjc-leaderboard"


def log_run(
    cfg: AppConfig, name: str, result: BacktestResult, *, data_hash: str, market_weight: float
) -> None:
    """Log one model's params + metrics to the local MLflow store (best-effort)."""
    try:
        import mlflow

        cfg.paths.mlruns_dir.mkdir(parents=True, exist_ok=True)
        # MLflow 3.x deprecates the file store; use a local sqlite backend (PLAN.md §3).
        mlflow.set_tracking_uri(f"sqlite:///{(cfg.paths.mlruns_dir / 'mlflow.db').as_posix()}")
        mlflow.set_experiment(EXPERIMENT)
        with mlflow.start_run(run_name=name):
            mlflow.log_params(
                {
                    "model": name,
                    "feature_version": result.feature_version,
                    "data_hash": data_hash,
                    "market_weight": market_weight,
                }
            )
            metrics = {
                "win_log_loss": result.win_log_loss,
                "brier": result.brier,
                "top1_hit_rate": result.top1_hit_rate,
                "ece": result.ece,
                "n_oos_races": float(result.n_oos_races),
            }
            for key, pol in result.policies.items():
                metrics[f"{key}_roi"] = pol.roi
                metrics[f"{key}_sharpe"] = pol.sharpe
            mlflow.log_metrics(metrics)
    except Exception as exc:
        log.warning("mlflow.log_failed", model=name, error=str(exc))
