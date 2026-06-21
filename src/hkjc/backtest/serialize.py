"""Serialize backtest results to JSON snapshots for the API to read (M6).

The API is a pure reader: ``run_backtest`` and ``run_leaderboard`` persist these snapshots so
the dashboards render real M2/M3 output without re-running the (expensive) walk-forward.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from hkjc.backtest.engine import BacktestResult
from hkjc.backtest.metrics import CalibrationBins


def calibration_to_list(bins: CalibrationBins) -> list[dict[str, float]]:
    """Calibration bins -> a list of points for the UI reliability curve."""
    return [
        {"bin_mid": m, "pred_mean": p, "obs_rate": o, "count": c}
        for m, p, o, c in zip(bins.bin_mid, bins.pred_mean, bins.obs_rate, bins.count, strict=True)
    ]


def result_to_dict(
    result: BacktestResult, calibration: list[dict[str, float]] | None = None
) -> dict[str, Any]:
    """A BacktestResult as a plain JSON-able dict (policies + optional calibration curve)."""
    return {
        "feature_version": result.feature_version,
        "n_oos_races": result.n_oos_races,
        "n_oos_runners": result.n_oos_runners,
        "test_span": list(result.test_span),
        "win_log_loss": result.win_log_loss,
        "brier": result.brier,
        "top1_hit_rate": result.top1_hit_rate,
        "ece": result.ece,
        "canary_coef_ratio": result.canary_coef_ratio,
        "canary_roi": result.canary_roi,
        "policies": {
            key: {
                "name": pol.name,
                "n_bets": pol.n_bets,
                "staked": pol.staked,
                "profit": pol.profit,
                "roi": pol.roi,
                "roi_lo": pol.roi_lo,
                "roi_hi": pol.roi_hi,
                "sharpe": pol.sharpe,
            }
            for key, pol in result.policies.items()
        },
        "calibration": calibration or [],
    }


def write_json(path: Path, obj: Any) -> Path:
    """Write ``obj`` as indented JSON, creating parent dirs. Returns the path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    return path
