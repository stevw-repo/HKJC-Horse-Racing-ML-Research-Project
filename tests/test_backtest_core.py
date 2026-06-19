"""Tests for walk-forward splitting, metrics, and bootstrap CIs."""

from __future__ import annotations

import numpy as np

from hkjc.backtest import metrics
from hkjc.backtest.bootstrap import bootstrap_roi_ci
from hkjc.backtest.walk_forward import iter_season_splits
from hkjc.models.base import group_codes


def test_season_splits_are_time_ordered() -> None:
    seasons = np.array(["2018-19"] * 2 + ["2019-20"] * 2 + ["2020-21"] * 2)
    splits = list(iter_season_splits(seasons, min_train_seasons=1))
    assert [s for s, _, _ in splits] == ["2019-20", "2020-21"]
    for test_season, train_mask, test_mask in splits:
        # No row is both train and test; train precedes test in season order.
        assert not (train_mask & test_mask).any()
        train_seasons = set(seasons[train_mask].tolist())
        assert all(ts < test_season for ts in train_seasons)


def test_roi_and_sharpe() -> None:
    assert metrics.roi(50.0, 200.0) == 0.25
    assert metrics.roi(0.0, 0.0) == 0.0
    assert metrics.sharpe(np.array([0.1, 0.1, 0.1])) == 0.0  # zero variance
    assert metrics.sharpe(np.array([1.0])) == 0.0


def test_win_log_loss_perfect_vs_uniform() -> None:
    # Two 2-horse races; winner is row 0 of each.
    won = np.array([1.0, 0.0, 1.0, 0.0])
    codes, ng = group_codes(np.array([0, 0, 1, 1]))
    perfect = np.array([1.0, 0.0, 1.0, 0.0])
    uniform = np.array([0.5, 0.5, 0.5, 0.5])
    assert metrics.win_log_loss(perfect, won, codes, ng) < 1e-6
    assert abs(metrics.win_log_loss(uniform, won, codes, ng) - np.log(2)) < 1e-9


def test_calibration_bins_recover_rate() -> None:
    prob = np.concatenate([np.full(100, 0.1), np.full(100, 0.9)])
    rng = np.random.default_rng(0)
    outcome = np.concatenate([(rng.random(100) < 0.1), (rng.random(100) < 0.9)]).astype(float)
    bins = metrics.calibration_bins(prob, outcome, n_bins=10)
    assert len(bins.pred_mean) == 2
    assert abs(bins.obs_rate[0] - 0.1) < 0.12
    assert abs(bins.obs_rate[1] - 0.9) < 0.12


def test_bootstrap_ci_brackets_point() -> None:
    rng = np.random.default_rng(0)
    profit = rng.normal(1.0, 5.0, size=500)
    stake = np.full(500, 10.0)
    ci = bootstrap_roi_ci(profit, stake, n_iter=500, seed=1)
    assert ci.lo <= ci.roi <= ci.hi
    assert ci.n_races == 500
