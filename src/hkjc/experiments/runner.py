"""Generic walk-forward evaluation for any ``ProbabilityModel`` (M3).

Mirrors the M2 engine's honest protocol -- expanding per-season folds, WIN softmax + Harville
PLACE out-of-sample, flat bets paid at stored dividends, two ROIs -- but for an arbitrary
model and design matrix, so the leaderboard scores every learner identically. Reuses the
engine's dividend/profit/summary helpers (no canary; that was M2's one-time proof).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from hkjc.backtest import metrics
from hkjc.backtest.dataset import ModelData
from hkjc.backtest.engine import (
    BacktestResult,
    _place_profit,
    _record,
    _segments,
    _summarise,
    _win_profit,
)
from hkjc.backtest.pari_mutuel import round_stake
from hkjc.backtest.walk_forward import iter_season_splits
from hkjc.common.config import AppConfig, get_config
from hkjc.models.base import FloatArray, ProbabilityModel, group_codes
from hkjc.models.place import harville_place_probs

Factory = Callable[[], ProbabilityModel]


def evaluate_model(
    factory: Factory,
    x: FloatArray,
    data: ModelData,
    *,
    market_weight: float,
    ev_threshold: float,
    stake: float,
    min_train_seasons: int = 1,
    seed: int = 0,
    cfg: AppConfig | None = None,
) -> BacktestResult:
    """Walk-forward train/predict ``factory()`` on ``x`` and score it like the M2 engine."""
    cfg = cfg or get_config()
    oos_idx: list[np.typing.NDArray[np.int64]] = []
    wp_parts: list[FloatArray] = []
    pp_parts: list[FloatArray] = []
    for _season, train_mask, test_mask in iter_season_splits(data.season, min_train_seasons):
        if data.y[train_mask].sum() < 10:
            continue
        model = factory().fit(x[train_mask], data.race_id[train_mask], data.y[train_mask])
        ti = np.where(test_mask)[0]
        wp = model.win_probs(x[ti], data.race_id[ti])
        codes_t, ng_t = group_codes(data.race_id[ti])
        npg = np.zeros(ng_t, dtype=np.int64)
        npg[codes_t] = data.n_places[ti]
        oos_idx.append(ti)
        wp_parts.append(wp)
        pp_parts.append(harville_place_probs(wp, codes_t, ng_t, npg))

    if not oos_idx:
        msg = "Not enough seasons to walk forward."
        raise RuntimeError(msg)
    oos = np.concatenate(oos_idx)
    wp = np.concatenate(wp_parts)
    pp = np.concatenate(pp_parts)
    ocode, ong = group_codes(data.race_id[oos])
    return _eval_policies(
        data, oos, wp, pp, ocode, ong, market_weight, ev_threshold, stake, seed, cfg
    )


def _eval_policies(
    data: ModelData,
    oos: np.typing.NDArray[np.int64],
    wp: FloatArray,
    pp: FloatArray,
    ocode: np.typing.NDArray[np.int64],
    ong: int,
    market_weight: float,
    ev_threshold: float,
    stake: float,
    seed: int,
    cfg: AppConfig,
) -> BacktestResult:
    o_y = data.y[oos]
    o_placed = data.placed[oos]
    o_odds = data.win_odds[oos]
    o_mkt = np.where(np.isfinite(data.market_prob[oos]), data.market_prob[oos], wp)
    o_windiv = data.win_div[oos]
    o_placediv = data.place_div[oos]
    flat = round_stake(stake, unit=cfg.risk.bet_unit, min_bet=cfg.risk.min_bet)
    pol: dict[str, tuple[list[float], list[float], list[float]]] = {
        name: ([], [], []) for name in ("model_win", "model_place", "blend_win")
    }
    for start, stop in _segments(ocode):
        sl = slice(start, stop)
        i = start + int(np.argmax(wp[sl]))
        _record(pol["model_win"], _win_profit(flat, o_y[i] > 0, o_windiv[i], o_odds[i]), flat)
        j = start + int(np.argmax(pp[sl]))
        _record(pol["model_place"], _place_profit(flat, o_placed[j] > 0, o_placediv[j]), flat)
        blended = (1.0 - market_weight) * wp[sl] + market_weight * o_mkt[sl]
        odds_sl = o_odds[sl]
        race_profit = 0.0
        race_stake = 0.0
        for k in range(stop - start):
            if not (np.isfinite(odds_sl[k]) and odds_sl[k] > 0):
                continue
            if blended[k] * odds_sl[k] - 1.0 >= ev_threshold:
                p = _win_profit(flat, o_y[start + k] > 0, o_windiv[start + k], odds_sl[k])
                race_profit += p
                race_stake += flat
                pol["blend_win"][2].append(p / flat)
        pol["blend_win"][0].append(race_profit)
        pol["blend_win"][1].append(race_stake)

    policies = {
        "model_win": _summarise("model-only WIN", *pol["model_win"], seed=seed),
        "model_place": _summarise("model-only PLACE", *pol["model_place"], seed=seed),
        "blend_win": _summarise("market-blended WIN", *pol["blend_win"], seed=seed),
    }
    return BacktestResult(
        feature_version=cfg.features.feature_version,
        n_oos_races=ong,
        n_oos_runners=int(oos.size),
        test_span=(str(data.season[oos][0]), str(data.season[oos][-1])),
        win_log_loss=metrics.win_log_loss(wp, o_y, ocode, ong),
        brier=metrics.brier_win(wp, o_y),
        top1_hit_rate=metrics.top1_hit_rate(wp, o_y, ocode, ong),
        policies=policies,
        ece=metrics.expected_calibration_error(wp, o_y),
    )
