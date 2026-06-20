"""Walk-forward backtest engine (PLAN.md §1A, §2): the honest end-to-end run.

Trains the PL-strength baseline season-by-season (expanding window), predicts out-of-sample
WIN (softmax) and PLACE (Harville) probabilities, then simulates flat bets paid at the
*stored final dividends* (the pool truth). It reports **two ROIs** per PLAN.md §1A:

* **model-only** (conservative): selections come from model probabilities alone -- never from
  the odds -- so the decision cannot peek at the closing line. We back the model's top WIN
  and PLACE pick per race.
* **market-blended** (optimistic upper bound): the model is blended with the SP-implied
  market probability and bets are taken on a positive EV edge -- which *does* lean on the
  closing line, hence an upper bound.

A **leakage canary** rides through the same fit: a pure-noise feature whose fitted weight
must be ~0, and whose stand-alone (random-pick) ROI must show no edge.
"""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from hkjc.backtest import metrics
from hkjc.backtest.bootstrap import bootstrap_roi_ci
from hkjc.backtest.pari_mutuel import round_stake
from hkjc.backtest.walk_forward import iter_season_splits
from hkjc.common.config import AppConfig, get_config
from hkjc.features import store
from hkjc.features.base import BASELINE_FEATURES
from hkjc.models.base import FloatArray, IntArray, group_codes
from hkjc.models.logit import ConditionalLogit
from hkjc.models.place import harville_place_probs

WIN_UNIT = 10.0


@dataclass(frozen=True, slots=True)
class PolicyResult:
    """ROI + risk summary for one betting policy."""

    name: str
    n_bets: int
    staked: float
    profit: float
    roi: float
    roi_lo: float
    roi_hi: float
    sharpe: float


@dataclass(frozen=True, slots=True)
class BacktestResult:
    """End-to-end walk-forward outcome (the M2 exit-criterion artifact)."""

    feature_version: str
    n_oos_races: int
    n_oos_runners: int
    test_span: tuple[str, str]
    win_log_loss: float
    brier: float
    top1_hit_rate: float
    policies: dict[str, PolicyResult]
    ece: float = 0.0
    canary_coef_ratio: float | None = None
    canary_roi: float | None = None
    calibration_png: str | None = None


@dataclass(frozen=True, slots=True)
class _Arrays:
    x: FloatArray  # design matrix incl. canary as the last column
    canary_idx: int
    y: FloatArray
    placed: FloatArray
    n_places: IntArray
    win_odds: FloatArray
    market_prob: FloatArray
    win_div: FloatArray
    place_div: FloatArray
    canary: FloatArray
    race_id: IntArray
    season: np.typing.NDArray[np.str_]
    race_date: np.typing.NDArray[np.str_]  # 'YYYY-MM-DD' day key (per-day cap, M5 sweep)


def _dividend_lookup(cfg: AppConfig, pool: str) -> pl.DataFrame:
    """(race_key, saddle) -> dividend for a WIN/PLACE pool, from raw dividends Parquet."""
    base = cfg.paths.raw_dir / "dividends"
    files = [str(p) for p in base.rglob("*.parquet")] if base.exists() else []
    if not files:
        return pl.DataFrame(
            schema={
                "race_date": pl.Date,
                "venue": pl.String,
                "race_no": pl.Int64,
                "saddle": pl.Int64,
                f"{pool.lower()}_div": pl.Float64,
            }
        )
    div = pl.read_parquet(
        files, columns=["race_date", "venue", "race_no", "pool", "combination", "dividend"]
    )
    return (
        div.filter(pl.col("pool") == pool)
        .with_columns(saddle=pl.col("combination").str.strip_chars().cast(pl.Int64, strict=False))
        .filter(pl.col("saddle").is_not_null())
        .group_by(["race_date", "venue", "race_no", "saddle"])
        .agg(pl.col("dividend").min().alias(f"{pool.lower()}_div"))
    )


def _load_arrays(cfg: AppConfig) -> _Arrays:
    df = store.load_features(cfg)
    key = pl.concat_str(
        [pl.col("race_date").cast(pl.String), pl.col("venue"), pl.col("race_no").cast(pl.String)],
        separator="|",
    )
    df = df.with_columns(_chg=(key != key.shift(1)).fill_null(True))
    df = df.with_columns(race_id=(pl.col("_chg").cum_sum() - 1))
    df = df.join(
        _dividend_lookup(cfg, "WIN"), on=["race_date", "venue", "race_no", "saddle"], how="left"
    )
    df = df.join(
        _dividend_lookup(cfg, "PLACE"), on=["race_date", "venue", "race_no", "saddle"], how="left"
    )

    feat = df.select(BASELINE_FEATURES).to_numpy().astype(np.float64)
    canary = df["canary_random"].to_numpy().astype(np.float64)
    x = np.column_stack([feat, canary])
    return _Arrays(
        x=x,
        canary_idx=x.shape[1] - 1,
        y=df["won"].to_numpy().astype(np.float64),
        placed=df["placed"].to_numpy().astype(np.float64),
        n_places=df["n_places"].to_numpy().astype(np.int64),
        win_odds=df["win_odds"].to_numpy().astype(np.float64),
        market_prob=df["market_prob"].to_numpy().astype(np.float64),
        win_div=df["win_div"].to_numpy().astype(np.float64),
        place_div=df["place_div"].to_numpy().astype(np.float64),
        canary=canary,
        race_id=df["race_id"].to_numpy().astype(np.int64),
        season=df["season"].to_numpy().astype(str),
        race_date=df["race_date"].cast(pl.String).to_numpy().astype(str),
    )


def _segments(codes: IntArray) -> list[tuple[int, int]]:
    """Contiguous (start, stop) slices, one per race (codes are sorted/contiguous)."""
    _, starts, counts = np.unique(codes, return_index=True, return_counts=True)
    return [(int(s), int(s) + int(c)) for s, c in zip(starts, counts, strict=True)]


def _win_profit(stake: float, won: bool, div: float, odds: float) -> float:
    """Realised WIN profit: pay the stored dividend (per HK$10); fall back to SP if absent."""
    if not won:
        return -stake
    if np.isfinite(div) and div > 0:
        return stake * div / WIN_UNIT - stake
    if np.isfinite(odds) and odds > 0:
        return stake * (odds - 1.0)
    return -stake


def _place_profit(stake: float, placed: bool, div: float) -> float:
    if not placed:
        return -stake
    if np.isfinite(div) and div > 0:
        return stake * div / WIN_UNIT - stake
    return 0.0  # placed but dividend missing (rare) -> treat as push


def _summarise(
    name: str,
    race_profit: list[float],
    race_stake: list[float],
    bet_returns: list[float],
    seed: int,
) -> PolicyResult:
    profit_arr = np.asarray(race_profit, dtype=np.float64)
    stake_arr = np.asarray(race_stake, dtype=np.float64)
    ci = bootstrap_roi_ci(profit_arr, stake_arr, seed=seed)
    return PolicyResult(
        name=name,
        n_bets=len(bet_returns),
        staked=float(stake_arr.sum()),
        profit=float(profit_arr.sum()),
        roi=ci.roi,
        roi_lo=ci.lo,
        roi_hi=ci.hi,
        sharpe=metrics.sharpe(np.asarray(bet_returns, dtype=np.float64)),
    )


@dataclass(frozen=True, slots=True)
class WalkForwardOOS:
    """The walk-forward out-of-sample predictions, reused by the backtest and the M5 sweep."""

    arrays: _Arrays
    oos: IntArray  # indices into ``arrays`` for the OOS runners (season-ordered, concatenated)
    win_prob: FloatArray  # model WIN probability per OOS runner
    place_prob: FloatArray  # model PLACE probability (Harville) per OOS runner
    model: ConditionalLogit  # the last-season fit (for the canary-coefficient check)


def walk_forward_oos(
    cfg: AppConfig | None = None, *, l2: float = 1.0, min_train_seasons: int = 1
) -> WalkForwardOOS:
    """Fit the PL-logit season-by-season (expanding window) and collect OOS WIN/PLACE probs."""
    cfg = cfg or get_config()
    a = _load_arrays(cfg)

    oos_idx: list[IntArray] = []
    wp_parts: list[FloatArray] = []
    pp_parts: list[FloatArray] = []
    last_model: ConditionalLogit | None = None
    for _season, train_mask, test_mask in iter_season_splits(a.season, min_train_seasons):
        if a.y[train_mask].sum() < 10:
            continue
        model = ConditionalLogit(l2=l2).fit(a.x[train_mask], a.race_id[train_mask], a.y[train_mask])
        last_model = model
        test_idx = np.where(test_mask)[0]
        wp = model.win_probs(a.x[test_idx], a.race_id[test_idx])
        codes_t, ng_t = group_codes(a.race_id[test_idx])
        npg = np.zeros(ng_t, dtype=np.int64)
        npg[codes_t] = a.n_places[test_idx]
        oos_idx.append(test_idx)
        wp_parts.append(wp)
        pp_parts.append(harville_place_probs(wp, codes_t, ng_t, npg))

    if not oos_idx or last_model is None:
        msg = "Not enough seasons to walk forward; scrape/backfill more data first."
        raise RuntimeError(msg)

    return WalkForwardOOS(
        arrays=a,
        oos=np.concatenate(oos_idx),
        win_prob=np.concatenate(wp_parts),
        place_prob=np.concatenate(pp_parts),
        model=last_model,
    )


def run_backtest(
    cfg: AppConfig | None = None,
    *,
    l2: float = 1.0,
    market_weight: float | None = None,
    ev_threshold: float | None = None,
    flat_stake: float | None = None,
    min_train_seasons: int = 1,
    seed: int = 0,
    make_plot: bool = True,
) -> BacktestResult:
    """Run the honest walk-forward backtest and return the result summary."""
    cfg = cfg or get_config()
    market_weight = cfg.models.market_blend_weight if market_weight is None else market_weight
    ev_threshold = cfg.risk.ev_threshold if ev_threshold is None else ev_threshold
    stake = cfg.risk.min_bet if flat_stake is None else flat_stake

    wf = walk_forward_oos(cfg, l2=l2, min_train_seasons=min_train_seasons)
    ocode, ong = group_codes(wf.arrays.race_id[wf.oos])
    return _evaluate(
        wf.arrays,
        wf.oos,
        wf.win_prob,
        wf.place_prob,
        ocode,
        ong,
        wf.model,
        market_weight=market_weight,
        ev_threshold=ev_threshold,
        stake=stake,
        seed=seed,
        make_plot=make_plot,
        cfg=cfg,
    )


def _evaluate(
    a: _Arrays,
    oos: IntArray,
    wp: FloatArray,
    pp: FloatArray,
    ocode: IntArray,
    ong: int,
    model: ConditionalLogit,
    *,
    market_weight: float,
    ev_threshold: float,
    stake: float,
    seed: int,
    make_plot: bool,
    cfg: AppConfig,
) -> BacktestResult:
    o_y = a.y[oos]
    o_placed = a.placed[oos]
    o_odds = a.win_odds[oos]
    o_mkt = np.where(np.isfinite(a.market_prob[oos]), a.market_prob[oos], wp)
    o_windiv = a.win_div[oos]
    o_placediv = a.place_div[oos]
    o_canary = a.canary[oos]

    pol: dict[str, tuple[list[float], list[float], list[float]]] = {
        name: ([], [], []) for name in ("model_win", "model_place", "blend_win", "canary_win")
    }
    flat = round_stake(stake, unit=cfg.risk.bet_unit, min_bet=cfg.risk.min_bet)

    for start, stop in _segments(ocode):
        sl = slice(start, stop)
        # model-only WIN: back the top model pick.
        i = start + int(np.argmax(wp[sl]))
        _record(pol["model_win"], _win_profit(flat, o_y[i] > 0, o_windiv[i], o_odds[i]), flat)
        # model-only PLACE: back the top place pick.
        j = start + int(np.argmax(pp[sl]))
        _record(pol["model_place"], _place_profit(flat, o_placed[j] > 0, o_placediv[j]), flat)
        # canary sentinel: random pick by noise -> must show no edge.
        c = start + int(np.argmax(o_canary[sl]))
        _record(pol["canary_win"], _win_profit(flat, o_y[c] > 0, o_windiv[c], o_odds[c]), flat)
        # market-blended WIN: bet positive-EV runners at the SP line.
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
    canary = _summarise("canary WIN (sentinel)", *pol["canary_win"], seed=seed)

    beta = np.abs(model.coefficients)
    canary_ratio = float(beta[a.canary_idx] / beta.mean()) if beta.mean() > 0 else 0.0

    png = _calibration_plot(wp, o_y, cfg) if make_plot else None
    return BacktestResult(
        feature_version=cfg.features.feature_version,
        n_oos_races=ong,
        n_oos_runners=int(oos.size),
        test_span=(str(a.season[oos][0]), str(a.season[oos][-1])),
        win_log_loss=metrics.win_log_loss(wp, o_y, ocode, ong),
        brier=metrics.brier_win(wp, o_y),
        top1_hit_rate=metrics.top1_hit_rate(wp, o_y, ocode, ong),
        policies=policies,
        ece=metrics.expected_calibration_error(wp, o_y),
        canary_coef_ratio=canary_ratio,
        canary_roi=canary.roi,
        calibration_png=png,
    )


def _record(
    bucket: tuple[list[float], list[float], list[float]], profit: float, stake: float
) -> None:
    bucket[0].append(profit)
    bucket[1].append(stake)
    bucket[2].append(profit / stake if stake > 0 else 0.0)


def _calibration_plot(wp: FloatArray, won: FloatArray, cfg: AppConfig) -> str:
    bins = metrics.calibration_bins(wp, won, n_bins=12)
    out_dir = cfg.paths.processed_dir / "backtest"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "calibration_win.png"
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "--", color="grey", label="perfect")
    ax.plot(bins.pred_mean, bins.obs_rate, "o-", color="#1f77b4", label="model")
    ax.set_xlabel("Predicted WIN probability")
    ax.set_ylabel("Observed win rate")
    ax.set_title("WIN calibration (walk-forward OOS)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return str(path)
