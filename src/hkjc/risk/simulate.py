"""Day-ordered bankroll simulator for a staking policy (M5, PLAN.md §2 M5).

Replays the out-of-sample races **in calendar order**, compounding one bankroll. Each day is
deployed off the *start-of-day* bankroll (so a day's races are sized simultaneously), capped
per race (10%) and per day (25%), rounded to the legal HK$10 unit, paid at the **stored final
dividends**, and credited any losing-turnover rebate. The realised path yields the geometric
growth, max drawdown and (bootstrapped) risk-of-ruin that distinguish staking methods even
when -- as PLAN.md §1F predicts -- every method loses to the takeout.

Pool-impact dilution is *not* applied: the historical store carries no pool totals, and a
HK$10k per-race cap (the most at the HK$100k bankroll) is <0.2% of a typical HK$-million HKJC
WIN pool, so dilution is negligible across all four bankrolls -- the rebate threshold, not
pool impact, is the large-bankroll lever.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hkjc.backtest import metrics
from hkjc.backtest.bootstrap import bootstrap_roi_ci
from hkjc.models.base import FloatArray
from hkjc.risk.rebate import RebateRule
from hkjc.risk.staking import StakingConfig, desired_stakes, round_stakes, scale_to_cap

WIN_UNIT = 10.0


@dataclass(frozen=True, slots=True)
class RacePanel:
    """One race's aligned WIN + PLACE inputs (probabilities, prices, outcomes, dividends)."""

    p_win: FloatArray  # bettor's WIN probability (model blended with the market)
    b_win: FloatArray  # WIN decimal odds (the SP line)
    won: FloatArray
    win_div: FloatArray
    p_place: FloatArray  # model PLACE probability (Harville)
    b_place: FloatArray  # constructed PLACE line (market Harville x place takeout)
    placed: FloatArray
    place_div: FloatArray


@dataclass(frozen=True, slots=True)
class StakingOutcome:
    """Realised path summary for one (policy, bankroll) cell of the sweep."""

    method: str
    bankroll0: float
    correlated: bool
    kelly_lambda: float
    n_bets: int
    total_staked: float
    total_profit: float
    roi: float
    roi_lo: float
    roi_hi: float
    terminal_bankroll: float
    growth_per_day: float
    max_drawdown: float
    sharpe: float
    rounding_loss: float
    rebate_credit: float
    rebate_days: int
    ruin_prob: float
    ruined: bool


def _bet_profit_safe(stake: float, dividend: float, won: bool, odds: float) -> float:
    """Profit paid at the stored dividend; fall back to the SP line, else push (PLACE)."""
    if not won:
        return -stake
    if np.isfinite(dividend) and dividend > 0:
        return stake * dividend / WIN_UNIT - stake
    if np.isfinite(odds) and odds > 0:
        return stake * (odds - 1.0)  # WIN with a missing dividend -> pay the SP line
    return 0.0  # placed but the place dividend is missing -> treat as a push


def _race_bets(
    rp: RacePanel, start_bk: float, cfg: StakingConfig, pools: tuple[str, ...]
) -> tuple[list[float], list[float], list[bool], list[float], list[str]]:
    """Desired stakes for one race (WIN+PLACE), already scaled to the per-race exposure cap."""
    stakes: list[float] = []
    divs: list[float] = []
    wons: list[bool] = []
    odds: list[float] = []
    pool: list[str] = []
    if "win" in pools:
        d = desired_stakes(rp.p_win, rp.b_win, start_bk, cfg, win=True)
        for i in np.nonzero(d > 0.0)[0]:
            stakes.append(float(d[i]))
            divs.append(float(rp.win_div[i]))
            wons.append(bool(rp.won[i] > 0))
            odds.append(float(rp.b_win[i]))
            pool.append("win")
    if "place" in pools:
        d = desired_stakes(rp.p_place, rp.b_place, start_bk, cfg, win=False)
        for i in np.nonzero(d > 0.0)[0]:
            stakes.append(float(d[i]))
            divs.append(float(rp.place_div[i]))
            wons.append(bool(rp.placed[i] > 0))
            odds.append(float("nan"))
            pool.append("place")
    if not stakes:
        return stakes, divs, wons, odds, pool
    capped = scale_to_cap(np.asarray(stakes, dtype=np.float64), cfg.per_race_cap * start_bk)
    return capped.tolist(), divs, wons, odds, pool


def _ruin_probability(returns: FloatArray, ruin_floor: float, *, seed: int, n_iter: int) -> float:
    """Bootstrap P(bankroll ever falls below ``ruin_floor`` of its start) by resampling the
    realised daily multiplicative returns -- the variance lens that separates full from
    fractional Kelly on the same (losing) edge."""
    n = returns.size
    if n == 0:
        return 0.0
    rng = np.random.default_rng(seed)
    ruined = 0
    for _ in range(n_iter):
        path = np.cumprod(returns[rng.integers(0, n, size=n)])
        if float(path.min()) < ruin_floor:
            ruined += 1
    return ruined / n_iter


def simulate_path(
    card: list[list[RacePanel]],
    cfg: StakingConfig,
    bankroll0: float,
    *,
    pools: tuple[str, ...] = ("win", "place"),
    rebate: RebateRule | None = None,
    seed: int = 0,
    ruin_floor: float = 0.10,  # "ruin" = bankroll falls below 10% of its starting value
    ruin_iters: int = 300,
) -> StakingOutcome:
    """Compound ``bankroll0`` through the day-ordered ``card`` under one staking policy."""
    rebate = rebate or RebateRule()
    bankroll = bankroll0
    peak = bankroll0
    max_dd = 0.0
    n_bets = 0
    total_staked = 0.0
    total_profit = 0.0
    rounding_loss = 0.0
    rebate_credit = 0.0
    rebate_days = 0
    ruined = False
    day_profit: list[float] = []
    day_stake: list[float] = []
    daily_return: list[float] = []
    ruin_cash = ruin_floor * bankroll0

    for races in card:
        start_bk = bankroll
        if start_bk < cfg.min_bet:  # bankrupt -> can no longer bet
            ruined = True
            daily_return.append(1.0)
            day_profit.append(0.0)
            day_stake.append(0.0)
            continue

        stakes: list[float] = []
        divs: list[float] = []
        wons: list[bool] = []
        odds: list[float] = []
        pool: list[str] = []
        for rp in races:
            rs, rd, rw, ro, rpl = _race_bets(rp, start_bk, cfg, pools)
            stakes.extend(rs)
            divs.extend(rd)
            wons.extend(rw)
            odds.extend(ro)
            pool.extend(rpl)

        if not stakes:
            daily_return.append(1.0)
            day_profit.append(0.0)
            day_stake.append(0.0)
            continue

        sized = scale_to_cap(np.asarray(stakes, dtype=np.float64), cfg.per_day_cap * start_bk)
        desired_after_caps = float(sized.sum())
        rounded = round_stakes(sized, unit=cfg.bet_unit, min_bet=cfg.min_bet)
        rounding_loss += max(desired_after_caps - float(rounded.sum()), 0.0)

        profit = 0.0
        staked = 0.0
        losing = {"win": 0.0, "place": 0.0}
        for s, dv, wn, od, pl in zip(rounded.tolist(), divs, wons, odds, pool, strict=True):
            if s <= 0.0:
                continue
            staked += s
            n_bets += 1
            profit += _bet_profit_safe(s, dv, wn, od)
            if not wn:
                losing[pl] += s

        credit = rebate.credit(losing["win"]) + rebate.credit(losing["place"])
        if rebate.triggered(losing["win"]) or rebate.triggered(losing["place"]):
            rebate_days += 1
        rebate_credit += credit
        profit += credit

        bankroll += profit
        total_profit += profit
        total_staked += staked
        day_profit.append(profit)
        day_stake.append(staked)
        daily_return.append(bankroll / start_bk if start_bk > 0 else 1.0)
        peak = max(peak, bankroll)
        if peak > 0:
            max_dd = max(max_dd, (peak - bankroll) / peak)
        if bankroll < ruin_cash:
            ruined = True

    ci = bootstrap_roi_ci(
        np.asarray(day_profit, dtype=np.float64), np.asarray(day_stake, dtype=np.float64), seed=seed
    )
    returns = np.asarray(daily_return, dtype=np.float64)
    log_m = np.log(np.clip(returns, 1e-12, None))
    growth = float(np.exp(log_m.mean()) - 1.0) if returns.size else 0.0
    return StakingOutcome(
        method=cfg.method,
        bankroll0=bankroll0,
        correlated=cfg.correlated,
        kelly_lambda=cfg.kelly_lambda,
        n_bets=n_bets,
        total_staked=total_staked,
        total_profit=total_profit,
        roi=ci.roi,
        roi_lo=ci.lo,
        roi_hi=ci.hi,
        terminal_bankroll=bankroll,
        growth_per_day=growth,
        max_drawdown=max_dd,
        sharpe=metrics.sharpe(returns - 1.0),
        rounding_loss=rounding_loss,
        rebate_credit=rebate_credit,
        rebate_days=rebate_days,
        ruin_prob=_ruin_probability(returns, ruin_floor, seed=seed + 1, n_iter=ruin_iters),
        ruined=ruined,
    )
