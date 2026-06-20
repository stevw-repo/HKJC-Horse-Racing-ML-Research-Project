"""Pydantic v2 response models for the API (M6).

These are the typed contracts the React dashboards consume; every endpoint serves **real**
M2-M5 output (read from the DuckDB views + persisted ``processed/`` snapshots), except the
race-day card, which is a clearly-flagged mock until the M7 live-odds logger exists.
"""

from __future__ import annotations

from pydantic import BaseModel


class PolicyOut(BaseModel):
    """One betting policy's ROI summary in the backtest."""

    name: str
    n_bets: int
    staked: float
    profit: float
    roi: float
    roi_lo: float
    roi_hi: float
    sharpe: float


class CalibrationPoint(BaseModel):
    """One reliability-diagram bin (predicted vs observed WIN rate)."""

    bin_mid: float
    pred_mean: float
    obs_rate: float
    count: int


class BacktestResponse(BaseModel):
    """The honest walk-forward backtest result (M2)."""

    feature_version: str
    n_oos_races: int
    n_oos_runners: int
    test_span: list[str]
    win_log_loss: float
    brier: float
    top1_hit_rate: float
    ece: float
    canary_coef_ratio: float | None = None
    canary_roi: float | None = None
    policies: dict[str, PolicyOut]
    calibration: list[CalibrationPoint]


class LeaderboardRow(BaseModel):
    """One model's walk-forward metrics in the M3 leaderboard."""

    name: str
    n_oos_races: int
    win_log_loss: float
    brier: float
    top1_hit_rate: float
    ece: float
    model_win_roi: float | None = None
    model_place_roi: float | None = None
    blend_win_roi: float | None = None


class StakingRow(BaseModel):
    """One (policy, bankroll) cell of the M5 staking sweep."""

    bankroll: float
    policy: str
    n_bets: int
    staked: float
    roi: float
    roi_lo: float
    roi_hi: float
    terminal: float
    max_dd: float
    sharpe: float
    round_loss: float
    rebate_days: int
    ruin_prob: float
    ruined: bool


class RaceSummary(BaseModel):
    """A stored race (for the data-health / explorer context)."""

    race_date: str
    venue: str
    race_no: int
    distance: int | None = None
    going: str | None = None
    field_size: int


class RaceDayRunner(BaseModel):
    """A mocked race-day runner: model probabilities + value vs the (mock) live odds."""

    saddle: int
    horse: str
    win_prob: float
    place_prob: float
    win_odds: float
    ev: float
    stake: float


class RaceDayRace(BaseModel):
    """A mocked upcoming race with value-staking recommendations."""

    race_no: int
    distance: int
    going: str
    runners: list[RaceDayRunner]


class RaceDayResponse(BaseModel):
    """A mocked upcoming card. ``mock`` is always true until the M7 live logger lands."""

    mock: bool
    meeting_date: str
    venue: str
    note: str
    races: list[RaceDayRace]
