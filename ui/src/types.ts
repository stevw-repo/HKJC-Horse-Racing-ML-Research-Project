// TypeScript mirrors of the FastAPI Pydantic response models (src/hkjc/api/schemas.py).

export interface Policy {
  name: string;
  n_bets: number;
  staked: number;
  profit: number;
  roi: number;
  roi_lo: number;
  roi_hi: number;
  sharpe: number;
}

export interface CalibrationPoint {
  bin_mid: number;
  pred_mean: number;
  obs_rate: number;
  count: number;
}

export interface Backtest {
  feature_version: string;
  n_oos_races: number;
  n_oos_runners: number;
  test_span: string[];
  win_log_loss: number;
  brier: number;
  top1_hit_rate: number;
  ece: number;
  canary_coef_ratio: number | null;
  canary_roi: number | null;
  policies: Record<string, Policy>;
  calibration: CalibrationPoint[];
}

export interface LeaderboardRow {
  name: string;
  n_oos_races: number;
  win_log_loss: number;
  brier: number;
  top1_hit_rate: number;
  ece: number;
  model_win_roi: number | null;
  model_place_roi: number | null;
  blend_win_roi: number | null;
}

export interface StakingRow {
  bankroll: number;
  policy: string;
  n_bets: number;
  staked: number;
  roi: number;
  roi_lo: number;
  roi_hi: number;
  terminal: number;
  max_dd: number;
  sharpe: number;
  round_loss: number;
  rebate_days: number;
  ruin_prob: number;
  ruined: boolean;
}

export interface RaceSummary {
  race_date: string;
  venue: string;
  race_no: number;
  distance: number | null;
  going: string | null;
  field_size: number;
}

export interface Health {
  meetings: number;
  date_min: string | null;
  date_max: string | null;
  races_rows: number;
  results_rows: number;
  dividends_rows: number;
  horses_rows: number;
  horse_form_rows: number;
  people_rows: number;
  weather_rows: number;
  public_holidays_rows: number;
  barrier_trials_rows: number;
  trackwork_rows: number;
  sectionals_rows: number;
  comments_on_running_rows: number;
  manifest_urls: number;
  seasons: Record<string, number>;
}

export interface RaceDayRunner {
  saddle: number;
  horse: string;
  win_prob: number;
  place_prob: number;
  win_odds: number;
  ev: number;
  stake: number;
}

export interface RaceDayRace {
  race_no: number;
  distance: number;
  going: string;
  runners: RaceDayRunner[];
}

export interface RaceDay {
  mock: boolean;
  meeting_date: string;
  venue: string;
  note: string;
  races: RaceDayRace[];
}
