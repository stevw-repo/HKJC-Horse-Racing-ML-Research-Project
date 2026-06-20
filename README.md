# HKJC Horse-Racing ML Research Platform

A **local, single-user research platform** that predicts HKJC **WIN + PLACE** probabilities
for Sha Tin (`ST`) and Happy Valley (`HV`) races, detects value against the live
pari-mutuel odds, sizes stakes with Kelly variants, and **backtests honestly**.

> It is a methodology / research sandbox first, with eventual real-money use — and it
> **recommends only; it never places bets.**

See [`PLAN.md`](PLAN.md) for the authoritative build plan (critique, phased roadmap,
data scope, schema, scraper strategy) and [`CLAUDE.md`](CLAUDE.md) for the working
conventions and current state of this repo.

## Status

| Milestone | Scope | State |
|---|---|---|
| **M0** | Foundations: env, config, logging, storage layout, tooling, CLI | ✅ done |
| **M1** | Incremental scraper + storage | ✅ done — backfill stored (1,697 meetings, 2006–2026) |
| **M2** | Features + baseline conditional-logit + honest backtest | ✅ done |
| **M3** | Model zoo (GBMs + LambdaMART + tabular NNs) + calibration + market blend | ✅ done |
| **M4** | NLP track (English): comments-on-running -> lagged signals + ablation | ✅ done |
| **M5** | Risk / staking sweeps (Kelly variants, caps, rounding, multi-bankroll) | ✅ done |
| **M6** | UI: read-only FastAPI + React/Vite/TS dashboards | ✅ done |
| M7 | Live ops + odds logging | ◻ next |

## Quickstart

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12.

```bash
uv sync                 # create .venv and install pinned deps from uv.lock
uv run hkjc --help      # CLI entry point
uv run hkjc doctor      # resolved config, paths, locked data scope
uv run pytest           # test suite
uv run ruff check . && uv run ruff format .   # lint + format
uv run mypy             # strict type-check
uv run pre-commit install   # one-time: git hooks
```

## Data layer (M1)

Scrape HKJC + open data into a local DuckDB + partitioned-Parquet lake. Every fetch is
recorded in a `_scrape_manifest`, so re-runs are **idempotent** (frozen past pages fetch
zero rows). Politeness: concurrency-capped, rate-limited, retried, cached.

```bash
# Historical results backfill (fixtures calendar enumerates meetings back to ~2006).
uv run hkjc backfill --since 2024-09-01      # bound it; omit --since for full history (~2006)
uv run hkjc scrape --date 2026-06-03         # a single meeting

# Profiles + alternative sources (run after results so ids/dates are known).
uv run hkjc scrape-horses                    # horse bio + form records
uv run hkjc scrape-people                    # jockey + trainer season stats
uv run hkjc scrape-weather --since-year 2006 # HKO daily-climate temperatures
uv run hkjc scrape-trials                    # barrier trials
uv run hkjc scrape-trackwork                 # trackwork (gallop) records
uv run hkjc scrape-sectionals                # per-200m sectional times (#7)
uv run hkjc scrape-text                      # comments-on-running per race (#9, NLP)
uv run hkjc scrape-holidays                  # HK public holidays

uv run hkjc data-health                      # coverage report (meetings/races/rows by season)
```

**What's collected:** results (finish order, SP/win-odds, running positions, finish times,
full WIN→QUARTET dividends), going + rail position, horse profiles (locked bio block +
per-run form), jockey/trainer season stats, HKO daily temperatures, barrier trials,
trackwork, public holidays. See `PLAN.md` §0 for the locked scope (alt sources
3,4,5,7,9,11,14; #6 vet-list and #8 gear-change declarations were dropped as forward-only).
Per-200m **sectionals (#7)** are captured via `hkjc scrape-sectionals` (run it to backfill).

> A full ~20-season backfill is a multi-hour crawl (~1,600 meetings × ~11 pages). It's
> idempotent and cached, so bound the first run with `--since` or run it overnight.

## Features + backtest (M2)

Build the as-of feature store, then run an honest, time-ordered walk-forward backtest of the
PL-strength conditional-logit baseline (WIN softmax + Harville PLACE).

```bash
uv run hkjc features build    # -> features_runner (one as-of row per runner; ~197k rows)
uv run hkjc backtest          # walk-forward, two ROIs, calibration PNG, leakage canary
```

Every feature is computable from `event_time ≤ race_off_time`; the closing-line SP is walled
off from the model and a deterministic **leakage canary** must score ~0. The backtest reports
**two ROIs** (PLAN §1A): a conservative *model-only* (selections from model probability alone)
and an optimistic *market-blended* (positive-EV bets at the SP line). Payouts use the stored
final dividends. The baseline's honest model-only WIN ROI sits around the ~17.5% takeout —
i.e. **no edge beyond the market**, the expected starting point (PLAN §1F); beating it is the
job of M3+.

## Model zoo + leaderboard (M3)

Train the whole zoo (LightGBM, XGBoost, CatBoost, LambdaMART, MLP, FT-Transformer, ensemble)
behind one `ProbabilityModel` interface and rank them on the same honest walk-forward.

```bash
uv run hkjc features build    # rebuild as features v1 (adds the debut-age heuristic)
uv run hkjc train             # walk-forward leaderboard (log-loss / top-1 / ECE / two ROIs)
uv run hkjc train --models catboost,logit --seasons 5   # quick subset over recent seasons
uv run hkjc tune --model catboost --trials 15           # Optuna HPO (min walk-forward log-loss)
```

GBMs use the GPU when present (`HKJC_FORCE_CPU=1` forces CPU; CI is CPU-only); runs are logged
to a local MLflow sqlite store with the feature-store data hash for reproducibility. The
honest takeaway holds across the zoo: **every model loses ≈ the takeout — no edge beyond the
market yet.** A clean finding is that models trained on the *grouped within-race* likelihood
(logit, the NNs) are much better calibrated than the pointwise GBMs, which is what the
calibration layer (temperature/isotonic/Platt) is for.

> Heavy jobs: run them via the venv directly (`.venv/Scripts/hkjc.exe`) — `uv run` re-syncs
> the env on each call and can deadlock on the editable-`hkjc.exe` lock against a running job.

## NLP track (M4)

Scrape English race text and fold it in as a **lagged** feature group (a comment describes a
run, so it is only a feature for the horse's *later* races — PLAN §1C), then ablate it.

```bash
uv run hkjc scrape-text       # comments-on-running per race (corunning, #9)
uv run hkjc features nlp      # encode comments -> lexicon flags + MiniLM anchor-similarities
uv run hkjc features build    # rebuild as v2 (joins + lags the nlp_text group)
uv run hkjc ablate            # walk-forward logit with vs without the NLP group (marginal effect)
```

Signals = spaCy rules/lexicon (interpretable trouble/ran-on/easing/… counts) + MiniLM sentence
embeddings reduced to a few interpretable anchor-similarity scores. The group is **ablatable**
and kept out of the baseline. With a full text backfill the ablation quantifies NLP's marginal
ROI/log-loss; on a small pilot it is ~0 (low lagged coverage), as expected.

## Risk / staking sweeps (M5)

Reuse the walk-forward OOS predictions and sweep **staking methods × bankrolls** to size value
bets honestly: flat, fixed-fraction, full and fractional Kelly (incl. the exact within-race
**correlated/simultaneous** Kelly), under per-race 10% / per-day 25% caps and legal HK$10
rounding, at HK$1k/10k/50k/100k.

```bash
uv run hkjc risk sweep                       # full sweep -> comparison table + CSV/Parquet + ROI PNG
uv run hkjc risk sweep --pools win           # WIN only (default win,place)
uv run hkjc risk sweep --rebate-rate 0.1     # assume a 10% losing-turnover rebate above HK$10k
```

The honest takeaway holds here too: **no staking rule manufactures an edge** — every method
loses ≈ the takeout (best is fractional Kelly λ≈0.05–0.10 at ~−15%; flat/fixed −22% to −30%),
with wide, overlapping CIs. What staking *does* change is structural, and the sweep surfaces the
two headline effects: the **HK$10 granularity** loss (at HK$1,000 flat/fixed can place *no*
legal diversified bets; Kelly loses ~98% of intended stake to rounding, falling to ~23% at
HK$100k) and the **HK$10k rebate threshold** (crossed on 0 days at HK$1k–10k, but 16–17 days at
HK$100k). Pari-mutuel pool dilution is negligible (a HK$10k cap is <0.2% of HKJC's pools) and is
not modelled; HKJC's real rebate schedule is parameterised, not fabricated. Outputs land in
`data/processed/risk/`.

## Dashboards (M6)

A local, **read-only** FastAPI backend + a React/Vite/TS dashboard suite — it surfaces
recommendations and **never places a bet** (there is no write/bet endpoint).

```bash
uv run hkjc serve              # FastAPI backend on http://127.0.0.1:8000 (/api/*)

# In a second terminal (needs Node.js LTS — not a Python dep; install once):
cd ui && npm install           # first run only
npm run dev                    # Vite dev server on http://localhost:5173 (proxies /api -> :8000)
```

Four dashboards, all on real M2–M5 output (race-day on a mocked card, flagged **MOCK** until
the M7 live logger lands): **Data Health** (coverage + meetings/season + recent races),
**Backtest Explorer** (policy ROIs + WIN calibration curve + the M5 staking sweep),
**Experiment Compare** (the model-zoo leaderboard + ROI-vs-takeout bars), and **Race Day**
(value/stake recommendations). The backend reads the DuckDB views live and the persisted
`processed/` snapshots (`run_backtest`/`run_leaderboard`/`run_sweep` write them) — no training
happens in a request. The frontend (`ui/`) builds with `npm run build` (`tsc` + `vite`); the
Python CI stays Python-only.

## Configuration

YAML under [`config/`](config/), loaded and validated via `pydantic-settings`
(`src/hkjc/common/config.py`). `HKJC_`-prefixed env vars override YAML (nested via `__`,
e.g. `HKJC_RISK__BANKROLL=5000`). `config/local.yaml` is a gitignored override layer.

## Layout

```
config/            # YAML: paths, sources, features, risk, backtest, models
src/hkjc/
  common/          # config, logging, keys, time (HKT)
  data/            # scrape · parse · store (DuckDB+Parquet) · weather · holidays · live(M7)
  features/        # as-of feature store · canary · design matrix (M2/M3) · nlp/ lagged text (M4)
  models/          # ProbabilityModel · logit · place (M2) · gbm · nn · ensemble · calibrate · blend (M3)
  backtest/        # walk-forward engine · pari-mutuel sim · metrics · bootstrap · dataset (M2/M3)
  experiments/     # leaderboard · MLflow tracking · Optuna tuning · NLP ablation (M3/M4)
  risk/            # kelly · staking · rebate · simulate · sweep · report (M5)
  api/             # FastAPI: app · routes · schemas · service (M6, read-only)
  cli.py           # Typer entry point (`hkjc`)
ui/                # React + Vite + TS dashboards (M6; node_modules/dist gitignored)
tests/             # pytest suite
fixtures/          # checked-in HTML/JSON for offline parser tests
data/              # gitignored data lake: raw/ processed/ cache/ live_odds/ mlruns/
```
