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
| M3 | Model zoo + calibration + market blend | ◻ next |
| M4 | NLP track (English) | — |
| M5 | Risk / staking sweeps | — |
| M6 | UI (React + FastAPI) | — |
| M7 | Live ops + odds logging | — |

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
uv run hkjc scrape-holidays                  # HK public holidays

uv run hkjc data-health                      # coverage report (meetings/races/rows by season)
```

**What's collected:** results (finish order, SP/win-odds, running positions, finish times,
full WIN→QUARTET dividends), going + rail position, horse profiles (locked bio block +
per-run form), jockey/trainer season stats, HKO daily temperatures, barrier trials,
trackwork, public holidays. See `PLAN.md` §0 for the locked scope (alt sources
3,4,5,7,9,11,14; #6 vet-list and #8 gear-change declarations were dropped as forward-only).
Per-200m **sectionals (#7) are not yet captured** — backlogged for M3.

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
  features/        # as-of feature store + leakage canary (M2); nlp/ is M4
  models/          # ProbabilityModel (PL strength) · logit · place (M2); gbm/nn/blend (M3)
  backtest/        # walk-forward engine · pari-mutuel sim · metrics · bootstrap (M2)
  risk/ experiments/ api/   # M5 / M3 / M6 (skeletons)
  cli.py           # Typer entry point (`hkjc`)
tests/             # pytest suite
fixtures/          # checked-in HTML/JSON for offline parser tests
data/              # gitignored data lake: raw/ processed/ cache/ live_odds/ mlruns/
```
