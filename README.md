# HKJC Horse-Racing ML Research Platform

A **local, single-user research platform** that predicts HKJC **WIN + PLACE** probabilities
for Sha Tin (`ST`) and Happy Valley (`HV`) races, detects value against the live
pari-mutuel odds, sizes stakes with Kelly variants, and **backtests honestly**.

> It is a methodology / research sandbox first, with eventual real-money use — and it
> **recommends only; it never places bets.**

See [`PLAN.md`](PLAN.md) for the authoritative build plan (critique, phased roadmap,
data scope, schema, and scraper strategy) and [`CLAUDE.md`](CLAUDE.md) for the working
conventions of this repo.

## Status

| Milestone | Scope | State |
|---|---|---|
| **M0** | Foundations: env, config, logging, storage layout, tooling, CLI skeleton | ✅ in place |
| M1 | Incremental scraper + storage | ⏳ next |
| M2 | Features + baseline + honest backtest | — |
| M3 | Model zoo + calibration + market blend | — |
| M4 | NLP track (English) | — |
| M5 | Risk / staking sweeps | — |
| M6 | UI (React + FastAPI) | — |
| M7 | Live ops + odds logging | — |

## Quickstart

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12.

```bash
uv sync                 # create .venv and install pinned deps from uv.lock
uv run hkjc --help      # CLI entry point
uv run hkjc doctor      # show resolved config, paths, and the locked data scope
uv run pytest           # run the test suite
uv run ruff check .     # lint
uv run ruff format .    # format
uv run mypy             # type-check
```

Optionally install git hooks: `uv run pre-commit install`.

## Configuration

Config is YAML under [`config/`](config/), loaded and validated via `pydantic-settings`
(`src/hkjc/common/config.py`). Environment variables prefixed `HKJC_` override YAML
(nested via `__`, e.g. `HKJC_RISK__BANKROLL=5000`). The data scope is **locked** per
PLAN.md §0.

## Layout

```
config/            # YAML: paths, sources, features, risk, backtest, models
src/hkjc/          # the package (common, data, features, models, risk, backtest, experiments, api)
tests/             # pytest suite
fixtures/          # checked-in sample HTML/JSON for offline parser regression tests (M1+)
data/              # gitignored local data lake: raw/ processed/ cache/ live_odds/ mlruns/
```
