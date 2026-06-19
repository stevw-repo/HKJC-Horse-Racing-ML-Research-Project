# CLAUDE.md — working guide for this repo

Context for AI assistants (and humans) working on the HKJC horse-racing ML research
platform. Read this plus **[`PLAN.md`](PLAN.md)** (the authoritative build plan) before
making changes. `prompt.md` is the original brief that produced the plan.

## What this is

A **local, single-user** research platform that predicts HKJC **WIN + PLACE**
probabilities for Sha Tin (`ST`) and Happy Valley (`HV`), detects value vs the live
pari-mutuel odds, sizes stakes with Kelly variants, and **backtests honestly**.

> **Hard invariant: it recommends only — it NEVER places bets.** No code path may submit
> a wager to HKJC or anywhere else.

## Golden rules (domain invariants — do not violate)

- **Never place bets.** Output recommendations only.
- **Pari-mutuel, not fixed odds.** WIN/PLACE are pools; displayed odds move until close;
  the final dividend isn't known at bet time. EV is net of **~17.5% takeout** (rebate 0
  at HK$1,000). Model the pool, not fixed odds.
- **Market wall.** SP / win-odds on the results page is the **closing line = market data**.
  Never use it as a fundamental pre-race feature (leakage). `features.groups.market.wall`
  marks this; keep market features separate from fundamentals.
- **Lagged text only.** Stewards' reports / comments-on-running describe the race that just
  happened → only valid as features for a horse's **prior** runs (`text_event_time <
  race_off_time`).
- **As-of features only.** Every feature must be computable from `event_time ≤
  race_off_time`. A leakage canary (M2) must score ≈0 importance and ≈0 backtest ROI.
- **Honest backtest.** Report **two ROIs**: conservative model-only and upper-bound
  market-blended (final-dividend honesty caveat, PLAN.md §1A).
- **Pedigree is HKJC on-site only.** No external pedigree DBs. The external WBRR list is
  not used (but the per-runner `internationalRating` field is captured).
- All times are **HKT** (UTC+8). Use `hkjc.common.time` (needs `tzdata` on Windows).

## Commands

```bash
uv sync                      # install pinned deps + editable package (creates .venv)
uv run hkjc --help           # CLI
uv run hkjc doctor           # show resolved config, paths, locked data scope
uv run pytest                # tests
uv run ruff check . && uv run ruff format .   # lint + format
uv run mypy                  # strict type-check
uv run pre-commit install    # one-time: enable git hooks
```

CI (`.github/workflows/ci.yml`) runs ruff + ruff-format-check + mypy + pytest on
ubuntu-latest with `uv sync --frozen`. Keep `uv.lock` committed and in sync.

## Layout

```
config/            YAML config (one top-level key per file): paths, sources, features, risk, backtest, models
src/hkjc/
  common/          config (pydantic-settings), logging (structlog), keys, time (HKT)
  data/            scrape · parse · store (DuckDB+Parquet) · live (GraphQL, M7) · weather (HKO CSV)
  features/        as-of feature store (M2) + nlp/ (English, M4)
  models/          ProbabilityModel zoo: logit (M2); gbm/nn/ensemble/calibrate/place/blend (M3)
  risk/            Kelly variants, caps, legal rounding (M5)
  backtest/        walk-forward engine, pari-mutuel sim, metrics, bootstrap (M2)
  experiments/     MLflow(local) + Optuna + leaderboard (M3)
  api/             FastAPI (M6)
  cli.py           Typer entry point (`hkjc`)
tests/             pytest suite
fixtures/          checked-in sample HTML/JSON for offline parser tests (M1+)
data/              gitignored data lake: raw/ processed/ cache/ live_odds/ mlruns/
```

Note: `models/base.py` (the `ProbabilityModel` interface emitting a per-runner
**Plackett–Luce strength vector**), `models/logit/`, `models/place/` (M2) plus `models/gbm/`,
`models/nn/`, `models/ensemble/`, `models/calibrate/`, `models/blend/`, `experiments/`
(M3) and `features/nlp/` (M4) all exist. `risk/` is M5; `api/` is M6.

## Configuration

`src/hkjc/common/config.py` loads every `config/*.yaml`, merges them, and validates into
`AppConfig` (pydantic-settings). Precedence: **explicit kwargs > `HKJC_` env vars
(nested via `__`) > merged YAML**. Example override: `HKJC_RISK__BANKROLL=5000`.
`config/local.yaml` is a gitignored override layer. Use `get_config()` (cached).

Config sub-models use `extra="forbid"` — adding a YAML key requires adding the field to
the model (and vice versa). Tests in `tests/test_config.py` assert the **locked data
scope** (PLAN.md §0); update them if the scope intentionally changes.

## Conventions

- Python 3.12, `from __future__ import annotations` at the top of every module.
- **mypy strict** must pass; **ruff** (line length 100) lints and formats.
- Keep **user-facing CLI strings ASCII** (Windows console can mangle em/en-dashes; ruff
  flags ambiguous dashes via RUF00x). Docstrings/comments may use em-dashes (`—`).
- Add new runtime deps at the milestone that first needs them (don't front-load the whole
  stack); pin via `uv add` so `uv.lock` updates. Dev tools live in the `dev` group.
- Tests live in `tests/`, named `test_*.py`, fully typed (`-> None`). Property tests
  (hypothesis) are reserved for the backtest/Kelly/dividend math (M2+).

## M1 scraper (in progress)

The **results** pipeline is built and pilot-validated. Flow: `data/pipeline.py`
orchestrates `scrape/client.py` (async fetcher: cache + retry + rate-limit + concurrency)
→ `parse/results.py` (per-race page is the source of truth — only it carries Win Odds,
finish time, running positions, and horse/jockey/trainer ids; the meeting `resultsall`
page is abridged, used only for venue + race-count enumeration) → `store/writer.py`
(partitioned raw Parquet + DuckDB views) + `store/manifest.py` (`_scrape_manifest` for
idempotency). Parsed shapes live in `data/models.py`.

**Horse profiles** are also built (`parse/profiles.py` → `store.write_horse_profile`):
the locked bio block (PLAN.md §0) + the form-records table (per-run rating/gear/going/draw,
with a `RaceIndex` that joins to `races`). Form records are how we reconstruct historical
pre-race fundamentals, because **race cards are forward-only** — HKJC removes a meeting's
card once results publish, so cards cannot be backfilled (capture upcoming cards going
forward, for race-day/M7).

**Retired-horse bio:** a retired horse's profile drops the combined `Country of Origin / Age`
row for a standalone `Country of Origin` (no current age) and shows `Last Rating` instead of
`Current Rating` — the parser handles both, so `country_of_origin`/`last_rating` are
recovered for old horses (only `age` is genuinely absent). HKJC publishes **no foaling date**,
so **age-at-race is derived in M2** as `race_year - birth_year`, where `birth_year =
scrape_year - age` (calendar convention, stored at scrape time). `birth_year`/`age` are null
for horses retired before they were ever scraped — age-at-race is then unavailable for that
deep-history era (acceptable; it isn't the live-betting era).

**Jockey/trainer profiles** are built too (`parse_person_profile` → `people` view): one
parser for both roles capturing current-season strike-rate inputs (wins/2-3-4ths, rides or
runners, win %, stakes).

**HKO weather** (`data/weather/hko.py` → `weather` view): daily mean/max/min temperature
per station (full history is one request each), mapped HKO→HV and SHA→ST. Only temperature
is exposed per-station (no humidity/rainfall). Note HKO's **current-month lag** — daily
climate publishes after the month completes, so very recent meetings have no weather yet.

CLI: `hkjc scrape --date YYYY-MM-DD [--force]`, `hkjc backfill [--limit N] [--since ...]`,
`hkjc scrape-horses [--limit N]`, `hkjc scrape-people [--limit N]`,
`hkjc scrape-weather [--since-year Y]`, `hkjc scrape-holidays`,
`hkjc scrape-trials [--limit N]`, `hkjc scrape-trackwork [--limit N]`,
`hkjc scrape-sectionals [--limit N]` (#7), `hkjc scrape-text [--limit N]` (#9), `hkjc
data-health`.
**Public holidays** (#14) are ingested from gov.hk open data (`data/holidays.py` →
`public_holidays` view); the feed is served with a BOM and spans only ~current +/- 1 year.
**Barrier trials** (#4, `parse/trials.py` → `barrier_trials` view): per-batch runs from
`btresult?date=YYYY/MM/DD` (param is `date`, not `racedate`); trials run at ST/HV/Conghua
and do **not** link jockey/trainer ids, so those are stored as names.
**Trackwork** (#5, `data/trackwork.py` → `trackwork` view): a paginated JSON endpoint
(`/racing/information/json/TrackworkOneDayRecords/<YYYYMMDD>1E.aspx?PageNum=N`) discovered
via the browser — the `trackworksearch` page is JS-driven, so the data isn't in its HTML.
Records carry names, not ids.
Meeting URLs (`resultsall?racedate=`) discover venue + race count without a `Racecourse`
param; per-race URLs are `localresults?...&RaceNo=N`. Re-running a **frozen** (past) meeting
recorded in the manifest fetches 0 rows; horse profiles are **mutable** and refetched.

**Historical enumeration (RESOLVED 2026-06-15):** the results `selectId` dropdown only lists
~2 recent seasons, but the **fixtures calendar** covers the full history —
`fixture?calyear=Y&calmonth=M` (`parse/fixtures.py`, `<td class="calendar">` day cells) lists
meeting days back to ~2006. `pipeline.list_fixture_dates` drives `hkjc backfill` (default
start ~2006-09; `--since` to bound). Caveat: on old result pages HKJC only hyperlinks
**currently-active** jockeys/trainers, so `jockey_code`/`trainer_code` are partial for old
meetings — the always-present `jockey_name`/`trainer_name` text is captured alongside.

**Going/rail** (#3): `going` and `rail` are first-class on the `races` view — going comes
straight from the results meta, and `rail` is parsed from the course token (`TURF - "C"
Course` → `C`; AWT has none). HKJC publishes no going-stick / separate rail page, so this
is the complete #3 signal from results.

**Scope change 2026-06-15:** #6 vet-list and #8 gear-change declarations were **dropped**
(forward-only / tied to declared starters; not historically backfillable). Per-run gear is
still captured via horse-form. Race cards are **kept** (needed for M7 race-day predictions).

**Remaining M1:** the historical data layer is complete (results, profiles, weather,
holidays, trials, trackwork, going/rail). Left: **forward race-card capture** (with M7
live-ops, when a meeting is upcoming) and the **full backfill** + a richer coverage/gap
report. The enabled alternative sources are now: going/rail #3, barrier trials #4,
trackwork #5, sectional archive #7, racing news #9 (M4 NLP), pedigree #11, holidays #14.

## Milestone status

M0, **M1 (scraper + storage), M2 (features + baseline + honest backtest), M3 (model zoo +
calibration + blend), and M4 (English NLP track) are implementation-complete.** M1: every
locked source has a parser + DuckDB view + offline fixture test, idempotency is proven,
enumeration reaches ~2006, and the full backfill is **stored** (1,697 meetings, 2006-09 ->
2026-06; `hkjc data-health` reports coverage). M2: the as-of feature store + leakage canary,
the PL-strength conditional-logit baseline + Harville PLACE, and an honest walk-forward
backtest. M3: a GPU model zoo (GBMs + LambdaMART + tabular NNs + ensemble) behind one
`ProbabilityModel`, calibration + market-blend, MLflow(sqlite)+Optuna, and a reproducible
leaderboard. M4: comments-on-running + report text capture, spaCy-rules/lexicon + MiniLM
embedding signals, a **lagged** `nlp_text` feature group, and an ablation harness. All green
under ruff/mypy/pytest. Each milestone's exit criterion is in PLAN.md §2 — treat it as the
definition of done. Forward race-card capture is parked with M7.

## M2 (features + baseline + honest backtest) — implementation-complete

Pipeline: `features/build.py` -> `features_runner` (processed Parquet + DuckDB view) ->
`backtest/engine.py` (walk-forward). CLI: `hkjc features build`, `hkjc backtest [--l2 ...]
[--market-weight ...] [--ev ...] [--no-plot]`.

- **As-of feature store** (`features/build.py`; column contracts/roles in `features/base.py`):
  one row per runner per race (196,947 rows x 56 cols, 2006-2026), every feature from
  `event_time <= race_off_time`. Reads raw Parquet via **polars column-projection** (~4s; the
  DuckDB `union_by_name` glob over ~9.8k per-horse files took 7 min). Use
  `missing_columns="insert"` for results' schema drift (older files predate
  `jockey_name`/`trainer_name`). Feature groups: career-stage
  (`career_run_number`/`days_since_debut`/`days_since_last_run`/`seasons_active`), prior form
  (win/place rate, last-3 finish & lbw, dist/going match-rate), as-of official rating + trend
  (from `horse_form`, joined on `race_index`), rolling jockey/trainer strike rates,
  bio/pedigree, weather, public holiday, trial recency, and a finish-time **speed proxy**.
- **Leakage discipline:** strictly-prior cumulative aggregates (`cum_* - current`), per-horse
  rolling on a date-sorted frame, connection rates joined back by a stable `_row` id. The
  **market wall** keeps SP (`win_odds`, `market_prob`) tagged `role="market"` and out of
  `BASELINE_FEATURES`. A deterministic noise **canary** rides through fit + backtest.
- **Connections key (important):** old result pages carry the jockey/trainer **name** (no
  link), recent pages the **code** (no name) — complementary (~100% via coalesce). A
  name->code map (from rows that have both) **canonicalizes identity** so a long career isn't
  split at the era boundary. Rates are computed **as-of from prior `results`**, never from the
  `people` snapshot (current-season = leakage).
- **Age (as decided 2026-06-15, now implemented):** career-stage features are the backbone
  (~95%+ coverage); `age_at_race` is `race_year - birth_year` where known (~9%) else null +
  an `age_imputed` flag. The debut-age heuristic is **not yet** built (a clean M3 add).
- **Baseline** (`models/base.py` = `ProbabilityModel` -> PL strength vector; `models/logit/`):
  Benter-style within-race conditional logit, convex MLE via `scipy` L-BFGS (analytic
  gradient), median-impute + standardize, small L2. WIN = within-race softmax;
  **PLACE = Harville** (`models/place/`, top-k closed forms; verified exact vs brute-force
  enumeration; identity `sum_i P(top-k)=k`).
- **Backtest** (`backtest/`): `pari_mutuel.py` dividend math (property-tested: place-count by
  field size, dead-heat split, HK$10 rounding, pool-impact dilution), `walk_forward.py`
  per-season expanding splits, `metrics.py` (log-loss/Brier/top-1/calibration bins),
  `bootstrap.py` ROI CIs (resampling races), matplotlib calibration PNG. **Payouts use the
  stored final dividends** (the honest pool truth, dead-heat already encoded).
- **Two ROIs (PLAN §1A):** *model-only* = back the model's top WIN/PLACE pick (no odds in the
  decision; conservative). *market-blended* = bet positive-EV runners at the SP line using a
  model+market blend (uses the line; optimistic upper bound). Two lenses, not strict
  lower/upper bounds of one strategy (the conservative rule can't use a price at all, so it is
  necessarily a different selection).
- **Honest result (15,083 OOS races, seasons 2007-08..2025-26, feature_version v0):**
  model-only WIN ROI **-17.5%** [-20.4, -14.3] — ~the takeout, i.e. **no edge beyond the
  market** (the expected baseline outcome, PLAN §1F); top-1 hit 24% vs 8% base = real ranking
  signal. **Leakage canary clean**: fitted weight 0.011 of mean |coef|; random-pick sentinel
  ROI -19% (no edge). Calibration PNG at `data/processed/backtest/calibration_win.png`.
- **Sectionals (#7) — now captured (M3 follow-up):** `parse/sectionals.py` parses the
  `displaysectionaltime` page (per runner per section: position, margin, section time, and the
  per-200m split; section times sum to the final time). `sectionals` DuckDB view + meeting
  partitions; `hkjc scrape-sectionals [--limit N] [--since ...]` (idempotent, frozen-skip;
  URL date is `DD/MM/YYYY`, no `Racecourse` param). Offline fixture test. **Backfill pending**
  (user runs `hkjc scrape-sectionals` like the M1 backfill); once stored, the M2 speed-figure
  proxy can switch to real splits (a clean feature-store add).

## M3 (model zoo + calibration + blend) — implementation-complete

Pipeline: `experiments/leaderboard.py` runs every model through `experiments/runner.py`
(the generic walk-forward, reusing `backtest/`), logs to MLflow, and ranks them. CLI:
`hkjc train [--models ...] [--seasons N] [--nn-epochs N]`, `hkjc tune [--model ...]`.

- **Model zoo** (all behind `ProbabilityModel` -> PL strength -> within-race softmax):
  `models/gbm/` LightGBM, XGBoost, CatBoost, LightGBM-**LambdaMART** (lambdarank, race=group);
  `models/nn/` **MLP** + **FT-Transformer** in PyTorch trained with the *grouped within-race
  conditional-logit (Plackett-Luce) NLL* (same likelihood as the logit), GPU + CPU-fallback,
  minibatched by race, early-stopped; `models/ensemble/` averages member within-race probs.
- **GPU (RTX 4060, cu124):** XGBoost/CatBoost use `device="cuda"`/`task_type="GPU"`; torch on
  CUDA. `models/device.gpu_available()` is the single switch (`HKJC_FORCE_CPU=1` forces CPU;
  CI is CPU-only). **Run via the venv directly** (`.venv/Scripts/python.exe`,
  `.venv/Scripts/hkjc.exe`) for heavy jobs — `uv run` re-syncs/rebuilds each call and the
  editable-`hkjc.exe` lock deadlocks against a running job.
- **Design matrix** (`features/design.py`): numeric set (= `BASELINE_FEATURES`, for logit/NNs)
  + integer-encoded categoricals (sire/dam/dam's-sire/import-type/sex/country/venue) appended
  for the GBMs. **CatBoost** takes them natively; **XGBoost** treats them as ordinal numerics
  (its native categoricals reject categories unseen in a training fold — common in
  walk-forward; CatBoost/LightGBM tolerate unseen). Plus the **debut-age heuristic** ->
  `age_at_race` now ~100% covered; `feature_version` bumped to **v1** (rebuild required).
- **Calibration** (`models/calibrate/`): temperature (within-race; primary secondary layer),
  isotonic, Platt. **Blend** (`models/blend/`): tunable model+market weight, renormalized.
- **Experiments** (`experiments/`): MLflow local **sqlite** backend (the file store is rejected
  by MLflow 3.x) under `data/mlruns`, logging params/metrics + the feature-store **data hash**;
  Optuna HPO (`tuning.py`, time-boxed, minimizes walk-forward log-loss).
- **Leaderboard result (full walk-forward, feature_version v1, ranked by model-only WIN ROI):**
  xgboost -17.1%, logit -17.3%, mlp/ft_transformer -19.1%, ensemble -19.5%, catboost -20.4%,
  lightgbm -20.7%, lambdamart -22.0%. **Every model loses ~the takeout — still no edge beyond
  the market** (PLAN §1F holds across the zoo). Key finding: models trained on the *grouped*
  within-race likelihood (logit, MLP, FT-Transformer) are far better **calibrated** (ECE
  ~0.002, top-1 ~0.24) than the pointwise GBMs (ECE ~0.05, top-1 ~0.15) — the GBMs optimize
  binary log-loss, so their raw-margin softmax is over-confident and *needs* the calibration
  layer / a grouped objective. Reproducible from MLflow (config + data hash).

## M4 (English NLP track) — implementation-complete

Pipeline: `data/parse/text.py` (scrape) -> `features/nlp/` (encode) -> lagged `nlp_text`
feature group in `features/build.py` -> `experiments/ablation.py`. CLI: `hkjc scrape-text`,
`hkjc features nlp`, `hkjc ablate`.

- **Text capture (#9):** `corunning` (Comments on Running -- a clean **per-runner** table:
  Placing/HorseNo/Horse/Jockey/Gear/Comment, horse_id in the anchor) -> `comments_on_running`
  view. **URL is `corunning?Date=YYYYMMDD&RaceNo=N` (no `Racecourse` param).** Critical gotcha:
  the page **silently ignores `racedate=YYYY/MM/DD`** (and every other param form) and returns
  the *latest* meeting -- so the first cut stored the same latest-meeting comments for every
  date. Always validate a per-date page by checking its data actually differs across dates (an
  old date with no comments correctly returns empty with the right param). `hkjc scrape-text
  [--limit N] [--since ...]` (per-race; idempotent, frozen). Offline fixture test. The
  meeting-level prose reports (`racereportfull`/`veterinaryrecord`/`exceptionalfactors`) were
  **dropped** -- those endpoints don't reliably honour the date (vet/exceptional return the
  latest *available* record), so they'd store garbage; a browser-recon follow-up could recover
  them later (like trackwork). NOTE sectionals' `displaysectionaltime?racedate=DD/MM/YYYY`
  **does** honour the date (verified) -- only the text endpoints are quirky.
- **Lagged discipline (PLAN §1C):** a comment describes the run it belongs to, so each NLP
  signal is **shifted one run forward per horse** in `_add_nlp` -- the value seen for a target
  race is the horse's *previous* comment (`text_event_time < race_off_time`). The leakage
  canary still rides through.
- **NLP signals (`features/nlp/`):** spaCy **blank-pipeline `PhraseMatcher`** over a curated
  `lexicon.py` -> interpretable counts (trouble / slow_start / ran_on / easing / weakened /
  wide / health); **MiniLM** (`all-MiniLM-L6-v2`, GPU/CPU) sentence embeddings reduced to a few
  **anchor similarities** (closeness to "troubled run" / "won easing" / "no excuse") so the
  384-dim vector becomes ablatable features. Cached to `processed/nlp_comment_features`
  (`build_comment_features`; the embedding pass is the cost). No spaCy model download needed
  (blank pipeline); MiniLM auto-downloads (~80MB) on first use.
- **Ablatable group:** `NLP_FEATURES` is kept out of `BASELINE_FEATURES`; `numeric_design_
  features(include_nlp)` + `load_model_data(include_nlp=...)` toggle it. `feature_version` ->
  **v2**.
- **Ablation (exit criterion):** `hkjc ablate` walk-forwards the logit with vs without the
  group and reports the delta. **Current result: ~0** (delta 0.0000 over recent seasons) --
  but only because the **text is a 39-meeting pilot** (~1% lagged coverage). The pipeline is
  complete + leakage-safe; the **definitive ablation needs the full text backfill** (`hkjc
  scrape-text`, multi-hour, then `features nlp` + `features build` + `ablate`).

## Next: M5 (risk / staking sweeps)

flat / fixed-fraction / full + fractional Kelly grid {0.05..0.5}, correlated/simultaneous
Kelly within & across races; per-race 10% / per-day 25% caps; legal HK$10 rounding;
multi-bankroll (1k/10k/50k/100k) sims; EV edge >=5% net takeout (PLAN.md §2 M5, config in
`config/risk.yaml`). Build in `risk/`; reuse the `backtest/` pari-mutuel sim. Still-open data
adds: **backfill sectionals + text** (then switch the speed-figure proxy to real splits and
get a real NLP ablation), GBM calibration/grouped-objective tuning.
