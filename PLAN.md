# HKJC Horse-Racing ML Research Platform — Build Plan

## Context

Greenfield project (repo currently holds only a standard Python `.gitignore`, an empty README, and `prompt.md`). The goal is a **local, single-user research platform** that predicts HKJC **WIN + PLACE** probabilities for Sha Tin (`ST`) and Happy Valley (`HV`) races, detects value vs live pari-mutuel odds, sizes stakes with Kelly variants, and **backtests honestly**. It is a methodology/research sandbox first, with eventual real-money use — and it **recommends only; it never places bets.**

This file is the plan deliverable requested by `prompt.md` (critique → roadmap → repo/stack → schema/scraper → "what I need from you" → open questions), updated with four confirmed decisions: **modern-markup backfill only**, **English-only NLP**, **recommended staking grid**, **live-odds logger built at M7 (standard order)**.

### Grounding I verified live (I have internet access)
| Source | Result |
|---|---|
| Fixtures page (`.../local/information/fixture`) | ✅ Server-rendered, real meeting dates → date enumeration works |
| Results page (`.../localresults?...`) | ✅ Server-rendered HTML; all fields present incl. full dividend table (WIN→QUARTET), sectionals, SP |
| HKO weather | ✅ **Open-data CSV API** (`data.weather.gov.hk/weatherAPI/opendata/...`) returns clean machine-readable daily climate — use this, not the JS `dailyExtract.htm` |
| GraphQL odds API (`info.cld.hkjc.com/graphql/base/`) | ✅ Reachable, auth-free, valid schema. Returns 0 meetings for *past* dates → live service only serves current/upcoming meetings; full validation needs a race day (next: **ST, 21 Jun 2026**) |
| Race card for a date 1 week out | "No information" → cards publish only a few days ahead (confirm exact lead time in M1) |
| NLP text URL guess | 404 → exact base paths for text pages must be confirmed at build time (implementation detail, not a blocker) |

**Consequence:** I do **not** need you to save sample HTML pages for me to write parsers offline. I will fetch live pages and check a few in as test fixtures myself during M1.

---

## 0. Finalized data scope (LOCKED 2026-06-14)

This section is authoritative and supersedes any ambiguity below. The data list was reviewed and approved; the two pre-build acknowledgements (§5) are **accepted**. Machine-readable home: `config/sources.yaml` (`alternative_sources`) + `config/features.yaml` (`groups`, `horse_bio`).

**Core HKJC race data (always on):** meetings, races, runners, results, full dividend table (all pools → exotics-ready), sectionals, horse/jockey/trainer profiles, plus the per-runner `internationalRating` field (free in card/results; kept as a class proxy for imports).

**Horse bio — full set (from horse profile):** Country of Origin, **Age-at-race-date** (derived, not just current age), Colour, Sex, Import Type, Sire / Dam / Dam's-sire, Owner, start-of-season & current official rating.

**Alternative sources — ENABLED (user picks 3,4,5,6,7,8,9,11,14):**

| # | Source | Primary use | Time-gating |
|---|---|---|---|
| 3 | Going / track-condition + **rail position** | rail position → first-class draw-bias feature | pre-race |
| 4 | Barrier-trial results | trial form (first-starters / returnees) | pre-race |
| 5 | Trackwork / gallop reports | work intent (rolling archive; M1 reports earliest date) | pre-race |
| 6 | Veterinary / vet-list | soundness & fitness flags | **pre-race vet-list = feature; post-race record = lagged only** |
| 7 | Sectional-time archive | home-grown speed figures | pre-race (of prior runs) |
| 8 | Gear-change declarations | first-time-gear (blinkers/visor) flags | pre-race |
| 9 | Racing news / press releases | English NLP signals, ablatable (M4) | lagged (`text_event_time < race_off_time`) |
| 11 | Pedigree | sire / dam / dam's-sire aptitude — **HKJC on-site only, no external DBs** | static |
| 14 | HK public-holiday / festival calendar | crowd / pool-size, day-vs-night context (gov.hk open data) | pre-race |

**DROPPED / not selected:** #2 rainfall nowcast (no clean historical archive → not a training feature; dropped per user), #1 real-time weather stations, #10 exotic-pool odds (v1), #12 external WBRR ratings (per-runner field still captured), #13 overseas prior form, #15 AQHI/tide.

**Weather:** historical = HKO daily-climate CSV (verified). No race-day microclimate source in v1.

---

## 1. Critique & better approaches

The plan in `prompt.md` is unusually sound — pari-mutuel mechanics, takeout, leakage, calibration, and honesty are all already correct. The high-value refinements:

**A. The deepest honesty trap is decision-time information, not payout.** In pari-mutuel, *everyone is paid the final dividend regardless of when they bet* — so backtesting **payouts** against final dividends is correct. The lookahead risk is in the **decision**: if the backtester decides "this is value" using the *final* dividend-implied probability, it is peeking at the closing line (the most efficient estimate available). Because no historical *provisional* odds exist, an honest backtest cannot perfectly reproduce the bet-time decision. Mitigations baked into the plan:
   - Report **two numbers** per config: a **conservative model-only ROI** (decisions use model probability alone, never final odds) and an **upper-bound blended ROI** (decisions use final-dividend-implied market as a proxy for the closing line). The truth lives between them.
   - You chose to build the live-odds logger at M7 (standard order). **Implication to accept:** provisional-odds time series only begin accumulating from M7 onward, so the market-blend's *historical* evaluation stays a proxy until enough live data is collected. (Reversible — we can pull the logger forward later; the next uncapturable meeting is 21 Jun 2026.)

**B. SP / "win odds" on the results page is the market closing line — the thing you are trying to beat.** It must be modeled as **market data**, never used as a fundamental pre-race feature for the model (that would be circular/leakage). Keep a hard wall between "fundamental features" and "market features."

**C. Post-race text can only be used lagged.** Stewards' reports and comments-on-running describe *the race that just happened*, so they can **never** be features for that same race. They are only valid as features describing a horse's **prior** runs (e.g., "blocked/unlucky last start" → positive regression signal). Vet records / exceptional factors *may* be pre-race. The NLP pipeline will enforce `text_event_time < race_off_time`.

**D. Make a Plackett–Luce strength vector the canonical model output.** Every model emits one latent **strength `sᵢ` per runner**; everything else is derived:
   - WIN = softmax over strengths (normalized within race — this is exactly Benter-style conditional logit).
   - PLACE = Harville recursion over strengths, plus Henery / Lo–Bacon–Shek corrections and a directly-modeled PLACE head, compared.
   - Exotics (v2+) = ordering probabilities via Monte-Carlo over the same PL strengths.
   This single abstraction satisfies the "future-proofing" requirement **and** gives v1 a clean, comparable interface. Note **Harville's known bias** (overstates favorites placing) → that's why we compare corrections + direct modeling.

**E. Learning-to-rank (LambdaMART) needs a calibration + normalization layer.** It optimizes ranking (NDCG), not calibrated probabilities. Convert scores → within-race softmax → isotonic. **Within-race normalization is the primary calibration step; pooled isotonic/Platt is secondary.**

**F. The HK$1,000 + HK$10-unit granularity is a headline result, not a footnote.** 1 unit = 1% of bankroll, so Kelly stakes on small edges round to 0 or 1 unit, forcing over-/under-betting. The backtester will run the **same configs at HK$1k / 10k / 50k / 100k** to quantify how granularity (and the HK$10k-loss rebate threshold, which never triggers at HK$1k) changes the picture. Realistic prior: most configs show **negative ROI after ~17.5% takeout** (you must beat the closing line by ~1/0.825 ≈ 21% just to break even). **Success = a bootstrap-significant edge in specific segments, or an honest "no edge" conclusion** — not guaranteed profit. This framing is stated up front in the UI and reports.

**G. Tabular NNs are research comparators, not the workhorse.** At tens of thousands of races, gradient boosting (esp. CatBoost for native categorical horse/jockey/trainer IDs, LightGBM for LambdaMART) will almost certainly win. We'll include TabNet/FT-Transformer for completeness but won't over-invest.

**H. Leakage gets an architectural control, not just discipline.** An **as-of feature store** where every feature is provably computed from `event_time ≤ race_off_time`, plus an automated **leakage canary** (shuffle-label / future-data sentinel feature that must score ~0 importance and ~0 backtest ROI). This is the single highest-ROI piece of test infrastructure.

**I. Place-dividend edge cases break backtests.** Must-handle: place-count by field size (3 places @ 7+ runners, 2 @ 5–6, none @ <5 / WALKOVER), dead-heat dividend splitting, scratchings/refunds and reduced pools, reserves. These get property-based tests.

---

## 2. Phased roadmap

Each milestone has an explicit **exit criterion** (= its verification).

| M | Milestone | Scope | Exit criterion |
|---|---|---|---|
| **M0** | Foundations | uv env + pinned lockfile; repo scaffold; pydantic-settings + YAML config; structured logging; DuckDB/Parquet layout; pytest/ruff/mypy + pre-commit; Typer CLI skeleton | `uv run hkjc --help` works; empty test suite green; CI lints clean |
| **M1** | Scraper + storage | Idempotent incremental scraper (httpx+selectolax) for **modern-markup** results, fixtures, race cards, profiles, sectionals, draw/trial/trackwork; HKO climate CSV ingest; raw Parquet + processed DuckDB; `_scrape_manifest`; data-health/coverage report. Backfill all current-format seasons; report earliest reliable date. Legacy-season parser → backlog. | Full backfill stored; re-running the crawl fetches **0** new rows (idempotency proven); coverage report shows no unexplained gaps; parser tests pass on checked-in HTML fixtures |
| **M2** | Features + baseline + honest backtest | As-of feature store + leakage canary; PL-strength **conditional-logit** baseline; Harville PLACE; walk-forward backtest engine w/ takeout, HK$10 rounding, pool-impact, place-count/dead-heat logic; bootstrap CIs | End-to-end walk-forward backtest of baseline producing honest ROI/Sharpe + calibration plots + CIs; **leakage canary scores ≈0** |
| **M3** | Model zoo + calibration + blend | LightGBM/XGBoost(GPU)/CatBoost + LambdaMART, tabular NNs, ensembles behind one `ProbabilityModel` interface; isotonic/Platt/temperature calibration; market-blend stage (tunable weight); MLflow (local) + Optuna | Leaderboard ranking all models on walk-forward ROI/Sharpe/log-loss/calibration; reproducible from logged config + data hash |
| **M4** | NLP track (English) | Scrape/parse English stewards' reports, comments-on-running, vet, incidents, exceptional factors → **lagged** structured signals (trouble/unlucky flags, vet concerns, "won easing") via spaCy rules + lexicon + sentence-transformer embeddings; ablatable feature group | Ablation report quantifying NLP's marginal contribution to ROI/log-loss |
| **M5** | Risk / staking sweeps | flat, fixed-fraction, full + fractional Kelly grid {0.05,0.1,0.15,0.25,0.5}, **correlated/simultaneous Kelly** within & across races; per-race 10% / per-day 25% caps; legal rounding; **multi-bankroll (1k/10k/50k/100k)** sims; EV edge ≥5% net takeout | Staking comparison report incl. the granularity & rebate-threshold findings |
| **M6** | UI (React + FastAPI) | FastAPI exposing data/predictions/value-staking/backtest; React+Vite+TS dashboards: race-day, experiment-compare, backtest-explorer, data/scraper-health | All four dashboards usable locally against real backtest + (mocked until race day) live data |
| **M7** | Live ops + odds logging | GraphQL **live-odds snapshot logger** (B1/B2/B3) keyed on `lastUpdateTime`; race-day pipeline: scrape card → predict → blend → value vs live odds → stake recommendation at a fixed cutoff; Windows Task Scheduler automation | Automated race-day card with recommendations + logged odds snapshots; **no bet is ever placed** |

Dependency notes: M2 depends on M1; M3/M4 on M2; M5 on M3; M6 surfaces M2–M5; M7 needs M1 (cards) + M3 (models).

---

## 3. Repository structure & tech stack

```
hkjc-racing/
  pyproject.toml                 # uv-managed; pinned deps + lockfile
  config/                        # YAML: sources, features, models, risk, backtest, paths
  src/hkjc/
    common/                      # config models, logging, ids/keys, time (HKT) utils
    data/
      scrape/                    # httpx clients, per-source fetchers, rate-limit, retry, cache
      parse/                     # per-source parsers (results, racecard, profile, sectional, text)
      store/                     # duckdb+parquet io, schema, _scrape_manifest, incremental logic
      live/                      # GraphQL odds poller + snapshot logger (M7)
      weather/                   # HKO open-data CSV ingest + station mapping
    features/
      build.py base.py           # as-of feature builders + leakage guards
      nlp/                       # English text → lagged structured signals/embeddings
    models/
      base.py                    # ProbabilityModel: race -> per-runner PL strength vector
      logit/ gbm/ nn/ ensemble/
      calibrate/                 # isotonic/platt/temperature + within-race normalization
      place/                     # harville / henery / lbs / direct
      blend/                     # market-blend stage
    risk/                        # kelly variants, correlated kelly, caps, legal rounding
    backtest/                    # walk-forward engine, pari-mutuel sim, metrics, bootstrap
    experiments/                 # mlflow + optuna harness, leaderboard
    api/                         # fastapi app, routers, pydantic schemas
    cli.py                       # typer: scrape/backfill/train/backtest/predict/poll
  ui/                            # react + vite + ts frontend
  tests/                         # parsers(fixtures) / features(leakage) / backtest+kelly(property)
  fixtures/                      # checked-in sample HTML/JSON for offline parser regression tests
  data/                          # gitignored: raw/ processed/ cache/ live_odds/ mlruns/
  notebooks/                     # exploration
```

**Tech stack (right-sized to RTX 4060 8 GB / i9-13900 / 64 GB):**

| Layer | Choice | Why |
|---|---|---|
| Lang / env | **Python 3.12 + uv** | Fast, reproducible lockfile; matches "Python everywhere" |
| Scrape | **httpx** (async/HTTP2) + **selectolax** + **tenacity** + **aiolimiter** + local cache | Fast parsing at volume; clean retry/backoff + polite rate-limiting + idempotent caching |
| Store | **DuckDB + partitioned Parquet** (pyarrow) | Endorsed: ideal single-user analytical store at this scale; raw=immutable Parquet, processed=DuckDB views/tables |
| Dataframes | **Polars** (+ pandas where needed) | Speed on as-of joins/feature builds |
| Weather | **HKO open-data CSV API** (verified) | Machine-readable; no scraping |
| NLP (English) | **spaCy** + lexicon + **sentence-transformers** (MiniLM) | Rules/NER + embeddings; runs fine on 8 GB |
| Models | **LightGBM, XGBoost(GPU), CatBoost**; sklearn (logit/isotonic); PyTorch + pytorch-tabular (TabNet/FT-Transformer) | GBMs are the workhorse; CatBoost handles ID categoricals; NNs as comparators |
| Calibration / PL | sklearn isotonic + **custom numpy/scipy** (softmax, Harville, PL) | Within-race normalization is bespoke |
| Experiments | **MLflow (local file/sqlite backend) + Optuna** | Local-only privacy (no cloud), integrates with Optuna; W&B not needed |
| Backtest tests | **pytest + hypothesis** | Property tests for Kelly/dividend/place-count/rounding math |
| API / UI | **FastAPI + Pydantic v2 / uvicorn**; **React + Vite + TS + TanStack Query + Recharts/Plotly + Tailwind + shadcn/ui** | Matches stack; fast clean dashboards |
| Orchestration | **Typer CLI + Windows Task Scheduler** | Simplest race-day automation for single-user Windows |
| Data versioning | Immutable partitioned raw Parquet + `_scrape_manifest` (content hashes) + dataset-snapshot hash logged to MLflow | Reproducibility without DVC overhead (DVC deferred) |

---

## 4. Data schema & scraper strategy

**Canonical keys:** `race_key = (race_date, venue ∈ {ST,HV}, race_no)`; `horse_id = HK_YYYY_XXXX`; `jockey_code`; `trainer_code`. GraphQL `runners[].no` ↔ HTML saddle number; GraphQL `horse.code/id` ↔ HTML `horseid` — reconciled into one model.

**Core tables (raw → processed):**
- `meetings` — race_date, venue, going, rail position, weather snapshot, fixture meta
- `races` — race_key, class, distance, surface/track, going, prize, field_size, post_time(HKT), place_count
- `runners` — race_key+saddle, horse_id, jockey_code, trainer_code, draw, declared_wt, actual_wt, rating, intl_rating, gear, allowance, last6, scratched
- `results` — race_key+saddle → finish_pos, finish_time, running_positions, sectionals, **SP win odds (= market)**, dead_heat, beaten_margin
- `dividends` — race_key+pool_type+combination → dividend, pool_size *(generic across all pools → exotics-ready)*
- `horses` / `jockeys` / `trainers` — profile + season splits (course/dist/going/class strike rates). **`horses` bio block (locked):** country_of_origin, age-at-race (derived), colour, sex, import_type, sire/dam/dam's-sire (= pedigree #11, HKJC-only), owner, start-of-season & current rating
- `sectionals` — race_key+saddle → per-section times
- `text_raw` / `text_parsed` — corunning, stewards, vet, incidents, exceptional factors; keyed by race_key(+saddle), with `text_event_time` for lagging
- `live_odds_snapshots` (M7) — snapshot_ts, race_key, pool_type, comb, odds_value, odds_drop, hot_fav, status, last_update_time *(append-only)*
- `pool_investments` (M7) — snapshot_ts, race_key, pool_type, investment, total
- `_scrape_manifest` — url, source, fetched_at, content_hash, status, n_rows *(idempotency + provenance)*
- `features_runner` — race_key+saddle → as-of features + feature_version + computed_at + win/place labels

**Alternative-source tables (enabled — see §0):**
- `going_rail` — race_key → going description, going-stick (where available), **rail position** (#3) → draw-bias features
- `barrier_trials` — horse_id+trial_date → trial finish/time/comment (#4)
- `trackwork` — horse_id+work_date → gallop/work info (#5; rolling archive)
- `vet_records` — horse_id(+race_key) → pre-race vet-list entries (feature) & post-race vet notes (lagged) (#6)
- `gear_changes` — race_key+saddle → gear + derived first-time-gear flags (#8)
- `pedigree` — horse_id → sire / dam / dam's-sire (#11, HKJC-only); folded into `horses` bio block
- `public_holidays` — date → holiday/festival flags (#14, gov.hk open data)
- racing news (#9) → flows into `text_raw` / `text_parsed` (English NLP, lagged)

**Scraper strategy per source:**

| Source | Method | Pattern | Mutability |
|---|---|---|---|
| Fixtures (date enumeration) | HTTP parse | `.../local/information/fixture` (verified) | re-fetch (schedule changes) |
| Results (+ full dividends, sectionals, SP) | HTTP parse | `.../localresults?racedate=YYYY/MM/DD&Racecourse=ST\|HV&RaceNo=N`; meeting: `.../resultsall?...` | **frozen** once results final |
| Race card / entries | HTTP parse | `.../racecard?racedate=...&Racecourse=...&RaceNo=N` | mutable until off, then frozen |
| Profiles (horse/jockey/trainer) | HTTP parse | `.../horse?horseid=` etc. | refresh periodically |
| Draw / barrier trials / trackwork | HTTP parse | `.../draw`, `.../btresult`, `.../trackworksearch` | refresh |
| NLP text | HTTP parse | corunning / racereportfull / veterinaryrecord / exceptionalfactors / formline *(exact base paths TBC in M1)* | frozen post-race |
| Weather (history) | CSV API | `data.weather.gov.hk/weatherAPI/opendata/...` (verified) | frozen |
| Live WIN/PLA odds, card+odds, pool investments | GraphQL POST | `info.cld.hkjc.com/graphql/base/` (B1/B2/B3; verified reachable) | live (M7) |

**Incremental / idempotent crawl:** enumerate `(date, venue)` from the fixtures page → for each meeting discover race count → fetch via `resultsall`/per-race. A fetch is **skipped** when `_scrape_manifest` shows the URL fetched AND the target data is already stored AND the page is non-mutable. **Mutable** pages (upcoming cards, live odds) are re-fetchable until results finalize, then **frozen**. Raw layer is immutable, partitioned by `season/venue/date`. Content hashes detect silent upstream changes. Polite: rate-limited, exponential backoff, cached.

---

## 5. What I need from you

**Must-have to start** — all currently satisfied:
1. **Internet access** — RESOLVED: I have it; HTML + HKO CSV + GraphQL all reachable. No sample pages needed from you (I'll fetch + check in fixtures myself in M1).
2. **Bankroll / risk defaults** — CONFIRMED: HK$1,000, rebate 0, recommended sweep grid (frac-Kelly {0.05–0.5}+full+flat, per-race 10%, per-day 25%, EV ≥5%, multi-bankroll 1k/10k/50k/100k).
3. **Backfill scope** — CONFIRMED: modern-markup seasons only; legacy parser backlogged.
4. **NLP scope** — CONFIRMED: English only.
5. **Two acknowledgements — ACCEPTED 2026-06-14:**
   - ✅ The **final-dividend honesty caveat** (1A): historical blended ROI is an optimistic upper bound until live provisional odds accumulate.
   - ✅ **Scraping HKJC/HKO public pages for personal research** is the user's call re: their terms of use; I'll be polite/rate-limited but won't make a legal determination.

**Nice-to-have later:**
- ✅ **RESOLVED 2026-06-14:** alternative sources enabled = 3,4,5,6,7,8,9,11,14; #2 dropped; full horse-bio block added (see §0).
- ✅ **CONFIRMED:** MLflow-local (not W&B).
- Decide later whether to **enable the legacy-season parser** and/or **pull the live-odds logger earlier** (next uncapturable meeting: 21 Jun 2026).
- Any **segments you suspect hold edge** (class/distance/going/field-size) to prioritize in analysis.

### 15 alternative data sources (free/public) — **SELECTION LOCKED (see §0)**
Legend: ✅ enabled · ⬜ not enabled.
1. ⬜ **HKO automatic weather stations real-time API** — per-station temp/rain/humidity/wind incl. Sha Tin & HV vicinity (race-day microclimate).
2. ⬜ **HKO rainfall nowcast / radar / warnings** — going-change prediction on the day. *(Dropped: no clean historical archive.)*
3. ✅ **HKJC going-stick / track-condition & rail-position reports** — rail position strongly affects draw bias.
4. ✅ **HKJC barrier-trial results** — trial form, a strong public pre-race signal.
5. ✅ **HKJC trackwork / gallop reports** — recent work intensity/intent. *(Rolling archive; M1 reports earliest date.)*
6. ✅ **HKJC veterinary / horses-on-vet-list** — soundness & fitness flags. *(Pre-race vet-list = feature; post-race record = lagged only.)*
7. ✅ **HKJC sectional-time archive** — basis for home-grown speed figures.
8. ✅ **HKJC gear-change declarations** — first-time blinkers/visors etc.
9. ✅ **HKJC racing news / press releases** — trainer comments, intentions (English NLP, M4).
10. ⬜ **GraphQL exotic-pool odds/investments** (QIN/QPL/FCT/TRI/…) — market-structure & liquidity signals.
11. ✅ **Pedigree data (HKJC profile sire/dam)** — distance/going aptitude. *(HKJC on-site only; external pedigree DBs NOT used.)*
12. ⬜ **International ratings** (World's Best Racehorse Rankings) — *external WBRR not used; the per-runner `internationalRating` field is still captured in core data.*
13. ⬜ **Overseas prior form for newly-imported horses** (partly-public racing authorities) — pre-HK record.
14. ✅ **HK public-holiday / festival calendar (gov.hk open data)** — crowd/pool-size & day-vs-night effects.
15. ⬜ **EPD air-quality (AQHI) open data + HKO sunrise/sunset/tide** — heat/pollution stress & twilight-meeting lighting (cheap, weak signals).

---

## 6. Open questions & assumptions

**Assumptions (proceeding unless you object):**
- Modern markup is stable back to ~2006–07; M1 reports the actual earliest clean date.
- "Speed figures" aren't official → we **compute our own** from finish/sectional times normalized by going/distance/class (a sub-project in M2/M3).
- Place rules: 3 places @ 7+ runners, 2 @ 5–6, none below; dead-heats split; WIN dividend per HK$10 — all encoded + property-tested.
- All times HKT; SP on results = closing line = market (never a fundamental feature).
- Everything local: MLflow local backend, no cloud, GPU optional (GBMs primary).
- NLP signals strictly **lagged** (prior runs only).
- Pedigree limited to **HKJC-provided sire/dam/dam's-sire** (no external pedigree DB scraping).
- **Age-at-race** is derived from the horse's age + racing season (HKJC does not publish foaling dates prominently).
- #2 rainfall nowcast is **dropped** (no clean historical archive); historical weather = HKO daily-climate CSV only.

**Open questions (non-blocking; defaults chosen):**
- Exact base URLs for the English text pages (resolve in M1; one guess 404'd).
- Race-card publication lead time (resolve in M1; ~a few days, not a week).
- Nearest-station mapping for weather joins (default: HKO/urban for HV, Sha Tin station for ST).
- Retraining cadence for walk-forward (default: per-season rolling refit; revisit in M2).

---

## Verification (how we'll know it works)
- **Idempotency:** re-run full crawl → `_scrape_manifest` shows 0 new fetches/rows.
- **Parsers:** pytest against checked-in `fixtures/` HTML/JSON (no live-site dependency).
- **Leakage:** canary feature (future/shuffled) scores ≈0 importance and ≈0 backtest ROI; CI gate.
- **Backtest/Kelly math:** hypothesis property tests for dividend calc, place-count-by-field-size, dead-heat splitting, stake rounding, Kelly fractions.
- **End-to-end:** `hkjc backtest --config ...` reproduces logged ROI/Sharpe + calibration from a logged data hash; A/B compare two configs.
- **UI smoke:** all four dashboards render real backtest output locally.
- **Live (M7):** dry-run on a real meeting (e.g. 21 Jun 2026) logs odds snapshots and emits a recommendation card — and places no bet.
