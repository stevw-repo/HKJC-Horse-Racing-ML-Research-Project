"""Feature-store contracts: the spec registry, group/role taxonomy, and as-of guards.

Every column the builder emits is declared here with a :class:`FeatureSpec`. The ``role``
encodes the hard rules from PLAN.md:

* ``fundamental`` -- legal pre-race model inputs (computable from ``event_time <=
  race_off_time``).
* ``market`` -- the closing line (SP / win-odds-implied probability). Walled off from the
  fundamental model (PLAN.md §1B); only the market-blend backtest policy may read it.
* ``label`` -- post-race outcome (win / place). Never an input.
* ``canary`` -- a pure-noise sentinel that must score ~0 (PLAN.md §1H); the highest-ROI
  leakage test.
* ``meta`` / ``id`` -- keys and bookkeeping, not model inputs.

``BASELINE_FEATURES`` is the compact, fully-numeric fundamental subset the M2
conditional-logit baseline consumes; the store emits a wider set (incl. high-cardinality
categoricals) for the M3 GBMs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Role = Literal["id", "meta", "fundamental", "market", "label", "canary"]


@dataclass(frozen=True, slots=True)
class FeatureSpec:
    """One column emitted by the feature builder."""

    name: str
    role: Role
    group: str  # matches config/features.yaml `groups` (or "key"/"label"/"canary")
    description: str


# --------------------------------------------------------------------------- #
# Registry (the authoritative list of feature_runner columns)
# --------------------------------------------------------------------------- #
FEATURE_SPECS: tuple[FeatureSpec, ...] = (
    # --- keys / meta -------------------------------------------------------- #
    FeatureSpec("race_date", "id", "key", "Meeting date (HKT)."),
    FeatureSpec("venue", "id", "key", "ST or HV."),
    FeatureSpec("race_no", "id", "key", "Race number within the meeting."),
    FeatureSpec("saddle", "id", "key", "Saddle/cloth number (runner key)."),
    FeatureSpec("horse_id", "id", "key", "HK_YYYY_XXXX."),
    FeatureSpec("jockey_code", "id", "key", "Jockey code (may be null on old pages)."),
    FeatureSpec("trainer_code", "id", "key", "Trainer code (may be null on old pages)."),
    FeatureSpec("season", "meta", "key", "HK season label, e.g. 2025-26 (walk-forward split key)."),
    FeatureSpec("field_size", "fundamental", "field_size", "Number of runners declared."),
    FeatureSpec("n_places", "meta", "field_size", "Paid place positions (by field size)."),
    # --- target-race fundamentals (declared / known pre-race) --------------- #
    FeatureSpec("distance_m", "fundamental", "form", "Race distance (m)."),
    FeatureSpec("is_turf", "fundamental", "going_track", "1 if turf, 0 if all-weather."),
    FeatureSpec("draw", "fundamental", "draw_bias", "Barrier draw (declared)."),
    FeatureSpec("draw_rel", "fundamental", "draw_bias", "Draw / field_size in [0,1]."),
    FeatureSpec("rail_offset", "fundamental", "draw_bias", "Rail displacement (m) from token."),
    FeatureSpec("actual_weight", "fundamental", "weight", "Weight carried (lbs)."),
    FeatureSpec("declared_weight", "fundamental", "weight", "Declared body weight (lbs)."),
    FeatureSpec(
        "as_of_rating", "fundamental", "class_moves", "Official rating going into the race."
    ),
    FeatureSpec("rating_trend3", "fundamental", "class_moves", "Rating delta vs ~3 runs prior."),
    FeatureSpec("month", "fundamental", "seasonality", "Calendar month (1-12)."),
    FeatureSpec("day_of_week", "fundamental", "seasonality", "0=Mon .. 6=Sun."),
    FeatureSpec(
        "is_public_holiday", "fundamental", "public_holiday", "1 if meeting on a HK holiday."
    ),
    FeatureSpec("mean_temp", "fundamental", "weather", "HKO daily mean temperature (C)."),
    # --- horse career-stage (run-history backbone; ~universal coverage) ----- #
    FeatureSpec("career_run_number", "fundamental", "days_since_run", "# prior HK runs (0=debut)."),
    FeatureSpec("days_since_debut", "fundamental", "days_since_run", "Days since first HK run."),
    FeatureSpec("days_since_last_run", "fundamental", "days_since_run", "Days since previous run."),
    FeatureSpec(
        "seasons_active", "fundamental", "days_since_run", "# distinct prior seasons raced."
    ),
    # --- horse prior form (strictly prior runs) ----------------------------- #
    FeatureSpec("win_rate_prior", "fundamental", "form", "Career win rate before this race."),
    FeatureSpec("place_rate_prior", "fundamental", "form", "Career place rate before this race."),
    FeatureSpec("avg_finish_last3", "fundamental", "form", "Mean finish pos over up to 3 prior."),
    FeatureSpec("avg_lbw_last3", "fundamental", "form", "Mean beaten-lengths over up to 3 prior."),
    FeatureSpec(
        "recent_speed", "fundamental", "speed_figures", "Mean m/s over up to 3 prior runs."
    ),
    FeatureSpec("dist_match_rate", "fundamental", "form", "Prior place rate at similar distance."),
    FeatureSpec(
        "going_match_rate", "fundamental", "going_track", "Prior place rate on same going."
    ),
    # --- connections (rolling as-of, NOT the people snapshot) --------------- #
    FeatureSpec(
        "jockey_win_rate", "fundamental", "form", "Jockey career win rate before this race."
    ),
    FeatureSpec("jockey_runs_prior", "fundamental", "form", "Jockey # prior rides in our data."),
    FeatureSpec("trainer_win_rate", "fundamental", "form", "Trainer career win rate before race."),
    FeatureSpec(
        "trainer_runs_prior", "fundamental", "form", "Trainer # prior runners in our data."
    ),
    # --- trial form (#4) ---------------------------------------------------- #
    FeatureSpec(
        "days_since_trial", "fundamental", "trial_form", "Days since most recent prior trial."
    ),
    FeatureSpec(
        "had_recent_trial", "fundamental", "trial_form", "1 if a trial in the prior 60 days."
    ),
    # --- bio (#0 locked block; pedigree #11 emitted for M3, not in baseline) - #
    FeatureSpec("age_at_race", "fundamental", "form", "race_year - birth_year (sparse; see flag)."),
    FeatureSpec("age_imputed", "fundamental", "form", "1 if age_at_race is unknown/imputed."),
    FeatureSpec("import_type", "fundamental", "pedigree", "PPG/ISG/PP/... (categorical; M3 GBM)."),
    FeatureSpec("sex", "fundamental", "pedigree", "Gelding/Mare/... (categorical; M3 GBM)."),
    FeatureSpec(
        "country_of_origin", "fundamental", "pedigree", "Foaling country (categorical; M3)."
    ),
    FeatureSpec("sire", "fundamental", "pedigree", "Sire name (high-cardinality; M3 GBM)."),
    FeatureSpec("dam", "fundamental", "pedigree", "Dam name (high-cardinality; M3 GBM)."),
    FeatureSpec("dams_sire", "fundamental", "pedigree", "Dam's sire (high-cardinality; M3 GBM)."),
    # --- nlp_text (#9, M4; LAGGED = prior run's comment-on-running) ---------- #
    FeatureSpec("nlp_trouble", "fundamental", "nlp_text", "Prior run: trouble-phrase count."),
    FeatureSpec("nlp_slow_start", "fundamental", "nlp_text", "Prior run: slow-start count."),
    FeatureSpec("nlp_ran_on", "fundamental", "nlp_text", "Prior run: ran-on/kept-on count."),
    FeatureSpec("nlp_easing", "fundamental", "nlp_text", "Prior run: easing/in-hand count."),
    FeatureSpec("nlp_weakened", "fundamental", "nlp_text", "Prior run: weakened/faded count."),
    FeatureSpec("nlp_wide", "fundamental", "nlp_text", "Prior run: raced-wide count."),
    FeatureSpec("nlp_health", "fundamental", "nlp_text", "Prior run: health/soundness count."),
    FeatureSpec(
        "nlp_sim_trouble", "fundamental", "nlp_text", "Prior run: MiniLM sim to 'trouble'."
    ),
    FeatureSpec(
        "nlp_sim_easywin", "fundamental", "nlp_text", "Prior run: MiniLM sim to 'easy win'."
    ),
    FeatureSpec(
        "nlp_sim_noexcuse", "fundamental", "nlp_text", "Prior run: MiniLM sim to 'no excuse'."
    ),
    # --- market wall (closing line) ----------------------------------------- #
    FeatureSpec("market_prob", "market", "market", "Overround-adjusted SP-implied win prob."),
    FeatureSpec("win_odds", "market", "market", "Starting price (closing line; market data)."),
    # --- leakage canary ----------------------------------------------------- #
    FeatureSpec("canary_random", "canary", "canary", "Deterministic noise; must score ~0."),
    # --- labels (post-race outcomes) ---------------------------------------- #
    FeatureSpec("won", "label", "label", "1 if finished 1st (incl. dead-heat win)."),
    FeatureSpec("placed", "label", "label", "1 if finished within the paid places."),
    FeatureSpec("finish_pos", "label", "label", "Official finishing position (null=DNF)."),
)

_BY_ROLE: dict[Role, tuple[str, ...]] = {}
for _spec in FEATURE_SPECS:
    _BY_ROLE.setdefault(_spec.role, ())
    _BY_ROLE[_spec.role] = (*_BY_ROLE[_spec.role], _spec.name)


def names_by_role(role: Role) -> tuple[str, ...]:
    """Column names with the given role."""
    return _BY_ROLE.get(role, ())


# The compact numeric fundamental design matrix for the M2 conditional-logit baseline.
# (High-cardinality categoricals -- sire/dam/import_type/etc. -- are left to the M3 GBMs.)
BASELINE_FEATURES: tuple[str, ...] = (
    "as_of_rating",
    "rating_trend3",
    "draw_rel",
    "rail_offset",
    "actual_weight",
    "declared_weight",
    "distance_m",
    "is_turf",
    "field_size",
    "career_run_number",
    "days_since_last_run",
    "days_since_debut",
    "seasons_active",
    "win_rate_prior",
    "place_rate_prior",
    "avg_finish_last3",
    "avg_lbw_last3",
    "recent_speed",
    "dist_match_rate",
    "going_match_rate",
    "jockey_win_rate",
    "trainer_win_rate",
    "had_recent_trial",
    "age_at_race",
    "age_imputed",
    "mean_temp",
    "is_public_holiday",
)

# The lagged NLP feature group (#9, M4) -- prior run's comment-on-running signals. Ablatable:
# kept out of BASELINE_FEATURES so the ablation can add it and measure the marginal effect.
NLP_FEATURES: tuple[str, ...] = (
    "nlp_trouble",
    "nlp_slow_start",
    "nlp_ran_on",
    "nlp_easing",
    "nlp_weakened",
    "nlp_wide",
    "nlp_health",
    "nlp_sim_trouble",
    "nlp_sim_easywin",
    "nlp_sim_noexcuse",
)


def numeric_design_features(include_nlp: bool = False) -> tuple[str, ...]:
    """The numeric design columns, optionally with the lagged NLP group appended (M4 ablation)."""
    return (*BASELINE_FEATURES, *NLP_FEATURES) if include_nlp else BASELINE_FEATURES


# The canary rides alongside the real features through fit + backtest; it must score ~0.
CANARY_FEATURE = "canary_random"
