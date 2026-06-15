"""Tests that the YAML config loads and encodes the LOCKED data scope (PLAN.md §0)."""

from __future__ import annotations

from hkjc.common.config import AppConfig, get_config

ENABLED_ALT_SOURCES = {
    "going_rail",
    "barrier_trials",
    "trackwork",
    "sectional_archive",
    "racing_news",
    "pedigree",
    "public_holidays",
}
# #6 vet_records and #8 gear_changes dropped 2026-06-15 (forward-only sources).
DROPPED_ALT_SOURCES = {
    "weather_stations_realtime",
    "rainfall_nowcast",
    "vet_records",
    "gear_changes",
    "exotic_pool_odds",
    "external_wbrr",
    "overseas_prior_form",
    "aqhi_tide",
}


def test_config_loads_with_confirmed_defaults() -> None:
    cfg = get_config()
    assert isinstance(cfg, AppConfig)
    assert cfg.risk.bankroll == 1000.0
    assert cfg.risk.min_bet == 10.0
    assert cfg.risk.rebate_rate == 0.0
    assert cfg.risk.kelly_fractions == [0.05, 0.10, 0.15, 0.25, 0.50]
    assert cfg.backtest.takeout_win == 0.175
    assert cfg.backtest.takeout_place == 0.175
    assert cfg.models.baseline == "conditional_logit"


def test_enabled_alternative_sources_match_lock() -> None:
    cfg = get_config()
    assert set(cfg.sources.enabled_alt_sources) == ENABLED_ALT_SOURCES


def test_dropped_sources_are_disabled() -> None:
    cfg = get_config()
    for name in DROPPED_ALT_SOURCES:
        assert cfg.sources.alternative_sources[name].enabled is False


def test_horse_bio_full_set_enabled() -> None:
    bio = get_config().features.horse_bio
    assert bio.age_at_race
    assert bio.sire and bio.dam and bio.dams_sire
    assert bio.owner
    assert bio.season_start_rating and bio.current_rating


def test_market_features_walled_and_nlp_ablatable() -> None:
    groups = get_config().features.groups
    assert groups["market"].wall is True
    assert groups["nlp_text"].ablatable is True


def test_paths_resolved_to_absolute() -> None:
    paths = get_config().paths
    assert paths.data_root.is_absolute()
    assert paths.duckdb_path.name == "hkjc.duckdb"
