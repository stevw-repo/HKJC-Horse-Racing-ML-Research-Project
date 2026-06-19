"""Application configuration: ``pydantic-settings`` models loaded from ``config/*.yaml``.

Configuration is layered (highest priority first):

1. Explicit keyword arguments to :class:`AppConfig`.
2. Environment variables prefixed ``HKJC_`` (nested via ``__``), e.g.
   ``HKJC_RISK__BANKROLL=5000`` overrides ``risk.bankroll``.
3. YAML files in the ``config/`` directory (one top-level key per file), merged.

The data scope encoded here is LOCKED per ``PLAN.md`` §0 (reviewed 2026-06-14).
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

CONFIG_DIR_ENV = "HKJC_CONFIG_DIR"


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
class Paths(BaseModel):
    """Filesystem layout for the local data lake (raw Parquet + processed DuckDB)."""

    model_config = ConfigDict(extra="forbid")

    data_root: Path = Path("data")
    fixtures_dir: Path = Path("fixtures")
    raw_subdir: str = "raw"
    processed_subdir: str = "processed"
    cache_subdir: str = "cache"
    live_odds_subdir: str = "live_odds"
    mlruns_subdir: str = "mlruns"
    duckdb_filename: str = "hkjc.duckdb"

    @property
    def raw_dir(self) -> Path:
        return self.data_root / self.raw_subdir

    @property
    def processed_dir(self) -> Path:
        return self.data_root / self.processed_subdir

    @property
    def cache_dir(self) -> Path:
        return self.data_root / self.cache_subdir

    @property
    def live_odds_dir(self) -> Path:
        return self.data_root / self.live_odds_subdir

    @property
    def mlruns_dir(self) -> Path:
        return self.data_root / self.mlruns_subdir

    @property
    def duckdb_path(self) -> Path:
        return self.processed_dir / self.duckdb_filename


# --------------------------------------------------------------------------- #
# Sources (PLAN.md §0 — LOCKED data scope)
# --------------------------------------------------------------------------- #
class CoreSources(BaseModel):
    """Core HKJC race data — always collected."""

    model_config = ConfigDict(extra="forbid")

    meetings: bool = True
    races: bool = True
    runners: bool = True
    results: bool = True
    dividends: bool = True
    sectionals: bool = True
    profiles: bool = True
    international_rating: bool = True


class AltSource(BaseModel):
    """One alternative data source with its enable flag and time-gating note."""

    model_config = ConfigDict(extra="forbid")

    id: int
    enabled: bool
    gating: str | None = None


class WeatherSource(BaseModel):
    """Weather provider config (HKO daily-climate CSV, historical only in v1)."""

    model_config = ConfigDict(extra="forbid")

    provider: str = "hko_opendata_csv"
    historical_only: bool = True


class Sources(BaseModel):
    """All data-source configuration. Defaults match the verified endpoints."""

    model_config = ConfigDict(extra="forbid")

    hkjc_base_url: str = "https://racing.hkjc.com/en-us/local/information"
    hkjc_graphql_url: str = "https://info.cld.hkjc.com/graphql/base/"
    hko_weather_api: str = "https://data.weather.gov.hk/weatherAPI/opendata"
    gov_holidays_url: str = "https://www.1823.gov.hk/common/ical/en.json"
    venues: list[str] = Field(default_factory=lambda: ["ST", "HV"])
    language: str = "en"
    core: CoreSources = Field(default_factory=CoreSources)
    alternative_sources: dict[str, AltSource] = Field(default_factory=dict)
    weather: WeatherSource = Field(default_factory=WeatherSource)

    @property
    def enabled_alt_sources(self) -> list[str]:
        """Names of alternative sources that are enabled."""
        return [name for name, src in self.alternative_sources.items() if src.enabled]


# --------------------------------------------------------------------------- #
# Features
# --------------------------------------------------------------------------- #
class FeatureGroup(BaseModel):
    """A toggleable feature group. ``wall`` marks market features kept off-limits to
    the fundamental model; ``ablatable`` marks groups measured by ablation (e.g. NLP)."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    ablatable: bool = False
    wall: bool = False


class HorseBio(BaseModel):
    """Locked horse-bio block (PLAN.md §0). ``age_at_race`` is derived from age + season."""

    model_config = ConfigDict(extra="forbid")

    country_of_origin: bool = True
    age_at_race: bool = True
    colour: bool = True
    sex: bool = True
    import_type: bool = True
    sire: bool = True
    dam: bool = True
    dams_sire: bool = True
    owner: bool = True
    season_start_rating: bool = True
    current_rating: bool = True


class Leakage(BaseModel):
    """Leakage controls (PLAN.md §1H)."""

    model_config = ConfigDict(extra="forbid")

    enable_canary: bool = True
    enforce_as_of: bool = True


class Features(BaseModel):
    """Feature-engineering configuration."""

    model_config = ConfigDict(extra="forbid")

    feature_version: str = "v0"
    groups: dict[str, FeatureGroup] = Field(default_factory=dict)
    horse_bio: HorseBio = Field(default_factory=HorseBio)
    leakage: Leakage = Field(default_factory=Leakage)


# --------------------------------------------------------------------------- #
# Risk / staking (confirmed defaults)
# --------------------------------------------------------------------------- #
class Risk(BaseModel):
    """Risk / staking defaults (PLAN.md §5.2)."""

    model_config = ConfigDict(extra="forbid")

    bankroll: float = 1000.0
    currency: str = "HKD"
    rebate_rate: float = 0.0
    min_bet: float = 10.0
    bet_unit: float = 10.0
    ev_threshold: float = 0.05
    staking_methods: list[str] = Field(default_factory=list)
    kelly_fractions: list[float] = Field(default_factory=list)
    per_race_cap: float = 0.10
    per_day_cap: float = 0.25
    correlated_kelly: bool = True
    multi_bankroll: list[float] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Backtest
# --------------------------------------------------------------------------- #
class PlaceRule(BaseModel):
    """One field-size -> place-count rule."""

    model_config = ConfigDict(extra="forbid")

    min_runners: int
    places: int


class PlaceRules(BaseModel):
    """Place-dividend rules (PLAN.md §1I)."""

    model_config = ConfigDict(extra="forbid")

    places_by_field_size: list[PlaceRule] = Field(default_factory=list)
    dead_heat_split: bool = True
    win_dividend_unit: float = 10.0


class Backtest(BaseModel):
    """Backtest-engine defaults."""

    model_config = ConfigDict(extra="forbid")

    takeout_win: float = 0.175
    takeout_place: float = 0.175
    rounding_unit: float = 10.0
    pool_impact: bool = True
    walk_forward: bool = True
    retrain_cadence: str = "per_season"
    bootstrap_iterations: int = 1000
    report_two_rois: bool = True
    place_rules: PlaceRules = Field(default_factory=PlaceRules)


# --------------------------------------------------------------------------- #
# Models (firms up at M3)
# --------------------------------------------------------------------------- #
class Models(BaseModel):
    """Modeling configuration (canonical choices; full zoo lands at M3)."""

    model_config = ConfigDict(extra="forbid")

    baseline: str = "conditional_logit"
    place_methods: list[str] = Field(default_factory=list)
    calibration: list[str] = Field(default_factory=list)
    within_race_normalization: bool = True
    market_blend_enabled: bool = True
    market_blend_weight: float = 0.5
    experiment_tracking: str = "mlflow_local"
    hpo: str = "optuna"


# --------------------------------------------------------------------------- #
# Loader
# --------------------------------------------------------------------------- #
def find_repo_root(start: Path | None = None) -> Path:
    """Return the repo root (nearest ancestor containing ``pyproject.toml``).

    Searches upward from ``start`` (if given), then the CWD, then this module's
    location (which sits inside the repo under an editable install). Falls back to
    the CWD if no marker is found.
    """
    bases: list[Path] = []
    if start is not None:
        bases.append(start)
    bases.append(Path.cwd())
    bases.append(Path(__file__).resolve().parent)
    for base in bases:
        base = base.resolve()
        for candidate in (base, *base.parents):
            if (candidate / "pyproject.toml").is_file():
                return candidate
    return Path.cwd().resolve()


def resolve_config_dir() -> Path:
    """Return the config directory (``$HKJC_CONFIG_DIR`` or ``<repo>/config``)."""
    env = os.environ.get(CONFIG_DIR_ENV)
    if env:
        return Path(env).expanduser().resolve()
    return find_repo_root() / "config"


def _deep_merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``update`` into ``base`` (in place) and return ``base``."""
    for key, value in update.items():
        existing = base.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            _deep_merge(existing, value)
        else:
            base[key] = value
    return base


class _YamlDirSource(PydanticBaseSettingsSource):
    """Settings source that merges every ``*.yaml`` file in the config directory."""

    def __init__(self, settings_cls: type[BaseSettings]) -> None:
        super().__init__(settings_cls)
        self._data = self._read()

    @staticmethod
    def _read() -> dict[str, Any]:
        config_dir = resolve_config_dir()
        merged: dict[str, Any] = {}
        if config_dir.is_dir():
            for path in sorted(config_dir.glob("*.yaml")):
                with path.open(encoding="utf-8") as handle:
                    loaded = yaml.safe_load(handle) or {}
                if not isinstance(loaded, dict):
                    msg = f"Config file {path} must contain a top-level mapping"
                    raise ValueError(msg)
                _deep_merge(merged, loaded)
        return merged

    def get_field_value(self, field: Any, field_name: str) -> tuple[Any, str, bool]:
        return self._data.get(field_name), field_name, False

    def __call__(self) -> dict[str, Any]:
        return self._data


class AppConfig(BaseSettings):
    """Top-level application configuration, assembled from YAML + env overrides."""

    model_config = SettingsConfigDict(
        env_prefix="HKJC_",
        env_nested_delimiter="__",
        extra="ignore",
        nested_model_default_partial_update=True,
    )

    paths: Paths = Field(default_factory=Paths)
    sources: Sources = Field(default_factory=Sources)
    features: Features = Field(default_factory=Features)
    risk: Risk = Field(default_factory=Risk)
    backtest: Backtest = Field(default_factory=Backtest)
    models: Models = Field(default_factory=Models)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        # Priority: explicit kwargs > env vars > merged YAML files.
        return (init_settings, env_settings, _YamlDirSource(settings_cls))

    @model_validator(mode="after")
    def _resolve_paths(self) -> Self:
        root = find_repo_root()
        if not self.paths.data_root.is_absolute():
            self.paths.data_root = (root / self.paths.data_root).resolve()
        if not self.paths.fixtures_dir.is_absolute():
            self.paths.fixtures_dir = (root / self.paths.fixtures_dir).resolve()
        return self


@lru_cache(maxsize=1)
def get_config() -> AppConfig:
    """Return the cached, fully-resolved application configuration."""
    return AppConfig()
