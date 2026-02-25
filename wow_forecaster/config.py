"""
Application configuration management.

Load order (each layer overrides the previous):
  1. ``config/default.toml``      — committed static defaults
  2. ``config/local.toml``        — optional local overrides (gitignored)
  3. ``.env``                     — local secrets and env overrides (gitignored)
  4. Environment variables        — ``WOW_FORECASTER_*`` prefix

Entry point: ``load_config(config_path=None) -> AppConfig``

All pipeline stages and CLI commands receive an ``AppConfig`` instance —
never raw dicts or individual env var lookups scattered through the codebase.
"""

from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, field_validator

# ── Sub-config models ─────────────────────────────────────────────────────────


class DatabaseConfig(BaseModel):
    """SQLite database connection settings."""

    model_config = ConfigDict(frozen=True)

    db_path: str = "data/db/wow_forecaster.db"
    wal_mode: bool = True
    busy_timeout_ms: int = 5000


class DataConfig(BaseModel):
    """Filesystem paths for raw and processed data."""

    model_config = ConfigDict(frozen=True)

    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    events_seed_file: str = "config/events/tww_events.json"


class ExpansionsConfig(BaseModel):
    """Expansion registry and active/transfer settings."""

    model_config = ConfigDict(frozen=True)

    known: list[str] = [
        "classic", "tbc", "wotlk", "cata", "mop", "wod",
        "legion", "bfa", "shadowlands", "dragonflight", "tww", "midnight",
    ]
    active: str = "tww"
    transfer_target: str = "midnight"


class RealmsConfig(BaseModel):
    """Realm tracking preferences."""

    model_config = ConfigDict(frozen=True)

    defaults: list[str] = ["area-52", "illidan", "stormrage", "tichondrius"]
    default_faction: str = "neutral"


class PipelineConfig(BaseModel):
    """Pipeline execution parameters."""

    model_config = ConfigDict(frozen=True)

    normalize_batch_size: int = 1000
    outlier_z_threshold: float = 3.0
    min_obs_for_feature: int = 30


class ForecastConfig(BaseModel):
    """Forecast generation settings."""

    model_config = ConfigDict(frozen=True)

    horizons: list[str] = ["1d", "7d", "30d"]
    confidence_pct: float = 0.80
    default_model_slug: str = "stub_linear_v0"

    @field_validator("confidence_pct")
    @classmethod
    def validate_confidence(cls, v: float) -> float:
        if not 0.0 < v < 1.0:
            raise ValueError(f"confidence_pct must be in (0.0, 1.0), got {v}.")
        return v


class LoggingConfig(BaseModel):
    """Logging output settings."""

    model_config = ConfigDict(frozen=True)

    level: str = "INFO"
    log_file: str = "data/logs/forecaster.log"
    json_format: bool = False

    @field_validator("level")
    @classmethod
    def validate_level(cls, v: str) -> str:
        valid = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v.upper() not in valid:
            raise ValueError(f"Log level must be one of {sorted(valid)}, got '{v}'.")
        return v.upper()


class BacktestConfig(BaseModel):
    """Walk-forward backtesting parameters."""

    model_config = ConfigDict(frozen=True)

    window_days: int = 30
    step_days: int = 7
    horizons_days: list[int] = [1, 3]
    min_train_rows: int = 14


class FeatureConfig(BaseModel):
    """Feature engineering parameters for the dataset builder.

    These values control which lag/rolling windows are computed, when an
    archetype series is considered "cold start", and how far back the
    training window extends by default.
    """

    model_config = ConfigDict(frozen=True)

    lag_days: list[int] = [1, 3, 7, 14, 28]
    rolling_windows: list[int] = [7, 14, 28]
    cold_start_threshold: int = 30       # obs below this → is_cold_start=True
    training_lookback_days: int = 180    # default training window length
    target_horizons_days: list[int] = [1, 7, 28]  # forward-looking price targets


class AppConfig(BaseModel):
    """Complete application configuration — the single source of truth.

    All pipeline stages and CLI commands receive an ``AppConfig`` instance.
    It is constructed by ``load_config()`` which merges TOML + .env.
    """

    model_config = ConfigDict(frozen=True)

    database: DatabaseConfig = DatabaseConfig()
    data: DataConfig = DataConfig()
    expansions: ExpansionsConfig = ExpansionsConfig()
    realms: RealmsConfig = RealmsConfig()
    pipeline: PipelineConfig = PipelineConfig()
    forecast: ForecastConfig = ForecastConfig()
    logging: LoggingConfig = LoggingConfig()
    backtest: BacktestConfig = BacktestConfig()
    features: FeatureConfig = FeatureConfig()
    debug: bool = False


# ── Loader ────────────────────────────────────────────────────────────────────

_PROJECT_ROOT = Path(__file__).parent.parent


def _find_project_root() -> Path:
    """Walk up from this file to find the project root (contains pyproject.toml)."""
    candidate = Path(__file__).parent
    for _ in range(5):
        if (candidate / "pyproject.toml").exists():
            return candidate
        candidate = candidate.parent
    return _PROJECT_ROOT


def load_config(config_path: Optional[Path] = None) -> AppConfig:
    """Load and merge application configuration.

    Args:
        config_path: Explicit path to a TOML config file. Defaults to
            ``<project_root>/config/default.toml``.

    Returns:
        Fully validated ``AppConfig`` instance.

    Raises:
        FileNotFoundError: If the specified ``config_path`` does not exist.
        pydantic.ValidationError: If merged config values fail validation.
    """
    root = _find_project_root()

    # 1. Load .env file (silently skip if missing)
    dotenv_path = root / ".env"
    load_dotenv(dotenv_path=dotenv_path, override=False)

    # 2. Load TOML config
    if config_path is None:
        config_path = root / "config" / "default.toml"

    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_path}\n"
            "Run 'wow-forecaster init-db' or create config/default.toml first."
        )

    with open(config_path, "rb") as f:
        raw: dict[str, Any] = tomllib.load(f)

    # Also merge local.toml if present (gitignored local overrides)
    local_config_path = config_path.parent / "local.toml"
    if local_config_path.exists():
        with open(local_config_path, "rb") as f:
            local_raw: dict[str, Any] = tomllib.load(f)
        raw = _deep_merge(raw, local_raw)

    # 3. Apply WOW_FORECASTER_* environment variable overrides
    raw = _apply_env_overrides(raw)

    # 4. Build and validate AppConfig
    return _build_app_config(raw)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into ``base``."""
    result = dict(base)
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = val
    return result


def _apply_env_overrides(raw: dict[str, Any]) -> dict[str, Any]:
    """Apply WOW_FORECASTER_* env vars to the raw config dict.

    Supported overrides:
      WOW_FORECASTER_DB_PATH    → raw["database"]["db_path"]
      WOW_FORECASTER_LOG_LEVEL  → raw["logging"]["level"]
      WOW_FORECASTER_DEBUG      → raw["debug"]
    """
    if db_path := os.environ.get("WOW_FORECASTER_DB_PATH"):
        raw.setdefault("database", {})["db_path"] = db_path

    if log_level := os.environ.get("WOW_FORECASTER_LOG_LEVEL"):
        raw.setdefault("logging", {})["level"] = log_level

    if debug := os.environ.get("WOW_FORECASTER_DEBUG"):
        raw["debug"] = debug.lower() in ("1", "true", "yes")

    return raw


def _build_app_config(raw: dict[str, Any]) -> AppConfig:
    """Map raw TOML dict to ``AppConfig`` model structure."""
    # Flatten top-level keys that may be nested under [project]
    project = raw.pop("project", {})

    return AppConfig(
        database=DatabaseConfig(**raw.get("database", {})),
        data=DataConfig(**raw.get("data", {})),
        expansions=ExpansionsConfig(**raw.get("expansions", {})),
        realms=RealmsConfig(**raw.get("realms", {})),
        pipeline=PipelineConfig(**raw.get("pipeline", {})),
        forecast=ForecastConfig(**raw.get("forecast", {})),
        logging=LoggingConfig(**raw.get("logging", {})),
        backtest=BacktestConfig(**raw.get("backtest", {})),
        features=FeatureConfig(**raw.get("features", {})),
        debug=raw.get("debug", project.get("debug", False)),
    )
