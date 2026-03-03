"""
Unit tests for config.py — covering paths not exercised by indirect fixtures.

Targeted gaps:
  - _deep_merge: recursive TOML merge
  - _apply_env_overrides: WOW_FORECASTER_* env-var injection
  - Pydantic validators: ForecastConfig.confidence_pct, LoggingConfig.level
  - load_config: FileNotFoundError on missing path
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from pydantic import ValidationError

from wow_forecaster.config import (
    AppConfig,
    ForecastConfig,
    LoggingConfig,
    _apply_env_overrides,
    _deep_merge,
    load_config,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _clean_env():
    """Restore WOW_FORECASTER_* env vars after each test."""
    keys = ["WOW_FORECASTER_DB_PATH", "WOW_FORECASTER_LOG_LEVEL", "WOW_FORECASTER_DEBUG"]
    before = {k: os.environ.get(k) for k in keys}
    yield
    for k, v in before.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


# ── _deep_merge ───────────────────────────────────────────────────────────────

class TestDeepMerge:
    def test_flat_override(self):
        base = {"a": 1, "b": 2}
        override = {"b": 99}
        assert _deep_merge(base, override) == {"a": 1, "b": 99}

    def test_nested_merge_preserves_unoverridden_keys(self):
        base = {"db": {"path": "old.db", "wal": True}}
        override = {"db": {"path": "new.db"}}
        result = _deep_merge(base, override)
        assert result == {"db": {"path": "new.db", "wal": True}}

    def test_nested_override_does_not_mutate_base(self):
        base = {"db": {"path": "old.db"}}
        override = {"db": {"path": "new.db"}}
        _deep_merge(base, override)
        assert base["db"]["path"] == "old.db"

    def test_override_with_new_key(self):
        base = {"a": 1}
        override = {"b": 2}
        assert _deep_merge(base, override) == {"a": 1, "b": 2}

    def test_scalar_overrides_dict(self):
        """If override replaces a dict key with a scalar, the scalar wins."""
        base = {"a": {"nested": 1}}
        override = {"a": 99}
        assert _deep_merge(base, override) == {"a": 99}

    def test_dict_overrides_scalar(self):
        base = {"a": 99}
        override = {"a": {"nested": 1}}
        assert _deep_merge(base, override) == {"a": {"nested": 1}}

    def test_deeply_nested(self):
        base = {"l1": {"l2": {"l3": "old"}}}
        override = {"l1": {"l2": {"l3": "new", "extra": True}}}
        result = _deep_merge(base, override)
        assert result["l1"]["l2"] == {"l3": "new", "extra": True}

    def test_empty_override_returns_base_copy(self):
        base = {"a": 1}
        assert _deep_merge(base, {}) == {"a": 1}

    def test_empty_base_returns_override_copy(self):
        override = {"a": 1}
        assert _deep_merge({}, override) == {"a": 1}


# ── _apply_env_overrides ──────────────────────────────────────────────────────

class TestApplyEnvOverrides:
    def test_db_path_override(self):
        os.environ["WOW_FORECASTER_DB_PATH"] = "/tmp/test.db"
        raw: dict = {}
        result = _apply_env_overrides(raw)
        assert result["database"]["db_path"] == "/tmp/test.db"

    def test_db_path_override_merges_with_existing_database_section(self):
        os.environ["WOW_FORECASTER_DB_PATH"] = "/tmp/test.db"
        raw = {"database": {"wal_mode": False}}
        result = _apply_env_overrides(raw)
        assert result["database"]["db_path"] == "/tmp/test.db"
        assert result["database"]["wal_mode"] is False

    def test_log_level_override(self):
        os.environ["WOW_FORECASTER_LOG_LEVEL"] = "DEBUG"
        raw: dict = {}
        result = _apply_env_overrides(raw)
        assert result["logging"]["level"] == "DEBUG"

    def test_debug_true_variants(self):
        for truthy in ("1", "true", "yes", "True", "YES"):
            os.environ["WOW_FORECASTER_DEBUG"] = truthy
            result = _apply_env_overrides({})
            assert result["debug"] is True, f"Expected True for {truthy!r}"

    def test_debug_false_variants(self):
        for falsy in ("0", "false", "no", "False", "NO"):
            os.environ["WOW_FORECASTER_DEBUG"] = falsy
            result = _apply_env_overrides({})
            assert result["debug"] is False, f"Expected False for {falsy!r}"

    def test_no_env_vars_leaves_raw_unchanged(self):
        for k in ["WOW_FORECASTER_DB_PATH", "WOW_FORECASTER_LOG_LEVEL", "WOW_FORECASTER_DEBUG"]:
            os.environ.pop(k, None)
        raw = {"database": {"db_path": "original.db"}}
        result = _apply_env_overrides(raw)
        assert result == {"database": {"db_path": "original.db"}}


# ── Pydantic validators ───────────────────────────────────────────────────────

class TestForecastConfigValidator:
    def test_valid_confidence(self):
        fc = ForecastConfig(confidence_pct=0.95)
        assert fc.confidence_pct == 0.95

    def test_confidence_zero_raises(self):
        with pytest.raises(ValidationError):
            ForecastConfig(confidence_pct=0.0)

    def test_confidence_one_raises(self):
        with pytest.raises(ValidationError):
            ForecastConfig(confidence_pct=1.0)

    def test_confidence_negative_raises(self):
        with pytest.raises(ValidationError):
            ForecastConfig(confidence_pct=-0.5)

    def test_confidence_above_one_raises(self):
        with pytest.raises(ValidationError):
            ForecastConfig(confidence_pct=1.5)


class TestLoggingConfigValidator:
    def test_valid_levels(self):
        for level in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
            lc = LoggingConfig(level=level)
            assert lc.level == level

    def test_case_insensitive_normalised_to_upper(self):
        lc = LoggingConfig(level="debug")
        assert lc.level == "DEBUG"

    def test_invalid_level_raises(self):
        with pytest.raises(ValidationError):
            LoggingConfig(level="VERBOSE")

    def test_empty_level_raises(self):
        with pytest.raises(ValidationError):
            LoggingConfig(level="")


# ── load_config error path ────────────────────────────────────────────────────

class TestLoadConfig:
    def test_missing_path_raises_file_not_found(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            load_config(config_path=tmp_path / "nonexistent.toml")

    def test_returns_app_config(self):
        cfg = load_config()
        assert isinstance(cfg, AppConfig)

    def test_defaults_are_frozen(self):
        cfg = load_config()
        with pytest.raises(ValidationError):
            cfg.database.db_path = "mutated"  # type: ignore[misc]
