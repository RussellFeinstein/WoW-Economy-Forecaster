"""Tests for the ForecastStage freshness gate (issue #12).

The gate refuses to generate forecasts when the newest normalized
observation is older than config.forecast.max_data_age_hours.  This is the
Python-level seam that also covers manual `run-daily-forecast` invocations;
the batch-level seam (run_daily.bat prologue) is tested in
tests/test_scripts/test_run_daily.py.

All tests inject a fixed `now` so results do not depend on the wall clock.
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta

import pytest

from wow_forecaster.config import (
    AppConfig,
    DatabaseConfig,
    DataConfig,
    ForecastConfig,
    ModelConfig,
)
from wow_forecaster.db.schema import apply_schema
from wow_forecaster.pipeline.forecast import (
    ForecastStage,
    StaleDataError,
    _fetch_max_observation_age_hours,
)

# Fixed reference time: all observation ages are computed relative to this.
NOW = datetime(2026, 4, 1, 12, 0, 0, tzinfo=UTC)


# ── Harness ───────────────────────────────────────────────────────────────────


def _make_db(tmp_path) -> str:
    db_file = str(tmp_path / "test.db")
    conn = sqlite3.connect(db_file)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    apply_schema(conn)
    conn.commit()
    conn.close()
    return db_file


def _insert_observation(
    db_file: str,
    observed_at: datetime,
    realm_slug: str = "us",
    is_outlier: int = 0,
    item_id: int = 100,
) -> None:
    """Insert one raw + normalized observation pair at a fixed timestamp."""
    ts = observed_at.strftime("%Y-%m-%dT%H:%M:%SZ")
    conn = sqlite3.connect(db_file)
    try:
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute(
            "INSERT OR IGNORE INTO item_categories (slug, display_name, archetype_tag) "
            "VALUES ('test.cat', 'Test', 'test');"
        )
        cat_id = conn.execute(
            "SELECT category_id FROM item_categories WHERE slug='test.cat';"
        ).fetchone()[0]
        conn.execute(
            "INSERT OR IGNORE INTO items "
            "(item_id, name, category_id, expansion_slug, quality) "
            "VALUES (?, 'Item', ?, 'midnight', 'common');",
            (item_id, cat_id),
        )
        obs_id = conn.execute(
            "INSERT INTO market_observations_raw "
            "(item_id, realm_slug, faction, observed_at, source, is_processed) "
            "VALUES (?, ?, 'neutral', ?, 'test', 1) RETURNING obs_id;",
            (item_id, realm_slug, ts),
        ).fetchone()[0]
        conn.execute(
            "INSERT INTO market_observations_normalized "
            "(obs_id, item_id, realm_slug, observed_at, price_gold, "
            " quantity_listed, is_outlier) "
            "VALUES (?, ?, ?, ?, 10.0, 100, ?);",
            (obs_id, item_id, realm_slug, ts, is_outlier),
        )
        conn.commit()
    finally:
        conn.close()


def _make_config(tmp_path, db_file: str, max_age_hours: float = 26.0) -> AppConfig:
    """AppConfig pointing every path at tmp_path so no repo data is touched."""
    return AppConfig(
        database=DatabaseConfig(db_path=db_file),
        data=DataConfig(
            raw_dir=str(tmp_path / "raw"),
            processed_dir=str(tmp_path / "processed"),
        ),
        forecast=ForecastConfig(max_data_age_hours=max_age_hours),
        model=ModelConfig(
            artifact_dir=str(tmp_path / "artifacts"),
            forecast_output_dir=str(tmp_path / "forecasts"),
            recommendation_output_dir=str(tmp_path / "recommendations"),
        ),
    )


# ── _fetch_max_observation_age_hours ──────────────────────────────────────────


class TestFetchMaxObservationAgeHours:
    def test_returns_age_of_newest_observation(self, tmp_path):
        db_file = _make_db(tmp_path)
        _insert_observation(db_file, NOW - timedelta(hours=48), item_id=100)
        _insert_observation(db_file, NOW - timedelta(hours=5), item_id=101)
        conn = sqlite3.connect(db_file)
        try:
            age = _fetch_max_observation_age_hours(conn, "us", now=NOW)
        finally:
            conn.close()
        assert age == pytest.approx(5.0)

    def test_none_when_no_observations(self, tmp_path):
        db_file = _make_db(tmp_path)
        conn = sqlite3.connect(db_file)
        try:
            assert _fetch_max_observation_age_hours(conn, "us", now=NOW) is None
        finally:
            conn.close()

    def test_outlier_rows_are_ignored(self, tmp_path):
        db_file = _make_db(tmp_path)
        _insert_observation(db_file, NOW - timedelta(hours=48), item_id=100)
        _insert_observation(
            db_file, NOW - timedelta(hours=1), is_outlier=1, item_id=101
        )
        conn = sqlite3.connect(db_file)
        try:
            age = _fetch_max_observation_age_hours(conn, "us", now=NOW)
        finally:
            conn.close()
        assert age == pytest.approx(48.0)

    def test_scoped_to_realm(self, tmp_path):
        db_file = _make_db(tmp_path)
        _insert_observation(
            db_file, NOW - timedelta(hours=1), realm_slug="eu", item_id=100
        )
        conn = sqlite3.connect(db_file)
        try:
            assert _fetch_max_observation_age_hours(conn, "us", now=NOW) is None
        finally:
            conn.close()


# ── ForecastStage gate behavior ───────────────────────────────────────────────


class TestForecastStageFreshnessGate:
    def test_stale_data_raises_and_records_failed_run(self, tmp_path):
        """Observations past the threshold abort the stage with StaleDataError."""
        db_file = _make_db(tmp_path)
        _insert_observation(db_file, NOW - timedelta(hours=100))
        config = _make_config(tmp_path, db_file)
        stage = ForecastStage(config=config, db_path=db_file)

        with pytest.raises(StaleDataError, match="100.0h old"):
            stage.run(realm_slug="us", now=NOW)

        conn = sqlite3.connect(db_file)
        try:
            row = conn.execute(
                "SELECT status, error_message FROM run_metadata "
                "WHERE pipeline_stage='forecast' ORDER BY run_id DESC LIMIT 1;"
            ).fetchone()
        finally:
            conn.close()
        assert row is not None
        assert row[0] == "failed"
        assert "refusing to forecast" in row[1]

    def test_empty_db_raises(self, tmp_path):
        """A realm with no observations at all is refused, not forecast blind."""
        db_file = _make_db(tmp_path)
        config = _make_config(tmp_path, db_file)
        stage = ForecastStage(config=config, db_path=db_file)

        with pytest.raises(StaleDataError, match="no normalized observations"):
            stage.run(realm_slug="us", now=NOW)

    def test_fresh_data_passes_gate(self, tmp_path):
        """Fresh observations clear the gate; the stage then completes with
        zero rows because no model artifacts exist in the tmp tree."""
        db_file = _make_db(tmp_path)
        _insert_observation(db_file, NOW - timedelta(hours=2))
        config = _make_config(tmp_path, db_file)
        stage = ForecastStage(config=config, db_path=db_file)

        result = stage.run(realm_slug="us", now=NOW)
        assert result.status == "success"
        assert result.rows_processed == 0

    def test_age_exactly_at_threshold_passes(self, tmp_path):
        """The gate trips strictly above the threshold, not at it."""
        db_file = _make_db(tmp_path)
        _insert_observation(db_file, NOW - timedelta(hours=26))
        config = _make_config(tmp_path, db_file, max_age_hours=26.0)
        stage = ForecastStage(config=config, db_path=db_file)

        result = stage.run(realm_slug="us", now=NOW)
        assert result.status == "success"

    def test_zero_threshold_disables_gate(self, tmp_path):
        """max_data_age_hours=0 turns the gate off entirely."""
        db_file = _make_db(tmp_path)
        _insert_observation(db_file, NOW - timedelta(days=365))
        config = _make_config(tmp_path, db_file, max_age_hours=0.0)
        stage = ForecastStage(config=config, db_path=db_file)

        result = stage.run(realm_slug="us", now=NOW)
        assert result.status == "success"

    def test_config_default_is_26_hours(self):
        """Config-code sync: the ForecastConfig default matches default.toml."""
        assert ForecastConfig().max_data_age_hours == 26.0
