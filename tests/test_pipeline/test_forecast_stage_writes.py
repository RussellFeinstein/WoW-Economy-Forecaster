"""Tests for the shape of ForecastStage's database writes (issue #107).

The forecast stage used to insert the archetype forecasts and then build the
item-level forecasts on the same connection, so the write transaction the
inserts opened stayed open through a scan of the whole normalized table and
the 07:16 hourly's IngestStage failed with ``database is locked`` on every day
the daily ran.  These tests pin the two properties the fix relies on:

* no write transaction spans the item-forecast reads, which is checked the way
  production found it, by a second connection inserting a ``run_metadata`` row
  while ``_generate_item_forecasts`` is running; and
* the item forecasts come from the rollup tables, so a database with populated
  rollups and empty observation tables (the shape a restored durable backup
  has, since the backup excludes both observation tables) still yields item
  rows.

Everything the stage needs from the ML side (artifact discovery, model load,
the inference Parquet, inference itself) is replaced at its source module,
because ``_execute`` imports those names at call time.  The freshness gate is
off: it is a separate concern with its own tests, and the restored-backup case
has no observations for it to read.
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pytest

from wow_forecaster.config import (
    AppConfig,
    DatabaseConfig,
    DataConfig,
    ForecastConfig,
    ModelConfig,
)
from wow_forecaster.db.rollup import upsert_rollups_for_date
from wow_forecaster.db.schema import apply_schema
from wow_forecaster.models.forecast import ForecastOutput
from wow_forecaster.pipeline import forecast as forecast_module
from wow_forecaster.pipeline.forecast import ForecastStage

# A fixed midday anchor: the stage derives the item-forecast window from it, so
# the fixture rows land on the same calendar date at any wall-clock time.
NOW = datetime(2026, 3, 9, 12, 0, 0, tzinfo=UTC)
RUN_DATE: date = NOW.date()
REALM = "us"
ARCHETYPE_ID = 10
ITEM_ID = 100


# ── Harness ───────────────────────────────────────────────────────────────────


def _make_db(tmp_path) -> str:
    db_file = str(tmp_path / "test.db")
    conn = sqlite3.connect(db_file)
    try:
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        apply_schema(conn)
        conn.commit()
    finally:
        conn.close()
    return db_file


def _seed_item_history(db_file: str, days: int = 3) -> None:
    """One archetype, one recipe-linked item, priced on each of the last ``days``
    calendar dates ending at RUN_DATE, with the rollups built from those rows
    through the real aggregation."""
    conn = sqlite3.connect(db_file)
    try:
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute(
            "INSERT OR IGNORE INTO item_categories (slug, display_name, archetype_tag) "
            "VALUES ('test.cat', 'Test', 'test');"
        )
        cat_id = conn.execute(
            "SELECT category_id FROM item_categories WHERE slug='test.cat';"
        ).fetchone()[0]
        conn.execute(
            "INSERT OR IGNORE INTO economic_archetypes "
            "(archetype_id, slug, display_name, category_tag, sub_tag, "
            " is_transferable, transfer_confidence) "
            "VALUES (?, 'mat.herb.common', 'Test', 'mat', NULL, 1, 0.8);",
            (ARCHETYPE_ID,),
        )
        conn.execute(
            "INSERT OR IGNORE INTO items "
            "(item_id, name, category_id, expansion_slug, quality, archetype_id) "
            "VALUES (?, 'Item', ?, 'midnight', 'common', ?);",
            (ITEM_ID, cat_id, ARCHETYPE_ID),
        )
        conn.execute(
            "INSERT OR IGNORE INTO recipes "
            "(recipe_id, profession_slug, output_item_id, output_quantity, expansion_slug) "
            "VALUES (1, 'alchemy', ?, 1, 'midnight');",
            (ITEM_ID,),
        )
        for offset in range(days):
            day = RUN_DATE - timedelta(days=offset)
            ts = f"{day.isoformat()}T12:00:00Z"
            obs_id = conn.execute(
                "INSERT INTO market_observations_raw "
                "(item_id, realm_slug, faction, observed_at, source, is_processed) "
                "VALUES (?, ?, 'neutral', ?, 'test', 1) RETURNING obs_id;",
                (ITEM_ID, REALM, ts),
            ).fetchone()[0]
            conn.execute(
                "INSERT INTO market_observations_normalized "
                "(obs_id, item_id, realm_slug, observed_at, price_gold, "
                " quantity_listed, is_outlier) "
                "VALUES (?, ?, ?, ?, 50.0, 100, 0);",
                (obs_id, ITEM_ID, REALM, ts),
            )
            upsert_rollups_for_date(conn, REALM, day)
        conn.commit()
    finally:
        conn.close()


def _empty_observation_tables(db_file: str) -> None:
    """The restored-backup shape: durable tables intact, observation tables empty."""
    conn = sqlite3.connect(db_file)
    try:
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute("DELETE FROM market_observations_normalized;")
        conn.execute("DELETE FROM market_observations_raw;")
        conn.commit()
    finally:
        conn.close()


def _make_config(tmp_path, db_file: str) -> AppConfig:
    return AppConfig(
        database=DatabaseConfig(db_path=db_file),
        data=DataConfig(
            raw_dir=str(tmp_path / "raw"),
            processed_dir=str(tmp_path / "processed"),
        ),
        forecast=ForecastConfig(max_data_age_hours=0.0),
        model=ModelConfig(
            artifact_dir=str(tmp_path / "artifacts"),
            forecast_output_dir=str(tmp_path / "forecasts"),
            recommendation_output_dir=str(tmp_path / "recommendations"),
        ),
    )


def _fake_inference(config, run, forecasters, inference_parquet_path, realm_slug, **kwargs):
    """Stand-in for ml.predictor.run_inference: one archetype forecast per horizon."""
    return [
        ForecastOutput(
            run_id=run.run_id,
            archetype_id=ARCHETYPE_ID,
            item_id=None,
            realm_slug=realm_slug,
            forecast_horizon=label,  # type: ignore[arg-type]
            target_date=RUN_DATE + timedelta(days=days),
            predicted_price_gold=60.0,
            confidence_lower=54.0,
            confidence_upper=66.0,
            confidence_pct=0.80,
            model_slug="lgbm_test",
        )
        for label, days in (("1d", 1), ("7d", 7), ("28d", 28))
    ]


@pytest.fixture
def ml_seams(monkeypatch, tmp_path):
    """Replace artifact discovery, model loading and inference at their source modules."""
    monkeypatch.setattr(
        "wow_forecaster.ml.trainer.find_latest_model_artifact",
        lambda artifact_dir, realm_slug, horizon_days: tmp_path / f"model_{horizon_days}d.pkl",
    )
    monkeypatch.setattr(
        "wow_forecaster.ml.lgbm_model.LightGBMForecaster.load",
        classmethod(lambda cls, artifact_path: object()),
    )
    monkeypatch.setattr(
        "wow_forecaster.ml.predictor.find_latest_inference_parquet",
        lambda processed_dir, realm_slug: Path(tmp_path / "inference.parquet"),
    )
    monkeypatch.setattr("wow_forecaster.ml.predictor.run_inference", _fake_inference)


def _forecast_rows(db_file: str) -> tuple[int, int]:
    """(archetype rows, item rows) in forecast_outputs."""
    conn = sqlite3.connect(db_file)
    try:
        arch = conn.execute(
            "SELECT COUNT(*) FROM forecast_outputs WHERE archetype_id IS NOT NULL"
        ).fetchone()[0]
        item = conn.execute(
            "SELECT COUNT(*) FROM forecast_outputs WHERE item_id IS NOT NULL"
        ).fetchone()[0]
        return int(arch), int(item)
    finally:
        conn.close()


# ── Tests ─────────────────────────────────────────────────────────────────────


class TestForecastStageWriteTransaction:
    def test_second_writer_can_insert_while_item_forecasts_run(
        self, tmp_path, monkeypatch, ml_seams
    ):
        """The hourly's first write (a run_metadata row) must not wait on the
        forecast stage while the item forecasts are being built.  Before #107
        the archetype inserts had already opened the write transaction, so a
        one-second busy timeout here raised ``database is locked``."""
        db_file = _make_db(tmp_path)
        _seed_item_history(db_file)
        real_generate = forecast_module._generate_item_forecasts
        second_writer_outcome: list[str] = []

        def generate_with_concurrent_writer(conn, run_id, archetype_forecasts, realm_slug, **kw):
            other = sqlite3.connect(db_file, timeout=1.0)
            try:
                other.execute("PRAGMA busy_timeout = 1000;")
                other.execute(
                    "INSERT INTO run_metadata "
                    "(run_slug, pipeline_stage, status, config_snapshot, started_at) "
                    "VALUES ('hourly-during-forecast', 'ingest', 'started', '{}', ?);",
                    (NOW.strftime("%Y-%m-%dT%H:%M:%SZ"),),
                )
                other.commit()
                second_writer_outcome.append("inserted")
            finally:
                other.close()
            return real_generate(conn, run_id, archetype_forecasts, realm_slug, **kw)

        monkeypatch.setattr(
            forecast_module, "_generate_item_forecasts", generate_with_concurrent_writer
        )
        config = _make_config(tmp_path, db_file)
        stage = ForecastStage(config=config, db_path=db_file)

        result = stage.run(realm_slug=REALM, now=NOW)

        assert result.status == "success"
        assert second_writer_outcome == ["inserted"]
        arch_rows, item_rows = _forecast_rows(db_file)
        assert arch_rows == 3
        assert item_rows == 3, "the item forecasts must still be persisted after the reorder"

    def test_item_forecasts_from_rollups_with_empty_observation_tables(
        self, tmp_path, ml_seams
    ):
        """A restored durable backup has every rollup row and no observation rows;
        the daily forecast must still produce item-level forecasts from it."""
        db_file = _make_db(tmp_path)
        _seed_item_history(db_file)
        _empty_observation_tables(db_file)
        config = _make_config(tmp_path, db_file)
        stage = ForecastStage(config=config, db_path=db_file)

        result = stage.run(realm_slug=REALM, now=NOW)

        assert result.status == "success"
        arch_rows, item_rows = _forecast_rows(db_file)
        assert arch_rows == 3
        assert item_rows == 3
