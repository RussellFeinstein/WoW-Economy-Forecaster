"""Tests for wow_forecaster.viz.data_queries — data fetching for charts."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pandas as pd
import pytest

from wow_forecaster.db.schema import apply_schema
from wow_forecaster.viz.data_queries import (
    fetch_archetypes,
    fetch_backtest_predictions,
    fetch_backtest_summary,
    fetch_crafting_margins,
    fetch_drift_history,
    fetch_events,
    fetch_forecast_data,
    fetch_historical_prices,
    fetch_recommendation_scores,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def db_path(tmp_path: Path) -> str:
    """Create a temp SQLite DB with schema applied."""
    path = tmp_path / "test.db"
    conn = sqlite3.connect(str(path))
    conn.execute("PRAGMA foreign_keys = ON;")
    apply_schema(conn)
    conn.close()
    return str(path)


@pytest.fixture
def seeded_db(db_path: str) -> str:
    """DB with some sample data for query testing."""
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON;")

    # Seed an archetype
    conn.execute("""
        INSERT INTO economic_archetypes (archetype_id, slug, display_name, category_tag)
        VALUES (1, 'consumable.flask.stat', 'Stat Flasks', 'consumable')
    """)

    # Seed an item category
    conn.execute("""
        INSERT INTO item_categories (category_id, slug, display_name, archetype_tag)
        VALUES (1, 'flask', 'Flasks', 'consumable.flask.stat')
    """)

    # Seed an item
    conn.execute("""
        INSERT INTO items (item_id, name, category_id, archetype_id, expansion_slug, quality)
        VALUES (12345, 'Flask of Power', 1, 1, 'tww', 'epic')
    """)

    # Seed normalized observations
    for i, day in enumerate(range(10)):
        price = 500.0 + i * 10
        conn.execute("""
            INSERT INTO market_observations_raw
                (obs_id, item_id, realm_slug, observed_at, source)
            VALUES (?, 12345, 'us', date('now', ? || ' days'), 'blizzard_api')
        """, (i + 1, f"-{day}"))
        conn.execute("""
            INSERT INTO market_observations_normalized
                (obs_id, item_id, archetype_id, realm_slug, observed_at,
                 price_gold, is_outlier)
            VALUES (?, 12345, 1, 'us', date('now', ? || ' days'), ?, 0)
        """, (i + 1, f"-{day}", price))

    # Seed a run + forecast
    conn.execute("""
        INSERT INTO run_metadata (run_id, run_slug, pipeline_stage, status, config_snapshot)
        VALUES (1, 'test-run-001', 'forecast', 'completed', '{}')
    """)
    conn.execute("""
        INSERT INTO forecast_outputs
            (run_id, archetype_id, realm_slug, forecast_horizon, target_date,
             predicted_price_gold, confidence_lower, confidence_upper, model_slug)
        VALUES (1, 1, 'us', '1d', date('now', '+1 day'), 550.0, 500.0, 600.0, 'lgbm_1d')
    """)
    conn.execute("""
        INSERT INTO forecast_outputs
            (run_id, archetype_id, realm_slug, forecast_horizon, target_date,
             predicted_price_gold, confidence_lower, confidence_upper, model_slug)
        VALUES (1, 1, 'us', '7d', date('now', '+7 days'), 580.0, 480.0, 680.0, 'lgbm_7d')
    """)

    # Seed an event
    conn.execute("""
        INSERT INTO wow_events
            (slug, display_name, event_type, scope, severity, expansion_slug, start_date)
        VALUES ('tww-s2-launch', 'TWW Season 2', 'SEASON_LAUNCH', 'GLOBAL', 'MAJOR', 'tww',
                date('now', '-5 days'))
    """)

    conn.commit()
    conn.close()
    return db_path


# ── Empty DB tests (graceful degradation) ─────────────────────────────────────


class TestEmptyDB:
    def test_fetch_forecast_data_empty(self, db_path: str):
        df = fetch_forecast_data(db_path, "us")
        assert isinstance(df, pd.DataFrame)
        assert df.empty

    def test_fetch_historical_prices_empty(self, db_path: str):
        df = fetch_historical_prices(db_path, "us", 999)
        assert isinstance(df, pd.DataFrame)
        assert df.empty

    def test_fetch_archetypes_empty(self, db_path: str):
        df = fetch_archetypes(db_path)
        assert isinstance(df, pd.DataFrame)
        assert df.empty

    def test_fetch_events_empty(self, db_path: str):
        df = fetch_events(db_path)
        assert isinstance(df, pd.DataFrame)
        assert df.empty

    def test_fetch_crafting_margins_empty(self, db_path: str):
        df = fetch_crafting_margins(db_path, "us")
        assert isinstance(df, pd.DataFrame)
        assert df.empty


class TestNonexistentDB:
    def test_forecast_data_bad_path(self):
        df = fetch_forecast_data("/nonexistent/path.db", "us")
        assert isinstance(df, pd.DataFrame)
        assert df.empty

    def test_historical_prices_bad_path(self):
        df = fetch_historical_prices("/nonexistent/path.db", "us", 1)
        assert isinstance(df, pd.DataFrame)
        assert df.empty


# ── Seeded DB tests ───────────────────────────────────────────────────────────


class TestFetchForecastData:
    def test_returns_forecasts(self, seeded_db: str):
        df = fetch_forecast_data(seeded_db, "us")
        assert not df.empty
        assert "predicted_price_gold" in df.columns
        assert "forecast_horizon" in df.columns
        assert len(df) == 2

    def test_filter_by_horizon(self, seeded_db: str):
        df = fetch_forecast_data(seeded_db, "us", horizon="1d")
        assert len(df) == 1
        assert df.iloc[0]["forecast_horizon"] == "1d"

    def test_wrong_realm_returns_empty(self, seeded_db: str):
        df = fetch_forecast_data(seeded_db, "eu")
        assert df.empty


class TestFetchHistoricalPrices:
    def test_returns_prices(self, seeded_db: str):
        df = fetch_historical_prices(seeded_db, "us", 1, days=30)
        assert not df.empty
        assert "avg_price_gold" in df.columns
        assert "obs_date" in df.columns

    def test_wrong_archetype_returns_empty(self, seeded_db: str):
        df = fetch_historical_prices(seeded_db, "us", 999, days=30)
        assert df.empty


class TestFetchArchetypes:
    def test_returns_archetypes(self, seeded_db: str):
        df = fetch_archetypes(seeded_db)
        assert not df.empty
        assert "category_tag" in df.columns
        assert df.iloc[0]["slug"] == "consumable.flask.stat"


class TestFetchEvents:
    def test_returns_events(self, seeded_db: str):
        df = fetch_events(seeded_db)
        assert not df.empty
        assert "display_name" in df.columns


# ── File-based query tests ────────────────────────────────────────────────────


class TestFetchBacktestPredictions:
    def test_empty_dir(self, tmp_path: Path):
        df = fetch_backtest_predictions(str(tmp_path), "us")
        assert isinstance(df, pd.DataFrame)
        assert df.empty

    def test_nonexistent_dir(self):
        df = fetch_backtest_predictions("/nonexistent", "us")
        assert df.empty

    def test_reads_predictions(self, tmp_path: Path):
        run_dir = tmp_path / "us_2025-01-01_2025-06-01" / "horizon_1d"
        run_dir.mkdir(parents=True)
        pd.DataFrame({
            "fold_index": [0, 0],
            "archetype_id": [1, 2],
            "actual_price": [100.0, 200.0],
            "predicted_price": [105.0, 195.0],
        }).to_csv(run_dir / "per_prediction.csv", index=False)

        df = fetch_backtest_predictions(str(tmp_path), "us")
        assert len(df) == 2
        assert "actual_price" in df.columns


class TestFetchBacktestSummary:
    def test_empty_dir(self, tmp_path: Path):
        df = fetch_backtest_summary(str(tmp_path), "us")
        assert df.empty

    def test_reads_summary(self, tmp_path: Path):
        run_dir = tmp_path / "us_2025-01-01_2025-06-01" / "horizon_1d"
        run_dir.mkdir(parents=True)
        pd.DataFrame({
            "model_name": ["naive_mean"],
            "mae": [50.0],
            "rmse": [60.0],
            "mape": [0.1],
        }).to_csv(run_dir / "summary.csv", index=False)

        df = fetch_backtest_summary(str(tmp_path), "us")
        assert not df.empty
        assert "mae" in df.columns


class TestFetchDriftHistory:
    def test_empty_dir(self, tmp_path: Path):
        df = fetch_drift_history(str(tmp_path), "us")
        assert df.empty

    def test_reads_drift_files(self, tmp_path: Path):
        data = {
            "realm_slug": "us",
            "checked_at": "2026-03-01T12:00:00Z",
            "overall_drift_level": "low",
            "uncertainty_multiplier": 1.25,
            "retrain_recommended": False,
            "data_drift": {"drift_level": "low", "drift_fraction": 0.15},
            "error_drift": {"drift_level": "none", "mae_ratio": 1.1},
            "event_shock": {"shock_active": False},
        }
        (tmp_path / "drift_status_us_2026-03-01.json").write_text(
            json.dumps(data), encoding="utf-8"
        )

        df = fetch_drift_history(str(tmp_path), "us")
        assert len(df) == 1
        assert df.iloc[0]["overall_drift_level"] == "low"
        assert df.iloc[0]["uncertainty_multiplier"] == 1.25


class TestFetchRecommendationScores:
    def test_empty_dir(self, tmp_path: Path):
        df = fetch_recommendation_scores(str(tmp_path), "us")
        assert df.empty

    def test_reads_recommendations(self, tmp_path: Path):
        data = {
            "schema_version": "v1",
            "realm_slug": "us",
            "generated_at": "2026-03-06T12:00:00Z",
            "categories": {
                "consumable": [
                    {
                        "rank": 1,
                        "archetype_id": 1,
                        "horizon": "1d",
                        "action": "buy",
                        "score": 72.5,
                        "score_components": {
                            "opportunity": 80.0,
                            "liquidity": 60.0,
                            "volatility": 20.0,
                            "event_boost": 10.0,
                            "uncertainty": 15.0,
                        },
                        "roi_pct": 0.12,
                        "current_price": 500.0,
                        "predicted_price": 560.0,
                        "risk_level": "low",
                        "ci_lower": 480.0,
                        "ci_upper": 640.0,
                        "model_slug": "lgbm_1d",
                    }
                ]
            },
        }
        (tmp_path / "recommendations_us_2026-03-06.json").write_text(
            json.dumps(data), encoding="utf-8"
        )

        df = fetch_recommendation_scores(str(tmp_path), "us")
        assert len(df) == 1
        assert df.iloc[0]["category"] == "consumable"
        assert df.iloc[0]["score"] == 72.5
        assert df.iloc[0]["sc_opportunity"] == 80.0
