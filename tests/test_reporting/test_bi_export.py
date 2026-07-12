"""Tests for wow_forecaster.reporting.bi_export — star-schema BI exports."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from wow_forecaster.db.schema import apply_schema
from wow_forecaster.reporting.bi_export import (
    export_star_schema,
    generate_data_dictionary,
)


@pytest.fixture
def seeded_db(tmp_path: Path) -> str:
    """Create a DB with sample data for BI export testing."""
    path = tmp_path / "test.db"
    conn = sqlite3.connect(str(path))
    conn.execute("PRAGMA foreign_keys = ON;")
    apply_schema(conn)
    from wow_forecaster.db.migrations import run_migrations
    run_migrations(conn)

    # Seed archetype
    conn.execute("""
        INSERT INTO economic_archetypes (archetype_id, slug, display_name, category_tag)
        VALUES (1, 'consumable.flask.stat', 'Stat Flasks', 'consumable')
    """)

    # Seed item category
    conn.execute("""
        INSERT INTO item_categories (category_id, slug, display_name, archetype_tag)
        VALUES (1, 'flask', 'Flasks', 'consumable.flask.stat')
    """)

    # Seed item
    conn.execute("""
        INSERT INTO items (item_id, name, category_id, archetype_id, expansion_slug, quality)
        VALUES (12345, 'Flask of Power', 1, 1, 'tww', 'epic')
    """)

    # Seed observations
    for i in range(5):
        conn.execute("""
            INSERT INTO market_observations_raw
                (obs_id, item_id, realm_slug, observed_at, source)
            VALUES (?, 12345, 'us', date('now', ? || ' days'), 'blizzard_api')
        """, (i + 1, f"-{i}"))
        conn.execute("""
            INSERT INTO market_observations_normalized
                (obs_id, item_id, archetype_id, realm_slug, observed_at,
                 price_gold, quantity_listed, is_outlier)
            VALUES (?, 12345, 1, 'us', date('now', ? || ' days'), ?, 100, 0)
        """, (i + 1, f"-{i}", 500.0 + i * 10))

    # Seed event
    conn.execute("""
        INSERT INTO wow_events
            (slug, display_name, event_type, scope, severity, expansion_slug, start_date)
        VALUES ('tww-s2', 'TWW S2', 'SEASON_LAUNCH', 'GLOBAL', 'MAJOR', 'tww',
                date('now', '-5 days'))
    """)

    # Seed run + forecast + recommendation
    conn.execute("""
        INSERT INTO run_metadata (run_id, run_slug, pipeline_stage, status, config_snapshot)
        VALUES (1, 'test-001', 'forecast', 'completed', '{}')
    """)
    conn.execute("""
        INSERT INTO forecast_outputs
            (forecast_id, run_id, archetype_id, realm_slug, forecast_horizon, target_date,
             predicted_price_gold, confidence_lower, confidence_upper, model_slug)
        VALUES (1, 1, 1, 'us', '1d', date('now', '+1 day'), 550.0, 500.0, 600.0, 'lgbm_1d')
    """)
    conn.execute("""
        INSERT INTO recommendation_outputs
            (forecast_id, action, reasoning, priority, score, risk_level,
             score_components, category_tag)
        VALUES (1, 'buy', 'Strong upward signal', 1, 72.5, 'low',
                '{"opportunity": 80, "liquidity": 60}', 'consumable')
    """)

    conn.commit()
    conn.close()
    return str(path)


class TestExportStarSchema:
    def test_generates_all_default_tables(self, seeded_db: str, tmp_path: Path):
        out = tmp_path / "bi_output"
        result = export_star_schema(seeded_db, str(out), "us")

        assert "dim_archetypes" in result
        assert "dim_events" in result
        assert "dim_items" in result
        assert "dim_dates" in result
        assert "fact_prices" in result
        assert "fact_forecasts" in result
        assert "fact_recommendations" in result
        assert "fact_backtest" not in result  # not included by default

    def test_all_files_exist(self, seeded_db: str, tmp_path: Path):
        out = tmp_path / "bi_output"
        result = export_star_schema(seeded_db, str(out), "us")

        for name, path in result.items():
            assert path.exists(), f"{name} file does not exist"
            assert path.stat().st_size > 0, f"{name} file is empty"

    def test_csv_format(self, seeded_db: str, tmp_path: Path):
        out = tmp_path / "bi_output"
        result = export_star_schema(seeded_db, str(out), "us", fmt="csv")

        for _name, path in result.items():
            assert path.suffix == ".csv"

    def test_parquet_format(self, seeded_db: str, tmp_path: Path):
        out = tmp_path / "bi_output"
        result = export_star_schema(seeded_db, str(out), "us", fmt="parquet")

        for _name, path in result.items():
            assert path.suffix == ".parquet"

    def test_includes_backtest_when_requested(self, seeded_db: str, tmp_path: Path):
        out = tmp_path / "bi_output"
        result = export_star_schema(seeded_db, str(out), "us", include_backtest=True)
        assert "fact_backtest" in result

    def test_dim_archetypes_has_data(self, seeded_db: str, tmp_path: Path):
        import csv

        out = tmp_path / "bi_output"
        export_star_schema(seeded_db, str(out), "us")

        with open(out / "dim_archetypes.csv", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 1
        assert rows[0]["slug"] == "consumable.flask.stat"
        assert rows[0]["category_tag"] == "consumable"

    def test_fact_prices_has_data(self, seeded_db: str, tmp_path: Path):
        import csv

        out = tmp_path / "bi_output"
        export_star_schema(seeded_db, str(out), "us")

        with open(out / "fact_prices.csv", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) > 0
        assert "avg_price_gold" in rows[0]
        assert "archetype_id" in rows[0]

    def test_fact_forecasts_has_data(self, seeded_db: str, tmp_path: Path):
        import csv

        out = tmp_path / "bi_output"
        export_star_schema(seeded_db, str(out), "us")

        with open(out / "fact_forecasts.csv", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 1
        assert rows[0]["model_slug"] == "lgbm_1d"

    def test_fact_recommendations_has_data(self, seeded_db: str, tmp_path: Path):
        import csv

        out = tmp_path / "bi_output"
        export_star_schema(seeded_db, str(out), "us")

        with open(out / "fact_recommendations.csv", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 1
        assert rows[0]["action"] == "buy"

    def test_wrong_realm_produces_empty_facts(self, seeded_db: str, tmp_path: Path):
        import csv

        out = tmp_path / "bi_output"
        export_star_schema(seeded_db, str(out), "eu")

        with open(out / "fact_prices.csv", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 0

    def test_creates_output_directory(self, seeded_db: str, tmp_path: Path):
        out = tmp_path / "deep" / "nested" / "dir"
        export_star_schema(seeded_db, str(out), "us")
        assert out.exists()


class TestEmptyDB:
    def test_empty_db_does_not_crash(self, tmp_path: Path):
        db_path = tmp_path / "empty.db"
        conn = sqlite3.connect(str(db_path))
        apply_schema(conn)
        from wow_forecaster.db.migrations import run_migrations
        run_migrations(conn)
        conn.close()

        out = tmp_path / "bi_output"
        result = export_star_schema(str(db_path), str(out), "us")
        assert len(result) >= 6


class TestGenerateDataDictionary:
    def test_creates_markdown_file(self, tmp_path: Path):
        path = generate_data_dictionary(tmp_path / "DATA_DICTIONARY.md")
        assert path.exists()
        content = path.read_text(encoding="utf-8")
        assert "dim_archetypes" in content
        assert "fact_forecasts" in content
        assert "fact_recommendations" in content

    def test_documents_all_tables(self, tmp_path: Path):
        path = generate_data_dictionary(tmp_path / "DATA_DICTIONARY.md")
        content = path.read_text(encoding="utf-8")
        for table in ["dim_archetypes", "dim_events", "dim_items", "dim_dates",
                       "fact_prices", "fact_forecasts", "fact_recommendations",
                       "fact_backtest"]:
            assert table in content, f"Missing table: {table}"

    def test_creates_parent_dirs(self, tmp_path: Path):
        path = generate_data_dictionary(tmp_path / "sub" / "dir" / "dict.md")
        assert path.exists()
