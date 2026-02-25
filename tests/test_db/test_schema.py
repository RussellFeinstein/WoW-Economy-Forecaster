"""Tests for SQLite schema — idempotency, table/index creation, FK enforcement."""

from __future__ import annotations

import sqlite3

import pytest

from wow_forecaster.db.schema import (
    ALL_TABLE_NAMES,
    apply_schema,
    get_existing_indexes,
    get_existing_tables,
)


class TestApplySchema:
    def test_applies_without_error(self, in_memory_db):
        # Schema was already applied in the fixture — just verify no error occurred
        tables = get_existing_tables(in_memory_db)
        assert len(tables) >= len(ALL_TABLE_NAMES)

    def test_all_tables_created(self, in_memory_db):
        tables = get_existing_tables(in_memory_db)
        for expected_table in ALL_TABLE_NAMES:
            assert expected_table in tables, (
                f"Expected table '{expected_table}' not found in database. "
                f"Found: {tables}"
            )

    def test_table_count(self, in_memory_db):
        tables = get_existing_tables(in_memory_db)
        # At minimum we have all defined tables (plus schema_versions may be added later)
        assert len(tables) >= len(ALL_TABLE_NAMES)

    def test_idempotent_double_apply(self, in_memory_db):
        """apply_schema() called twice must not raise errors."""
        apply_schema(in_memory_db)  # second call
        tables = get_existing_tables(in_memory_db)
        assert len(tables) >= len(ALL_TABLE_NAMES)

    def test_key_indexes_created(self, in_memory_db):
        indexes = get_existing_indexes(in_memory_db)
        expected_indexes = [
            "idx_obs_raw_item_time",
            "idx_obs_norm_item_time",
            "idx_events_type_date",
            "idx_forecast_archetype_date",
        ]
        for idx in expected_indexes:
            assert idx in indexes, (
                f"Expected index '{idx}' not found. Found: {indexes}"
            )


class TestForeignKeyEnforcement:
    def test_fk_enforcement_is_on(self, in_memory_db):
        """Foreign key constraints should be enabled by get_connection()."""
        row = in_memory_db.execute("PRAGMA foreign_keys;").fetchone()
        assert row[0] == 1, "PRAGMA foreign_keys should be 1 (enabled)"

    def test_fk_violation_raises(self, in_memory_db):
        """Inserting a row with an invalid FK should raise an OperationalError."""
        with pytest.raises(sqlite3.IntegrityError):
            in_memory_db.execute(
                """
                INSERT INTO items (item_id, name, category_id, expansion_slug, quality)
                VALUES (99999, 'Nonexistent', 99999, 'tww', 'common');
                """
            )


class TestTableStructure:
    def test_item_categories_has_slug_column(self, in_memory_db):
        cols = [
            row[1]
            for row in in_memory_db.execute("PRAGMA table_info(item_categories);").fetchall()
        ]
        assert "slug" in cols
        assert "display_name" in cols
        assert "archetype_tag" in cols

    def test_wow_events_has_announced_at(self, in_memory_db):
        cols = [
            row[1]
            for row in in_memory_db.execute("PRAGMA table_info(wow_events);").fetchall()
        ]
        assert "announced_at" in cols, "wow_events must have announced_at for backtest bias guard"
        assert "start_date" in cols
        assert "end_date" in cols

    def test_market_obs_raw_has_is_processed(self, in_memory_db):
        cols = [
            row[1]
            for row in in_memory_db.execute(
                "PRAGMA table_info(market_observations_raw);"
            ).fetchall()
        ]
        assert "is_processed" in cols

    def test_run_metadata_has_config_snapshot(self, in_memory_db):
        cols = [
            row[1]
            for row in in_memory_db.execute("PRAGMA table_info(run_metadata);").fetchall()
        ]
        assert "config_snapshot" in cols
        assert "run_slug" in cols
        assert "pipeline_stage" in cols
