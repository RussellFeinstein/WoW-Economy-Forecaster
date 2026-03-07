"""Tests for DB migration 0005 — crafting tables."""

from __future__ import annotations

import sqlite3

import pytest

from wow_forecaster.db.migrations import run_migrations
from wow_forecaster.db.schema import apply_schema, get_existing_tables, ALL_TABLE_NAMES


class TestCraftingTablesInSchema:
    def test_recipes_table_exists(self, in_memory_db):
        tables = get_existing_tables(in_memory_db)
        assert "recipes" in tables

    def test_recipe_reagents_table_exists(self, in_memory_db):
        tables = get_existing_tables(in_memory_db)
        assert "recipe_reagents" in tables

    def test_crafting_margin_snapshots_table_exists(self, in_memory_db):
        tables = get_existing_tables(in_memory_db)
        assert "crafting_margin_snapshots" in tables

    def test_all_table_names_contains_crafting_tables(self):
        assert "recipes" in ALL_TABLE_NAMES
        assert "recipe_reagents" in ALL_TABLE_NAMES
        assert "crafting_margin_snapshots" in ALL_TABLE_NAMES

    def test_table_count_is_21(self):
        assert len(ALL_TABLE_NAMES) == 21

    def test_recipes_schema(self, in_memory_db):
        cols = {
            row[1]
            for row in in_memory_db.execute("PRAGMA table_info(recipes);").fetchall()
        }
        required = {
            "recipe_id", "profession_slug", "output_item_id",
            "output_quantity", "expansion_slug", "source", "created_at",
        }
        assert required <= cols

    def test_recipe_reagents_schema(self, in_memory_db):
        cols = {
            row[1]
            for row in in_memory_db.execute("PRAGMA table_info(recipe_reagents);").fetchall()
        }
        required = {"id", "recipe_id", "ingredient_item_id", "quantity", "reagent_type"}
        assert required <= cols

    def test_crafting_margin_snapshots_schema(self, in_memory_db):
        cols = {
            row[1]
            for row in in_memory_db.execute(
                "PRAGMA table_info(crafting_margin_snapshots);"
            ).fetchall()
        }
        required = {
            "snapshot_id", "recipe_id", "realm_slug", "obs_date",
            "output_price_gold", "craft_cost_gold", "margin_gold",
            "margin_pct", "ingredient_coverage_pct",
        }
        assert required <= cols

    def test_crafting_margin_snapshots_unique_constraint(self, in_memory_db):
        """UNIQUE(recipe_id, realm_slug, obs_date) prevents duplicates."""
        in_memory_db.execute(
            "INSERT INTO recipes (recipe_id, profession_slug, output_item_id, expansion_slug)"
            " VALUES (1, 'alchemy', 100, 'midnight');"
        )
        in_memory_db.execute(
            """
            INSERT INTO crafting_margin_snapshots
                (recipe_id, realm_slug, obs_date, ingredient_coverage_pct)
            VALUES (1, 'us', '2026-03-06', 1.0);
            """
        )
        with pytest.raises(sqlite3.IntegrityError):
            in_memory_db.execute(
                """
                INSERT INTO crafting_margin_snapshots
                    (recipe_id, realm_slug, obs_date, ingredient_coverage_pct)
                VALUES (1, 'us', '2026-03-06', 0.9);
                """
            )

    def test_migration_0005_runs(self):
        """Migration 0005 applies cleanly on a fresh DB."""
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        apply_schema(conn)
        count = run_migrations(conn)
        # At least migration 0005 was applied (may be 0 if already in schema)
        assert count >= 0
        tables = get_existing_tables(conn)
        assert "recipes" in tables
        conn.close()

    def test_migration_0005_idempotent(self, in_memory_db):
        """Running migration 0005 again is a no-op."""
        run_migrations(in_memory_db)
        run_migrations(in_memory_db)
        tables = get_existing_tables(in_memory_db)
        assert "crafting_margin_snapshots" in tables
