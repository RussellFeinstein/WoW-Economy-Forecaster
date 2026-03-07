"""Tests for RecipeSeeder — fetch->normalise->upsert flow (mocked API)."""

from __future__ import annotations

import sqlite3
from unittest.mock import MagicMock, patch

import pytest

from wow_forecaster.db.schema import apply_schema
from wow_forecaster.recipes.recipe_seeder import RecipeSeeder, SeedStats


def _make_db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    apply_schema(conn)
    return conn


def _make_mock_client():
    """Return a BlizzardClient mock with pre-wired profession/recipe API responses."""
    client = MagicMock()

    # fetch_profession: returns skill tiers
    client.fetch_profession.return_value = {
        "id": 171,
        "name": "Alchemy",
        "skill_tiers": [
            {"id": 2751, "name": "Midnight Alchemy"},
            {"id": 2500, "name": "The War Within Alchemy"},
        ],
    }

    # fetch_profession_skill_tier: returns recipe stubs
    def skill_tier_side_effect(profession_id, skill_tier_id):
        if skill_tier_id == 2751:
            return {
                "categories": [
                    {
                        "name": "Flasks",
                        "recipes": [
                            {"id": 101, "name": "Midnight Flask"},
                            {"id": 102, "name": "Midnight Potion"},
                        ],
                    }
                ]
            }
        return {"categories": []}

    client.fetch_profession_skill_tier.side_effect = skill_tier_side_effect

    # fetch_recipe: returns full recipe data
    def recipe_side_effect(recipe_id):
        if recipe_id == 101:
            return {
                "id": 101,
                "name": "Midnight Flask",
                "crafted_item": {"id": 5001, "quantity": 1},
                "reagents": [
                    {"reagent": {"id": 200, "name": "Midnight Herb"}, "quantity": 8},
                    {"reagent": {"id": 201, "name": "Midnight Ore"}, "quantity": 2},
                ],
            }
        if recipe_id == 102:
            return {
                "id": 102,
                "name": "Midnight Potion",
                "crafted_item": {"id": 5002, "quantity": 2},
                "reagents": [
                    {"reagent": {"id": 200, "name": "Midnight Herb"}, "quantity": 4},
                ],
            }
        return {}

    client.fetch_recipe.side_effect = recipe_side_effect
    return client


class TestRecipeSeeder:
    def test_seed_upserts_recipes(self):
        conn = _make_db()
        client = _make_mock_client()
        seeder = RecipeSeeder(conn, client, delay_s=0.0)
        stats = seeder.seed(
            expansion_slug="midnight",
            professions=["alchemy"],
            tier_keywords=["midnight"],
        )
        assert stats.recipes_upserted == 2
        assert len(stats.errors) == 0

        recipes = conn.execute("SELECT * FROM recipes;").fetchall()
        assert len(recipes) == 2
        ids = {r["recipe_id"] for r in recipes}
        assert ids == {101, 102}

    def test_seed_upserts_reagents(self):
        conn = _make_db()
        client = _make_mock_client()
        seeder = RecipeSeeder(conn, client, delay_s=0.0)
        seeder.seed(
            expansion_slug="midnight",
            professions=["alchemy"],
            tier_keywords=["midnight"],
        )
        reagents = conn.execute(
            "SELECT * FROM recipe_reagents WHERE recipe_id = 101;"
        ).fetchall()
        assert len(reagents) == 2
        item_ids = {r["ingredient_item_id"] for r in reagents}
        assert item_ids == {200, 201}

    def test_seed_sets_expansion_slug(self):
        conn = _make_db()
        client = _make_mock_client()
        seeder = RecipeSeeder(conn, client, delay_s=0.0)
        seeder.seed(
            expansion_slug="midnight",
            professions=["alchemy"],
            tier_keywords=["midnight"],
        )
        rows = conn.execute("SELECT DISTINCT expansion_slug FROM recipes;").fetchall()
        assert rows[0]["expansion_slug"] == "midnight"

    def test_seed_tier_keyword_filters_tiers(self):
        """When keywords=["midnight"], the TWW tier should NOT be fetched."""
        conn = _make_db()
        client = _make_mock_client()
        seeder = RecipeSeeder(conn, client, delay_s=0.0)
        seeder.seed(
            expansion_slug="midnight",
            professions=["alchemy"],
            tier_keywords=["midnight"],
        )
        # fetch_profession_skill_tier called once (2751 only, not 2500)
        calls = client.fetch_profession_skill_tier.call_args_list
        called_tier_ids = {call.args[1] for call in calls}
        assert 2751 in called_tier_ids
        assert 2500 not in called_tier_ids

    def test_seed_is_idempotent(self):
        """Running seed twice does not duplicate rows."""
        conn = _make_db()
        client = _make_mock_client()
        seeder = RecipeSeeder(conn, client, delay_s=0.0)
        seeder.seed(
            expansion_slug="midnight",
            professions=["alchemy"],
            tier_keywords=["midnight"],
        )
        seeder.seed(
            expansion_slug="midnight",
            professions=["alchemy"],
            tier_keywords=["midnight"],
        )
        count = conn.execute("SELECT COUNT(*) FROM recipes;").fetchone()[0]
        assert count == 2

    def test_seed_unknown_profession_skipped(self):
        conn = _make_db()
        client = _make_mock_client()
        seeder = RecipeSeeder(conn, client, delay_s=0.0)
        stats = seeder.seed(
            expansion_slug="midnight",
            professions=["nonexistent_profession_xyz"],
            tier_keywords=["midnight"],
        )
        assert stats.recipes_upserted == 0
        assert len(stats.errors) > 0

    def test_seed_recipe_with_no_crafted_item_skipped(self):
        """Recipes that have no crafted_item are silently dropped."""
        conn = _make_db()
        client = _make_mock_client()

        # Override recipe 101 to have no crafted item
        def bad_recipe(recipe_id):
            if recipe_id == 101:
                return {"id": 101, "name": "Broken Recipe", "reagents": []}
            return client.fetch_recipe.side_effect.__wrapped__(recipe_id)

        original = client.fetch_recipe.side_effect
        def patched(recipe_id):
            if recipe_id == 101:
                return {"id": 101, "name": "Broken Recipe", "reagents": []}
            return original(recipe_id)

        client.fetch_recipe.side_effect = patched
        seeder = RecipeSeeder(conn, client, delay_s=0.0)
        stats = seeder.seed(
            expansion_slug="midnight",
            professions=["alchemy"],
            tier_keywords=["midnight"],
        )
        # Only recipe 102 should be upserted (101 has no crafted_item)
        assert stats.recipes_upserted == 1
