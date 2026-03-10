"""Tests for RecipeSeeder — fetch->normalise->upsert flow (mocked API)."""

from __future__ import annotations

import sqlite3
from unittest.mock import MagicMock

import pytest

from wow_forecaster.db.schema import apply_schema
from wow_forecaster.recipes.blizzard_recipe_client import (
    _is_optional_slot,
    _resolve_recipe_by_name,
)
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
    # Name search returns None by default; override per-test for resolution tests.
    client.search_item_by_name.return_value = None
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

    def test_seed_recipe_with_no_crafted_item_and_no_name_match_skipped(self):
        """Recipes without crafted_item that fail name lookup are dropped."""
        conn = _make_db()
        client = _make_mock_client()

        original = client.fetch_recipe.side_effect
        def patched(recipe_id):
            if recipe_id == 101:
                return {"id": 101, "name": "Broken Recipe", "reagents": []}
            return original(recipe_id)

        client.fetch_recipe.side_effect = patched
        # search_item_by_name returns None (already set in _make_mock_client)
        seeder = RecipeSeeder(conn, client, delay_s=0.0)
        stats = seeder.seed(
            expansion_slug="midnight",
            professions=["alchemy"],
            tier_keywords=["midnight"],
        )
        # Only recipe 102 upserted (101 has no crafted_item and name lookup fails)
        assert stats.recipes_upserted == 1

    def test_seed_resolves_output_item_via_name_lookup(self):
        """Recipes without crafted_item are resolved when name search succeeds."""
        conn = _make_db()
        client = _make_mock_client()

        original = client.fetch_recipe.side_effect
        def patched(recipe_id):
            if recipe_id == 101:
                # No crafted_item — like a TWW/Midnight recipe
                return {
                    "id": 101,
                    "name": "Midnight Flask",
                    "reagents": [
                        {"reagent": {"id": 200, "name": "Midnight Herb"}, "quantity": 8},
                    ],
                }
            return original(recipe_id)

        client.fetch_recipe.side_effect = patched
        client.search_item_by_name.side_effect = (
            lambda name: 5001 if name == "Midnight Flask" else None
        )

        seeder = RecipeSeeder(conn, client, delay_s=0.0)
        stats = seeder.seed(
            expansion_slug="midnight",
            professions=["alchemy"],
            tier_keywords=["midnight"],
        )
        assert stats.recipes_upserted == 2
        row = conn.execute(
            "SELECT output_item_id FROM recipes WHERE recipe_id = 101;"
        ).fetchone()
        assert row["output_item_id"] == 5001

    def test_seed_extracts_modified_slots_as_reagents(self):
        """modified_crafting_slots primary materials become required reagents."""
        conn = _make_db()
        client = _make_mock_client()

        original = client.fetch_recipe.side_effect
        def patched(recipe_id):
            if recipe_id == 101:
                return {
                    "id": 101,
                    "name": "Midnight Flask",
                    "reagents": [
                        {"reagent": {"id": 200, "name": "Mote of Energy"}, "quantity": 2},
                    ],
                    "modified_crafting_slots": [
                        {"slot_type": {"name": "Sunglass Vial", "id": 405}, "display_order": 0},
                        {"slot_type": {"name": "Peacebloom", "id": 432}, "display_order": 1},
                        # Optional slot — should be skipped
                        {"slot_type": {"name": "Artisan's Authenticity", "id": 396}, "display_order": 2},
                    ],
                }
            return original(recipe_id)

        client.fetch_recipe.side_effect = patched
        client.search_item_by_name.side_effect = {
            "Midnight Flask": 5001,
            "Sunglass Vial": 9001,
            "Peacebloom": 9002,
            "Artisan's Authenticity": None,
        }.get

        seeder = RecipeSeeder(conn, client, delay_s=0.0)
        seeder.seed(
            expansion_slug="midnight",
            professions=["alchemy"],
            tier_keywords=["midnight"],
        )
        reagents = conn.execute(
            "SELECT ingredient_item_id FROM recipe_reagents WHERE recipe_id = 101 ORDER BY ingredient_item_id;"
        ).fetchall()
        ingredient_ids = {r["ingredient_item_id"] for r in reagents}
        # Mote (200) + Sunglass Vial (9001) + Peacebloom (9002); authenticity skipped
        assert ingredient_ids == {200, 9001, 9002}


class TestIsOptionalSlot:
    def test_authenticity_is_optional(self):
        assert _is_optional_slot("Artisan's Authenticity") is True

    def test_embellishment_is_optional(self):
        assert _is_optional_slot("Add Embellishment") is True

    def test_herb_is_not_optional(self):
        assert _is_optional_slot("Peacebloom") is False

    def test_vial_is_not_optional(self):
        assert _is_optional_slot("Sunglass Vial") is False

    def test_ore_is_not_optional(self):
        assert _is_optional_slot("Refulgent Copper Ore") is False


class TestResolveRecipeByName:
    def test_returns_none_when_name_lookup_fails(self):
        client = MagicMock()
        client.search_item_by_name.return_value = None
        result = _resolve_recipe_by_name(
            {"id": 1, "name": "Unknown Recipe"},
            profession_slug="alchemy",
            expansion_slug="midnight",
            skill_level_required=0,
            client=client,
        )
        assert result is None

    def test_returns_recipe_with_resolved_output(self):
        client = MagicMock()
        client.search_item_by_name.return_value = 9999
        result = _resolve_recipe_by_name(
            {
                "id": 1,
                "name": "Potion of Recklessness",
                "reagents": [
                    {"reagent": {"id": 100, "name": "Mote"}, "quantity": 2}
                ],
            },
            profession_slug="alchemy",
            expansion_slug="midnight",
            skill_level_required=0,
            client=client,
        )
        assert result is not None
        assert result.output_item_id == 9999
        assert result.recipe_name == "Potion of Recklessness"
        assert len(result.reagents) == 1
        assert result.reagents[0].ingredient_item_id == 100
        assert result.reagents[0].quantity == 2

    def test_optional_slots_excluded(self):
        client = MagicMock()
        client.search_item_by_name.side_effect = {
            "Potion of Recklessness": 9999,
            "Peacebloom": 8001,
            "Artisan's Authenticity": 8002,
        }.get
        result = _resolve_recipe_by_name(
            {
                "id": 1,
                "name": "Potion of Recklessness",
                "reagents": [],
                "modified_crafting_slots": [
                    {"slot_type": {"name": "Peacebloom", "id": 432}, "display_order": 0},
                    {"slot_type": {"name": "Artisan's Authenticity", "id": 396}, "display_order": 1},
                ],
            },
            profession_slug="alchemy",
            expansion_slug="midnight",
            skill_level_required=0,
            client=client,
        )
        assert result is not None
        ingredient_ids = {r.ingredient_item_id for r in result.reagents}
        assert 8001 in ingredient_ids       # Peacebloom included
        assert 8002 not in ingredient_ids   # Authenticity excluded
