"""Tests for RecipeRepository — CRUD operations on recipes/reagents tables."""

from __future__ import annotations

import pytest

from wow_forecaster.recipes.recipe_repo import RecipeRepository


def _insert_recipe(repo: RecipeRepository, recipe_id: int = 1001, profession: str = "alchemy") -> None:
    repo.upsert_recipe(
        recipe_id=recipe_id,
        profession_slug=profession,
        output_item_id=5000 + recipe_id,
        output_quantity=1,
        recipe_name=f"Test Recipe {recipe_id}",
        skill_level_required=50,
        expansion_slug="midnight",
    )


class TestRecipeRepo:
    def test_upsert_and_get(self, in_memory_db):
        repo = RecipeRepository(in_memory_db)
        _insert_recipe(repo, 1001)
        in_memory_db.commit()

        result = repo.get_recipe(1001)
        assert result is not None
        assert result.recipe_id == 1001
        assert result.profession_slug == "alchemy"
        assert result.output_item_id == 6001
        assert result.expansion_slug == "midnight"

    def test_upsert_updates_existing(self, in_memory_db):
        repo = RecipeRepository(in_memory_db)
        _insert_recipe(repo, 1001, "alchemy")
        # Update with a different profession (shouldn't happen IRL but tests the conflict path)
        repo.upsert_recipe(
            recipe_id=1001,
            profession_slug="enchanting",
            output_item_id=9999,
            output_quantity=2,
            recipe_name="Updated Recipe",
            skill_level_required=100,
            expansion_slug="midnight",
        )
        in_memory_db.commit()

        result = repo.get_recipe(1001)
        assert result.profession_slug == "enchanting"
        assert result.output_item_id == 9999
        assert result.output_quantity == 2

    def test_get_recipe_not_found_returns_none(self, in_memory_db):
        repo = RecipeRepository(in_memory_db)
        assert repo.get_recipe(99999) is None

    def test_get_recipes_by_expansion(self, in_memory_db):
        repo = RecipeRepository(in_memory_db)
        _insert_recipe(repo, 101, "alchemy")
        _insert_recipe(repo, 102, "blacksmithing")
        repo.upsert_recipe(
            recipe_id=200,
            profession_slug="alchemy",
            output_item_id=7000,
            output_quantity=1,
            recipe_name="Old Recipe",
            skill_level_required=10,
            expansion_slug="tww",
        )
        in_memory_db.commit()

        midnight = repo.get_recipes_by_expansion("midnight")
        assert len(midnight) == 2
        assert all(r.expansion_slug == "midnight" for r in midnight)

        tww = repo.get_recipes_by_expansion("tww")
        assert len(tww) == 1

    def test_replace_reagents(self, in_memory_db):
        repo = RecipeRepository(in_memory_db)
        _insert_recipe(repo, 1001)
        in_memory_db.commit()

        repo.replace_reagents(1001, [
            (101, 4, "required"),
            (102, 8, "required"),
        ])
        in_memory_db.commit()

        reagents = repo.get_reagents_for_recipe(1001)
        assert len(reagents) == 2
        item_ids = {r.ingredient_item_id for r in reagents}
        assert item_ids == {101, 102}

    def test_replace_reagents_replaces_old(self, in_memory_db):
        repo = RecipeRepository(in_memory_db)
        _insert_recipe(repo, 1001)
        repo.replace_reagents(1001, [(101, 4, "required")])
        repo.replace_reagents(1001, [(200, 2, "required"), (201, 1, "required")])
        in_memory_db.commit()

        reagents = repo.get_reagents_for_recipe(1001)
        assert len(reagents) == 2
        assert {r.ingredient_item_id for r in reagents} == {200, 201}

    def test_get_all_craftable_item_ids(self, in_memory_db):
        repo = RecipeRepository(in_memory_db)
        _insert_recipe(repo, 1001)
        _insert_recipe(repo, 1002)
        in_memory_db.commit()

        ids = repo.get_all_craftable_item_ids()
        assert 6001 in ids
        assert 6002 in ids

    def test_get_recipe_count(self, in_memory_db):
        repo = RecipeRepository(in_memory_db)
        assert repo.get_recipe_count() == 0
        _insert_recipe(repo, 1)
        _insert_recipe(repo, 2)
        in_memory_db.commit()
        assert repo.get_recipe_count() == 2

    def test_get_all_reagents_by_recipe_batch(self, in_memory_db):
        repo = RecipeRepository(in_memory_db)
        _insert_recipe(repo, 1001)
        _insert_recipe(repo, 1002)
        repo.replace_reagents(1001, [(10, 2, "required"), (11, 3, "required")])
        repo.replace_reagents(1002, [(20, 1, "required")])
        in_memory_db.commit()

        result = repo.get_all_reagents_by_recipe([1001, 1002])
        assert len(result[1001]) == 2
        assert len(result[1002]) == 1
        assert result[1001][0].recipe_id == 1001
