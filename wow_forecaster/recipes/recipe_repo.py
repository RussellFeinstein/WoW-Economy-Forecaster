"""
Data access layer for recipes and recipe_reagents tables.

All queries are read-only except for ``upsert_recipe`` and
``upsert_reagents``, which are used exclusively by RecipeSeeder.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass


@dataclass
class RecipeRow:
    """A row from the recipes table."""

    recipe_id: int
    profession_slug: str
    output_item_id: int
    output_quantity: int
    recipe_name: str | None
    skill_level_required: int
    expansion_slug: str
    source: str


@dataclass
class ReagentRow:
    """A row from the recipe_reagents table."""

    recipe_id: int
    ingredient_item_id: int
    quantity: int
    reagent_type: str


class RecipeRepository:
    """Read/write access to recipes and recipe_reagents tables."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    # ── Reads ──────────────────────────────────────────────────────────────────

    def get_recipes_by_expansion(self, expansion_slug: str) -> list[RecipeRow]:
        """Return all recipes for an expansion."""
        rows = self._conn.execute(
            """
            SELECT recipe_id, profession_slug, output_item_id, output_quantity,
                   recipe_name, skill_level_required, expansion_slug, source
            FROM recipes
            WHERE expansion_slug = ?
            ORDER BY profession_slug, recipe_id
            """,
            (expansion_slug,),
        ).fetchall()
        return [_row_to_recipe(r) for r in rows]

    def get_all_recipes(self) -> list[RecipeRow]:
        """Return all recipes regardless of expansion."""
        rows = self._conn.execute(
            """
            SELECT recipe_id, profession_slug, output_item_id, output_quantity,
                   recipe_name, skill_level_required, expansion_slug, source
            FROM recipes
            ORDER BY expansion_slug, profession_slug, recipe_id
            """,
        ).fetchall()
        return [_row_to_recipe(r) for r in rows]

    def get_recipe(self, recipe_id: int) -> RecipeRow | None:
        """Return a single recipe by ID, or None if not found."""
        row = self._conn.execute(
            """
            SELECT recipe_id, profession_slug, output_item_id, output_quantity,
                   recipe_name, skill_level_required, expansion_slug, source
            FROM recipes WHERE recipe_id = ?
            """,
            (recipe_id,),
        ).fetchone()
        return _row_to_recipe(row) if row else None

    def get_reagents_for_recipe(self, recipe_id: int) -> list[ReagentRow]:
        """Return required reagents for a recipe."""
        rows = self._conn.execute(
            """
            SELECT recipe_id, ingredient_item_id, quantity, reagent_type
            FROM recipe_reagents
            WHERE recipe_id = ? AND reagent_type = 'required'
            ORDER BY ingredient_item_id
            """,
            (recipe_id,),
        ).fetchall()
        return [_row_to_reagent(r) for r in rows]

    def get_all_reagents_by_recipe(
        self, recipe_ids: list[int]
    ) -> dict[int, list[ReagentRow]]:
        """Batch-fetch reagents for multiple recipe IDs.

        Returns dict mapping recipe_id -> list[ReagentRow].
        """
        if not recipe_ids:
            return {}
        placeholders = ",".join("?" * len(recipe_ids))
        rows = self._conn.execute(
            f"""
            SELECT recipe_id, ingredient_item_id, quantity, reagent_type
            FROM recipe_reagents
            WHERE recipe_id IN ({placeholders}) AND reagent_type = 'required'
            ORDER BY recipe_id, ingredient_item_id
            """,
            recipe_ids,
        ).fetchall()

        result: dict[int, list[ReagentRow]] = {rid: [] for rid in recipe_ids}
        for r in rows:
            result[r[0]].append(_row_to_reagent(r))
        return result

    def get_all_craftable_item_ids(self) -> set[int]:
        """Return set of all item IDs that are outputs of at least one recipe."""
        rows = self._conn.execute(
            "SELECT DISTINCT output_item_id FROM recipes;"
        ).fetchall()
        return {row[0] for row in rows}

    def get_recipe_count(self) -> int:
        """Return total number of recipes in the DB."""
        row = self._conn.execute("SELECT COUNT(*) FROM recipes;").fetchone()
        return int(row[0]) if row else 0

    # ── Writes ─────────────────────────────────────────────────────────────────

    def upsert_recipe(
        self,
        recipe_id: int,
        profession_slug: str,
        output_item_id: int,
        output_quantity: int,
        recipe_name: str | None,
        skill_level_required: int,
        expansion_slug: str,
        source: str = "blizzard_api",
    ) -> None:
        """Insert or replace a recipe row."""
        self._conn.execute(
            """
            INSERT INTO recipes
                (recipe_id, profession_slug, output_item_id, output_quantity,
                 recipe_name, skill_level_required, expansion_slug, source)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(recipe_id) DO UPDATE SET
                profession_slug      = excluded.profession_slug,
                output_item_id       = excluded.output_item_id,
                output_quantity      = excluded.output_quantity,
                recipe_name          = excluded.recipe_name,
                skill_level_required = excluded.skill_level_required,
                expansion_slug       = excluded.expansion_slug,
                source               = excluded.source
            """,
            (
                recipe_id, profession_slug, output_item_id, output_quantity,
                recipe_name, skill_level_required, expansion_slug, source,
            ),
        )

    def replace_reagents(
        self,
        recipe_id: int,
        reagents: list[tuple[int, int, str]],
    ) -> None:
        """Replace all reagents for a recipe.

        Args:
            recipe_id: Recipe to update.
            reagents:  List of (ingredient_item_id, quantity, reagent_type).
        """
        self._conn.execute(
            "DELETE FROM recipe_reagents WHERE recipe_id = ?;", (recipe_id,)
        )
        self._conn.executemany(
            """
            INSERT INTO recipe_reagents (recipe_id, ingredient_item_id, quantity, reagent_type)
            VALUES (?, ?, ?, ?)
            """,
            [(recipe_id, iid, qty, rtype) for iid, qty, rtype in reagents],
        )


# ── Row mappers ────────────────────────────────────────────────────────────────

def _row_to_recipe(row) -> RecipeRow:
    return RecipeRow(
        recipe_id=int(row[0]),
        profession_slug=row[1],
        output_item_id=int(row[2]),
        output_quantity=int(row[3]),
        recipe_name=row[4],
        skill_level_required=int(row[5]),
        expansion_slug=row[6],
        source=row[7],
    )


def _row_to_reagent(row) -> ReagentRow:
    return ReagentRow(
        recipe_id=int(row[0]),
        ingredient_item_id=int(row[1]),
        quantity=int(row[2]),
        reagent_type=row[3],
    )
