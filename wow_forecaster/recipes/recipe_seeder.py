"""
Recipe seeder — fetches profession/recipe data from Blizzard static API
and upserts into the ``recipes`` + ``recipe_reagents`` DB tables.

Expansion slug mapping to Blizzard tier keywords:
  "midnight"    -> "Midnight"   (or whatever Blizzard names the tier)
  "tww"         -> "The War Within"
  "dragonflight"-> "Dragon Isles"

The keyword match is best-effort: if a tier's display name contains the
keyword (case-insensitive), it is included.  Pass ``tier_keywords=[""]``
to include all tiers (wildcard).

Usage::

    from wow_forecaster.db.connection import get_connection
    from wow_forecaster.ingestion.blizzard_client import BlizzardClient
    from wow_forecaster.recipes.recipe_seeder import RecipeSeeder

    client = BlizzardClient(client_id=..., client_secret=...)
    with get_connection(db_path) as conn:
        seeder = RecipeSeeder(conn, client)
        stats = seeder.seed(expansion_slug="midnight", professions=["alchemy"])
        print(stats)
"""

from __future__ import annotations

import logging
import sqlite3
import time
from dataclasses import dataclass, field

from wow_forecaster.recipes.blizzard_recipe_client import (
    NormalisedRecipe,
    fetch_all_recipes_for_profession,
)
from wow_forecaster.recipes.recipe_repo import RecipeRepository

logger = logging.getLogger(__name__)

# Blizzard profession name -> Blizzard API profession ID (static, as of TWW/Midnight)
# https://develop.battle.net/documentation/world-of-warcraft/game-data-apis
_PROFESSION_IDS: dict[str, int] = {
    "alchemy":        171,
    "blacksmithing":  164,
    "cooking":        185,
    "enchanting":     333,
    "engineering":    202,
    "fishing":        356,
    "herbalism":      182,
    "inscription":    773,
    "jewelcrafting":  755,
    "leatherworking": 165,
    "mining":         186,
    "skinning":       393,
    "tailoring":      197,
}

# Expansion slug -> keywords to match against Blizzard skill tier display names
_EXPANSION_TIER_KEYWORDS: dict[str, list[str]] = {
    "midnight":     ["midnight"],
    "tww":          ["war within"],
    "dragonflight": ["dragon isles"],
    "shadowlands":  ["shadowlands"],
    "bfa":          ["kul tiran", "zandalari", "battle for azeroth"],
    "legion":       ["legion"],
    "wod":          ["warlords", "draenor"],
    "mop":          ["pandaria"],
    "cata":         ["cataclysm"],
    "wotlk":        ["northrend"],
    "tbc":          ["outland"],
    "classic":      ["classic"],
}


@dataclass
class SeedStats:
    """Summary of a seeding run."""

    expansion_slug: str
    professions_attempted: list[str] = field(default_factory=list)
    recipes_upserted: int = 0
    recipes_skipped: int = 0
    errors: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        return (
            f"SeedStats(expansion={self.expansion_slug}, "
            f"professions={len(self.professions_attempted)}, "
            f"upserted={self.recipes_upserted}, "
            f"skipped={self.recipes_skipped}, "
            f"errors={len(self.errors)})"
        )


class RecipeSeeder:
    """Seeds recipe data from Blizzard static API into the DB.

    Args:
        conn:    Open sqlite3 connection with FK enforcement on.
        client:  Authenticated BlizzardClient.
        delay_s: Seconds to wait between recipe fetches (rate limiting).
    """

    def __init__(
        self,
        conn: sqlite3.Connection,
        client,
        delay_s: float = 0.1,
    ) -> None:
        self._conn = conn
        self._client = client
        self._repo = RecipeRepository(conn)
        self._delay_s = delay_s

    def seed(
        self,
        expansion_slug: str,
        professions: list[str] | None = None,
        tier_keywords: list[str] | None = None,
    ) -> SeedStats:
        """Seed recipes for the given expansion and professions.

        Args:
            expansion_slug: e.g. "midnight", "tww". Used to tag recipes and
                            filter skill tiers unless ``tier_keywords`` overrides.
            professions:    Slugs to seed (e.g. ["alchemy", "enchanting"]).
                            None = all professions in ``_PROFESSION_IDS``.
            tier_keywords:  Override tier name matching keywords. None = use
                            the default mapping for ``expansion_slug``.
                            Pass ``[""]`` to include ALL tiers (full history).

        Returns:
            SeedStats summary.
        """
        stats = SeedStats(expansion_slug=expansion_slug)

        target_profs = professions or list(_PROFESSION_IDS.keys())
        keywords = tier_keywords if tier_keywords is not None else (
            _EXPANSION_TIER_KEYWORDS.get(expansion_slug)
        )
        if keywords is None:
            # Unknown expansion — include all tiers
            keywords = [""]
            logger.warning(
                "No tier keyword mapping for expansion '%s'; including all tiers.",
                expansion_slug,
            )

        # Fetch profession index to resolve IDs by name for any new professions
        all_profession_ids = self._resolve_profession_ids(target_profs)

        for prof_slug in target_profs:
            prof_id = all_profession_ids.get(prof_slug)
            if prof_id is None:
                logger.warning("No profession ID found for '%s' — skipping.", prof_slug)
                stats.errors.append(f"no_id:{prof_slug}")
                continue

            stats.professions_attempted.append(prof_slug)
            logger.info("Seeding profession: %s (id=%d)", prof_slug, prof_id)

            try:
                recipes = fetch_all_recipes_for_profession(
                    client=self._client,
                    profession_id=prof_id,
                    profession_name=prof_slug,
                    expansion_slug=expansion_slug,
                    target_tier_keywords=keywords,
                )
            except Exception as exc:
                logger.error("Failed to fetch profession %s: %s", prof_slug, exc)
                stats.errors.append(f"fetch_error:{prof_slug}:{exc}")
                continue

            upserted = self._upsert_recipes(recipes, stats)
            logger.info(
                "Profession %s: %d recipes upserted", prof_slug, upserted
            )
            if self._delay_s > 0:
                time.sleep(self._delay_s)

        self._conn.commit()
        logger.info("Recipe seeding complete. %s", stats)
        return stats

    def _resolve_profession_ids(self, prof_slugs: list[str]) -> dict[str, int]:
        """Build profession_slug -> profession_id mapping.

        Starts from the hardcoded ``_PROFESSION_IDS`` table (reliable).
        Falls back to fetching the profession index from the API only when
        a slug is missing from the hardcoded map.
        """
        result: dict[str, int] = {}
        missing: list[str] = []

        for slug in prof_slugs:
            if slug in _PROFESSION_IDS:
                result[slug] = _PROFESSION_IDS[slug]
            else:
                missing.append(slug)

        if missing:
            logger.info(
                "Fetching profession index to resolve unknown slugs: %s", missing
            )
            try:
                profs = self._client.fetch_professions()
                for prof in profs:
                    name = prof.get("name", "")
                    pid = prof.get("id")
                    if not name or not pid:
                        continue
                    slug = name.lower().replace(" ", "_")
                    if slug in missing:
                        result[slug] = int(pid)
            except Exception as exc:
                logger.error("Failed to fetch profession index: %s", exc)

        return result

    def _upsert_recipes(
        self, recipes: list[NormalisedRecipe], stats: SeedStats
    ) -> int:
        count = 0
        for recipe in recipes:
            try:
                self._repo.upsert_recipe(
                    recipe_id=recipe.recipe_id,
                    profession_slug=recipe.profession_slug,
                    output_item_id=recipe.output_item_id,
                    output_quantity=recipe.output_quantity,
                    recipe_name=recipe.recipe_name,
                    skill_level_required=recipe.skill_level_required,
                    expansion_slug=recipe.expansion_slug,
                )
                self._repo.replace_reagents(
                    recipe_id=recipe.recipe_id,
                    reagents=[
                        (r.ingredient_item_id, r.quantity, r.reagent_type)
                        for r in recipe.reagents
                    ],
                )
                stats.recipes_upserted += 1
                count += 1
                if self._delay_s > 0:
                    time.sleep(self._delay_s)
            except Exception as exc:
                logger.warning(
                    "Failed to upsert recipe %d: %s", recipe.recipe_id, exc
                )
                stats.recipes_skipped += 1
                stats.errors.append(f"upsert_error:{recipe.recipe_id}:{exc}")
        return count
