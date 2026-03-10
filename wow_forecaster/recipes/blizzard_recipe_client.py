"""
Blizzard static Game Data API — profession and recipe data.

Thin wrapper around BlizzardClient that normalises profession/recipe API
responses into plain dicts suitable for upsert into the DB.

Key normalisation decisions:
  - Only ``required`` reagents are extracted (optional/finishing excluded).
  - ``crafted_item`` is checked first for output_item_id; falls back to
    ``alliance_crafted_item`` / ``horde_crafted_item`` for faction-specific
    recipes (rare, mostly old content).
  - For TWW/Midnight recipes where ``crafted_item`` is absent, output is
    resolved via ``BlizzardClient.search_item_by_name(recipe_name)``.
    Primary materials from ``modified_crafting_slots`` are also extracted;
    optional quality/embellishment slots are filtered by keyword.
    Slot quantities are assumed to be 1 (not exposed in the recipe API).
  - Recipes with no resolvable output item are silently dropped.
  - ``skill_level_required`` is set to 0 when absent (field is optional in API).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Slot type name keywords that indicate optional/quality-only reagent slots in
# the modified crafting system (TWW/Midnight).  These slots do not represent
# primary crafting materials and are excluded from ingredient cost calculations.
_OPTIONAL_SLOT_KEYWORDS: frozenset[str] = frozenset({
    "authenticity",   # Artisan's Authenticity — crafting quality reagent
    "embellishment",  # Add Embellishment — gear customisation slot
    "acuity",         # Artisan's Acuity — profession currency
    "empower",        # Empower — gear power level slot
    "customize",      # Customize Secondary Stats
    "moxie",          # Artisan's Moxie — profession stat knowledge
})


def _is_optional_slot(slot_name: str) -> bool:
    """Return True if the slot is a quality/optional reagent (not a primary material)."""
    lower = slot_name.lower()
    return any(kw in lower for kw in _OPTIONAL_SLOT_KEYWORDS)


# Known TWW-era profession slugs mapped from display names returned by API.
# Used to normalise "Alchemy" -> "alchemy" etc.
_PROFESSION_SLUG_MAP: dict[str, str] = {
    "Alchemy":       "alchemy",
    "Blacksmithing": "blacksmithing",
    "Cooking":       "cooking",
    "Enchanting":    "enchanting",
    "Engineering":   "engineering",
    "Fishing":       "fishing",
    "Herbalism":     "herbalism",
    "Inscription":   "inscription",
    "Jewelcrafting": "jewelcrafting",
    "Leatherworking": "leatherworking",
    "Mining":        "mining",
    "Skinning":      "skinning",
    "Tailoring":     "tailoring",
}


@dataclass
class NormalisedRecipe:
    """A recipe record ready for DB upsert."""

    recipe_id: int
    profession_slug: str
    output_item_id: int
    output_quantity: int
    recipe_name: str
    skill_level_required: int
    expansion_slug: str
    reagents: list[NormalisedReagent] = field(default_factory=list)


@dataclass
class NormalisedReagent:
    """A required reagent for a recipe."""

    ingredient_item_id: int
    quantity: int
    reagent_type: str = "required"


def normalise_profession_name(api_name: str) -> str:
    """Map an API display name to a slug (lower-case, no spaces)."""
    return _PROFESSION_SLUG_MAP.get(api_name, api_name.lower().replace(" ", "_"))


def fetch_all_recipes_for_profession(
    client,
    profession_id: int,
    profession_name: str,
    expansion_slug: str,
    target_tier_keywords: list[str] | None = None,
) -> list[NormalisedRecipe]:
    """Fetch and normalise all recipes for a profession, filtered to relevant tiers.

    Iterates skill tiers, collects recipe stubs, then fetches each full recipe.
    Only tiers whose name contains any keyword in ``target_tier_keywords`` are
    fetched (case-insensitive).  If ``target_tier_keywords`` is None, all tiers
    are fetched.

    Args:
        client:               BlizzardClient instance (authenticated).
        profession_id:        Blizzard profession ID.
        profession_name:      Display name (e.g. "Alchemy").
        expansion_slug:       e.g. "midnight", "tww".
        target_tier_keywords: Subset of tier names to include (e.g. ["Midnight"]).

    Returns:
        List of NormalisedRecipe (with reagents populated).
    """
    profession_slug = normalise_profession_name(profession_name)
    prof_data = client.fetch_profession(profession_id)

    skill_tiers = prof_data.get("skill_tiers", [])
    results: list[NormalisedRecipe] = []

    for tier in skill_tiers:
        tier_name: str = tier.get("name", "")
        tier_id: int = tier.get("id", 0)

        if target_tier_keywords is not None:
            keywords_lower = [k.lower() for k in target_tier_keywords]
            if not any(kw in tier_name.lower() for kw in keywords_lower):
                logger.debug(
                    "Skipping tier '%s' (id=%d) for profession %s",
                    tier_name, tier_id, profession_slug,
                )
                continue

        logger.info(
            "Fetching tier '%s' (id=%d) for profession %s",
            tier_name, tier_id, profession_slug,
        )

        try:
            tier_data = client.fetch_profession_skill_tier(profession_id, tier_id)
        except Exception as exc:
            logger.warning(
                "Failed to fetch tier %d for profession %d: %s", tier_id, profession_id, exc
            )
            continue

        recipe_stubs = _collect_recipe_stubs(tier_data)
        logger.info(
            "  Found %d recipe stubs in tier '%s'", len(recipe_stubs), tier_name
        )

        for stub_id, stub_name in recipe_stubs:
            try:
                recipe_data = client.fetch_recipe(stub_id)
            except Exception as exc:
                logger.warning("Failed to fetch recipe %d (%s): %s", stub_id, stub_name, exc)
                continue

            skill_level = _extract_skill_level(tier_data, stub_id)
            normalised = _normalise_recipe(
                recipe_data,
                profession_slug=profession_slug,
                expansion_slug=expansion_slug,
                skill_level_required=skill_level,
            )

            # Fallback for TWW/Midnight recipes where crafted_item is absent from API.
            if normalised is None and "crafted_item" not in recipe_data:
                normalised = _resolve_recipe_by_name(
                    recipe_data=recipe_data,
                    profession_slug=profession_slug,
                    expansion_slug=expansion_slug,
                    skill_level_required=skill_level,
                    client=client,
                )

            if normalised is not None:
                results.append(normalised)

    logger.info(
        "Profession %s: collected %d recipes with output items",
        profession_slug, len(results),
    )
    return results


def _resolve_recipe_by_name(
    recipe_data: dict,
    profession_slug: str,
    expansion_slug: str,
    skill_level_required: int,
    client,
) -> "NormalisedRecipe | None":
    """Resolve a recipe where ``crafted_item`` is absent via item name search.

    Used for TWW/Midnight recipes where the Blizzard static API no longer
    includes the output item in the recipe response.

    Output item is found by searching for an item whose English name exactly
    matches the recipe name.  Primary materials are extracted from both the
    ``reagents`` field (fixed-quantity catalysts) and ``modified_crafting_slots``
    (primary materials like herbs, ores, vials).  Optional quality/embellishment
    slots are skipped.  Slot quantities default to 1 (not available in the API).

    Args:
        recipe_data:          Raw Blizzard recipe API response.
        profession_slug:      e.g. "alchemy".
        expansion_slug:       e.g. "midnight".
        skill_level_required: Skill level (0 if unknown).
        client:               Authenticated BlizzardClient (for name search).

    Returns:
        NormalisedRecipe on success, or None if output item cannot be resolved.
    """
    recipe_id = recipe_data.get("id")
    recipe_name = (recipe_data.get("name") or "").strip()
    if not recipe_id or not recipe_name:
        return None

    output_item_id = client.search_item_by_name(recipe_name)
    if output_item_id is None:
        logger.debug(
            "Name lookup failed for recipe %d ('%s') — skipped",
            recipe_id, recipe_name,
        )
        return None

    # Fixed-quantity reagents (quality catalysts, Mote of Primal Energy, etc.)
    reagents: list[NormalisedReagent] = []
    for r in recipe_data.get("reagents", []):
        reagent_item = r.get("reagent", {})
        item_id = reagent_item.get("id")
        quantity = r.get("quantity", 1) or 1
        if item_id:
            reagents.append(
                NormalisedReagent(
                    ingredient_item_id=int(item_id),
                    quantity=int(quantity),
                    reagent_type="required",
                )
            )

    # Primary materials from modified_crafting_slots (herbs, ores, vials, etc.).
    # Slot quantities are not in the API response; assumed to be 1.
    for slot in recipe_data.get("modified_crafting_slots", []):
        slot_name = (slot.get("slot_type", {}).get("name") or "").strip()
        if not slot_name or _is_optional_slot(slot_name):
            continue
        slot_item_id = client.search_item_by_name(slot_name)
        if slot_item_id is not None and slot_item_id != output_item_id:
            reagents.append(
                NormalisedReagent(
                    ingredient_item_id=slot_item_id,
                    quantity=1,
                    reagent_type="required",
                )
            )

    return NormalisedRecipe(
        recipe_id=int(recipe_id),
        profession_slug=profession_slug,
        output_item_id=output_item_id,
        output_quantity=1,
        recipe_name=recipe_name,
        skill_level_required=skill_level_required,
        expansion_slug=expansion_slug,
        reagents=reagents,
    )


def _collect_recipe_stubs(tier_data: dict) -> list[tuple[int, str]]:
    """Extract (recipe_id, recipe_name) stubs from a skill tier response."""
    stubs: list[tuple[int, str]] = []
    for category in tier_data.get("categories", []):
        for recipe in category.get("recipes", []):
            r_id = recipe.get("id")
            r_name = recipe.get("name", "")
            if r_id:
                stubs.append((int(r_id), r_name))
    return stubs


def _normalise_recipe(
    data: dict,
    profession_slug: str,
    expansion_slug: str,
    skill_level_required: int = 0,
) -> NormalisedRecipe | None:
    """Normalise a raw Blizzard recipe API response.

    Returns None if the recipe has no resolvable crafted item.
    """
    recipe_id = data.get("id")
    if not recipe_id:
        return None

    recipe_name = data.get("name", "")

    # Resolve output item
    output_item_id: int | None = None
    output_quantity: int = 1

    crafted = data.get("crafted_item")
    if crafted:
        output_item_id = crafted.get("id")
        output_quantity = crafted.get("quantity", 1) or 1
    else:
        # Faction-specific fallback (rare)
        for key in ("alliance_crafted_item", "horde_crafted_item"):
            alt = data.get(key)
            if alt and alt.get("id"):
                output_item_id = alt["id"]
                output_quantity = alt.get("quantity", 1) or 1
                break

    if output_item_id is None:
        logger.debug(
            "Recipe %d (%s) has no crafted_item — skipped",
            recipe_id, recipe_name,
        )
        return None

    # Extract required reagents only
    reagents: list[NormalisedReagent] = []
    for r in data.get("reagents", []):
        reagent_item = r.get("reagent", {})
        item_id = reagent_item.get("id")
        quantity = r.get("quantity", 1) or 1
        if item_id:
            reagents.append(
                NormalisedReagent(
                    ingredient_item_id=int(item_id),
                    quantity=int(quantity),
                    reagent_type="required",
                )
            )

    return NormalisedRecipe(
        recipe_id=int(recipe_id),
        profession_slug=profession_slug,
        output_item_id=int(output_item_id),
        output_quantity=int(output_quantity),
        recipe_name=recipe_name,
        skill_level_required=skill_level_required,
        expansion_slug=expansion_slug,
        reagents=reagents,
    )


def _extract_skill_level(tier_data: dict, recipe_id: int) -> int:
    """Attempt to extract skill_level_required for a recipe from tier data.

    The Blizzard skill-tier endpoint does not include skill level per recipe
    in a consistent field; returns 0 when unavailable.
    """
    for category in tier_data.get("categories", []):
        for recipe in category.get("recipes", []):
            if recipe.get("id") == recipe_id:
                return int(recipe.get("skill_level_required", 0) or 0)
    return 0
