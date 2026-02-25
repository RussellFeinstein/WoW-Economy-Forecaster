"""
Item and category models.

``ItemCategory`` supports a hierarchical slug-based tree (e.g.
``consumable`` → ``consumable.flask`` → ``consumable.flask.stat``) that is
expansion-aware but designed to remain stable across expansions so that
category-to-category mappings support the TWW → Midnight transfer.

``Item`` stores WoW canonical item IDs with economy-relevant metadata.
Items are never hard-coded by name — they are looked up by ID and categorized
through the ``ItemCategory`` and ``EconomicArchetype`` systems.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, field_validator

VALID_QUALITIES = frozenset({
    "poor", "common", "uncommon", "rare", "epic", "legendary", "artifact", "heirloom",
})

VALID_EXPANSIONS = frozenset({
    "classic", "tbc", "wotlk", "cata", "mop", "wod",
    "legion", "bfa", "shadowlands", "dragonflight", "tww", "midnight",
})


class ItemCategory(BaseModel):
    """A hierarchical item category node.

    Categories use a dot-delimited slug scheme so that prefix-matching
    gives all children of a node (e.g. ``slug.startswith("consumable.")``).

    Attributes:
        category_id: Auto-assigned database PK; ``None`` before DB insertion.
        slug: Unique identifier, e.g. ``"consumable.flask"`` or ``"mat.ore"``.
        display_name: Human-readable label shown in CLI / future UI.
        parent_slug: Parent category slug, or ``None`` for top-level nodes.
        archetype_tag: Corresponding ``ArchetypeTag`` string value for this node.
        expansion_slug: Expansion this category applies to, or ``None`` for all.
    """

    model_config = ConfigDict(frozen=True)

    category_id: Optional[int] = None
    slug: str
    display_name: str
    parent_slug: Optional[str] = None
    archetype_tag: str
    expansion_slug: Optional[str] = None

    @field_validator("expansion_slug")
    @classmethod
    def validate_expansion(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and v not in VALID_EXPANSIONS:
            raise ValueError(
                f"Unknown expansion '{v}'. Must be one of {sorted(VALID_EXPANSIONS)}."
            )
        return v

    @field_validator("slug")
    @classmethod
    def validate_slug_format(cls, v: str) -> str:
        if not v or " " in v or v != v.lower():
            raise ValueError(
                f"Category slug '{v}' must be lowercase, non-empty, and contain no spaces."
            )
        return v


class Item(BaseModel):
    """A WoW item with economy-relevant metadata.

    Items are identified by their WoW canonical ``item_id`` (the integer ID used
    by Blizzard's API). Do NOT hardcode expansion-specific item names here —
    use ``item_id`` lookups and the ``ItemCategory`` / ``EconomicArchetype``
    systems for cross-expansion reasoning.

    Attributes:
        item_id: WoW canonical item ID (not auto-incremented — use Blizzard's ID).
        name: Display name at time of registration (may change in patches).
        category_id: FK to ``item_categories.category_id``.
        archetype_id: FK to ``economic_archetypes.archetype_id``; ``None`` if
            not yet classified.
        expansion_slug: Expansion this item belongs to.
        quality: Item quality tier.
        is_crafted: ``True`` if produced by a player profession.
        is_boe: ``True`` if Bind-on-Equip (tradeable after looting).
        ilvl: Item level for gear items; ``None`` for non-gear.
        notes: Optional free-form annotation.
    """

    model_config = ConfigDict(frozen=True)

    item_id: int
    name: str
    category_id: int
    archetype_id: Optional[int] = None
    expansion_slug: str
    quality: str
    is_crafted: bool = False
    is_boe: bool = False
    ilvl: Optional[int] = None
    notes: Optional[str] = None

    @field_validator("quality")
    @classmethod
    def validate_quality(cls, v: str) -> str:
        if v not in VALID_QUALITIES:
            raise ValueError(
                f"Invalid quality '{v}'. Must be one of {sorted(VALID_QUALITIES)}."
            )
        return v

    @field_validator("expansion_slug")
    @classmethod
    def validate_expansion(cls, v: str) -> str:
        if v not in VALID_EXPANSIONS:
            raise ValueError(
                f"Unknown expansion '{v}'. Must be one of {sorted(VALID_EXPANSIONS)}."
            )
        return v

    @field_validator("ilvl")
    @classmethod
    def validate_ilvl(cls, v: Optional[int]) -> Optional[int]:
        if v is not None and v <= 0:
            raise ValueError(f"ilvl must be a positive integer, got {v}.")
        return v
