"""
Item archetype taxonomy for cross-expansion economic behavior transfer.

Hierarchy: ``ArchetypeCategory`` (top-level) → ``ArchetypeTag`` (behavior group).

Design principle: archetypes represent **economic behavior**, not item identity.
A ``consumable.flask.stat`` in TWW and the equivalent in Midnight are different
items but share the same economic archetype — crafted consumable, ~1 hour duration,
consumed by raiders on progression nights, price spikes before RTWF, crashes at
season end. This abstraction powers the TWW → Midnight transfer learning.

The ``CATEGORY_TAG_MAP`` dict is the canonical integrity contract:
  - Every ``ArchetypeCategory`` must have an entry.
  - Every ``ArchetypeTag`` must appear in exactly one category's list.
  - Tags for a category must share the category's slug prefix.

Run ``tests/test_taxonomy/test_archetype_taxonomy.py`` to verify this contract.

This module has NO imports from any other ``wow_forecaster`` package.
"""

from enum import StrEnum


class ArchetypeCategory(StrEnum):
    """Top-level economic category for item grouping."""

    CONSUMABLE = "consumable"
    """Items consumed on use: flasks, potions, food buffs, runes."""

    CRAFTING_MAT = "mat"
    """Raw materials gathered from the world and used in crafting recipes."""

    GEAR = "gear"
    """Equippable items: BOE drops, crafted gear sets, optional reagents."""

    ENCHANT = "enchant"
    """Permanent enhancements applied to gear slots."""

    GEM = "gem"
    """Gems socketed into gear for stat bonuses."""

    PROFESSION_TOOL = "prof_tool"
    """Profession-specific tools and accessories (Dragonflight+ professions)."""

    REAGENT = "reagent"
    """Crafting reagents consumed in the crafting process (not the final product)."""

    TRADE_GOOD = "trade_good"
    """General tradeable commodities: vendor-quality materials, misc AH goods."""

    SERVICE = "service"
    """Crafting services (crafting order fulfillment) — placeholder for future."""

    COLLECTION = "collection"
    """Pets, mounts, transmog items — long-tail AH market segment."""


class ArchetypeTag(StrEnum):
    """Specific behavior-based archetype tag within a category.

    Naming convention: ``<category_slug>.<sub_group>[.<detail>]``
    This prefix scheme allows filtering all tags for a category via string prefix.
    """

    # ── Consumable ─────────────────────────────────────────────────────────────
    CONSUMABLE_FLASK_STAT = "consumable.flask.stat"
    """Primary stat flasks (Intellect, Agility, Strength, Stamina variants).
    Demand driver: raid nights, RTWF. Supply: Alchemy crafting."""

    CONSUMABLE_FLASK_UTILITY = "consumable.flask.utility"
    """Utility flasks (resistance, underwater breathing, etc.)."""

    CONSUMABLE_POTION_COMBAT = "consumable.potion.combat"
    """On-use DPS/healing/mana potions consumed mid-combat.
    Limit 1 per fight in retail; demand concentrated in progression content."""

    CONSUMABLE_POTION_UTILITY = "consumable.potion.utility"
    """Non-combat potions: invisibility, speed, cauldrons, etc."""

    CONSUMABLE_FOOD_STAT = "consumable.food.stat"
    """Individual stat buff food (secondary stats, haste, crit, etc.)."""

    CONSUMABLE_FOOD_FEAST = "consumable.food.feast"
    """Feasts that feed an entire raid group; higher mat cost, higher value."""

    CONSUMABLE_AUGMENT_RUNE = "consumable.augment_rune"
    """Augmentation runes providing a primary stat boost for 1 hour."""

    CONSUMABLE_SCROLL = "consumable.scroll"
    """Stat scrolls and miscellaneous short-duration consumable buffs."""

    # ── Crafting Mat ──────────────────────────────────────────────────────────
    MAT_ORE_COMMON = "mat.ore.common"
    """High-volume farmed ore (primary mining output in current content tier)."""

    MAT_ORE_RARE = "mat.ore.rare"
    """Rare ore nodes; lower supply, higher per-unit value."""

    MAT_HERB_COMMON = "mat.herb.common"
    """High-volume farmed herbs (Alchemy, Inscription inputs)."""

    MAT_HERB_RARE = "mat.herb.rare"
    """Rare herb spawns; used in top-tier alchemy recipes."""

    MAT_CLOTH = "mat.cloth"
    """Cloth drops from humanoid mobs; Tailoring and Bandage input."""

    MAT_LEATHER_COMMON = "mat.leather.common"
    """Standard leather from skinning; Leatherworking bulk input."""

    MAT_LEATHER_RARE = "mat.leather.rare"
    """Rare or specialty leather (e.g., thick, exotic); higher-tier LW recipes."""

    MAT_CRYSTAL_ENCHANTING = "mat.crystal.enchanting"
    """High-level disenchant output; top-tier enchant recipes."""

    MAT_DUST_ENCHANTING = "mat.dust.enchanting"
    """Common disenchant output; bulk enchanting material."""

    MAT_FISH = "mat.fish"
    """Fish for food crafting; volatile price tied to feast recipes."""

    MAT_ELEMENTAL = "mat.elemental"
    """Elemental drops (fire, water, earth, air); used across professions."""

    MAT_ESSENCE = "mat.essence"
    """Prismatic, Cosmic, or expansion-specific essences; cross-profession input."""

    # ── Gear ─────────────────────────────────────────────────────────────────
    GEAR_BOE_ENDGAME = "gear.boe.endgame"
    """High-ilvl Bind-on-Equip drops; most expensive individual AH items.
    Price peaks at patch/season start, falls as tier progresses."""

    GEAR_BOE_LEVELING = "gear.boe.leveling"
    """BOE gear for leveling alts; expansion launch demand spike."""

    GEAR_CRAFTED_ENDGAME = "gear.crafted.endgame"
    """Endgame crafted gear sets (e.g., Dragonflight Spark gear).
    Embellishment slots drive demand; new season resets demand."""

    GEAR_CRAFTED_LEVELING = "gear.crafted.leveling"
    """Crafted leveling gear; expansion launch + alt leveling demand."""

    GEAR_OPTIONAL_REAGENT = "gear.optional_reagent"
    """Optional reagents (embellishments, missives, finishing reagents)
    that modify crafted gear stats."""

    # ── Enchant ───────────────────────────────────────────────────────────────
    ENCHANT_WEAPON = "enchant.weapon"
    """Weapon enchants; high demand at season start and content patch."""

    ENCHANT_ARMOR_SLOT = "enchant.armor_slot"
    """Chest, boots, bracers, rings, cloak enchants."""

    ENCHANT_SECONDARY_STAT = "enchant.secondary_stat"
    """Haste/crit/mastery/versatility enchants; demand tied to BiS lists."""

    # ── Gem ──────────────────────────────────────────────────────────────────
    GEM_PRIMARY_STAT = "gem.primary_stat"
    """Primary stat gems (Int, Agi, Str, Stam); demand at content release."""

    GEM_SECONDARY_STAT = "gem.secondary_stat"
    """Secondary stat or special gems; demand driven by socket count in tier."""

    GEM_META = "gem.meta"
    """Meta gems with conditional bonuses (classic/WotLK era; placeholder)."""

    # ── Profession Tool ───────────────────────────────────────────────────────
    PROF_TOOL_MAIN = "prof_tool.main"
    """Primary profession tool (e.g., Dragonflight Blacksmithing Hammer)."""

    PROF_TOOL_ACCESSORY = "prof_tool.accessory"
    """Secondary tool or accessory providing bonus skill or specialization."""

    # ── Reagent ───────────────────────────────────────────────────────────────
    REAGENT_ALCHEMY = "reagent.alchemy"
    """Alchemy-specific crafting reagents (vials, solvent, etc.)."""

    REAGENT_INSCRIPTION = "reagent.inscription"
    """Inscription-specific reagents (inks, parchments)."""

    REAGENT_JEWELCRAFTING = "reagent.jewelcrafting"
    """Jewelcrafting-specific setting reagents."""

    REAGENT_ENGINEERING = "reagent.engineering"
    """Engineering components and schematics materials."""

    REAGENT_UNIVERSAL = "reagent.universal"
    """Cross-profession reagents (e.g., Rousing/Awakened elements in Dragonflight,
    or their Midnight equivalents). High volume, stable demand."""

    # ── Trade Good ────────────────────────────────────────────────────────────
    TRADE_GOOD_COMMODITY = "trade_good.commodity"
    """Stackable, fungible commodities priced per-unit on the AH."""

    TRADE_GOOD_UNIQUE = "trade_good.unique"
    """Non-stackable tradeable goods; bags, miscellaneous equipment."""


# ── Integrity contract ────────────────────────────────────────────────────────

CATEGORY_TAG_MAP: dict[ArchetypeCategory, list[ArchetypeTag]] = {
    ArchetypeCategory.CONSUMABLE: [
        ArchetypeTag.CONSUMABLE_FLASK_STAT,
        ArchetypeTag.CONSUMABLE_FLASK_UTILITY,
        ArchetypeTag.CONSUMABLE_POTION_COMBAT,
        ArchetypeTag.CONSUMABLE_POTION_UTILITY,
        ArchetypeTag.CONSUMABLE_FOOD_STAT,
        ArchetypeTag.CONSUMABLE_FOOD_FEAST,
        ArchetypeTag.CONSUMABLE_AUGMENT_RUNE,
        ArchetypeTag.CONSUMABLE_SCROLL,
    ],
    ArchetypeCategory.CRAFTING_MAT: [
        ArchetypeTag.MAT_ORE_COMMON,
        ArchetypeTag.MAT_ORE_RARE,
        ArchetypeTag.MAT_HERB_COMMON,
        ArchetypeTag.MAT_HERB_RARE,
        ArchetypeTag.MAT_CLOTH,
        ArchetypeTag.MAT_LEATHER_COMMON,
        ArchetypeTag.MAT_LEATHER_RARE,
        ArchetypeTag.MAT_CRYSTAL_ENCHANTING,
        ArchetypeTag.MAT_DUST_ENCHANTING,
        ArchetypeTag.MAT_FISH,
        ArchetypeTag.MAT_ELEMENTAL,
        ArchetypeTag.MAT_ESSENCE,
    ],
    ArchetypeCategory.GEAR: [
        ArchetypeTag.GEAR_BOE_ENDGAME,
        ArchetypeTag.GEAR_BOE_LEVELING,
        ArchetypeTag.GEAR_CRAFTED_ENDGAME,
        ArchetypeTag.GEAR_CRAFTED_LEVELING,
        ArchetypeTag.GEAR_OPTIONAL_REAGENT,
    ],
    ArchetypeCategory.ENCHANT: [
        ArchetypeTag.ENCHANT_WEAPON,
        ArchetypeTag.ENCHANT_ARMOR_SLOT,
        ArchetypeTag.ENCHANT_SECONDARY_STAT,
    ],
    ArchetypeCategory.GEM: [
        ArchetypeTag.GEM_PRIMARY_STAT,
        ArchetypeTag.GEM_SECONDARY_STAT,
        ArchetypeTag.GEM_META,
    ],
    ArchetypeCategory.PROFESSION_TOOL: [
        ArchetypeTag.PROF_TOOL_MAIN,
        ArchetypeTag.PROF_TOOL_ACCESSORY,
    ],
    ArchetypeCategory.REAGENT: [
        ArchetypeTag.REAGENT_ALCHEMY,
        ArchetypeTag.REAGENT_INSCRIPTION,
        ArchetypeTag.REAGENT_JEWELCRAFTING,
        ArchetypeTag.REAGENT_ENGINEERING,
        ArchetypeTag.REAGENT_UNIVERSAL,
    ],
    ArchetypeCategory.TRADE_GOOD: [
        ArchetypeTag.TRADE_GOOD_COMMODITY,
        ArchetypeTag.TRADE_GOOD_UNIQUE,
    ],
    # Placeholder categories — no tags defined yet (future expansion)
    ArchetypeCategory.SERVICE: [],
    ArchetypeCategory.COLLECTION: [],
}
