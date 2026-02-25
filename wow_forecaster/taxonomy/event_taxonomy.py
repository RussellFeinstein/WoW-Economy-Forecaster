"""
Event taxonomy for WoW economy events.

Three orthogonal dimensions describe every event:
  - ``EventType``     — the *what*: what kind of event is this?
  - ``EventScope``    — the *who*:  which players/realms are affected?
  - ``EventSeverity`` — the *how much*: expected market impact magnitude?

``ImpactDirection`` describes per-archetype price movement, used in the
``event_archetype_impacts`` DB table.

Usage example::

    from wow_forecaster.taxonomy.event_taxonomy import EventType, EventSeverity

    event_type = EventType.RTWF
    severity   = EventSeverity.MAJOR

This module has NO imports from any other ``wow_forecaster`` package.
"""

from enum import StrEnum


class EventType(StrEnum):
    """Category of in-game or real-world event affecting AH economy."""

    # ── Expansion lifecycle ───────────────────────────────────────────────────
    EXPANSION_LAUNCH = "expansion_launch"
    """New expansion goes live; largest possible demand shock."""

    EXPANSION_PREPATCH = "expansion_prepatch"
    """Pre-patch system changes (talent trees, class tuning, new currencies)."""

    EXPANSION_ANNOUNCEMENT = "expansion_announcement"
    """BlizzCon or press reveal of upcoming expansion; speculative buying begins."""

    # ── Patch lifecycle ───────────────────────────────────────────────────────
    MAJOR_PATCH = "major_patch"
    """x.1, x.2 content patches introducing new raids, dungeons, systems."""

    MINOR_PATCH = "minor_patch"
    """x.0.5 tuning patches; often recipe/drop-rate adjustments."""

    HOTFIX = "hotfix"
    """Server-side fixes that can affect item availability or crafting costs."""

    # ── Season lifecycle ──────────────────────────────────────────────────────
    SEASON_START = "season_start"
    """M+, PvP, or crafting season begins; consumable demand spikes."""

    SEASON_END = "season_end"
    """Season ends; crafting mat demand drops as progression stalls."""

    # ── Competitive events ────────────────────────────────────────────────────
    RTWF = "rtwf"
    """Race to World First; extreme consumable demand for ~2 weeks."""

    ARENA_TOURNAMENT = "arena_tournament"
    """Arena/PvP tournament; enchant, gem, and trinket demand spike."""

    # ── In-game recurring events ──────────────────────────────────────────────
    HOLIDAY_EVENT = "holiday_event"
    """Darkmoon Faire, Winter Veil, Brewfest, Hallow's End, etc."""

    WORLD_BOSS = "world_boss"
    """World boss availability window; rare mat / loot table impacts."""

    BONUS_WEEK = "bonus_week"
    """Timewalking, M+ bonus loot, PvP bonus weeks; targeted activity spikes."""

    TRADING_POST_RESET = "trading_post_reset"
    """Monthly Trading Post reset; craft mats for currency items in demand."""

    # ── Structured content ────────────────────────────────────────────────────
    NEW_RAID_TIER = "new_raid_tier"
    """New raid tier opens (Normal/Heroic first, then Mythic week)."""

    NEW_DUNGEON_POOL = "new_dungeon_pool"
    """M+ dungeon pool rotation; affects consumable tier (old = less demand)."""

    NEW_CRAFTING_SYSTEM = "new_crafting_system"
    """Major overhaul to a profession system (e.g., Dragonflight work orders)."""

    # ── Supply shocks ─────────────────────────────────────────────────────────
    ITEM_ADDED = "item_added"
    """New economy-relevant items introduced via patch or update."""

    ITEM_REMOVED = "item_removed"
    """Items removed, deprecated, or nerfed to near-zero AH value."""

    RECIPE_CHANGE = "recipe_change"
    """Crafting recipe material requirements changed; upstream mat prices shift."""

    DROP_RATE_CHANGE = "drop_rate_change"
    """Drop rate adjusted (buff or nerf) on a farmed mat or BOE item."""

    # ── Infrastructure ────────────────────────────────────────────────────────
    MAINTENANCE_WINDOW = "maintenance_window"
    """Scheduled weekly or patch maintenance; AH activity pause."""

    EMERGENCY_MAINTENANCE = "emergency_maintenance"
    """Unscheduled downtime; can cause pent-up demand on return."""

    SERVER_MERGE = "server_merge"
    """Realm consolidation; liquidity and price discovery changes."""

    CROSS_REALM_CHANGE = "cross_realm_change"
    """Connected realm additions or removals affecting market depth."""

    # ── External / meta ───────────────────────────────────────────────────────
    BLIZZCON = "blizzcon"
    """BlizzCon convention; speculation on reveals drives some mat movement."""

    CONTENT_DROUGHT = "content_drought"
    """Extended gap between patches; player activity and AH volume decline."""


class EventScope(StrEnum):
    """Spatial/demographic scope of event impact."""

    GLOBAL = "global"
    """All regions and all realms simultaneously affected."""

    REGION = "region"
    """One region (US, EU, KR, TW, CN) affected; server time differences matter."""

    REALM_CLUSTER = "realm_cluster"
    """Connected realm cluster only; server merge or local infrastructure event."""

    FACTION = "faction"
    """Alliance or Horde only within a realm (faction-specific content drops)."""


class EventSeverity(StrEnum):
    """Expected magnitude of AH price impact."""

    CRITICAL = "critical"
    """Expansion launch-level event; >50% price movement expected on affected archetypes."""

    MAJOR = "major"
    """RTWF, major patch; 20–50% price movement expected."""

    MODERATE = "moderate"
    """Season starts, holiday events; 10–20% movement expected."""

    MINOR = "minor"
    """Bonus weeks, minor patches, hotfixes; 5–10% movement expected."""

    NEGLIGIBLE = "negligible"
    """Maintenance windows, minor server changes; <5% movement expected."""


class ImpactDirection(StrEnum):
    """Direction of price/volume impact on a specific archetype during an event."""

    SPIKE = "spike"
    """Demand surge → prices and/or sales volume increase."""

    CRASH = "crash"
    """Supply flood or demand collapse → prices and/or volume drop."""

    MIXED = "mixed"
    """Some items within the archetype spike while others crash."""

    NEUTRAL = "neutral"
    """No meaningful expected impact on this archetype from this event."""
