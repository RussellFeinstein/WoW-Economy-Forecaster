"""
Item bootstrapper: seeds item_categories, economic_archetypes, and items
from the Blizzard commodity AH snapshot + Blizzard Item API.

Flow
----
1. Seed item_categories  — one row per ArchetypeCategory (idempotent).
2. Seed economic_archetypes — one row per ArchetypeTag (idempotent).
3. Read unique item IDs from the latest commodity snapshot on disk.
4. Fetch item metadata from the Blizzard Item API (async, rate-limited).
5. Map item_class + item_subclass → archetype_tag → archetype_id + category_id.
6. Insert all items (ON CONFLICT IGNORE — safe to re-run).

Run this ONCE after init-db and before run-hourly-refresh so that commodity
records can pass the FK guard in the ingestion pipeline.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
from pathlib import Path
from typing import Optional

import httpx

logger = logging.getLogger(__name__)


# ── Blizzard Item Class / Subclass → ArchetypeTag ─────────────────────────────

# Blizzard item_class_id → default archetype tag
_CLASS_TO_TAG: dict[int, str] = {
    0:  "consumable.flask.stat",      # Consumable (default; refined by subclass)
    2:  "gear.boe.endgame",           # Weapon
    3:  "gem.primary_stat",           # Gem
    4:  "gear.boe.endgame",           # Armor
    7:  "trade_good.commodity",       # Tradegoods (refined by subclass)
    9:  "trade_good.commodity",       # Recipe
    11: "trade_good.unique",          # Quiver
    14: "trade_good.unique",          # Miscellaneous
    15: "enchant.armor_slot",         # Glyph
    16: "enchant.armor_slot",         # Equipment Enhancement (refined by subclass)
    19: "trade_good.commodity",       # Battle Pets → trade_good (not tracked)
    21: "prof_tool.main",             # Profession Equipment
}

# class_id=0 (Consumable) subclass_id → archetype tag
_CONSUMABLE_SUBCLASS_TO_TAG: dict[int, str] = {
    0: "consumable.potion.combat",    # Generic consumable
    1: "consumable.potion.combat",    # Potion
    2: "consumable.flask.stat",       # Elixir
    3: "consumable.flask.stat",       # Flask / Phial
    4: "consumable.scroll",           # Scroll
    5: "consumable.food.stat",        # Food & Drink
    6: "consumable.scroll",           # Bandage (first aid)
    7: "consumable.potion.utility",   # Item Enhancement (sharpening stones, etc.)
    8: "consumable.augment_rune",     # Augmentation Rune
    9: "consumable.potion.utility",   # Other consumable
}

# class_id=7 (Tradegoods) subclass_id → archetype tag
_TRADEGOODS_SUBCLASS_TO_TAG: dict[int, str] = {
    1:  "trade_good.commodity",       # Parts
    2:  "trade_good.commodity",       # Explosives & Devices
    4:  "reagent.jewelcrafting",      # Jewelcrafting
    5:  "mat.cloth",                  # Cloth
    6:  "mat.leather.common",         # Leather
    7:  "mat.ore.common",             # Metal & Stone
    8:  "mat.fish",                   # Cooking ingredients
    9:  "mat.herb.common",            # Herb
    10: "mat.elemental",              # Elemental
    11: "trade_good.commodity",       # Other
    12: "mat.dust.enchanting",        # Enchanting
    13: "mat.cloth",                  # Cloth (alternate subclass)
    16: "reagent.inscription",        # Inscription
}

# class_id=16 (Equipment Enhancement) subclass_id → archetype tag
_ENHANCEMENT_SUBCLASS_TO_TAG: dict[int, str] = {
    0:  "enchant.armor_slot",         # Head
    1:  "enchant.armor_slot",         # Neck
    2:  "enchant.armor_slot",         # Shoulder
    3:  "enchant.armor_slot",         # Cloak
    4:  "enchant.armor_slot",         # Chest
    5:  "enchant.armor_slot",         # Wrist
    6:  "enchant.armor_slot",         # Hands
    7:  "enchant.armor_slot",         # Waist
    8:  "enchant.armor_slot",         # Legs
    9:  "enchant.armor_slot",         # Feet
    10: "enchant.armor_slot",         # Ring
    11: "enchant.weapon",             # Weapon Enchantment
    12: "gear.optional_reagent",      # Embellishment
    13: "reagent.universal",          # Tinker (engineering slot)
}

DEFAULT_TAG = "trade_good.commodity"

# Display names for ArchetypeCategory slugs
_CATEGORY_DISPLAY: dict[str, str] = {
    "consumable": "Consumable",
    "mat":        "Crafting Material",
    "gear":       "Gear & Equipment",
    "enchant":    "Enchant & Enhancement",
    "gem":        "Gem",
    "prof_tool":  "Profession Tool",
    "reagent":    "Reagent",
    "trade_good": "Trade Good",
    "service":    "Service",
    "collection": "Collection",
}


def _resolve_archetype_tag(class_id: int, subclass_id: int) -> str:
    """Map Blizzard item class/subclass IDs to our archetype tag string."""
    if class_id == 0:
        return _CONSUMABLE_SUBCLASS_TO_TAG.get(
            subclass_id, _CLASS_TO_TAG.get(class_id, DEFAULT_TAG)
        )
    if class_id == 7:
        return _TRADEGOODS_SUBCLASS_TO_TAG.get(subclass_id, DEFAULT_TAG)
    if class_id == 16:
        return _ENHANCEMENT_SUBCLASS_TO_TAG.get(subclass_id, "enchant.armor_slot")
    return _CLASS_TO_TAG.get(class_id, DEFAULT_TAG)


def _item_expansion(item_id: int) -> str:
    """Rough expansion slug from item ID range."""
    if item_id >= 224000:
        return "midnight"
    if item_id >= 190000:
        return "tww"
    if item_id >= 171000:
        return "shadowlands"
    if item_id >= 155000:
        return "bfa"
    if item_id >= 124000:
        return "legion"
    if item_id >= 109000:
        return "wod"
    return "legacy"


# ── Category / archetype seeding ──────────────────────────────────────────────

def _seed_categories_and_archetypes(
    conn,
) -> tuple[dict[str, int], dict[str, int]]:
    """Seed item_categories and economic_archetypes if not already present.

    Idempotent: uses INSERT OR IGNORE so re-runs are safe.

    Returns:
        Tuple of (category_slug_to_id, archetype_tag_to_id) dicts.
    """
    from wow_forecaster.taxonomy.archetype_taxonomy import (
        ArchetypeCategory,
        CATEGORY_TAG_MAP,
    )

    cur = conn.cursor()

    # ── item_categories ────────────────────────────────────────────────────────
    for cat in ArchetypeCategory:
        tags = CATEGORY_TAG_MAP.get(cat, [])
        default_tag = str(tags[0]) if tags else cat.value
        cur.execute(
            """
            INSERT OR IGNORE INTO item_categories
                (slug, display_name, parent_slug, archetype_tag, expansion_slug)
            VALUES (?, ?, NULL, ?, 'all')
            """,
            (
                cat.value,
                _CATEGORY_DISPLAY.get(cat.value, cat.value.replace("_", " ").title()),
                default_tag,
            ),
        )

    cat_id_map: dict[str, int] = {
        row[1]: row[0]
        for row in cur.execute(
            "SELECT category_id, slug FROM item_categories"
        ).fetchall()
    }

    # ── economic_archetypes ────────────────────────────────────────────────────
    for cat, tags in CATEGORY_TAG_MAP.items():
        for tag in tags:
            tag_str = str(tag)
            parts = tag_str.split(".")
            # Display name: title-case everything after the category prefix
            display = " ".join(
                p.replace("_", " ").title() for p in parts[1:]
            ) or tag_str.title()
            sub_tag = parts[1] if len(parts) > 1 else None
            cur.execute(
                """
                INSERT OR IGNORE INTO economic_archetypes
                    (slug, display_name, category_tag, sub_tag,
                     is_transferable, transfer_confidence)
                VALUES (?, ?, ?, ?, 1, 0.70)
                """,
                (tag_str, display, cat.value, sub_tag),
            )

    arch_id_map: dict[str, int] = {
        row[1]: row[0]
        for row in cur.execute(
            "SELECT archetype_id, slug FROM economic_archetypes"
        ).fetchall()
    }

    conn.commit()
    logger.info(
        "Seeded %d item_categories, %d economic_archetypes",
        len(cat_id_map), len(arch_id_map),
    )
    return cat_id_map, arch_id_map


# ── Blizzard Item API (async) ─────────────────────────────────────────────────

def _get_oauth_token(client_id: str, client_secret: str, region: str = "us") -> str:
    """Fetch a Blizzard OAuth2 access token (client credentials grant)."""
    creds = base64.b64encode(f"{client_id}:{client_secret}".encode()).decode()
    with httpx.Client() as client:
        resp = client.post(
            f"https://{region}.battle.net/oauth/token",
            headers={
                "Authorization": f"Basic {creds}",
                "Content-Type": "application/x-www-form-urlencoded",
            },
            data={"grant_type": "client_credentials"},
            timeout=30.0,
        )
        resp.raise_for_status()
        return resp.json()["access_token"]


async def _fetch_one_item(
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    item_id: int,
    token: str,
    region: str,
) -> tuple[int, Optional[dict]]:
    """Fetch a single item from the Blizzard Item API."""
    url = f"https://{region}.api.blizzard.com/data/wow/item/{item_id}"
    params = {"namespace": f"static-{region}", "locale": "en_US"}
    headers = {"Authorization": f"Bearer {token}"}

    async with semaphore:
        try:
            resp = await client.get(
                url, params=params, headers=headers, timeout=15.0
            )
            if resp.status_code == 200:
                return item_id, resp.json()
            # 404 = item removed from game; silently skip
            return item_id, None
        except Exception as exc:
            logger.debug("Item %d: fetch error — %s", item_id, exc)
            return item_id, None


async def _fetch_all_items_async(
    token: str,
    item_ids: list[int],
    region: str = "us",
    concurrency: int = 50,
) -> dict[int, dict]:
    """Concurrently fetch item metadata for all given IDs."""
    semaphore = asyncio.Semaphore(concurrency)
    results: dict[int, dict] = {}
    total = len(item_ids)

    async with httpx.AsyncClient() as client:
        tasks = [
            _fetch_one_item(client, semaphore, iid, token, region)
            for iid in item_ids
        ]
        done = 0
        for coro in asyncio.as_completed(tasks):
            item_id, data = await coro
            if data is not None:
                results[item_id] = data
            done += 1
            if done % 1000 == 0 or done == total:
                logger.info(
                    "Item API: %d/%d fetched (%.0f%%)",
                    done, total, 100.0 * done / total,
                )

    return results


# ── Public entry point ────────────────────────────────────────────────────────

def find_latest_commodity_snapshot(raw_dir: Path) -> Optional[Path]:
    """Return the most recently modified commodity snapshot on disk."""
    candidates = sorted(
        (raw_dir / "snapshots" / "blizzard_api").rglob("commodities_us_*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def bootstrap_items(
    db_path: str,
    client_id: str,
    client_secret: str,
    raw_dir: Path,
    region: str = "us",
    concurrency: int = 50,
) -> tuple[int, int, int]:
    """Bootstrap item_categories, economic_archetypes, and items tables.

    Args:
        db_path:       SQLite database path.
        client_id:     Blizzard API client ID.
        client_secret: Blizzard API client secret.
        raw_dir:       Root raw data directory (config.data.raw_dir as Path).
        region:        Blizzard region slug (default "us").
        concurrency:   Max simultaneous Item API requests (default 50).

    Returns:
        Tuple of ``(categories_seeded, archetypes_seeded, items_inserted)``.

    Raises:
        FileNotFoundError: If no commodity snapshot exists on disk.
    """
    from wow_forecaster.db.connection import get_connection

    # ── 1. Find snapshot ───────────────────────────────────────────────────────
    snapshot_path = find_latest_commodity_snapshot(raw_dir)
    if not snapshot_path:
        raise FileNotFoundError(
            "No commodity snapshot found in data/raw/snapshots/blizzard_api/. "
            "Run 'wowfc run-hourly-refresh' first to fetch commodity data."
        )

    logger.info("Loading commodity snapshot: %s", snapshot_path.name)
    records = json.loads(snapshot_path.read_text()).get("data", [])
    item_ids = sorted(set(r["item_id"] for r in records))
    logger.info(
        "Snapshot contains %d records, %d unique item IDs",
        len(records), len(item_ids),
    )

    # ── 2. OAuth2 token ────────────────────────────────────────────────────────
    logger.info("Fetching Blizzard OAuth2 token...")
    token = _get_oauth_token(client_id, client_secret, region)

    # ── 3. Fetch item metadata ─────────────────────────────────────────────────
    logger.info(
        "Fetching metadata for %d items from Blizzard Item API "
        "(concurrency=%d) — this takes ~2-5 minutes...",
        len(item_ids), concurrency,
    )
    item_meta = asyncio.run(
        _fetch_all_items_async(token, item_ids, region, concurrency)
    )
    fetched = len(item_meta)
    not_found = len(item_ids) - fetched
    logger.info(
        "Item API complete: %d fetched, %d not found/removed",
        fetched, not_found,
    )

    # ── 4. Seed categories + archetypes, get ID maps ───────────────────────────
    with get_connection(db_path) as conn:
        cat_id_map, arch_id_map = _seed_categories_and_archetypes(conn)

        default_cat_id  = cat_id_map.get("trade_good")
        default_arch_id = arch_id_map.get(DEFAULT_TAG)

        # ── 5. Insert items ────────────────────────────────────────────────────
        cur = conn.cursor()
        inserted = 0

        # Items with full metadata from the API
        for item_id, meta in item_meta.items():
            name        = meta.get("name") or f"Item #{item_id}"
            quality_raw = (meta.get("quality") or {}).get("type", "COMMON")
            quality     = quality_raw.lower()
            class_id    = (meta.get("item_class") or {}).get("id", -1)
            subclass_id = (meta.get("item_subclass") or {}).get("id", 0)
            is_equip    = bool(meta.get("is_equippable", False))

            tag     = _resolve_archetype_tag(class_id, subclass_id)
            arch_id = arch_id_map.get(tag, default_arch_id)

            # Category slug is the first dot-component of the tag
            cat_slug = tag.split(".")[0] if "." in tag else "trade_good"
            cat_id   = cat_id_map.get(cat_slug, default_cat_id)

            try:
                cur.execute(
                    """
                    INSERT OR IGNORE INTO items
                        (item_id, name, category_id, archetype_id,
                         expansion_slug, quality, is_crafted, is_boe, ilvl, notes)
                    VALUES (?, ?, ?, ?, ?, ?, 0, ?, NULL, NULL)
                    """,
                    (
                        item_id, name, cat_id, arch_id,
                        _item_expansion(item_id), quality,
                        1 if is_equip else 0,
                    ),
                )
                inserted += cur.rowcount
            except Exception as exc:
                logger.debug("Item %d insert failed: %s", item_id, exc)

        # Items in snapshot but not found via API (removed/deprecated items)
        # Insert as placeholders so commodity data still flows through.
        missing_ids = set(item_ids) - set(item_meta.keys())
        for item_id in missing_ids:
            try:
                cur.execute(
                    """
                    INSERT OR IGNORE INTO items
                        (item_id, name, category_id, archetype_id,
                         expansion_slug, quality, is_crafted, is_boe, ilvl, notes)
                    VALUES (?, ?, ?, ?, ?, 'common', 0, 0, NULL, 'api_not_found')
                    """,
                    (
                        item_id,
                        f"Unknown Item #{item_id}",
                        default_cat_id,
                        default_arch_id,
                        _item_expansion(item_id),
                    ),
                )
                inserted += cur.rowcount
            except Exception as exc:
                logger.debug("Placeholder item %d insert failed: %s", item_id, exc)

        conn.commit()

    logger.info(
        "bootstrap_items complete: %d categories | %d archetypes | %d items inserted",
        len(cat_id_map), len(arch_id_map), inserted,
    )
    return len(cat_id_map), len(arch_id_map), inserted
