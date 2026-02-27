"""
Auctionator addon Lua parser — imports AUCTIONATOR_POSTING_HISTORY into
market_observations_raw as historical backfill data.

Source:  WTF/Account/{account}/SavedVariables/Auctionator.lua
Section: AUCTIONATOR_POSTING_HISTORY (plain-text Lua table, not the binary
         AUCTIONATOR_PRICE_DATABASE section which uses LibSerialize+LibDeflate).

Format::

    AUCTIONATOR_POSTING_HISTORY = {
    ["__dbversion"] = 1,
    ["213220"] = {
    {
    ["price"] = 289900,
    ["quantity"] = 25,
    ["time"] = 1744068783,
    },
    ...
    },
    ...
    }

Realm context
-------------
Commodity AH prices are US-region-wide since patch 9.2.7.
All records are tagged ``realm_slug="us"`` / ``faction="neutral"``.

Default path (Windows retail WoW)::

    C:/Program Files (x86)/World of Warcraft/_retail_/WTF/Account/
        {account_name}/SavedVariables/Auctionator.lua
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from wow_forecaster.models.market import RawMarketObservation

logger = logging.getLogger(__name__)

# ── Section markers ───────────────────────────────────────────────────────────

_SECTION_START = "AUCTIONATOR_POSTING_HISTORY = {"
_NEXT_TOPLEVEL_RE = re.compile(r"^[A-Z]")   # Next top-level Lua variable

# ── Line matchers ─────────────────────────────────────────────────────────────

_ITEM_RE  = re.compile(r'^\["(\d+)"\] = \{$')        # ["12345"] = {
_PRICE_RE = re.compile(r'^\["price"\] = (\d+),$')     # ["price"] = N,
_QTY_RE   = re.compile(r'^\["quantity"\] = (\d+),$')  # ["quantity"] = N,
_TIME_RE  = re.compile(r'^\["time"\] = (\d+),$')      # ["time"] = N,

# Default WoW retail path on Windows
DEFAULT_LUA_PATH = Path(
    "C:/Program Files (x86)/World of Warcraft/_retail_/WTF/Account"
    "/60546360#1/SavedVariables/Auctionator.lua"
)


# ── Parser ────────────────────────────────────────────────────────────────────

def parse_auctionator_lua(path: Path) -> list[tuple[int, int, int, int]]:
    """Parse AUCTIONATOR_POSTING_HISTORY from an Auctionator.lua SavedVariables file.

    Uses a line-by-line state machine; does not attempt to decode the binary
    ``AUCTIONATOR_PRICE_DATABASE`` section.

    Args:
        path: Path to Auctionator.lua.

    Returns:
        List of ``(item_id, price_copper, quantity, unix_timestamp)`` tuples.
        One tuple per individual posting record.
    """
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        lines = fh.readlines()

    in_section = False
    current_item_id: int | None = None
    current_price:   int | None = None
    current_qty:     int | None = None
    current_time:    int | None = None
    results: list[tuple[int, int, int, int]] = []

    for raw_line in lines:
        stripped = raw_line.strip()

        # ── Find section start ─────────────────────────────────────────────
        if not in_section:
            if stripped == _SECTION_START:
                in_section = True
            continue

        # ── Detect end of section (next top-level Lua variable) ───────────
        if _NEXT_TOPLEVEL_RE.match(stripped) and stripped != _SECTION_START:
            break

        # ── Item ID key ────────────────────────────────────────────────────
        m = _ITEM_RE.match(stripped)
        if m:
            current_item_id = int(m.group(1))
            current_price = current_qty = current_time = None
            continue

        # ── Inner entry fields ─────────────────────────────────────────────
        m = _PRICE_RE.match(stripped)
        if m:
            current_price = int(m.group(1))
            continue

        m = _QTY_RE.match(stripped)
        if m:
            current_qty = int(m.group(1))
            continue

        m = _TIME_RE.match(stripped)
        if m:
            current_time = int(m.group(1))
            continue

        # ── Flush complete entry on closing brace ──────────────────────────
        if stripped == "}," and current_item_id is not None:
            if (
                current_price is not None
                and current_qty is not None
                and current_time is not None
            ):
                results.append(
                    (current_item_id, current_price, current_qty, current_time)
                )
            # Reset inner fields; keep current_item_id for the next entry
            current_price = current_qty = current_time = None

    logger.debug(
        "parse_auctionator_lua: %d raw records from %s", len(results), path.name
    )
    return results


# ── Converter ─────────────────────────────────────────────────────────────────

def build_raw_observations(
    records: list[tuple[int, int, int, int]],
    known_item_ids: set[int],
    realm_slug: str = "us",
    faction: str = "neutral",
) -> tuple[list[RawMarketObservation], int]:
    """Convert parsed Auctionator records to :class:`RawMarketObservation` objects.

    Args:
        records:        List of ``(item_id, price_copper, quantity, unix_ts)``.
        known_item_ids: Item IDs present in the ``items`` table (FK guard).
        realm_slug:     Realm context; "us" for region-wide commodity data.
        faction:        Always "neutral" for commodity AH data.

    Returns:
        Tuple of ``(observations, skipped_count)`` where ``skipped_count``
        is the number of records dropped because ``item_id`` was not in the
        item registry.
    """
    from wow_forecaster.models.market import RawMarketObservation

    observations: list[RawMarketObservation] = []
    skipped = 0

    for item_id, price_copper, quantity, unix_ts in records:
        if item_id not in known_item_ids:
            skipped += 1
            continue

        observed_at = datetime.fromtimestamp(unix_ts, tz=timezone.utc).replace(
            tzinfo=None
        )

        observations.append(
            RawMarketObservation(
                item_id=item_id,
                realm_slug=realm_slug,
                faction=faction,
                observed_at=observed_at,
                source="auctionator_posting",
                min_buyout_raw=price_copper,
                market_value_raw=None,
                historical_value_raw=None,
                quantity_listed=quantity,
                num_auctions=None,
                raw_json=None,
            )
        )

    return observations, skipped


# ── Top-level import function ─────────────────────────────────────────────────

def import_auctionator_data(
    lua_path: Path,
    db_path: str,
    realm_slug: str = "us",
) -> tuple[int, int]:
    """Parse Auctionator.lua and insert historical records into market_observations_raw.

    Args:
        lua_path:   Path to Auctionator.lua SavedVariables file.
        db_path:    SQLite database path string.
        realm_slug: Realm context for records ("us" for region-wide commodities).

    Returns:
        Tuple of ``(inserted_count, skipped_count)``.
    """
    from wow_forecaster.db.connection import get_connection
    from wow_forecaster.db.repositories.item_repo import ItemRepository
    from wow_forecaster.db.repositories.market_repo import MarketObservationRepository

    logger.info("Parsing Auctionator.lua: %s", lua_path)
    raw_records = parse_auctionator_lua(lua_path)
    logger.info(
        "Parsed %d raw price entries from AUCTIONATOR_POSTING_HISTORY", len(raw_records)
    )

    if not raw_records:
        logger.warning(
            "No records found. Check that AUCTIONATOR_POSTING_HISTORY "
            "exists in %s and WoW is closed (file is written on exit).",
            lua_path,
        )
        return 0, 0

    with get_connection(db_path) as conn:
        item_repo = ItemRepository(conn)
        market_repo = MarketObservationRepository(conn)
        known_item_ids = item_repo.get_all_item_ids()

        logger.info(
            "Item registry has %d known items. "
            "Records for unknown items are silently skipped.",
            len(known_item_ids),
        )

        observations, skipped = build_raw_observations(
            raw_records, known_item_ids, realm_slug=realm_slug
        )

        inserted = market_repo.insert_raw_batch(observations)
        conn.commit()

    logger.info(
        "Auctionator import complete: inserted=%d skipped_unknown=%d",
        inserted, skipped,
    )
    return inserted, skipped
