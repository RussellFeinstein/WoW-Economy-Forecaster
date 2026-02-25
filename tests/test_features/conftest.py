"""
Fixtures for the feature engineering test suite.

Provides ``feature_db``: an in-memory SQLite connection seeded with:
  - 2 economic archetypes (consumable.flask.stat, mat.ore.common)
  - 1 item per archetype (expansion_slug="tww")
  - 30 days × 3 obs/day × 2 archetypes of normalised observations
    with deterministic, non-outlier prices
  - 2 events: one announced before the data window, one announced after
  - 1 archetype mapping (archetype 1 → archetype 2, tww→midnight)

The seeded prices are designed so lag/rolling tests can verify exact values:
  - archetype 1: price_mean oscillates 100, 110, 105, 120, 115, 130, 125 … cycling
  - archetype 2: constant 50.0 (useful for std=0.0 tests)
"""

from __future__ import annotations

import sqlite3
from datetime import date, datetime, timezone, timedelta
from typing import Generator

import pytest

from wow_forecaster.db.schema import apply_schema


# Base observation date for the seed window.
_SEED_START = date(2025, 1, 1)
_SEED_DAYS  = 30
_REALM      = "area-52"

_PRICES_ARCH1 = [100.0, 110.0, 105.0, 120.0, 115.0, 130.0, 125.0]


def _price_arch1(day_index: int) -> float:
    return _PRICES_ARCH1[day_index % len(_PRICES_ARCH1)]


@pytest.fixture
def feature_db() -> Generator[sqlite3.Connection, None, None]:
    """In-memory SQLite with full schema + feature test seed data."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    apply_schema(conn)
    _seed(conn)
    yield conn
    conn.close()


def _seed(conn: sqlite3.Connection) -> None:
    """Insert all seed data into the in-memory database."""
    # Item categories (required FK for items).
    conn.execute(
        """
        INSERT INTO item_categories (category_id, slug, display_name, archetype_tag)
        VALUES (1, 'consumable.flask', 'Flasks', 'consumable.flask.stat'),
               (2, 'mat.ore',         'Ore',    'mat.ore.common')
        """
    )

    # Economic archetypes.
    conn.execute(
        """
        INSERT INTO economic_archetypes
            (archetype_id, slug, display_name, category_tag, sub_tag,
             is_transferable, transfer_confidence)
        VALUES
            (1, 'consumable.flask.stat', 'Stat Flask', 'consumable', 'consumable.flask.stat',
             1, 0.90),
            (2, 'mat.ore.common',        'Common Ore', 'mat',        'mat.ore.common',
             1, 0.75)
        """
    )

    # Items (1 per archetype, expansion_slug="tww").
    conn.execute(
        """
        INSERT INTO items
            (item_id, name, category_id, archetype_id, expansion_slug, quality, is_crafted, is_boe)
        VALUES
            (191528, 'Phial of Tepid Versatility', 1, 1, 'tww', 'rare',   1, 0),
            (190311, 'Bismuth',                    2, 2, 'tww', 'common', 0, 0)
        """
    )

    # Archetype mapping (archetype 1 → archetype 2, tww → midnight).
    conn.execute(
        """
        INSERT INTO archetype_mappings
            (source_archetype_id, target_archetype_id,
             source_expansion, target_expansion,
             confidence_score, mapping_rationale, created_by)
        VALUES (1, 2, 'tww', 'midnight', 0.85,
                'Test mapping for transfer learning fixture.', 'manual')
        """
    )

    # Two WoW events:
    #   - early_event: announced 2025-01-01, starts 2025-01-10 → known throughout window
    #   - late_event:  announced 2025-02-01, starts 2025-02-10 → unknown to rows before 2025-02-01
    conn.execute(
        """
        INSERT INTO wow_events
            (event_id, slug, display_name, event_type, scope, severity,
             expansion_slug, start_date, end_date, announced_at)
        VALUES
            (1, 'early-event', 'Early Test Event',  'rtwf',    'global', 'major',
             'tww', '2025-01-10', '2025-01-17', '2025-01-01T00:00:00+00:00'),
            (2, 'late-event',  'Late Test Event',   'major_patch', 'global', 'moderate',
             'tww', '2025-02-10', '2025-02-17', '2025-02-01T00:00:00+00:00'),
            (3, 'launch-event', 'TWW Launch',       'expansion_launch', 'global', 'critical',
             'tww', '2024-08-26', NULL,             '2024-05-01T00:00:00+00:00')
        """
    )

    # Archetype impact for early-event on archetype 1.
    conn.execute(
        """
        INSERT INTO event_archetype_impacts
            (event_id, archetype_id, impact_direction, lag_days, duration_days, source)
        VALUES (1, 1, 'bullish', 0, 7, 'manual')
        """
    )

    # Normalised observations: 30 days × 3 obs/day × 2 archetypes.
    rows = []
    obs_id = 1
    # We need raw obs first; FK requires market_observations_raw, but actually
    # the FK on market_observations_normalized references obs_id in
    # market_observations_raw.  To keep tests simple, insert raw obs first
    # then normalised obs referencing them.
    raw_rows: list[tuple] = []
    norm_rows: list[tuple] = []

    for day_idx in range(_SEED_DAYS):
        obs_date = _SEED_START + timedelta(days=day_idx)
        for hour_offset in [8, 14, 20]:
            observed_at = datetime(
                obs_date.year, obs_date.month, obs_date.day,
                hour_offset, 0, 0, tzinfo=timezone.utc
            ).isoformat()

            # Archetype 1 (flask): oscillating price
            price_a1 = _price_arch1(day_idx)
            raw_rows.append((obs_id, 191528, _REALM, "neutral", observed_at,
                             "blizzard_api", int(price_a1 * 10000), None, None, 50, 5))
            norm_rows.append((obs_id, obs_id, 191528, None, _REALM, "neutral",
                              observed_at, price_a1, None, None, 50, 5, None, 0))
            obs_id += 1

            # Archetype 2 (ore): constant price 50.0
            raw_rows.append((obs_id, 190311, _REALM, "neutral", observed_at,
                             "blizzard_api", 500000, None, None, 200, 20))
            norm_rows.append((obs_id, obs_id, 190311, None, _REALM, "neutral",
                              observed_at, 50.0, None, None, 200, 20, None, 0))
            obs_id += 1

    conn.executemany(
        """
        INSERT INTO market_observations_raw
            (obs_id, item_id, realm_slug, faction, observed_at,
             source, min_buyout_raw, market_value_raw, historical_value_raw,
             quantity_listed, num_auctions)
        VALUES (?,?,?,?,?,?,?,?,?,?,?)
        """,
        raw_rows,
    )
    conn.executemany(
        """
        INSERT INTO market_observations_normalized
            (norm_id, obs_id, item_id, archetype_id, realm_slug, faction,
             observed_at, price_gold, market_value_gold, historical_value_gold,
             quantity_listed, num_auctions, z_score, is_outlier)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        norm_rows,
    )
    conn.commit()
