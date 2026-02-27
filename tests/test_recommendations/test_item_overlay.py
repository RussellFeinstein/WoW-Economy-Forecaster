"""
Tests for wow_forecaster/recommendations/item_overlay.py.

What we test
------------
fetch_item_discounts():
  - Returns empty list when archetype_mean_gold <= 0.
  - Returns empty list when no matching normalized observations exist.
  - Computes discount_pct correctly: (mean - price) / mean.
  - Buy action: ranks items by discount descending (most underpriced first).
  - Sell action: ranks items by discount ascending (most overpriced first).
  - Other action: ranks items by abs(discount) descending.
  - Excludes outlier rows (is_outlier = 1).
  - Respects lookback_days cutoff (old observations excluded).
  - Respects min_obs threshold (items with too few obs excluded).
  - Respects top_n limit.
  - Returns at most top_n rows when more items exist.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

from wow_forecaster.recommendations.item_overlay import (
    ItemDiscountRow,
    fetch_item_discounts,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def conn() -> sqlite3.Connection:
    """In-memory DB with minimal schema for item_overlay tests."""
    db = sqlite3.connect(":memory:")
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA foreign_keys = OFF;")
    db.executescript("""
        CREATE TABLE items (
            item_id      INTEGER PRIMARY KEY,
            name         TEXT    NOT NULL,
            archetype_id INTEGER,
            category_id  INTEGER NOT NULL DEFAULT 1,
            expansion_slug TEXT NOT NULL DEFAULT 'tww',
            quality      TEXT NOT NULL DEFAULT 'common',
            is_crafted   INTEGER NOT NULL DEFAULT 0,
            is_boe       INTEGER NOT NULL DEFAULT 0
        );
        CREATE TABLE market_observations_normalized (
            norm_id      INTEGER PRIMARY KEY AUTOINCREMENT,
            obs_id       INTEGER NOT NULL DEFAULT 1,
            item_id      INTEGER NOT NULL,
            archetype_id INTEGER,
            realm_slug   TEXT    NOT NULL,
            faction      TEXT    NOT NULL DEFAULT 'neutral',
            observed_at  TEXT    NOT NULL,
            price_gold   REAL    NOT NULL,
            quantity_listed INTEGER,
            num_auctions INTEGER,
            z_score      REAL,
            is_outlier   INTEGER NOT NULL DEFAULT 0,
            normalized_at TEXT   NOT NULL DEFAULT (datetime('now'))
        );
    """)
    db.commit()
    return db


def _now_iso(delta_days: int = 0) -> str:
    """Return an ISO datetime string relative to now."""
    dt = datetime.now(tz=timezone.utc) + timedelta(days=delta_days)
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def _insert_item(conn: sqlite3.Connection, item_id: int, name: str, archetype_id: int) -> None:
    conn.execute(
        "INSERT INTO items(item_id, name, archetype_id) VALUES (?,?,?)",
        (item_id, name, archetype_id),
    )


def _insert_obs(
    conn: sqlite3.Connection,
    item_id: int,
    realm_slug: str,
    price_gold: float,
    observed_at: str,
    is_outlier: int = 0,
) -> None:
    conn.execute(
        """INSERT INTO market_observations_normalized
           (item_id, realm_slug, price_gold, observed_at, is_outlier)
           VALUES (?,?,?,?,?)""",
        (item_id, realm_slug, price_gold, observed_at, is_outlier),
    )


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestFetchItemDiscounts:

    def test_zero_mean_returns_empty(self, conn):
        result = fetch_item_discounts(conn, archetype_id=1, realm_slug="us",
                                      archetype_mean_gold=0.0)
        assert result == []

    def test_negative_mean_returns_empty(self, conn):
        result = fetch_item_discounts(conn, archetype_id=1, realm_slug="us",
                                      archetype_mean_gold=-10.0)
        assert result == []

    def test_no_items_returns_empty(self, conn):
        result = fetch_item_discounts(conn, archetype_id=1, realm_slug="us",
                                      archetype_mean_gold=100.0)
        assert result == []

    def test_no_observations_returns_empty(self, conn):
        _insert_item(conn, 1, "Herb A", archetype_id=1)
        result = fetch_item_discounts(conn, archetype_id=1, realm_slug="us",
                                      archetype_mean_gold=100.0)
        assert result == []

    def test_discount_computed_correctly(self, conn):
        # archetype mean = 100g; item price = 60g -> discount = 0.40 (40% underpriced)
        _insert_item(conn, 1, "Herb A", archetype_id=1)
        _insert_obs(conn, 1, "us", 60.0, _now_iso())
        conn.commit()

        rows = fetch_item_discounts(conn, archetype_id=1, realm_slug="us",
                                    archetype_mean_gold=100.0)
        assert len(rows) == 1
        assert rows[0].item_id == 1
        assert rows[0].name == "Herb A"
        assert rows[0].item_price_gold == pytest.approx(60.0)
        assert rows[0].discount_pct == pytest.approx(0.40)

    def test_negative_discount_for_overpriced_item(self, conn):
        # archetype mean = 100g; item price = 150g -> discount = -0.50 (50% overpriced)
        _insert_item(conn, 1, "Gem A", archetype_id=1)
        _insert_obs(conn, 1, "us", 150.0, _now_iso())
        conn.commit()

        rows = fetch_item_discounts(conn, archetype_id=1, realm_slug="us",
                                    archetype_mean_gold=100.0)
        assert rows[0].discount_pct == pytest.approx(-0.50)

    def test_buy_action_most_underpriced_first(self, conn):
        # Items: 40g (60% under), 70g (30% under), 90g (10% under)
        for item_id, name, price in [(1, "A", 40.0), (2, "B", 70.0), (3, "C", 90.0)]:
            _insert_item(conn, item_id, name, archetype_id=1)
            _insert_obs(conn, item_id, "us", price, _now_iso())
        conn.commit()

        rows = fetch_item_discounts(conn, archetype_id=1, realm_slug="us",
                                    archetype_mean_gold=100.0, action="buy")
        discounts = [r.discount_pct for r in rows]
        assert discounts == sorted(discounts, reverse=True)
        assert rows[0].name == "A"  # cheapest = most underpriced

    def test_sell_action_most_overpriced_first(self, conn):
        # Items: 120g (-20%), 150g (-50%), 200g (-100%)
        for item_id, name, price in [(1, "A", 120.0), (2, "B", 150.0), (3, "C", 200.0)]:
            _insert_item(conn, item_id, name, archetype_id=1)
            _insert_obs(conn, item_id, "us", price, _now_iso())
        conn.commit()

        rows = fetch_item_discounts(conn, archetype_id=1, realm_slug="us",
                                    archetype_mean_gold=100.0, action="sell")
        discounts = [r.discount_pct for r in rows]
        assert discounts == sorted(discounts)  # ascending: most negative first
        assert rows[0].name == "C"  # most overpriced

    def test_hold_action_most_deviant_first(self, conn):
        # Items at 60g (+40% under), 140g (-40% over), 100g (0%)
        for item_id, name, price in [(1, "A", 60.0), (2, "B", 140.0), (3, "C", 100.0)]:
            _insert_item(conn, item_id, name, archetype_id=1)
            _insert_obs(conn, item_id, "us", price, _now_iso())
        conn.commit()

        rows = fetch_item_discounts(conn, archetype_id=1, realm_slug="us",
                                    archetype_mean_gold=100.0, action="hold")
        abs_discounts = [abs(r.discount_pct) for r in rows]
        assert abs_discounts == sorted(abs_discounts, reverse=True)
        assert rows[-1].name == "C"  # exactly at mean = last

    def test_outliers_excluded(self, conn):
        _insert_item(conn, 1, "Herb A", archetype_id=1)
        _insert_obs(conn, 1, "us", 50.0, _now_iso(), is_outlier=1)
        conn.commit()

        rows = fetch_item_discounts(conn, archetype_id=1, realm_slug="us",
                                    archetype_mean_gold=100.0)
        assert rows == []

    def test_old_observations_excluded(self, conn):
        _insert_item(conn, 1, "Herb A", archetype_id=1)
        # Insert obs 10 days old — outside default lookback of 3 days
        _insert_obs(conn, 1, "us", 50.0, _now_iso(delta_days=-10))
        conn.commit()

        rows = fetch_item_discounts(conn, archetype_id=1, realm_slug="us",
                                    archetype_mean_gold=100.0, lookback_days=3)
        assert rows == []

    def test_recent_observations_included(self, conn):
        _insert_item(conn, 1, "Herb A", archetype_id=1)
        _insert_obs(conn, 1, "us", 60.0, _now_iso(delta_days=-1))
        conn.commit()

        rows = fetch_item_discounts(conn, archetype_id=1, realm_slug="us",
                                    archetype_mean_gold=100.0, lookback_days=3)
        assert len(rows) == 1

    def test_min_obs_filters_low_data_items(self, conn):
        _insert_item(conn, 1, "Herb A", archetype_id=1)
        _insert_item(conn, 2, "Herb B", archetype_id=1)
        # Herb A has 1 obs, Herb B has 3 obs
        _insert_obs(conn, 1, "us", 60.0, _now_iso())
        for _ in range(3):
            _insert_obs(conn, 2, "us", 70.0, _now_iso())
        conn.commit()

        rows = fetch_item_discounts(conn, archetype_id=1, realm_slug="us",
                                    archetype_mean_gold=100.0, min_obs=2)
        assert len(rows) == 1
        assert rows[0].item_id == 2

    def test_top_n_limits_results(self, conn):
        for i in range(1, 8):
            _insert_item(conn, i, f"Item {i}", archetype_id=1)
            _insert_obs(conn, i, "us", float(i * 10), _now_iso())
        conn.commit()

        rows = fetch_item_discounts(conn, archetype_id=1, realm_slug="us",
                                    archetype_mean_gold=100.0, top_n=3)
        assert len(rows) == 3

    def test_returns_ItemDiscountRow_instances(self, conn):
        _insert_item(conn, 1, "Herb A", archetype_id=1)
        _insert_obs(conn, 1, "us", 80.0, _now_iso())
        conn.commit()

        rows = fetch_item_discounts(conn, archetype_id=1, realm_slug="us",
                                    archetype_mean_gold=100.0)
        assert all(isinstance(r, ItemDiscountRow) for r in rows)

    def test_realm_slug_isolation(self, conn):
        # Item exists for realm "eu" only; query for "us" should return nothing
        _insert_item(conn, 1, "Herb A", archetype_id=1)
        _insert_obs(conn, 1, "eu", 60.0, _now_iso())
        conn.commit()

        rows = fetch_item_discounts(conn, archetype_id=1, realm_slug="us",
                                    archetype_mean_gold=100.0)
        assert rows == []

    def test_archetype_id_isolation(self, conn):
        # Item belongs to archetype 2; query for archetype 1 returns nothing
        _insert_item(conn, 1, "Gem A", archetype_id=2)
        _insert_obs(conn, 1, "us", 60.0, _now_iso())
        conn.commit()

        rows = fetch_item_discounts(conn, archetype_id=1, realm_slug="us",
                                    archetype_mean_gold=100.0)
        assert rows == []

    def test_obs_count_reported(self, conn):
        _insert_item(conn, 1, "Herb A", archetype_id=1)
        for _ in range(4):
            _insert_obs(conn, 1, "us", 60.0, _now_iso())
        conn.commit()

        rows = fetch_item_discounts(conn, archetype_id=1, realm_slug="us",
                                    archetype_mean_gold=100.0)
        assert rows[0].obs_count == 4

    def test_price_averaged_across_multiple_obs(self, conn):
        _insert_item(conn, 1, "Herb A", archetype_id=1)
        for price in [60.0, 80.0, 100.0]:
            _insert_obs(conn, 1, "us", price, _now_iso())
        conn.commit()

        rows = fetch_item_discounts(conn, archetype_id=1, realm_slug="us",
                                    archetype_mean_gold=100.0)
        assert rows[0].item_price_gold == pytest.approx(80.0)  # (60+80+100)/3
