"""Tests for wow_forecaster.db.rollup — pre-aggregated rollup tables."""

from __future__ import annotations

import sqlite3

import pytest

from wow_forecaster.db.migrations import run_migrations
from wow_forecaster.db.rollup import (
    backfill_rollups,
    upsert_archetype_rollup,
    upsert_item_rollup,
    upsert_rollups_for_date,
)
from wow_forecaster.db.schema import apply_schema

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def conn() -> sqlite3.Connection:
    """In-memory DB with full schema + migrations applied."""
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA foreign_keys = ON;")
    apply_schema(c)
    run_migrations(c)
    return c


@pytest.fixture
def seeded_conn(conn: sqlite3.Connection) -> sqlite3.Connection:
    """DB with sample normalized observations for rollup testing."""
    # Seed archetype + item
    conn.execute("""
        INSERT INTO economic_archetypes (archetype_id, slug, display_name, category_tag)
        VALUES (1, 'consumable.flask', 'Flasks', 'consumable')
    """)
    conn.execute("""
        INSERT INTO item_categories (category_id, slug, display_name, archetype_tag)
        VALUES (1, 'flask', 'Flasks', 'consumable.flask')
    """)
    conn.execute("""
        INSERT INTO items (item_id, name, category_id, archetype_id, expansion_slug, quality)
        VALUES (100, 'Flask A', 1, 1, 'tww', 'epic')
    """)
    conn.execute("""
        INSERT INTO items (item_id, name, category_id, archetype_id, expansion_slug, quality)
        VALUES (200, 'Flask B', 1, 1, 'tww', 'epic')
    """)

    # Seed raw observations (needed for FK)
    for obs_id in range(1, 11):
        conn.execute("""
            INSERT INTO market_observations_raw
                (obs_id, item_id, realm_slug, observed_at, source)
            VALUES (?, ?, 'us', '2026-03-15 12:00:00', 'blizzard_api')
        """, (obs_id, 100 if obs_id <= 6 else 200))

    # Seed normalized observations — mix of prices including a zero
    # Day 1: 2026-03-15
    test_data = [
        # (obs_id, item_id, price, market_val, hist_val, qty, auctions)
        (1, 100, 500.0, 550.0, 480.0, 10, 3),
        (2, 100, 520.0, 560.0, 490.0, 15, 4),
        (3, 100, 0.0, None, None, 5, 1),       # zero-price row
        (4, 200, 300.0, 320.0, None, None, 2),  # no quantity
        (5, 200, 310.0, 330.0, 290.0, 20, 5),
    ]
    # Day 2: 2026-03-16
    test_data_day2 = [
        (6, 100, 510.0, 555.0, 485.0, 12, 3),
        (7, 200, 305.0, 325.0, 292.0, 18, 4),
        (8, 200, 315.0, 335.0, 295.0, 22, 6),
    ]

    for obs_id, item_id, price, mv, hv, qty, auc in test_data:
        conn.execute("""
            INSERT INTO market_observations_normalized
                (obs_id, item_id, archetype_id, realm_slug, observed_at,
                 price_gold, market_value_gold, historical_value_gold,
                 quantity_listed, num_auctions, is_outlier)
            VALUES (?, ?, 1, 'us', '2026-03-15 12:00:00', ?, ?, ?, ?, ?, 0)
        """, (obs_id, item_id, price, mv, hv, qty, auc))

    # Need more raw obs for day 2
    for i, (_obs_id, item_id, price, mv, hv, qty, auc) in enumerate(test_data_day2):
        raw_id = 10 + i + 1
        conn.execute("""
            INSERT INTO market_observations_raw
                (obs_id, item_id, realm_slug, observed_at, source)
            VALUES (?, ?, 'us', '2026-03-16 12:00:00', 'blizzard_api')
        """, (raw_id, item_id))
        conn.execute("""
            INSERT INTO market_observations_normalized
                (obs_id, item_id, archetype_id, realm_slug, observed_at,
                 price_gold, market_value_gold, historical_value_gold,
                 quantity_listed, num_auctions, is_outlier)
            VALUES (?, ?, 1, 'us', '2026-03-16 12:00:00', ?, ?, ?, ?, ?, 0)
        """, (raw_id, item_id, price, mv, hv, qty, auc))

    # Also add an outlier row (should be excluded)
    conn.execute("""
        INSERT INTO market_observations_raw
            (obs_id, item_id, realm_slug, observed_at, source)
        VALUES (99, 100, 'us', '2026-03-15 12:00:00', 'blizzard_api')
    """)
    conn.execute("""
        INSERT INTO market_observations_normalized
            (obs_id, item_id, archetype_id, realm_slug, observed_at,
             price_gold, market_value_gold, historical_value_gold,
             quantity_listed, num_auctions, is_outlier)
        VALUES (99, 100, 1, 'us', '2026-03-15 12:00:00', 9999.0, 9999.0, 9999.0, 999, 99, 1)
    """)

    conn.commit()
    return conn


# ── Basic upsert tests ───────────────────────────────────────────────────────


class TestUpsertArchetypeRollup:
    def test_upserts_rows(self, seeded_conn):
        n = upsert_archetype_rollup(seeded_conn, "us", "2026-03-15")
        assert n >= 1

        row = seeded_conn.execute(
            "SELECT * FROM daily_rollup_archetype WHERE obs_date = '2026-03-15'"
        ).fetchone()
        assert row is not None
        assert row["archetype_id"] == 1
        assert row["realm_slug"] == "us"

    def test_obs_count_includes_all(self, seeded_conn):
        """obs_count counts ALL rows including zero-price."""
        upsert_archetype_rollup(seeded_conn, "us", "2026-03-15")
        row = seeded_conn.execute(
            "SELECT obs_count FROM daily_rollup_archetype WHERE obs_date = '2026-03-15'"
        ).fetchone()
        # Day 1: 5 non-outlier rows (3 for item 100, 2 for item 200)
        assert row["obs_count"] == 5

    def test_price_obs_count_excludes_zeros(self, seeded_conn):
        """price_obs_count counts only price_gold > 0 rows."""
        upsert_archetype_rollup(seeded_conn, "us", "2026-03-15")
        row = seeded_conn.execute(
            "SELECT price_obs_count FROM daily_rollup_archetype WHERE obs_date = '2026-03-15'"
        ).fetchone()
        # Day 1: 4 positive-price rows (item 100: 500, 520; item 200: 300, 310)
        assert row["price_obs_count"] == 4

    def test_price_mean_derivation(self, seeded_conn):
        """price_sum / price_obs_count should match AVG(CASE WHEN price > 0)."""
        upsert_archetype_rollup(seeded_conn, "us", "2026-03-15")
        row = seeded_conn.execute(
            "SELECT price_sum, price_obs_count FROM daily_rollup_archetype "
            "WHERE obs_date = '2026-03-15'"
        ).fetchone()
        expected_mean = (500.0 + 520.0 + 300.0 + 310.0) / 4
        actual_mean = row["price_sum"] / row["price_obs_count"]
        assert actual_mean == pytest.approx(expected_mean)

    def test_price_min_max_positive_only(self, seeded_conn):
        upsert_archetype_rollup(seeded_conn, "us", "2026-03-15")
        row = seeded_conn.execute(
            "SELECT price_min, price_max FROM daily_rollup_archetype "
            "WHERE obs_date = '2026-03-15'"
        ).fetchone()
        assert row["price_min"] == pytest.approx(300.0)
        assert row["price_max"] == pytest.approx(520.0)

    def test_outlier_excluded(self, seeded_conn):
        """The is_outlier=1 row (price 9999) should not appear."""
        upsert_archetype_rollup(seeded_conn, "us", "2026-03-15")
        row = seeded_conn.execute(
            "SELECT price_max FROM daily_rollup_archetype WHERE obs_date = '2026-03-15'"
        ).fetchone()
        assert row["price_max"] < 9999.0

    def test_market_value_sum_count(self, seeded_conn):
        upsert_archetype_rollup(seeded_conn, "us", "2026-03-15")
        row = seeded_conn.execute(
            "SELECT market_value_sum, market_value_count "
            "FROM daily_rollup_archetype WHERE obs_date = '2026-03-15'"
        ).fetchone()
        # market_value_gold: 550, 560, NULL, 320, 330 -> sum=1760, count=4
        assert row["market_value_sum"] == pytest.approx(1760.0)
        assert row["market_value_count"] == 4


class TestUpsertItemRollup:
    def test_upserts_rows(self, seeded_conn):
        n = upsert_item_rollup(seeded_conn, "us", "2026-03-15")
        assert n >= 2  # at least items 100 and 200

    def test_item_100_obs_count(self, seeded_conn):
        upsert_item_rollup(seeded_conn, "us", "2026-03-15")
        row = seeded_conn.execute(
            "SELECT obs_count FROM daily_rollup_item "
            "WHERE item_id = 100 AND obs_date = '2026-03-15'"
        ).fetchone()
        # Item 100 day 1: obs_ids 1, 2, 3 (including zero-price)
        assert row["obs_count"] == 3

    def test_item_100_positive_only(self, seeded_conn):
        upsert_item_rollup(seeded_conn, "us", "2026-03-15")
        row = seeded_conn.execute(
            "SELECT price_obs_count_pos, price_sum_pos FROM daily_rollup_item "
            "WHERE item_id = 100 AND obs_date = '2026-03-15'"
        ).fetchone()
        # Item 100 positive: 500, 520
        assert row["price_obs_count_pos"] == 2
        assert row["price_sum_pos"] == pytest.approx(1020.0)

    def test_qty_weighted_price(self, seeded_conn):
        """Verify qty_weighted_price_sum / qty_weight_sum matches manual calc."""
        upsert_item_rollup(seeded_conn, "us", "2026-03-15")
        row = seeded_conn.execute(
            "SELECT qty_weighted_price_sum, qty_weight_sum FROM daily_rollup_item "
            "WHERE item_id = 100 AND obs_date = '2026-03-15'"
        ).fetchone()
        # Item 100 all rows: (500*10 + 520*15 + 0*5) / (10+15+5)
        expected_num = 500.0 * 10 + 520.0 * 15 + 0.0 * 5
        expected_den = 10 + 15 + 5
        assert row["qty_weighted_price_sum"] == pytest.approx(expected_num)
        assert row["qty_weight_sum"] == pytest.approx(expected_den)


# ── Idempotency ───────────────────────────────────────────────────────────────


class TestIdempotency:
    def test_archetype_upsert_idempotent(self, seeded_conn):
        upsert_archetype_rollup(seeded_conn, "us", "2026-03-15")
        row1 = dict(seeded_conn.execute(
            "SELECT obs_count, price_sum FROM daily_rollup_archetype "
            "WHERE obs_date = '2026-03-15'"
        ).fetchone())

        upsert_archetype_rollup(seeded_conn, "us", "2026-03-15")
        row2 = dict(seeded_conn.execute(
            "SELECT obs_count, price_sum FROM daily_rollup_archetype "
            "WHERE obs_date = '2026-03-15'"
        ).fetchone())

        assert row1["obs_count"] == row2["obs_count"]
        assert row1["price_sum"] == pytest.approx(row2["price_sum"])

    def test_item_upsert_idempotent(self, seeded_conn):
        upsert_item_rollup(seeded_conn, "us", "2026-03-15")
        row1 = dict(seeded_conn.execute(
            "SELECT obs_count, price_sum FROM daily_rollup_item "
            "WHERE item_id = 100 AND obs_date = '2026-03-15'"
        ).fetchone())

        upsert_item_rollup(seeded_conn, "us", "2026-03-15")
        row2 = dict(seeded_conn.execute(
            "SELECT obs_count, price_sum FROM daily_rollup_item "
            "WHERE item_id = 100 AND obs_date = '2026-03-15'"
        ).fetchone())

        assert row1["obs_count"] == row2["obs_count"]
        assert row1["price_sum"] == pytest.approx(row2["price_sum"])


# ── Variance derivation ──────────────────────────────────────────────────────


class TestVarianceDerivation:
    def test_single_day_variance(self, seeded_conn):
        """Verify E[X^2] - E[X]^2 gives correct population variance."""
        upsert_archetype_rollup(seeded_conn, "us", "2026-03-15")
        row = seeded_conn.execute(
            "SELECT price_sum_all, price_sum_sq_all, obs_count "
            "FROM daily_rollup_archetype WHERE obs_date = '2026-03-15'"
        ).fetchone()

        n = row["obs_count"]
        mean = row["price_sum_all"] / n
        variance = row["price_sum_sq_all"] / n - mean * mean

        # Compute expected from raw data
        prices = [500.0, 520.0, 0.0, 300.0, 310.0]
        expected_mean = sum(prices) / len(prices)
        expected_var = sum((p - expected_mean) ** 2 for p in prices) / len(prices)

        assert mean == pytest.approx(expected_mean)
        assert variance == pytest.approx(expected_var, rel=1e-6)

    def test_multi_day_variance(self, seeded_conn):
        """Verify variance recovery across multiple daily rollup rows."""
        upsert_archetype_rollup(seeded_conn, "us", "2026-03-15")
        upsert_archetype_rollup(seeded_conn, "us", "2026-03-16")

        rows = seeded_conn.execute(
            "SELECT SUM(price_sum_all) AS total_sum, "
            "       SUM(price_sum_sq_all) AS total_sq, "
            "       SUM(obs_count) AS total_n "
            "FROM daily_rollup_archetype WHERE realm_slug = 'us'"
        ).fetchone()

        total_n = rows["total_n"]
        total_mean = rows["total_sum"] / total_n
        total_var = rows["total_sq"] / total_n - total_mean * total_mean

        # Expected from all raw prices (both days)
        all_prices = [500.0, 520.0, 0.0, 300.0, 310.0, 510.0, 305.0, 315.0]
        exp_mean = sum(all_prices) / len(all_prices)
        exp_var = sum((p - exp_mean) ** 2 for p in all_prices) / len(all_prices)

        assert total_mean == pytest.approx(exp_mean)
        assert total_var == pytest.approx(exp_var, rel=1e-6)


# ── Combined upsert ──────────────────────────────────────────────────────────


class TestUpsertRollupsForDate:
    def test_updates_both_tables(self, seeded_conn):
        arch, item = upsert_rollups_for_date(seeded_conn, "us", "2026-03-15")
        assert arch >= 1
        assert item >= 2

    def test_accepts_date_object(self, seeded_conn):
        from datetime import date
        arch, item = upsert_rollups_for_date(seeded_conn, "us", date(2026, 3, 15))
        assert arch >= 1


# ── Backfill ──────────────────────────────────────────────────────────────────


class TestBackfillRollups:
    def test_backfills_all_dates(self, seeded_conn):
        arch, item = backfill_rollups(seeded_conn, "us", batch_days=1)
        assert arch >= 2  # at least day 1 and day 2
        assert item >= 4  # 2 items x 2 days

    def test_progress_callback(self, seeded_conn):
        progress = []
        backfill_rollups(
            seeded_conn, "us", batch_days=1,
            progress_callback=lambda done, total: progress.append((done, total)),
        )
        assert len(progress) > 0
        assert progress[-1][0] == progress[-1][1]  # final: done == total

    def test_empty_realm(self, conn):
        arch, item = backfill_rollups(conn, "nonexistent")
        assert arch == 0
        assert item == 0

    def test_idempotent(self, seeded_conn):
        a1, i1 = backfill_rollups(seeded_conn, "us")
        a2, i2 = backfill_rollups(seeded_conn, "us")
        # Counts should be the same (upsert overwrites, doesn't duplicate)
        n = seeded_conn.execute("SELECT COUNT(*) FROM daily_rollup_archetype").fetchone()[0]
        assert n == 2  # 2 dates, 1 archetype each


# ── Edge cases ────────────────────────────────────────────────────────────────


class TestEdgeCases:
    def test_no_data_for_date(self, seeded_conn):
        """Upsert for a date with no observations returns 0."""
        n = upsert_archetype_rollup(seeded_conn, "us", "2025-01-01")
        assert n == 0

    def test_wrong_realm(self, seeded_conn):
        n = upsert_archetype_rollup(seeded_conn, "eu", "2026-03-15")
        assert n == 0

    def test_item_with_no_quantity(self, seeded_conn):
        """Item 200 day 1 has one row with NULL quantity_listed."""
        upsert_item_rollup(seeded_conn, "us", "2026-03-15")
        row = seeded_conn.execute(
            "SELECT qty_weight_sum FROM daily_rollup_item "
            "WHERE item_id = 200 AND obs_date = '2026-03-15'"
        ).fetchone()
        # Item 200: (300*1 + 310*20) — NULL quantity -> COALESCE to 1
        assert row["qty_weight_sum"] == pytest.approx(1 + 20)
