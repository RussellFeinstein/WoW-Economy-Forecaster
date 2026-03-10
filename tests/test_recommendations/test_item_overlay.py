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

fetch_item_rois():
  - Returns empty list when no item-level forecasts exist.
  - Computes roi_pct correctly: (forecast - current) / current.
  - Buy action: highest ROI first.
  - Sell action: lowest ROI first.
  - Respects top_n limit.
  - Only returns items in the given archetype.
  - Realm isolation: different realm returns nothing.
  - Uses the most recent forecast when multiple exist.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

from wow_forecaster.recommendations.item_overlay import (
    ItemDiscountRow,
    ItemForecastRoi,
    fetch_item_discounts,
    fetch_item_rois,
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


def _insert_item(
    conn: sqlite3.Connection,
    item_id: int,
    name: str,
    archetype_id: int,
    expansion_slug: str = "tww",
) -> None:
    conn.execute(
        "INSERT INTO items(item_id, name, archetype_id, expansion_slug) VALUES (?,?,?,?)",
        (item_id, name, archetype_id, expansion_slug),
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

    def test_expansion_slug_filter_excludes_other_expansions(self, conn):
        # TWW item and Midnight item in the same archetype.
        # Filtering by "midnight" should return only the Midnight item.
        _insert_item(conn, 1, "TWW Flask",      archetype_id=1, expansion_slug="tww")
        _insert_item(conn, 2, "Midnight Flask", archetype_id=1, expansion_slug="midnight")
        _insert_obs(conn, 1, "us", 80.0, _now_iso())
        _insert_obs(conn, 2, "us", 80.0, _now_iso())
        conn.commit()

        rows = fetch_item_discounts(conn, archetype_id=1, realm_slug="us",
                                    archetype_mean_gold=100.0, expansion_slug="midnight")
        assert len(rows) == 1
        assert rows[0].name == "Midnight Flask"

    def test_expansion_slug_none_returns_all_expansions(self, conn):
        # No filter (None) returns items from any expansion.
        _insert_item(conn, 1, "TWW Flask",      archetype_id=1, expansion_slug="tww")
        _insert_item(conn, 2, "Midnight Flask", archetype_id=1, expansion_slug="midnight")
        _insert_obs(conn, 1, "us", 80.0, _now_iso())
        _insert_obs(conn, 2, "us", 80.0, _now_iso())
        conn.commit()

        rows = fetch_item_discounts(conn, archetype_id=1, realm_slug="us",
                                    archetype_mean_gold=100.0, expansion_slug=None)
        assert len(rows) == 2

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


class TestPriceZScore:
    """price_z_score = (archetype_mean - item_price) / population_std of all item prices."""

    def test_single_item_z_score_is_zero(self, conn):
        # Only one item → std=0 → z_score fallback 0.0
        _insert_item(conn, 1, "Herb A", archetype_id=1)
        _insert_obs(conn, 1, "us", 70.0, _now_iso())
        conn.commit()

        rows = fetch_item_discounts(conn, archetype_id=1, realm_slug="us",
                                    archetype_mean_gold=100.0)
        assert rows[0].price_z_score == 0.0

    def test_two_items_z_scores_opposite_sign(self, conn):
        # Item A at 60g (below mean), Item B at 140g (above mean)
        # archetype_mean = 100g
        # deviations from mean: 40g and -40g → std = 40g
        # z_score A = (100 - 60) / 40 = +1.0  (underpriced)
        # z_score B = (100 - 140) / 40 = -1.0  (overpriced)
        _insert_item(conn, 1, "Cheap", archetype_id=1)
        _insert_item(conn, 2, "Pricey", archetype_id=1)
        _insert_obs(conn, 1, "us",  60.0, _now_iso())
        _insert_obs(conn, 2, "us", 140.0, _now_iso())
        conn.commit()

        rows = fetch_item_discounts(conn, archetype_id=1, realm_slug="us",
                                    archetype_mean_gold=100.0, action="buy")
        cheap  = next(r for r in rows if r.item_id == 1)
        pricey = next(r for r in rows if r.item_id == 2)
        assert cheap.price_z_score  == pytest.approx(+1.0)
        assert pricey.price_z_score == pytest.approx(-1.0)

    def test_item_at_mean_has_zero_z_score(self, conn):
        _insert_item(conn, 1, "At Mean",  archetype_id=1)
        _insert_item(conn, 2, "Off Mean", archetype_id=1)
        _insert_obs(conn, 1, "us", 100.0, _now_iso())
        _insert_obs(conn, 2, "us",  80.0, _now_iso())
        conn.commit()

        rows = fetch_item_discounts(conn, archetype_id=1, realm_slug="us",
                                    archetype_mean_gold=100.0, action="hold")
        at_mean = next(r for r in rows if r.item_id == 1)
        assert at_mean.price_z_score == pytest.approx(0.0)

    def test_z_score_field_present_on_all_rows(self, conn):
        for item_id, price in [(1, 50.0), (2, 100.0), (3, 150.0)]:
            _insert_item(conn, item_id, f"Item {item_id}", archetype_id=1)
            _insert_obs(conn, item_id, "us", price, _now_iso())
        conn.commit()

        rows = fetch_item_discounts(conn, archetype_id=1, realm_slug="us",
                                    archetype_mean_gold=100.0)
        assert all(hasattr(r, "price_z_score") for r in rows)
        assert all(isinstance(r.price_z_score, float) for r in rows)


# ── Fixtures and helpers for fetch_item_rois tests ────────────────────────────

@pytest.fixture
def roi_conn() -> sqlite3.Connection:
    """In-memory DB with schema for fetch_item_rois tests (includes forecast_outputs)."""
    from wow_forecaster.db.schema import apply_schema
    db = sqlite3.connect(":memory:")
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA foreign_keys = OFF;")
    apply_schema(db)
    db.commit()
    return db


def _insert_item_roi(conn, item_id: int, name: str, archetype_id: int) -> None:
    conn.execute(
        "INSERT OR IGNORE INTO item_categories (slug, display_name, archetype_tag) "
        "VALUES ('test.cat', 'Test', 'test');"
    )
    cat_id = conn.execute(
        "SELECT category_id FROM item_categories WHERE slug='test.cat';"
    ).fetchone()[0]
    conn.execute(
        "INSERT OR IGNORE INTO economic_archetypes "
        "(archetype_id, slug, display_name, category_tag, sub_tag, "
        " is_transferable, transfer_confidence) "
        "VALUES (?, 'test.arch', 'Test', 'mat', NULL, 1, 0.8);",
        (archetype_id,),
    )
    conn.execute(
        "INSERT OR IGNORE INTO items "
        "(item_id, name, category_id, expansion_slug, quality, archetype_id) "
        "VALUES (?, ?, ?, 'midnight', 'common', ?);",
        (item_id, name, cat_id, archetype_id),
    )


def _insert_obs_roi(conn, item_id: int, price: float, observed_at: str, realm: str = "us") -> None:
    conn.execute(
        "INSERT INTO market_observations_raw "
        "(item_id, realm_slug, faction, observed_at, source, is_processed) "
        "VALUES (?, ?, 'neutral', ?, 'test', 1);",
        (item_id, realm, observed_at),
    )
    obs_id = conn.execute("SELECT last_insert_rowid();").fetchone()[0]
    conn.execute(
        "INSERT INTO market_observations_normalized "
        "(obs_id, item_id, realm_slug, observed_at, price_gold, quantity_listed, is_outlier) "
        "VALUES (?, ?, ?, ?, ?, 1, 0);",
        (obs_id, item_id, realm, observed_at, price),
    )


def _insert_run(conn, run_slug: str = "test-run") -> int:
    row = conn.execute(
        "INSERT INTO run_metadata "
        "(run_slug, pipeline_stage, status, config_snapshot, started_at) "
        "VALUES (?, 'forecast', 'success', '{}', '2026-03-09T00:00:00') "
        "RETURNING run_id;",
        (run_slug,),
    ).fetchone()
    return int(row[0])


def _insert_item_forecast(
    conn,
    item_id: int,
    realm_slug: str,
    horizon: str,
    predicted: float,
    run_id: int,
    created_at: str = "2026-03-09T12:00:00Z",
) -> None:
    conn.execute(
        "INSERT INTO forecast_outputs "
        "(run_id, archetype_id, item_id, realm_slug, forecast_horizon, target_date, "
        " predicted_price_gold, confidence_lower, confidence_upper, confidence_pct, "
        " model_slug, created_at) "
        "VALUES (?, NULL, ?, ?, ?, '2026-03-16', ?, ?, ?, 0.80, 'item_ratio_lgbm_7d', ?);",
        (run_id, item_id, realm_slug, horizon,
         predicted, predicted * 0.9, predicted * 1.1, created_at),
    )


class TestFetchItemRois:
    """Tests for fetch_item_rois()."""

    def test_empty_when_no_forecasts(self, roi_conn):
        _insert_item_roi(roi_conn, 1, "Item A", archetype_id=10)
        _insert_obs_roi(roi_conn, 1, 50.0, _now_iso())
        roi_conn.commit()

        result = fetch_item_rois(roi_conn, archetype_id=10, realm_slug="us", horizon="7d")
        assert result == []

    def test_empty_when_no_current_price(self, roi_conn):
        _insert_item_roi(roi_conn, 1, "Item A", archetype_id=10)
        run_id = _insert_run(roi_conn)
        _insert_item_forecast(roi_conn, 1, "us", "7d", 75.0, run_id)
        roi_conn.commit()

        result = fetch_item_rois(roi_conn, archetype_id=10, realm_slug="us", horizon="7d")
        assert result == []

    def test_roi_computed_correctly(self, roi_conn):
        """roi_pct = (forecast - current) / current."""
        _insert_item_roi(roi_conn, 1, "Item A", archetype_id=10)
        _insert_obs_roi(roi_conn, 1, 50.0, _now_iso())
        run_id = _insert_run(roi_conn)
        _insert_item_forecast(roi_conn, 1, "us", "7d", 75.0, run_id)
        roi_conn.commit()

        result = fetch_item_rois(roi_conn, archetype_id=10, realm_slug="us", horizon="7d")
        assert len(result) == 1
        row = result[0]
        assert isinstance(row, ItemForecastRoi)
        assert row.item_id == 1
        assert row.name == "Item A"
        assert row.forecast_price == pytest.approx(75.0)
        # ROI = (75 - 50) / 50 = 0.50
        assert row.roi_pct == pytest.approx(0.50)

    def test_buy_action_highest_roi_first(self, roi_conn):
        """Buy action: items sorted by roi_pct descending."""
        for item_id, cur, fc in [(1, 50.0, 60.0), (2, 50.0, 80.0), (3, 50.0, 55.0)]:
            _insert_item_roi(roi_conn, item_id, f"Item {item_id}", archetype_id=10)
            _insert_obs_roi(roi_conn, item_id, cur, _now_iso())
        run_id = _insert_run(roi_conn)
        for item_id, _, fc in [(1, 50.0, 60.0), (2, 50.0, 80.0), (3, 50.0, 55.0)]:
            _insert_item_forecast(roi_conn, item_id, "us", "7d", fc, run_id)
        roi_conn.commit()

        result = fetch_item_rois(roi_conn, archetype_id=10, realm_slug="us",
                                 horizon="7d", action="buy")
        rois = [r.roi_pct for r in result]
        assert rois == sorted(rois, reverse=True)
        assert result[0].item_id == 2  # 60% ROI is highest

    def test_sell_action_lowest_roi_first(self, roi_conn):
        """Sell action: items sorted by roi_pct ascending (most bearish first)."""
        for item_id, cur, fc in [(1, 50.0, 40.0), (2, 50.0, 30.0), (3, 50.0, 45.0)]:
            _insert_item_roi(roi_conn, item_id, f"Item {item_id}", archetype_id=10)
            _insert_obs_roi(roi_conn, item_id, cur, _now_iso())
        run_id = _insert_run(roi_conn)
        for item_id, _, fc in [(1, 50.0, 40.0), (2, 50.0, 30.0), (3, 50.0, 45.0)]:
            _insert_item_forecast(roi_conn, item_id, "us", "7d", fc, run_id)
        roi_conn.commit()

        result = fetch_item_rois(roi_conn, archetype_id=10, realm_slug="us",
                                 horizon="7d", action="sell")
        rois = [r.roi_pct for r in result]
        assert rois == sorted(rois)
        assert result[0].item_id == 2  # most negative ROI

    def test_top_n_limits_results(self, roi_conn):
        for item_id in range(1, 8):
            _insert_item_roi(roi_conn, item_id, f"Item {item_id}", archetype_id=10)
            _insert_obs_roi(roi_conn, item_id, 50.0, _now_iso())
        run_id = _insert_run(roi_conn)
        for item_id in range(1, 8):
            _insert_item_forecast(roi_conn, item_id, "us", "7d", 60.0 + item_id, run_id)
        roi_conn.commit()

        result = fetch_item_rois(roi_conn, archetype_id=10, realm_slug="us",
                                 horizon="7d", top_n=3)
        assert len(result) == 3

    def test_archetype_isolation(self, roi_conn):
        """Items in a different archetype are not returned."""
        _insert_item_roi(roi_conn, 1, "Item A", archetype_id=10)
        _insert_item_roi(roi_conn, 2, "Item B", archetype_id=20)
        for item_id in (1, 2):
            _insert_obs_roi(roi_conn, item_id, 50.0, _now_iso())
        run_id = _insert_run(roi_conn)
        for item_id in (1, 2):
            _insert_item_forecast(roi_conn, item_id, "us", "7d", 70.0, run_id)
        roi_conn.commit()

        result = fetch_item_rois(roi_conn, archetype_id=10, realm_slug="us", horizon="7d")
        assert all(r.item_id == 1 for r in result)

    def test_realm_isolation(self, roi_conn):
        """Observations and forecasts for a different realm are not returned."""
        _insert_item_roi(roi_conn, 1, "Item A", archetype_id=10)
        _insert_obs_roi(roi_conn, 1, 50.0, _now_iso(), realm="eu")
        run_id = _insert_run(roi_conn)
        _insert_item_forecast(roi_conn, 1, "eu", "7d", 70.0, run_id)
        roi_conn.commit()

        result = fetch_item_rois(roi_conn, archetype_id=10, realm_slug="us", horizon="7d")
        assert result == []

    def test_most_recent_forecast_used(self, roi_conn):
        """When multiple forecasts exist for the same item, the most recent wins."""
        _insert_item_roi(roi_conn, 1, "Item A", archetype_id=10)
        _insert_obs_roi(roi_conn, 1, 50.0, _now_iso())
        run_id = _insert_run(roi_conn)
        _insert_item_forecast(roi_conn, 1, "us", "7d", 60.0, run_id,
                               created_at="2026-03-08T12:00:00Z")
        _insert_item_forecast(roi_conn, 1, "us", "7d", 90.0, run_id,
                               created_at="2026-03-09T12:00:00Z")
        roi_conn.commit()

        result = fetch_item_rois(roi_conn, archetype_id=10, realm_slug="us", horizon="7d")
        assert len(result) == 1
        # Should use the more recent forecast: 90g → ROI = (90-50)/50 = 0.80
        assert result[0].forecast_price == pytest.approx(90.0)
        assert result[0].roi_pct == pytest.approx(0.80)
