"""Tests for MarginCalculator — crafting margin computation."""

from __future__ import annotations

import sqlite3
from datetime import date, datetime, timezone

import pytest

from wow_forecaster.db.schema import apply_schema
from wow_forecaster.recipes.margin_calculator import MarginCalculator
from wow_forecaster.recipes.recipe_repo import RecipeRepository


def _setup_db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    apply_schema(conn)
    return conn


def _insert_item(conn, item_id: int, name: str = "Item") -> None:
    """Insert a minimal item row (category and archetype must exist first)."""
    # Insert category if not exists
    conn.execute(
        "INSERT OR IGNORE INTO item_categories (slug, display_name, archetype_tag) "
        "VALUES ('test.category', 'Test Category', 'test');"
    )
    cat_id = conn.execute(
        "SELECT category_id FROM item_categories WHERE slug='test.category';"
    ).fetchone()[0]
    conn.execute(
        "INSERT OR IGNORE INTO items (item_id, name, category_id, expansion_slug, quality) "
        "VALUES (?, ?, ?, 'midnight', 'common');",
        (item_id, name, cat_id),
    )


def _insert_price(
    conn,
    item_id: int,
    realm_slug: str,
    observed_at: str,
    price_gold: float,
    quantity: int = 100,
) -> None:
    """Insert a normalized market observation row."""
    # Need a raw observation first
    obs_id = conn.execute(
        "INSERT INTO market_observations_raw "
        "(item_id, realm_slug, faction, observed_at, source, min_buyout_raw, is_processed) "
        "VALUES (?, ?, 'neutral', ?, 'blizzard_api', ?, 1) RETURNING obs_id;",
        (item_id, realm_slug, observed_at, int(price_gold * 10000)),
    ).fetchone()[0]
    conn.execute(
        "INSERT INTO market_observations_normalized "
        "(obs_id, item_id, realm_slug, observed_at, price_gold, quantity_listed, is_outlier) "
        "VALUES (?, ?, ?, ?, ?, ?, 0);",
        (obs_id, item_id, realm_slug, observed_at, price_gold, quantity),
    )


def _insert_recipe(conn, recipe_id: int, output_item_id: int, profession: str = "alchemy") -> None:
    conn.execute(
        "INSERT OR IGNORE INTO recipes "
        "(recipe_id, profession_slug, output_item_id, output_quantity, expansion_slug) "
        "VALUES (?, ?, ?, 1, 'midnight');",
        (recipe_id, profession, output_item_id),
    )


def _insert_reagent(conn, recipe_id: int, ingredient_item_id: int, quantity: int) -> None:
    conn.execute(
        "INSERT INTO recipe_reagents (recipe_id, ingredient_item_id, quantity, reagent_type) "
        "VALUES (?, ?, ?, 'required');",
        (recipe_id, ingredient_item_id, quantity),
    )


class TestMarginCalculator:
    def test_simple_margin_computed(self):
        """Output price 100g, craft cost 40g -> margin 60g (60%)."""
        conn = _setup_db()
        obs_date = "2026-03-06"
        realm = "us"

        _insert_item(conn, 1000)   # output item
        _insert_item(conn, 2001)   # ingredient 1
        _insert_item(conn, 2002)   # ingredient 2

        _insert_price(conn, 1000, realm, f"{obs_date}T12:00:00Z", 100.0)
        _insert_price(conn, 2001, realm, f"{obs_date}T12:00:00Z", 10.0)
        _insert_price(conn, 2002, realm, f"{obs_date}T12:00:00Z", 6.0)

        _insert_recipe(conn, 501, 1000)
        _insert_reagent(conn, 501, 2001, 2)   # 2 × 10g = 20g
        _insert_reagent(conn, 501, 2002, 4)   # 4 × 6g  = 24g  => craft_cost = 44g (wait, output_qty=1)
        # Actually: craft_cost = (2*10 + 4*6) / 1 = 44g
        # margin = 100 - 44 = 56g

        conn.commit()

        calc = MarginCalculator(conn, min_coverage=0.5)
        stats = calc.compute_margins(realm_slug=realm, lookback_days=1, end_date=date(2026, 3, 6))

        assert stats.snapshots_written == 1
        row = conn.execute(
            "SELECT output_price_gold, craft_cost_gold, margin_gold, margin_pct "
            "FROM crafting_margin_snapshots WHERE recipe_id = 501;"
        ).fetchone()
        assert row is not None
        assert abs(row["output_price_gold"] - 100.0) < 0.01
        assert abs(row["craft_cost_gold"] - 44.0) < 0.01
        assert abs(row["margin_gold"] - 56.0) < 0.01
        assert abs(row["margin_pct"] - 0.56) < 0.01
        conn.close()

    def test_no_output_price_skipped(self):
        """Recipe with no price data for output item is skipped."""
        conn = _setup_db()
        realm = "us"
        obs_date = "2026-03-06"

        _insert_item(conn, 1000)   # output item — no price inserted
        _insert_item(conn, 2001)
        _insert_price(conn, 2001, realm, f"{obs_date}T12:00:00Z", 5.0)
        _insert_recipe(conn, 501, 1000)
        _insert_reagent(conn, 501, 2001, 1)
        conn.commit()

        calc = MarginCalculator(conn, min_coverage=0.5)
        stats = calc.compute_margins(realm_slug=realm, lookback_days=1, end_date=date(2026, 3, 6))
        assert stats.snapshots_written == 0
        conn.close()

    def test_low_coverage_skipped(self):
        """Recipe where only 1/3 ingredients have prices is skipped at default threshold."""
        conn = _setup_db()
        realm = "us"
        obs_date = "2026-03-06"

        _insert_item(conn, 1000)
        _insert_item(conn, 2001)
        _insert_item(conn, 2002)  # no price
        _insert_item(conn, 2003)  # no price

        _insert_price(conn, 1000, realm, f"{obs_date}T12:00:00Z", 50.0)
        _insert_price(conn, 2001, realm, f"{obs_date}T12:00:00Z", 5.0)

        _insert_recipe(conn, 501, 1000)
        _insert_reagent(conn, 501, 2001, 1)
        _insert_reagent(conn, 501, 2002, 1)
        _insert_reagent(conn, 501, 2003, 1)
        conn.commit()

        # 1/3 coverage = 0.33 < 0.5 threshold
        calc = MarginCalculator(conn, min_coverage=0.5)
        stats = calc.compute_margins(realm_slug=realm, lookback_days=1, end_date=date(2026, 3, 6))
        assert stats.snapshots_written == 0
        conn.close()

    def test_upsert_overwrites_existing(self):
        """Re-running build-margins updates existing snapshots."""
        conn = _setup_db()
        realm = "us"
        obs_date = "2026-03-06"

        _insert_item(conn, 1000)
        _insert_item(conn, 2001)
        _insert_price(conn, 1000, realm, f"{obs_date}T12:00:00Z", 100.0)
        _insert_price(conn, 2001, realm, f"{obs_date}T12:00:00Z", 20.0)
        _insert_recipe(conn, 501, 1000)
        _insert_reagent(conn, 501, 2001, 1)
        conn.commit()

        calc = MarginCalculator(conn, min_coverage=0.5)
        calc.compute_margins(realm_slug=realm, lookback_days=1, end_date=date(2026, 3, 6))

        # Update price and recompute
        conn.execute(
            "UPDATE market_observations_normalized SET price_gold = 30.0 WHERE item_id = 2001;"
        )
        conn.commit()

        stats = calc.compute_margins(realm_slug=realm, lookback_days=1, end_date=date(2026, 3, 6))
        # Second run updates the row
        row = conn.execute(
            "SELECT craft_cost_gold FROM crafting_margin_snapshots WHERE recipe_id = 501;"
        ).fetchone()
        assert abs(row["craft_cost_gold"] - 30.0) < 0.01
        conn.close()

    def test_output_quantity_divides_cost(self):
        """Recipes that output multiple units: cost is per unit."""
        conn = _setup_db()
        realm = "us"
        obs_date = "2026-03-06"

        _insert_item(conn, 1000)
        _insert_item(conn, 2001)
        _insert_price(conn, 1000, realm, f"{obs_date}T12:00:00Z", 10.0)
        _insert_price(conn, 2001, realm, f"{obs_date}T12:00:00Z", 30.0)

        # Recipe produces 5 units; ingredient = 30g; per-unit cost = 30/5 = 6g
        conn.execute(
            "INSERT OR IGNORE INTO recipes "
            "(recipe_id, profession_slug, output_item_id, output_quantity, expansion_slug) "
            "VALUES (502, 'alchemy', 1000, 5, 'midnight');"
        )
        _insert_reagent(conn, 502, 2001, 1)
        conn.commit()

        calc = MarginCalculator(conn, min_coverage=0.5)
        calc.compute_margins(realm_slug=realm, lookback_days=1, end_date=date(2026, 3, 6))

        row = conn.execute(
            "SELECT craft_cost_gold, margin_gold FROM crafting_margin_snapshots WHERE recipe_id = 502;"
        ).fetchone()
        assert abs(row["craft_cost_gold"] - 6.0) < 0.01   # 30 / 5 = 6
        assert abs(row["margin_gold"] - 4.0) < 0.01       # 10 - 6 = 4
        conn.close()
