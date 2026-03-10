"""Tests for crafting_advisor — window logic, compression/expansion, volume gate."""

from __future__ import annotations

import sqlite3
from datetime import date

import pytest

from wow_forecaster.db.schema import apply_schema
from wow_forecaster.recommendations.crafting_advisor import (
    CraftingWindow,
    _compute_margin_status,
    _compute_windows,
    _fetch_archetype_prices_bulk,
    _find_best_window,
    build_crafting_opportunities,
    rank_crafting_opportunities,
)


def _make_db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    apply_schema(conn)
    return conn


def _insert_category(conn) -> int:
    conn.execute(
        "INSERT OR IGNORE INTO item_categories (slug, display_name, archetype_tag) "
        "VALUES ('test.cat', 'Test', 'test');"
    )
    return conn.execute(
        "SELECT category_id FROM item_categories WHERE slug='test.cat';"
    ).fetchone()[0]


def _insert_item(conn, item_id: int, cat_id: int | None = None) -> None:
    if cat_id is None:
        cat_id = _insert_category(conn)
    conn.execute(
        "INSERT OR IGNORE INTO items (item_id, name, category_id, expansion_slug, quality) "
        "VALUES (?, 'Item', ?, 'midnight', 'common');",
        (item_id, cat_id),
    )


def _insert_price(conn, item_id: int, obs_date: str, price: float, qty: int = 200) -> None:
    obs_id = conn.execute(
        "INSERT INTO market_observations_raw "
        "(item_id, realm_slug, faction, observed_at, source, is_processed) "
        "VALUES (?, 'us', 'neutral', ?, 'test', 1) RETURNING obs_id;",
        (item_id, f"{obs_date}T12:00:00Z"),
    ).fetchone()[0]
    conn.execute(
        "INSERT INTO market_observations_normalized "
        "(obs_id, item_id, realm_slug, observed_at, price_gold, quantity_listed, is_outlier) "
        "VALUES (?, ?, 'us', ?, ?, ?, 0);",
        (obs_id, item_id, f"{obs_date}T12:00:00Z", price, qty),
    )


def _insert_recipe(conn, recipe_id: int, output_item_id: int) -> None:
    conn.execute(
        "INSERT OR IGNORE INTO recipes "
        "(recipe_id, profession_slug, output_item_id, output_quantity, expansion_slug) "
        "VALUES (?, 'alchemy', ?, 1, 'midnight');",
        (recipe_id, output_item_id),
    )


def _insert_reagent(conn, recipe_id: int, ingredient_id: int, qty: int = 1) -> None:
    conn.execute(
        "INSERT INTO recipe_reagents (recipe_id, ingredient_item_id, quantity, reagent_type) "
        "VALUES (?, ?, ?, 'required');",
        (recipe_id, ingredient_id, qty),
    )


def _insert_margin_snapshot(
    conn,
    recipe_id: int,
    obs_date: str,
    output_price: float,
    craft_cost: float,
    coverage: float = 1.0,
) -> None:
    margin = output_price - craft_cost
    pct = margin / output_price if output_price > 0 else None
    conn.execute(
        "INSERT OR REPLACE INTO crafting_margin_snapshots "
        "(recipe_id, realm_slug, obs_date, output_price_gold, craft_cost_gold, "
        " margin_gold, margin_pct, ingredient_coverage_pct) "
        "VALUES (?, 'us', ?, ?, ?, ?, ?, ?);",
        (recipe_id, obs_date, output_price, craft_cost, margin, pct, coverage),
    )


class TestFindBestWindow:
    def test_positive_margin_selected(self):
        windows = {
            CraftingWindow.NOW_NOW:  10.0,
            CraftingWindow.NOW_7D:   30.0,
            CraftingWindow._7D_7D:   20.0,
            CraftingWindow.NOW_28D:  25.0,
            CraftingWindow._7D_28D:  None,
            CraftingWindow._28D_28D: 15.0,
        }
        best_w, best_m = _find_best_window(windows, 100.0)
        assert best_w == CraftingWindow.NOW_7D
        assert best_m == 30.0

    def test_tie_broken_by_earliest_sell(self):
        windows = {
            CraftingWindow.NOW_NOW:  20.0,
            CraftingWindow.NOW_7D:   20.0,
            CraftingWindow._7D_7D:   20.0,
            CraftingWindow.NOW_28D:  20.0,
            CraftingWindow._7D_28D:  20.0,
            CraftingWindow._28D_28D: 20.0,
        }
        best_w, best_m = _find_best_window(windows, 100.0)
        assert best_w == CraftingWindow.NOW_NOW  # earliest
        assert best_m == 20.0

    def test_all_none_returns_now_now(self):
        windows = {w: None for w in CraftingWindow}
        best_w, best_m = _find_best_window(windows, 100.0)
        assert best_w == CraftingWindow.NOW_NOW
        assert best_m is None


class TestComputeMarginStatus:
    def test_expanding_positive_slope(self):
        # +0.05/day slope — well above expansion threshold of +0.02
        history = [(f"2026-03-{i:02d}", 0.10 + i * 0.05) for i in range(1, 8)]
        slope, rank, status = _compute_margin_status(history, history[-1][1], -0.02, 0.02)
        assert status == "expanding"
        assert slope is not None and slope > 0.02

    def test_compressing_negative_slope(self):
        # -0.05/day slope — well below compression threshold of -0.02
        history = [(f"2026-03-{i:02d}", 0.50 - i * 0.05) for i in range(1, 8)]
        slope, rank, status = _compute_margin_status(history, history[-1][1], -0.02, 0.02)
        assert status == "compressing"

    def test_stable_flat_slope(self):
        history = [(f"2026-03-{i:02d}", 0.20) for i in range(1, 8)]
        slope, rank, status = _compute_margin_status(history, 0.20, -0.02, 0.02)
        assert status == "stable"
        assert abs(slope) < 0.01

    def test_unknown_when_too_few_points(self):
        history = [("2026-03-01", 0.20), ("2026-03-02", 0.22)]
        slope, rank, status = _compute_margin_status(history, 0.22, -0.02, 0.02)
        assert status == "unknown"

    def test_percentile_rank_midpoint(self):
        history = [(f"2026-03-{i:02d}", float(i)) for i in range(1, 11)]
        slope, rank, status = _compute_margin_status(history, 5.0, -0.02, 0.02)
        assert rank is not None
        assert 0.0 <= rank <= 1.0


class TestBuildCraftingOpportunities:
    def test_returns_opportunity_for_recipe_with_data(self):
        conn = _make_db()
        cat_id = _insert_category(conn)
        _insert_item(conn, 1000, cat_id)
        _insert_item(conn, 2001, cat_id)

        _insert_recipe(conn, 501, 1000)
        _insert_reagent(conn, 501, 2001, 2)

        obs_date = "2026-03-06"
        _insert_price(conn, 1000, obs_date, 100.0, qty=300)
        _insert_price(conn, 2001, obs_date, 10.0)
        _insert_margin_snapshot(conn, 501, obs_date, 100.0, 20.0)
        conn.commit()

        opps = build_crafting_opportunities(
            conn=conn,
            realm_slug="us",
            run_date=date(2026, 3, 6),
            min_volume_units=50,
            min_ingredient_coverage=0.5,
        )
        assert len(opps) == 1
        opp = opps[0]
        assert opp.recipe_id == 501
        assert opp.output_item_id == 1000
        assert opp.current_output_price_gold == pytest.approx(100.0, rel=0.01)

    def test_volume_gate_excludes_low_volume(self):
        conn = _make_db()
        cat_id = _insert_category(conn)
        _insert_item(conn, 1000, cat_id)
        _insert_item(conn, 2001, cat_id)

        _insert_recipe(conn, 501, 1000)
        _insert_reagent(conn, 501, 2001, 1)

        obs_date = "2026-03-06"
        # Only 10 units listed — below min_volume_units=50
        _insert_price(conn, 1000, obs_date, 100.0, qty=10)
        _insert_price(conn, 2001, obs_date, 10.0)
        _insert_margin_snapshot(conn, 501, obs_date, 100.0, 20.0)
        conn.commit()

        opps = build_crafting_opportunities(
            conn=conn,
            realm_slug="us",
            run_date=date(2026, 3, 6),
            min_volume_units=50,
        )
        assert len(opps) == 0

    def test_no_margin_snapshot_excluded(self):
        conn = _make_db()
        cat_id = _insert_category(conn)
        _insert_item(conn, 1000, cat_id)
        _insert_item(conn, 2001, cat_id)

        _insert_recipe(conn, 501, 1000)
        _insert_reagent(conn, 501, 2001, 1)

        _insert_price(conn, 1000, "2026-03-06", 100.0, qty=200)
        _insert_price(conn, 2001, "2026-03-06", 10.0)
        # No margin snapshot inserted
        conn.commit()

        opps = build_crafting_opportunities(
            conn=conn, realm_slug="us", run_date=date(2026, 3, 6)
        )
        assert len(opps) == 0

    def test_now_now_window_matches_snapshot(self):
        conn = _make_db()
        cat_id = _insert_category(conn)
        _insert_item(conn, 1000, cat_id)
        _insert_item(conn, 2001, cat_id)
        _insert_recipe(conn, 501, 1000)
        _insert_reagent(conn, 501, 2001, 1)
        _insert_price(conn, 1000, "2026-03-06", 100.0, qty=300)
        _insert_price(conn, 2001, "2026-03-06", 40.0)
        _insert_margin_snapshot(conn, 501, "2026-03-06", 100.0, 40.0)
        conn.commit()

        opps = build_crafting_opportunities(
            conn=conn, realm_slug="us", run_date=date(2026, 3, 6), min_volume_units=50
        )
        assert len(opps) == 1
        # now->now margin should be ~60g (100 - 40)
        now_now = opps[0].windows.get(CraftingWindow.NOW_NOW)
        assert now_now is not None
        assert abs(now_now - 60.0) < 5.0


def _insert_archetype(conn, archetype_id: int, slug: str = "test.arch") -> int:
    conn.execute(
        "INSERT OR IGNORE INTO economic_archetypes "
        "(archetype_id, slug, display_name, category_tag, sub_tag, "
        " is_transferable, transfer_confidence) "
        "VALUES (?, ?, 'Test Archetype', 'mat', NULL, 1, 0.8);",
        (archetype_id, slug),
    )
    return archetype_id


def _assign_archetype(conn, item_id: int, archetype_id: int) -> None:
    conn.execute(
        "UPDATE items SET archetype_id = ? WHERE item_id = ?;",
        (archetype_id, item_id),
    )


class TestFetchArchetypePricesBulk:
    def test_returns_weighted_mean_for_archetype(self):
        conn = _make_db()
        cat_id = _insert_category(conn)
        arch_id = _insert_archetype(conn, 10)
        _insert_item(conn, 101, cat_id)
        _insert_item(conn, 102, cat_id)
        _assign_archetype(conn, 101, arch_id)
        _assign_archetype(conn, 102, arch_id)
        # Two items: 40g at qty=100, 60g at qty=100 → weighted avg = 50g
        _insert_price(conn, 101, "2026-03-06", 40.0, qty=100)
        _insert_price(conn, 102, "2026-03-06", 60.0, qty=100)
        conn.commit()

        result = _fetch_archetype_prices_bulk(conn, [arch_id], "us", date(2026, 3, 6))
        assert arch_id in result
        assert abs(result[arch_id] - 50.0) < 1.0

    def test_empty_archetype_ids_returns_empty(self):
        conn = _make_db()
        conn.commit()
        result = _fetch_archetype_prices_bulk(conn, [], "us", date(2026, 3, 6))
        assert result == {}

    def test_unknown_archetype_not_in_result(self):
        conn = _make_db()
        conn.commit()
        result = _fetch_archetype_prices_bulk(conn, [999], "us", date(2026, 3, 6))
        assert 999 not in result


class TestComputeWindowsTrendRatio:
    """Tests for the trend-ratio scaling in _compute_windows() for future horizons."""

    def _make_base_recipe(self):
        return {"recipe_id": 1, "output_item_id": 100, "output_quantity": 1}

    def test_ratio_scaling_applied_when_item_price_differs_from_archetype_avg(self):
        """Item at 20g in archetype averaging 50g; archetype forecast 60g → item projected at 24g."""
        recipe = self._make_base_recipe()
        # Archetype avg = 50g, forecast 7d = 60g → ratio = 1.2
        # Reagent item current = 20g → projected = 20 × 1.2 = 24g
        windows, _ = _compute_windows(
            recipe=recipe,
            reagents=[(200, 1)],  # 1 unit of item 200
            current_output_price=100.0,
            current_craft_cost=20.0,
            item_archetype_map={100: 10, 200: 10},
            forecasts={(10, "7d"): 60.0},
            current_price_map={100: 100.0, 200: 20.0},
            archetype_current_prices={10: 50.0},
        )
        # now->+7d: sell output at archetype 7d forecast; buy reagent at 20 × 1.2 = 24g
        # But output (item 100) is also in archetype 10: output projected = 100 × (60/50) = 120g
        # craft_cost projected = 24g; margin = 120 - 24 = 96g
        margin_now_7d = windows.get(CraftingWindow.NOW_7D)
        assert margin_now_7d is not None
        # Reagent projected via ratio: 20 × (60/50) = 24g
        # Output projected: current craft_cost used for buy=now; output_price from archetype forecast
        # For NOW_7D: buy_h=0 → uses current_craft_cost=20g; sell_h=7 → output = 100 × (60/50) = 120g
        assert abs(margin_now_7d - (120.0 - 20.0)) < 1.0

    def test_ratio_scaling_for_reagent_future_buy(self):
        """When buy horizon > 0, reagent cost uses trend-ratio on item's current price."""
        recipe = self._make_base_recipe()
        # Archetype current = 40g, forecast 7d = 50g → ratio = 1.25
        # Reagent item current = 16g → projected at 7d = 16 × 1.25 = 20g
        windows, _ = _compute_windows(
            recipe=recipe,
            reagents=[(200, 1)],
            current_output_price=100.0,
            current_craft_cost=16.0,
            item_archetype_map={100: 10, 200: 10},
            forecasts={(10, "7d"): 50.0},
            current_price_map={100: 100.0, 200: 16.0},
            archetype_current_prices={10: 40.0},
        )
        # +7d->+7d: buy reagent at 7d horizon (projected = 20g); sell output at 7d (projected = 100×1.25=125g)
        margin_7d_7d = windows.get(CraftingWindow._7D_7D)
        assert margin_7d_7d is not None
        # craft_cost = 20g (reagent at 7d); output = 125g
        assert abs(margin_7d_7d - (125.0 - 20.0)) < 1.0

    def test_fallback_to_archetype_forecast_when_no_archetype_current(self):
        """When archetype_current_prices is empty, falls back to raw archetype forecast."""
        recipe = self._make_base_recipe()
        windows, _ = _compute_windows(
            recipe=recipe,
            reagents=[(200, 1)],
            current_output_price=100.0,
            current_craft_cost=30.0,
            item_archetype_map={100: 10, 200: 10},
            forecasts={(10, "7d"): 55.0},
            current_price_map={100: 100.0, 200: 30.0},
            archetype_current_prices={},  # no archetype current price
        )
        # NOW_7D: sell at archetype forecast=55g (fallback, no ratio scaling)
        margin_now_7d = windows.get(CraftingWindow.NOW_7D)
        assert margin_now_7d is not None
        # Output uses archetype forecast=55g (no item current / archetype current ratio available)
        assert abs(margin_now_7d - (55.0 - 30.0)) < 1.0

    def test_fallback_to_current_price_when_no_forecast_at_all(self):
        """When no archetype forecast exists, falls back to current item price (no change assumed)."""
        recipe = self._make_base_recipe()
        windows, _ = _compute_windows(
            recipe=recipe,
            reagents=[(200, 1)],
            current_output_price=80.0,
            current_craft_cost=30.0,
            item_archetype_map={},  # no archetype mapping
            forecasts={},  # no forecasts
            current_price_map={100: 80.0, 200: 30.0},
            archetype_current_prices={},
        )
        # NOW_7D: no forecast → output falls back to current price=80g
        margin_now_7d = windows.get(CraftingWindow.NOW_7D)
        assert margin_now_7d is not None
        assert abs(margin_now_7d - (80.0 - 30.0)) < 1.0

    def test_zero_archetype_current_falls_back_to_forecast(self):
        """If archetype_current = 0, ratio is skipped to avoid division by zero."""
        recipe = self._make_base_recipe()
        windows, _ = _compute_windows(
            recipe=recipe,
            reagents=[(200, 1)],
            current_output_price=100.0,
            current_craft_cost=20.0,
            item_archetype_map={100: 10, 200: 10},
            forecasts={(10, "7d"): 60.0},
            current_price_map={100: 100.0, 200: 20.0},
            archetype_current_prices={10: 0.0},  # zero → division skipped
        )
        margin_now_7d = windows.get(CraftingWindow.NOW_7D)
        assert margin_now_7d is not None
        # Falls back to archetype forecast: output=60g, reagent=buy_now=20g
        assert abs(margin_now_7d - (60.0 - 20.0)) < 1.0


class TestRankCraftingOpportunities:
    def test_sorts_by_opportunity_score_desc(self):
        from wow_forecaster.recommendations.crafting_advisor import CraftingOpportunity

        opps = [
            CraftingOpportunity(
                recipe_id=i,
                recipe_name=f"Recipe {i}",
                profession_slug="alchemy",
                output_item_id=i,
                opportunity_score=float(i),
            )
            for i in [3, 1, 4, 1, 5]
        ]
        ranked = rank_crafting_opportunities(opps, top_n=3)
        assert len(ranked) == 3
        assert ranked[0].opportunity_score == 5.0
        assert ranked[1].opportunity_score == 4.0
        assert ranked[2].opportunity_score == 3.0

    def test_top_n_respected(self):
        from wow_forecaster.recommendations.crafting_advisor import CraftingOpportunity
        opps = [
            CraftingOpportunity(
                recipe_id=i,
                recipe_name=f"Recipe {i}",
                profession_slug="alchemy",
                output_item_id=i,
                opportunity_score=float(i),
            )
            for i in range(20)
        ]
        ranked = rank_crafting_opportunities(opps, top_n=5)
        assert len(ranked) == 5
