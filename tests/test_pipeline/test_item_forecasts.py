"""Tests for _generate_item_forecasts() and the item-level forecast helpers."""

from __future__ import annotations

import sqlite3
from datetime import date, timedelta

import pytest

from wow_forecaster.db.schema import apply_schema
from wow_forecaster.models.forecast import ForecastOutput
from wow_forecaster.pipeline.forecast import (
    _fetch_archetype_prices,
    _fetch_cold_start_blend_data,
    _fetch_item_archetypes,
    _fetch_item_prices,
    _fetch_recipe_item_ids,
    _generate_item_forecasts,
)


def _make_db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    apply_schema(conn)
    return conn


_DEFAULT_CAT_ID: int | None = None


def _ensure_category(conn) -> int:
    global _DEFAULT_CAT_ID
    conn.execute(
        "INSERT OR IGNORE INTO item_categories (slug, display_name, archetype_tag) "
        "VALUES ('test.cat', 'Test', 'test');"
    )
    row = conn.execute(
        "SELECT category_id FROM item_categories WHERE slug='test.cat';"
    ).fetchone()
    return int(row[0])


def _insert_archetype(conn, archetype_id: int, slug: str = "mat.herb.common") -> None:
    conn.execute(
        "INSERT OR IGNORE INTO economic_archetypes "
        "(archetype_id, slug, display_name, category_tag, sub_tag, "
        " is_transferable, transfer_confidence) "
        "VALUES (?, ?, 'Test', 'mat', NULL, 1, 0.8);",
        (archetype_id, slug),
    )


def _insert_item(conn, item_id: int, archetype_id: int | None = None) -> None:
    cat_id = _ensure_category(conn)
    conn.execute(
        "INSERT OR IGNORE INTO items (item_id, name, category_id, expansion_slug, quality, archetype_id) "
        "VALUES (?, 'Item', ?, 'midnight', 'common', ?);",
        (item_id, cat_id, archetype_id),
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


def _insert_price(conn, item_id: int, obs_date: str, price: float, qty: int = 100) -> None:
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


def _make_archetype_forecast(
    archetype_id: int,
    horizon: str,
    predicted: float,
    ci_lower: float,
    ci_upper: float,
    run_id: int = 1,
) -> ForecastOutput:
    return ForecastOutput(
        run_id=run_id,
        archetype_id=archetype_id,
        item_id=None,
        realm_slug="us",
        forecast_horizon=horizon,  # type: ignore[arg-type]
        target_date=date.today() + timedelta(days={"1d": 1, "7d": 7, "28d": 28}[horizon]),
        predicted_price_gold=predicted,
        confidence_lower=ci_lower,
        confidence_upper=ci_upper,
        confidence_pct=0.80,
        model_slug="lgbm_7d_v1",
    )


class TestFetchRecipeItemIds:
    def test_returns_output_and_reagent_items(self):
        conn = _make_db()
        _insert_item(conn, 100)
        _insert_item(conn, 200)
        _insert_recipe(conn, 1, 100)
        _insert_reagent(conn, 1, 200)
        conn.commit()

        ids = _fetch_recipe_item_ids(conn)
        assert set(ids) == {100, 200}

    def test_empty_when_no_recipes(self):
        conn = _make_db()
        conn.commit()
        assert _fetch_recipe_item_ids(conn) == []

    def test_deduplicates_items_across_recipes(self):
        conn = _make_db()
        _insert_item(conn, 100)
        _insert_item(conn, 200)
        _insert_recipe(conn, 1, 100)
        _insert_recipe(conn, 2, 100)  # same output, different recipe
        _insert_reagent(conn, 1, 200)
        _insert_reagent(conn, 2, 200)  # same reagent
        conn.commit()

        ids = _fetch_recipe_item_ids(conn)
        # Should have 100 and 200 each only once
        assert sorted(ids) == [100, 200]


class TestFetchItemArchetypes:
    def test_returns_items_with_archetypes(self):
        conn = _make_db()
        _insert_archetype(conn, 10)
        _insert_item(conn, 100, archetype_id=10)
        _insert_item(conn, 200)  # no archetype
        conn.commit()

        result = _fetch_item_archetypes(conn, [100, 200])
        assert result == {100: 10}
        assert 200 not in result

    def test_empty_item_ids_returns_empty(self):
        conn = _make_db()
        assert _fetch_item_archetypes(conn, []) == {}


class TestFetchItemAndArchetypePrices:
    def test_item_price_returns_weighted_mean(self):
        conn = _make_db()
        _insert_item(conn, 100)
        _insert_price(conn, 100, "2026-03-06", 40.0, qty=100)
        _insert_price(conn, 100, "2026-03-07", 60.0, qty=100)
        conn.commit()

        result = _fetch_item_prices(conn, [100], "us", date(2026, 3, 9))
        assert 100 in result
        assert abs(result[100] - 50.0) < 1.0

    def test_archetype_price_aggregates_across_items(self):
        conn = _make_db()
        _insert_archetype(conn, 10)
        _insert_item(conn, 101, archetype_id=10)
        _insert_item(conn, 102, archetype_id=10)
        _insert_price(conn, 101, "2026-03-06", 20.0, qty=100)
        _insert_price(conn, 102, "2026-03-06", 80.0, qty=100)
        conn.commit()

        result = _fetch_archetype_prices(conn, [10], "us", date(2026, 3, 9))
        assert 10 in result
        assert abs(result[10] - 50.0) < 1.0


class TestGenerateItemForecasts:
    def test_generates_forecast_for_recipe_item_with_archetype(self):
        conn = _make_db()
        _insert_archetype(conn, 10)
        _insert_item(conn, 100, archetype_id=10)
        _insert_recipe(conn, 1, 100)
        _insert_reagent(conn, 1, 100)
        _insert_price(conn, 100, "2026-03-06", 50.0)
        conn.commit()

        arch_forecasts = [
            _make_archetype_forecast(10, "7d", predicted=60.0, ci_lower=54.0, ci_upper=66.0),
        ]

        results = _generate_item_forecasts(conn, run_id=1, archetype_forecasts=arch_forecasts, realm_slug="us")
        assert len(results) == 1
        fc = results[0]
        assert fc.item_id == 100
        assert fc.archetype_id is None
        assert fc.forecast_horizon == "7d"
        # archetype_current ≈ 50g, ratio = 60/50 = 1.2, item_current = 50 → predicted ≈ 60g
        assert abs(fc.predicted_price_gold - 60.0) < 2.0

    def test_ratio_scaling_preserves_item_price_differentiation(self):
        """Two items in same archetype at 20g and 80g should forecast differently."""
        conn = _make_db()
        _insert_archetype(conn, 10)
        _insert_item(conn, 101, archetype_id=10)
        _insert_item(conn, 102, archetype_id=10)
        _insert_recipe(conn, 1, 101)
        _insert_reagent(conn, 1, 102)
        # Both items inserted; archetype avg = (20+80)/2 = 50g
        _insert_price(conn, 101, "2026-03-06", 20.0, qty=100)
        _insert_price(conn, 102, "2026-03-06", 80.0, qty=100)
        conn.commit()

        arch_forecasts = [
            _make_archetype_forecast(10, "7d", predicted=60.0, ci_lower=54.0, ci_upper=66.0),
        ]

        results = _generate_item_forecasts(conn, run_id=1, archetype_forecasts=arch_forecasts, realm_slug="us")
        # ratio = 60/50 = 1.2
        # item 101 (20g): forecast = 20 × 1.2 = 24g
        # item 102 (80g): forecast = 80 × 1.2 = 96g
        fc_map = {fc.item_id: fc.predicted_price_gold for fc in results}
        assert abs(fc_map[101] - 24.0) < 1.0
        assert abs(fc_map[102] - 96.0) < 1.0

    def test_skips_item_without_current_price(self):
        conn = _make_db()
        _insert_archetype(conn, 10)
        _insert_item(conn, 100, archetype_id=10)
        _insert_recipe(conn, 1, 100)
        # No price inserted for item 100
        conn.commit()

        arch_forecasts = [
            _make_archetype_forecast(10, "7d", predicted=60.0, ci_lower=54.0, ci_upper=66.0),
        ]

        results = _generate_item_forecasts(conn, run_id=1, archetype_forecasts=arch_forecasts, realm_slug="us")
        assert results == []

    def test_skips_item_without_archetype(self):
        conn = _make_db()
        _insert_item(conn, 100)  # no archetype
        _insert_recipe(conn, 1, 100)
        _insert_price(conn, 100, "2026-03-06", 50.0)
        conn.commit()

        arch_forecasts = [
            _make_archetype_forecast(10, "7d", predicted=60.0, ci_lower=54.0, ci_upper=66.0),
        ]

        results = _generate_item_forecasts(conn, run_id=1, archetype_forecasts=arch_forecasts, realm_slug="us")
        assert results == []

    def test_returns_empty_when_no_archetype_forecasts(self):
        conn = _make_db()
        _insert_archetype(conn, 10)
        _insert_item(conn, 100, archetype_id=10)
        _insert_recipe(conn, 1, 100)
        _insert_price(conn, 100, "2026-03-06", 50.0)
        conn.commit()

        results = _generate_item_forecasts(conn, run_id=1, archetype_forecasts=[], realm_slug="us")
        assert results == []

    def test_all_three_horizons_generated(self):
        conn = _make_db()
        _insert_archetype(conn, 10)
        _insert_item(conn, 100, archetype_id=10)
        _insert_recipe(conn, 1, 100)
        _insert_price(conn, 100, "2026-03-06", 50.0)
        conn.commit()

        arch_forecasts = [
            _make_archetype_forecast(10, "1d",  predicted=51.0, ci_lower=46.0, ci_upper=56.0),
            _make_archetype_forecast(10, "7d",  predicted=55.0, ci_lower=50.0, ci_upper=60.0),
            _make_archetype_forecast(10, "28d", predicted=60.0, ci_lower=54.0, ci_upper=66.0),
        ]

        results = _generate_item_forecasts(conn, run_id=1, archetype_forecasts=arch_forecasts, realm_slug="us")
        horizons = {fc.forecast_horizon for fc in results}
        assert horizons == {"1d", "7d", "28d"}

    def test_ci_lower_never_exceeds_ci_upper(self):
        conn = _make_db()
        _insert_archetype(conn, 10)
        _insert_item(conn, 100, archetype_id=10)
        _insert_recipe(conn, 1, 100)
        # Very small item current price — stress-tests the ci clamping
        _insert_price(conn, 100, "2026-03-06", 0.01)
        conn.commit()

        arch_forecasts = [
            _make_archetype_forecast(10, "7d", predicted=60.0, ci_lower=54.0, ci_upper=66.0),
        ]

        results = _generate_item_forecasts(conn, run_id=1, archetype_forecasts=arch_forecasts, realm_slug="us")
        for fc in results:
            assert fc.confidence_lower <= fc.confidence_upper, (
                f"CI ordering violated: lower={fc.confidence_lower} upper={fc.confidence_upper}"
            )


class TestFetchItemForecastsInAdvisor:
    """Verify crafting_advisor._fetch_item_forecasts() reads item-level forecasts correctly."""

    def test_reads_item_level_forecasts(self):
        from wow_forecaster.recommendations.crafting_advisor import _fetch_item_forecasts

        conn = _make_db()
        _insert_archetype(conn, 10)
        _insert_item(conn, 100, archetype_id=10)

        # Insert a run and an item-level forecast
        run_id = conn.execute(
            "INSERT INTO run_metadata (run_slug, pipeline_stage, status, config_snapshot, started_at) "
            "VALUES ('test-slug', 'forecast', 'success', '{}', '2026-03-09T00:00:00') RETURNING run_id;"
        ).fetchone()[0]
        conn.execute(
            "INSERT INTO forecast_outputs "
            "(run_id, archetype_id, item_id, realm_slug, forecast_horizon, target_date, "
            " predicted_price_gold, confidence_lower, confidence_upper, confidence_pct, model_slug) "
            "VALUES (?, NULL, ?, 'us', '7d', '2026-03-16', 75.0, 67.5, 82.5, 0.80, 'item_ratio_lgbm_7d');",
            (run_id, 100),
        )
        conn.commit()

        result = _fetch_item_forecasts(conn, "us")
        assert (100, "7d") in result
        assert abs(result[(100, "7d")] - 75.0) < 0.1

    def test_returns_empty_when_no_item_forecasts(self):
        from wow_forecaster.recommendations.crafting_advisor import _fetch_item_forecasts

        conn = _make_db()
        conn.commit()
        result = _fetch_item_forecasts(conn, "us")
        assert result == {}

    def test_item_forecasts_preferred_over_archetype_in_compute_windows(self):
        """When item-level forecasts exist, _compute_windows() should use them over trend-ratio."""
        from wow_forecaster.recommendations.crafting_advisor import _compute_windows

        recipe = {"recipe_id": 1, "output_item_id": 100, "output_quantity": 1}
        # archetype forecast = 60g, item forecast = 75g (different to confirm which is used)
        windows, _ = _compute_windows(
            recipe=recipe,
            reagents=[(200, 1)],
            current_output_price=50.0,
            current_craft_cost=20.0,
            item_archetype_map={100: 10, 200: 10},
            forecasts={(10, "7d"): 60.0},
            current_price_map={100: 50.0, 200: 20.0},
            archetype_current_prices={10: 50.0},
            item_forecasts={(100, "7d"): 75.0, (200, "7d"): 22.0},
        )
        # NOW_7D: sell output at item_forecast=75g; buy reagent at current_craft_cost=20g
        margin_now_7d = windows.get("now->+7d")
        assert margin_now_7d is not None
        assert abs(margin_now_7d - (75.0 - 20.0)) < 1.0


def _insert_mapping(
    conn,
    source_archetype_id: int,
    target_archetype_id: int,
    confidence: float = 0.8,
    source_expansion: str = "tww",
    target_expansion: str = "midnight",
) -> None:
    conn.execute(
        "INSERT INTO archetype_mappings "
        "(source_archetype_id, target_archetype_id, source_expansion, target_expansion, "
        " confidence_score, mapping_rationale) "
        "VALUES (?, ?, ?, ?, ?, 'test');",
        (source_archetype_id, target_archetype_id, source_expansion, target_expansion, confidence),
    )


class TestFetchColdStartBlendData:
    """Tests for _fetch_cold_start_blend_data()."""

    def test_returns_blend_entry_when_source_has_price(self):
        """Returns (price, confidence) keyed by target archetype_id."""
        conn = _make_db()
        _insert_archetype(conn, 1, "tww.herb")
        _insert_archetype(conn, 2, "mid.herb")
        _insert_item(conn, 100, archetype_id=1)
        _insert_price(conn, 100, "2026-03-06", 50.0, qty=100)
        _insert_mapping(conn, source_archetype_id=1, target_archetype_id=2, confidence=0.8)
        conn.commit()

        result = _fetch_cold_start_blend_data(conn, "us", "tww", "midnight")
        assert 2 in result
        source_price, confidence = result[2]
        assert abs(source_price - 50.0) < 1.0
        assert confidence == pytest.approx(0.8)

    def test_omits_entry_when_source_has_no_price(self):
        """If source archetype has no recent price data, target is omitted from result."""
        conn = _make_db()
        _insert_archetype(conn, 1, "tww.herb")
        _insert_archetype(conn, 2, "mid.herb")
        # No price for source archetype
        _insert_mapping(conn, source_archetype_id=1, target_archetype_id=2, confidence=0.8)
        conn.commit()

        result = _fetch_cold_start_blend_data(conn, "us", "tww", "midnight")
        assert result == {}

    def test_empty_when_no_mappings(self):
        """Returns empty dict when archetype_mappings has no rows."""
        conn = _make_db()
        conn.commit()
        result = _fetch_cold_start_blend_data(conn, "us", "tww", "midnight")
        assert result == {}

    def test_filters_by_expansion_pair(self):
        """Only mappings matching source_expansion + target_expansion are returned."""
        conn = _make_db()
        _insert_archetype(conn, 1, "src.herb")
        _insert_archetype(conn, 2, "tgt.herb")
        _insert_item(conn, 100, archetype_id=1)
        _insert_price(conn, 100, "2026-03-06", 60.0, qty=100)
        # Mapping for different expansion pair
        _insert_mapping(conn, 1, 2, confidence=0.9, source_expansion="dragonflight", target_expansion="tww")
        conn.commit()

        # Query for tww→midnight should return nothing
        result = _fetch_cold_start_blend_data(conn, "us", "tww", "midnight")
        assert result == {}

    def test_omits_zero_confidence_mappings(self):
        """Mappings with confidence_score = 0 are excluded (WHERE clause guard)."""
        conn = _make_db()
        _insert_archetype(conn, 1, "tww.mat")
        _insert_archetype(conn, 2, "mid.mat")
        _insert_item(conn, 100, archetype_id=1)
        _insert_price(conn, 100, "2026-03-06", 50.0, qty=100)
        _insert_mapping(conn, 1, 2, confidence=0.0)
        conn.commit()

        result = _fetch_cold_start_blend_data(conn, "us", "tww", "midnight")
        assert result == {}

    def test_multiple_mappings_returned(self):
        """Multiple TWW→Midnight mappings all appear in result."""
        conn = _make_db()
        _insert_archetype(conn, 1, "tww.herb")
        _insert_archetype(conn, 2, "mid.herb")
        _insert_archetype(conn, 3, "tww.ore")
        _insert_archetype(conn, 4, "mid.ore")
        _insert_item(conn, 100, archetype_id=1)
        _insert_item(conn, 200, archetype_id=3)
        _insert_price(conn, 100, "2026-03-06", 50.0, qty=100)
        _insert_price(conn, 200, "2026-03-06", 30.0, qty=100)
        _insert_mapping(conn, 1, 2, confidence=0.8)
        _insert_mapping(conn, 3, 4, confidence=0.7)
        conn.commit()

        result = _fetch_cold_start_blend_data(conn, "us", "tww", "midnight")
        assert set(result.keys()) == {2, 4}
        assert abs(result[2][0] - 50.0) < 1.0
        assert abs(result[4][0] - 30.0) < 1.0
        assert result[2][1] == pytest.approx(0.8)
        assert result[4][1] == pytest.approx(0.7)
