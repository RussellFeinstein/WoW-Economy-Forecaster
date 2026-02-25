"""
Tests for lag, rolling, momentum, and forward-looking target features.

All tests use synthetic DailyAggRow data — no DB dependency.
Prices are deterministic so expected values can be computed by hand.
"""

from __future__ import annotations

import math
from datetime import date, timedelta

import pytest

from wow_forecaster.config import FeatureConfig
from wow_forecaster.features.daily_agg import DailyAggRow
from wow_forecaster.features.lag_rolling import compute_lag_rolling_features

# Default config used across all tests unless overridden.
_CFG = FeatureConfig(
    lag_days=[1, 7],
    rolling_windows=[7],
    cold_start_threshold=30,
    training_lookback_days=180,
    target_horizons_days=[1, 7],
)


def _make_rows(prices: list[float | None], start: date = date(2025, 1, 1)) -> list[DailyAggRow]:
    """Build a single (archetype_id=1, realm='area-52') series with given prices."""
    rows = []
    for i, price in enumerate(prices):
        rows.append(
            DailyAggRow(
                archetype_id=1,
                realm_slug="area-52",
                obs_date=start + timedelta(days=i),
                price_mean=price,
                price_min=price,
                price_max=price,
                market_value_mean=None,
                historical_value_mean=None,
                obs_count=1 if price is not None else 0,
                quantity_sum=None,
                auctions_sum=None,
                is_volume_proxy=True,
            )
        )
    return rows


class TestLagFeatures:
    def test_first_row_lag_1d_is_none(self):
        """The very first row has no prior data — price_lag_1d must be None."""
        rows = _make_rows([100.0, 110.0])
        result = compute_lag_rolling_features(rows, _CFG)
        assert result[0]["price_lag_1d"] is None

    def test_second_row_lag_1d_equals_first_price(self):
        """price_lag_1d for the second row must equal the first row's price."""
        rows = _make_rows([100.0, 110.0])
        result = compute_lag_rolling_features(rows, _CFG)
        assert result[1]["price_lag_1d"] == pytest.approx(100.0)

    def test_lag_7d_correct_calendar_offset(self):
        """price_lag_7d for row 7 must equal row 0's price (calendar-accurate)."""
        prices = [100.0 + i * 10 for i in range(10)]   # 100, 110, 120, …
        rows = _make_rows(prices)
        result = compute_lag_rolling_features(rows, _CFG)
        assert result[7]["price_lag_7d"] == pytest.approx(100.0)

    def test_lag_28d_none_when_insufficient_history(self):
        """price_lag_28d is None for the first 28 rows of a new series."""
        cfg = FeatureConfig(lag_days=[28], rolling_windows=[7],
                            cold_start_threshold=30, training_lookback_days=180,
                            target_horizons_days=[1])
        rows = _make_rows([100.0] * 27)
        result = compute_lag_rolling_features(rows, cfg)
        assert all(r["price_lag_28d"] is None for r in result)

    def test_missing_day_makes_lag_none(self):
        """When day N is missing (price_mean=None), the next day's price_lag_1d is None."""
        rows = _make_rows([100.0, None, 120.0])
        result = compute_lag_rolling_features(rows, _CFG)
        # row index 2 is obs_date = start + 2 days; lag_1d looks back 1 day to the None row
        assert result[2]["price_lag_1d"] is None


class TestRollingFeatures:
    def test_rolling_mean_7d_correct_value(self):
        """rolling_mean_7d for day 6 is the mean of days 0–6."""
        prices = [float(i + 1) for i in range(10)]   # 1, 2, 3, …, 10
        rows = _make_rows(prices)
        result = compute_lag_rolling_features(rows, _CFG)
        # Day 6 window: days 0-6 = prices 1..7; mean = 4.0
        assert result[6]["price_roll_mean_7d"] == pytest.approx(4.0)

    def test_rolling_std_7d_correct_value(self):
        """rolling_std_7d is the population std of the 7-day window."""
        prices = [float(i + 1) for i in range(10)]
        rows = _make_rows(prices)
        result = compute_lag_rolling_features(rows, _CFG)
        # Population std of [1,2,3,4,5,6,7] = sqrt((49-16) / 1) hmm let me compute:
        # mean = 4.0; E[x²] = (1+4+9+16+25+36+49)/7 = 140/7 = 20.0
        # variance = 20.0 - 16.0 = 4.0; std = 2.0
        assert result[6]["price_roll_std_7d"] == pytest.approx(2.0)

    def test_rolling_std_zero_when_all_prices_identical(self):
        """Rolling std must be 0.0 (not negative) when all prices in window are equal."""
        rows = _make_rows([100.0] * 10)
        result = compute_lag_rolling_features(rows, _CFG)
        for r in result:
            std = r["price_roll_std_7d"]
            if std is not None:
                assert std >= 0.0

    def test_rolling_none_for_insufficient_initial_window(self):
        """For days 0–5, the 7-day window has fewer than 7 values; mean is still computed."""
        rows = _make_rows([100.0] * 3)
        result = compute_lag_rolling_features(rows, _CFG)
        # Even with only 1 value, rolling_mean_7d should return that value (not None)
        assert result[0]["price_roll_mean_7d"] == pytest.approx(100.0)


class TestMomentum:
    def test_pct_change_7d_correct_formula(self):
        """pct_change_7d = (current - lag_7d) / lag_7d."""
        prices = [100.0] * 7 + [110.0]   # day 7 price is 110, day 0 was 100
        rows = _make_rows(prices)
        result = compute_lag_rolling_features(rows, _CFG)
        # Day 7: lag_7d = 100.0, price = 110.0 → pct_change = 0.10
        assert result[7]["price_pct_change_7d"] == pytest.approx(0.10)

    def test_pct_change_none_when_lag_none(self):
        """pct_change_7d is None when price_lag_7d is None."""
        rows = _make_rows([100.0] * 5)
        result = compute_lag_rolling_features(rows, _CFG)
        assert result[0]["price_pct_change_7d"] is None


class TestTargets:
    def test_target_price_1d_equals_next_row_price(self):
        """target_price_1d for day 0 must equal day 1's price."""
        rows = _make_rows([100.0, 110.0, 120.0])
        result = compute_lag_rolling_features(rows, _CFG)
        assert result[0]["target_price_1d"] == pytest.approx(110.0)

    def test_target_price_7d_none_at_tail(self):
        """target_price_7d is None for the last 7 rows (no future data)."""
        rows = _make_rows([100.0] * 8)
        result = compute_lag_rolling_features(rows, _CFG)
        # Row 7 (index 7, last row) has no day+7 in the series
        assert result[7]["target_price_7d"] is None

    def test_series_boundary_no_bleed(self):
        """Two separate (archetype_id, realm) series must not influence each other."""
        rows_a = _make_rows([100.0] * 5)   # archetype 1
        rows_b = []
        for i, r in enumerate(_make_rows([200.0] * 5)):
            # Override archetype_id to 2 to create a second series.
            from dataclasses import replace
            rows_b.append(replace(r, archetype_id=2))

        result = compute_lag_rolling_features(rows_a + rows_b, _CFG)

        # Find rows for archetype 1 and archetype 2.
        arch1 = [r for r in result if r["archetype_id"] == 1]
        arch2 = [r for r in result if r["archetype_id"] == 2]

        # Lag features for arch1 should reflect arch1 prices (100), not arch2 (200).
        assert arch1[1]["price_lag_1d"] == pytest.approx(100.0)
        assert arch2[1]["price_lag_1d"] == pytest.approx(200.0)
