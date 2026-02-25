"""
Tests for baseline forecasting models.

What we test
------------
LastValueModel:
  - Predicts the most recent non-null price.
  - Horizon is ignored (prediction is always the last value).
  - Returns None if no training row has price_mean.

RollingMeanModel:
  - Predicts the mean of the last `window` non-null prices.
  - Returns None when fewer than min_rows prices are available.
  - Uses only the tail of training data (last `window` rows).

DayOfWeekModel:
  - Predicts the historical average for the target weekday.
  - Falls back to overall mean when the target weekday is under-represented.
  - Returns None if no training rows have valid prices/dates.

SimpleVolatilityModel:
  - Predicts the rolling mean (same as RollingMeanModel for point forecast).
  - Exposes predicted_volatility_pct (std / mean ratio).
  - Returns None for both when insufficient data.

All models:
  - Return None after fit on an empty row list.
  - Can be re-fit without side effects from a previous fit.
"""

from __future__ import annotations

from datetime import date

import pytest

from wow_forecaster.backtest.models import (
    DayOfWeekModel,
    LastValueModel,
    RollingMeanModel,
    SimpleVolatilityModel,
    all_baseline_models,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _row(price: float | None, obs_date: date | None = None) -> dict:
    """Create a minimal feature row dict."""
    return {
        "price_mean": price,
        "obs_date": obs_date or date(2024, 9, 1),
    }


def _rows_with_dates(prices: list[tuple[date, float | None]]) -> list[dict]:
    """Create rows from (date, price) pairs."""
    return [{"price_mean": p, "obs_date": d} for d, p in prices]


# ── LastValueModel ─────────────────────────────────────────────────────────────

def test_last_value_returns_most_recent_price() -> None:
    model = LastValueModel()
    rows  = [_row(100.0), _row(120.0), _row(150.0)]
    model.fit(rows)
    assert model.predict(1) == pytest.approx(150.0)


def test_last_value_skips_none_prices() -> None:
    """Uses the last non-None price, ignoring trailing Nones."""
    model = LastValueModel()
    rows  = [_row(100.0), _row(120.0), _row(None)]
    model.fit(rows)
    assert model.predict(1) == pytest.approx(120.0)


def test_last_value_ignores_horizon() -> None:
    """Prediction is the same regardless of horizon_days."""
    model = LastValueModel()
    model.fit([_row(100.0)])
    assert model.predict(1) == model.predict(7) == model.predict(28)


def test_last_value_returns_none_for_empty_rows() -> None:
    model = LastValueModel()
    model.fit([])
    assert model.predict(1) is None


def test_last_value_returns_none_when_all_prices_none() -> None:
    model = LastValueModel()
    model.fit([_row(None), _row(None)])
    assert model.predict(1) is None


def test_last_value_refit_clears_state() -> None:
    """Re-fitting replaces the previous state."""
    model = LastValueModel()
    model.fit([_row(100.0)])
    assert model.predict(1) == pytest.approx(100.0)
    model.fit([_row(200.0)])
    assert model.predict(1) == pytest.approx(200.0)


# ── RollingMeanModel ───────────────────────────────────────────────────────────

def test_rolling_mean_computes_correct_average() -> None:
    model = RollingMeanModel(window=3, min_rows=1)
    rows  = [_row(100.0), _row(200.0), _row(300.0)]
    model.fit(rows)
    assert model.predict(1) == pytest.approx(200.0)


def test_rolling_mean_uses_only_tail() -> None:
    """Only the last `window` rows are considered."""
    model = RollingMeanModel(window=2, min_rows=1)
    rows  = [_row(10.0), _row(20.0), _row(100.0), _row(200.0)]
    model.fit(rows)
    # Only last 2 rows: (100 + 200) / 2 = 150
    assert model.predict(1) == pytest.approx(150.0)


def test_rolling_mean_ignores_none_in_window() -> None:
    """None prices are skipped when computing the mean."""
    model = RollingMeanModel(window=3, min_rows=1)
    rows  = [_row(None), _row(100.0), _row(200.0)]
    model.fit(rows)
    assert model.predict(1) == pytest.approx(150.0)


def test_rolling_mean_returns_none_below_min_rows() -> None:
    model = RollingMeanModel(window=7, min_rows=3)
    rows  = [_row(100.0), _row(200.0)]  # only 2 valid prices
    model.fit(rows)
    assert model.predict(1) is None


def test_rolling_mean_returns_none_for_empty_rows() -> None:
    model = RollingMeanModel()
    model.fit([])
    assert model.predict(1) is None


def test_rolling_mean_horizon_ignored() -> None:
    """Point forecast is the same regardless of horizon."""
    model = RollingMeanModel(window=3, min_rows=1)
    model.fit([_row(100.0), _row(200.0), _row(300.0)])
    assert model.predict(1) == model.predict(7)


# ── DayOfWeekModel ─────────────────────────────────────────────────────────────

def test_day_of_week_correct_weekday_average() -> None:
    """Predicts the average of historical prices for the target weekday."""
    # Monday 2024-09-02, Wednesday 2024-09-04, Monday 2024-09-09
    mon_prices = [100.0, 120.0]
    rows = _rows_with_dates([
        (date(2024, 9, 2),  100.0),  # Monday
        (date(2024, 9, 4),  200.0),  # Wednesday
        (date(2024, 9, 9),  120.0),  # Monday
    ])
    model = DayOfWeekModel(min_rows=1)
    model.fit(rows)
    # Last training date = 2024-09-09 (Monday); horizon=7 → target is next Monday
    # Target weekday = Monday; avg of [100, 120] = 110
    pred = model.predict(horizon_days=7)
    assert pred == pytest.approx(sum(mon_prices) / len(mon_prices))


def test_day_of_week_falls_back_to_overall_mean() -> None:
    """Falls back to overall mean when target weekday has fewer than min_rows."""
    rows = _rows_with_dates([
        (date(2024, 9, 2), 100.0),   # Monday
        (date(2024, 9, 3), 200.0),   # Tuesday
        (date(2024, 9, 4), 300.0),   # Wednesday
    ])
    model = DayOfWeekModel(min_rows=2)  # need 2 observations per weekday
    model.fit(rows)
    # last date = Wed; horizon=1 → target is Thursday.
    # Thursday has 0 observations (< min_rows=2) → fallback to overall mean
    overall_mean = (100.0 + 200.0 + 300.0) / 3
    pred = model.predict(horizon_days=1)
    assert pred == pytest.approx(overall_mean)


def test_day_of_week_returns_none_for_empty_rows() -> None:
    model = DayOfWeekModel()
    model.fit([])
    assert model.predict(1) is None


def test_day_of_week_returns_none_when_no_valid_dates() -> None:
    """Returns None when all rows have a None obs_date (fit skips them)."""
    model = DayOfWeekModel()
    # Construct rows directly with obs_date=None so the fit loop skips them.
    model.fit([{"price_mean": 100.0, "obs_date": None}])
    # No last_date was recorded → predict returns None.
    assert model.predict(1) is None


def test_day_of_week_refit_clears_previous_state() -> None:
    """Re-fitting discards previous weekday averages."""
    model = DayOfWeekModel(min_rows=1)
    rows1 = _rows_with_dates([(date(2024, 9, 2), 1000.0)])  # Monday = 1000
    model.fit(rows1)
    rows2 = _rows_with_dates([(date(2024, 9, 2), 50.0)])    # Monday = 50
    model.fit(rows2)
    pred = model.predict(7)
    assert pred == pytest.approx(50.0)


# ── SimpleVolatilityModel ──────────────────────────────────────────────────────

def test_simple_volatility_mean_is_point_forecast() -> None:
    """Point forecast equals rolling mean of training prices."""
    model = SimpleVolatilityModel(window=3, min_rows=1)
    model.fit([_row(100.0), _row(200.0), _row(300.0)])
    assert model.predict(1) == pytest.approx(200.0)


def test_simple_volatility_exposes_volatility_pct() -> None:
    """predicted_volatility_pct is std/mean ratio for the training window."""
    # Prices: 90, 100, 110 → mean=100, std=~8.16, ratio=~0.0816
    model = SimpleVolatilityModel(window=3, min_rows=3)
    model.fit([_row(90.0), _row(100.0), _row(110.0)])
    assert model.predicted_volatility_pct is not None
    assert 0.0 < model.predicted_volatility_pct < 0.2


def test_simple_volatility_returns_none_below_min_rows() -> None:
    model = SimpleVolatilityModel(window=7, min_rows=3)
    model.fit([_row(100.0), _row(200.0)])
    assert model.predict(1) is None
    assert model.predicted_volatility_pct is None


def test_simple_volatility_zero_for_constant_prices() -> None:
    """Volatility is 0 when all prices are identical."""
    model = SimpleVolatilityModel(window=3, min_rows=3)
    model.fit([_row(100.0), _row(100.0), _row(100.0)])
    assert model.predicted_volatility_pct == pytest.approx(0.0)


def test_simple_volatility_returns_none_for_empty_rows() -> None:
    model = SimpleVolatilityModel()
    model.fit([])
    assert model.predict(1) is None
    assert model.predicted_volatility_pct is None


# ── all_baseline_models ────────────────────────────────────────────────────────

def test_all_baseline_models_returns_four_instances() -> None:
    models = all_baseline_models()
    assert len(models) == 4
    names = {m.name for m in models}
    assert names == {"last_value", "rolling_mean", "day_of_week", "simple_volatility"}


def test_all_baseline_models_are_independent_instances() -> None:
    """Two calls return completely independent model objects."""
    models_a = all_baseline_models()
    models_b = all_baseline_models()
    for a, b in zip(models_a, models_b):
        assert a is not b
