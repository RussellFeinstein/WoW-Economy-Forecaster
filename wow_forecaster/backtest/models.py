"""
Baseline forecasting models.

Why baselines first?
--------------------
Before investing in ML, we must establish what "doing well" actually means.
Baselines are intentionally simple — each one tests a specific hypothesis:

  LastValueModel       → "The market is a random walk; best prediction is
                          the current price."
                          Tests: is ANY signal present at all?

  RollingMeanModel     → "Prices revert to a recent average; short-term noise
                          should be smoothed away."
                          Tests: does mean-reversion beat the random walk?

  DayOfWeekModel       → "WoW markets are seasonal within the week; raid nights
                          (Tue/Wed) and reset days (Tue) drive predictable
                          demand spikes."
                          Tests: is weekly seasonality exploitable?

  SimpleVolatilityModel → "Market turbulence is persistent; yesterday's volatile
                           market is tomorrow's volatile market."
                           Tests: can we predict when prices will be unstable?

If an ML model cannot beat ALL of these baselines, it is not ready for use.

Interface contract
------------------
All models implement:

  fit(rows: list[dict]) → None
    Receive training rows for ONE (archetype_id, realm_slug) series,
    sorted by obs_date ascending.  Extracts any statistics needed for
    prediction.

  predict(horizon_days: int) → float | None
    Produce a price forecast for `horizon_days` days from the last
    training date.  Returns None if insufficient data.

This protocol is intentionally minimal so ML models can implement the same
interface later.  The evaluator calls fit() once per fold per series, then
predict() once per horizon.
"""

from __future__ import annotations

import math
from collections import defaultdict
from datetime import date, timedelta
from typing import Any


class LastValueModel:
    """Naive baseline: predict = most recent observed price.

    Implements the "random walk" hypothesis — tomorrow's price equals today's
    price.  Surprisingly strong in financial markets where price changes are
    hard to forecast.

    In WoW AH terms: "assume the price stays wherever it is right now."

    Notes:
        - horizon_days is ignored; prediction is always the last known price.
        - Returns None if no training row has a non-None price_mean.
    """

    name = "last_value"

    def __init__(self) -> None:
        self._last_price: float | None = None

    def fit(self, rows: list[dict[str, Any]]) -> None:
        """Record the most recent non-null price_mean from the training window."""
        self._last_price = None
        for r in reversed(rows):
            p = r.get("price_mean")
            if p is not None:
                self._last_price = float(p)
                break

    def predict(self, horizon_days: int) -> float | None:
        return self._last_price


class RollingMeanModel:
    """Rolling-mean baseline: predict = mean price over the last N training days.

    Tests whether smoothing reduces forecast error compared to using the single
    most recent observation.  The window N defaults to 7 days — one full WoW
    weekly cycle.

    Returns None if fewer than `min_rows` non-null prices are available in the
    window (prevents predictions based on too little evidence).
    """

    name = "rolling_mean"

    def __init__(self, window: int = 7, min_rows: int = 3) -> None:
        self._window = window
        self._min_rows = min_rows
        self._mean: float | None = None

    def fit(self, rows: list[dict[str, Any]]) -> None:
        """Compute rolling mean over the last `window` rows of training data."""
        self._mean = None
        tail = rows[-self._window:]
        prices = [float(r["price_mean"]) for r in tail if r.get("price_mean") is not None]
        if len(prices) >= self._min_rows:
            self._mean = sum(prices) / len(prices)

    def predict(self, horizon_days: int) -> float | None:
        return self._mean


class DayOfWeekModel:
    """Day-of-week seasonal baseline: predict = historical mean for that weekday.

    WoW markets have strong weekly patterns:
    - Tuesday reset: new raids open, consumable demand spikes.
    - Friday–Sunday: casual players active, demand for convenience items rises.
    - Monday: lowest demand, prices often depressed.

    This model learns the average price for each ISO weekday (1=Mon..7=Sun)
    from the training data, then forecasts using the weekday of the target date
    (train_end + horizon_days).

    Falls back to the overall training mean if the target weekday has
    fewer than `min_rows` observations.
    """

    name = "day_of_week"

    def __init__(self, min_rows: int = 2) -> None:
        self._min_rows = min_rows
        self._last_date: date | None = None
        self._dow_prices: dict[int, list[float]] = defaultdict(list)
        self._overall_mean: float | None = None

    def fit(self, rows: list[dict[str, Any]]) -> None:
        self._dow_prices = defaultdict(list)
        self._last_date = None
        all_prices: list[float] = []

        for r in rows:
            p = r.get("price_mean")
            obs_date = r.get("obs_date")
            if p is None or obs_date is None:
                continue
            price = float(p)
            d = obs_date if isinstance(obs_date, date) else date.fromisoformat(str(obs_date))
            dow = d.isoweekday()  # 1=Mon … 7=Sun
            self._dow_prices[dow].append(price)
            all_prices.append(price)
            self._last_date = d

        self._overall_mean = (sum(all_prices) / len(all_prices)) if all_prices else None

    def predict(self, horizon_days: int) -> float | None:
        if self._last_date is None:
            return None
        target_date = self._last_date + timedelta(days=horizon_days)
        target_dow = target_date.isoweekday()
        dow_prices = self._dow_prices.get(target_dow, [])
        if len(dow_prices) >= self._min_rows:
            return sum(dow_prices) / len(dow_prices)
        return self._overall_mean


class SimpleVolatilityModel:
    """Volatility-aware baseline: predicts mean price, exposes volatility estimate.

    For price prediction, this model behaves like RollingMeanModel — it predicts
    the rolling mean of the last N training prices.

    Its additional value is the ``predicted_volatility_pct`` property: the
    rolling standard deviation expressed as a fraction of the rolling mean.
    This exposes a volatility forecast that can be used to:
    - Evaluate whether the model's implied uncertainty is well-calibrated.
    - Identify high-uncertainty archetypes for risk-aware recommendations.

    Interpretation:
      volatility_pct ≈ 0.02 → "prices typically move ±2% over this window"
      volatility_pct ≈ 0.30 → "this archetype is highly volatile right now"
    """

    name = "simple_volatility"

    def __init__(self, window: int = 7, min_rows: int = 3) -> None:
        self._window = window
        self._min_rows = min_rows
        self._mean: float | None = None
        self._volatility_pct: float | None = None

    @property
    def predicted_volatility_pct(self) -> float | None:
        """Rolling-std / rolling-mean ratio, or None if insufficient data."""
        return self._volatility_pct

    def fit(self, rows: list[dict[str, Any]]) -> None:
        self._mean = None
        self._volatility_pct = None
        tail = rows[-self._window:]
        prices = [float(r["price_mean"]) for r in tail if r.get("price_mean") is not None]
        if len(prices) < self._min_rows:
            return
        mean = sum(prices) / len(prices)
        variance = max(0.0, sum((p - mean) ** 2 for p in prices) / len(prices))
        std = math.sqrt(variance)
        self._mean = mean
        self._volatility_pct = (std / mean) if mean > 0 else None

    def predict(self, horizon_days: int) -> float | None:
        return self._mean


def all_baseline_models() -> list[Any]:
    """Return one fresh instance of every baseline model.

    Used by the evaluator to get a clean set of models for each backtest run.
    Each call returns new instances (no state shared between runs).
    """
    return [
        LastValueModel(),
        RollingMeanModel(window=7),
        DayOfWeekModel(),
        SimpleVolatilityModel(window=7),
    ]
