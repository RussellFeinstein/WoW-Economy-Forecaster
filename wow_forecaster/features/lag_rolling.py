"""
Lag, rolling-window, momentum, and forward-looking target features.

Purpose
-------
Given the list of ``DailyAggRow`` objects produced by ``daily_agg.fetch_daily_agg()``,
this module computes all time-series-derived features for each row and returns a list
of flat dicts ready for assembly into the Parquet table.

How it works — step by step
----------------------------
1.  Group rows by (archetype_id, realm_slug) — each group is an independent time series.
2.  Build a ``{obs_date: price_mean}`` lookup dict for the group.
    This powers both lag lookups (backwards) and target lookups (forwards).
3.  For each row in the group:
    a. **Lag features**: directly index lookup[obs_date - timedelta(N)].
    b. **Rolling mean**: mean of non-None prices in the window
       [obs_date - timedelta(N-1), obs_date].
    c. **Rolling std**: ``sqrt(max(0.0, E[x²] - E[x]²))`` over the same window.
       The ``max(0.0, ...)`` clamp prevents negative values from floating-point
       cancellation when all prices in the window are nearly identical.
    d. **Momentum (pct_change)**: ``(price - lag_N) / lag_N``; None if lag_N is None or 0.
    e. **Target**: directly index lookup[obs_date + timedelta(H)].
       Targets are forward-looking labels — they are allowed to use future prices.
       They are excluded from the inference Parquet by the dataset builder.

Leakage notes
-------------
- Lag and rolling features only access obs_date ≤ current date.  No future information.
- Target features access obs_date + H.  This is deliberate — they are labels, not inputs.
- Event features (which have the principal leakage risk) are handled in ``event_features.py``.

Missing data handling
---------------------
- If no observation existed on a particular lookback date, the price in the lookup is None.
- Rolling mean / std skip None values (same semantics as SQL AVG which ignores NULLs).
- A window with zero non-None values produces None for mean and None for std.
- A window with exactly one non-None value produces that value for mean and 0.0 for std.
"""

from __future__ import annotations

import math
from collections import defaultdict
from datetime import date, timedelta
from typing import Any

from wow_forecaster.features.daily_agg import DailyAggRow
from wow_forecaster.config import FeatureConfig


def compute_lag_rolling_features(
    rows: list[DailyAggRow],
    config: FeatureConfig,
) -> list[dict[str, Any]]:
    """Compute lag, rolling, momentum, and target features for all rows.

    Args:
        rows:   Rows from ``daily_agg.fetch_daily_agg()``.  Must be sorted by
                (archetype_id, realm_slug, obs_date) — the SQL ORDER BY guarantees this.
        config: Feature configuration carrying ``lag_days``, ``rolling_windows``,
                and ``target_horizons_days``.

    Returns:
        A list of flat dicts, one per input row, containing the base DailyAggRow
        fields plus all lag / rolling / momentum / target columns.
        Columns are named exactly as declared in ``registry.FEATURE_REGISTRY``.
    """
    # Group rows by series key.
    groups: dict[tuple[int, str], list[DailyAggRow]] = defaultdict(list)
    for row in rows:
        groups[(row.archetype_id, row.realm_slug)].append(row)

    result: list[dict[str, Any]] = []
    for series_rows in groups.values():
        result.extend(_process_series(series_rows, config))
    return result


# ── Internal helpers ───────────────────────────────────────────────────────────


def _process_series(
    rows: list[DailyAggRow],
    config: FeatureConfig,
) -> list[dict[str, Any]]:
    """Process one (archetype_id, realm_slug) series."""
    # Build date → price_mean lookup for O(1) lag and target access.
    price_lookup: dict[date, float | None] = {r.obs_date: r.price_mean for r in rows}

    result: list[dict[str, Any]] = []
    for row in rows:
        d = row.obs_date
        price = row.price_mean

        out: dict[str, Any] = {
            # Pass through all DailyAggRow fields unchanged.
            "archetype_id":          row.archetype_id,
            "realm_slug":            row.realm_slug,
            "obs_date":              d,
            "price_mean":            price,
            "price_min":             row.price_min,
            "price_max":             row.price_max,
            "market_value_mean":     row.market_value_mean,
            "historical_value_mean": row.historical_value_mean,
            "obs_count":             row.obs_count,
            "quantity_sum":          row.quantity_sum,
            "auctions_sum":          row.auctions_sum,
            "is_volume_proxy":       row.is_volume_proxy,
        }

        # ── Lag features ────────────────────────────────────────────────────
        for n in config.lag_days:
            key = f"price_lag_{n}d"
            lag_date = d - timedelta(days=n)
            out[key] = price_lookup.get(lag_date)  # None if date not in spine

        # ── Rolling stats ────────────────────────────────────────────────────
        for n in config.rolling_windows:
            window = [
                price_lookup.get(d - timedelta(days=i))
                for i in range(n)         # i=0 is current day; i=n-1 is n-1 days ago
            ]
            valid = [v for v in window if v is not None]
            mean_key = f"price_roll_mean_{n}d"
            std_key  = f"price_roll_std_{n}d"
            if not valid:
                out[mean_key] = None
                out[std_key]  = None
            elif len(valid) == 1:
                out[mean_key] = valid[0]
                out[std_key]  = 0.0
            else:
                mu = sum(valid) / len(valid)
                mu2 = sum(v * v for v in valid) / len(valid)
                variance = max(0.0, mu2 - mu * mu)   # clamp for float precision
                out[mean_key] = mu
                out[std_key]  = math.sqrt(variance)

        # ── Momentum / pct-change ────────────────────────────────────────────
        for n in config.rolling_windows:
            lag_date = d - timedelta(days=n)
            lag_price = price_lookup.get(lag_date)
            pct_key = f"price_pct_change_{n}d"
            if price is None or lag_price is None or lag_price == 0.0:
                out[pct_key] = None
            else:
                out[pct_key] = (price - lag_price) / lag_price

        # ── Forward-looking targets (labels only, not features) ──────────────
        for h in config.target_horizons_days:
            target_date = d + timedelta(days=h)
            out[f"target_price_{h}d"] = price_lookup.get(target_date)

        result.append(out)
    return result
