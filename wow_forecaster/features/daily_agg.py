"""
Daily aggregation of normalised market observations.

Purpose
-------
This module collapses the per-scan ``market_observations_normalized`` rows into
one row per (archetype_id, realm_slug, calendar_date).  That daily-grain table
is the foundation for every downstream feature (lag, rolling, event distance,
archetype encoding).

Key design choices
------------------
1.  **Archetype join through items table** — ``market_observations_normalized``
    has ``archetype_id = NULL`` everywhere (the normalization TODO is not done).
    We JOIN through ``items.archetype_id`` instead.  Items without an archetype
    assignment are excluded from features; their count is surfaced in the quality
    report by ``quality.count_items_without_archetype()``.

2.  **Date spine via recursive CTE** — prices are sparse (an item may not be
    listed every day).  A recursive CTE generates the full calendar range, and a
    LEFT JOIN fills in NULLs for days with no observations.  This ensures that
    lag features are calendar-accurate: ``price_lag_7d`` always means "7 calendar
    days ago", not "7 prior observations".

3.  **Volume proxy flag** — ``quantity_listed`` is nullable (Blizzard API may not
    provide it).  When every observation on a given day has NULL quantity, the
    ``is_volume_proxy`` flag is set to True and ``obs_count`` acts as velocity proxy.

4.  **Outlier exclusion** — rows with ``is_outlier = 1`` are excluded from daily
    aggregation.  This prevents extreme price spikes from distorting features.

Input → Output
--------------
Input:  ``market_observations_normalized`` + ``items`` (via DB connection)
Output: ``list[DailyAggRow]`` sorted by (archetype_id, realm_slug, obs_date)

The returned list can be directly fed into ``lag_rolling.compute_lag_rolling_features()``.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import date, datetime


@dataclass
class DailyAggRow:
    """One row in the daily-aggregated price series.

    All price fields are in gold.  NULL values (Python ``None``) indicate that
    no observations existed for that archetype/realm/date combination after the
    date-spine LEFT JOIN.

    Attributes:
        archetype_id:          FK to economic_archetypes.
        realm_slug:            Blizzard realm slug.
        obs_date:              Calendar date of this aggregate.
        price_mean:            Mean non-outlier min-buyout price in gold.
                               None if no observations on this date.
        price_min:             Lowest price seen. None if no observations.
        price_max:             Highest price seen. None if no observations.
        market_value_mean:     Mean market value in gold (nullable source field).
        historical_value_mean: Mean historical value in gold (nullable source field).
        obs_count:             Number of price scans on this date. 0 if spine-only.
        quantity_sum:          Sum of quantity_listed. None if unavailable.
        auctions_sum:          Sum of num_auctions. None if unavailable.
        is_volume_proxy:       True when quantity_listed was absent from all
                               observations on this date (obs_count used as proxy).
    """

    archetype_id: int
    realm_slug: str
    obs_date: date
    price_mean: float | None
    price_min: float | None
    price_max: float | None
    market_value_mean: float | None
    historical_value_mean: float | None
    obs_count: int
    quantity_sum: float | None
    auctions_sum: float | None
    is_volume_proxy: bool


# ── SQL ────────────────────────────────────────────────────────────────────────

_BOUNDS_SQL = """
SELECT
    date(MIN(mon.observed_at)) AS min_d,
    date(MAX(mon.observed_at)) AS max_d
FROM market_observations_normalized mon
JOIN items i ON mon.item_id = i.item_id
WHERE i.archetype_id IS NOT NULL
  AND mon.is_outlier  = 0
  AND mon.realm_slug  = ?
"""

# The main aggregation query.
# Parameters (positional): realm_slug, realm_slug, min_d, max_d, realm_slug,
#                          start_date, end_date
_DAILY_AGG_SQL = """
WITH RECURSIVE date_spine(d) AS (
    SELECT ?
    UNION ALL
    SELECT date(d, '+1 day') FROM date_spine WHERE d < ?
),
archetypes_in_realm AS (
    SELECT DISTINCT i.archetype_id
    FROM market_observations_normalized mon
    JOIN items i ON mon.item_id = i.item_id
    WHERE i.archetype_id IS NOT NULL
      AND mon.is_outlier  = 0
      AND mon.realm_slug  = ?
),
full_spine AS (
    SELECT a.archetype_id, ds.d AS obs_date
    FROM archetypes_in_realm a
    CROSS JOIN date_spine ds
),
raw_daily AS (
    SELECT
        i.archetype_id,
        date(mon.observed_at)                                                   AS obs_date,
        AVG(CASE WHEN mon.price_gold > 0 THEN mon.price_gold END)              AS price_mean,
        MIN(CASE WHEN mon.price_gold > 0 THEN mon.price_gold END)              AS price_min,
        MAX(CASE WHEN mon.price_gold > 0 THEN mon.price_gold END)              AS price_max,
        AVG(mon.market_value_gold)                                              AS market_value_mean,
        AVG(mon.historical_value_gold)                                          AS historical_value_mean,
        COUNT(*)                                                                AS obs_count,
        SUM(mon.quantity_listed)                                                AS quantity_sum,
        SUM(mon.num_auctions)                                                   AS auctions_sum,
        CASE WHEN MAX(mon.quantity_listed) IS NULL THEN 1 ELSE 0 END           AS is_volume_proxy
    FROM market_observations_normalized mon
    JOIN items i ON mon.item_id = i.item_id
    WHERE i.archetype_id IS NOT NULL
      AND mon.is_outlier  = 0
      AND mon.realm_slug  = ?
      AND date(mon.observed_at) BETWEEN ? AND ?
    GROUP BY i.archetype_id, date(mon.observed_at)
)
SELECT
    fs.archetype_id,
    ? AS realm_slug,
    fs.obs_date,
    rd.price_mean,
    rd.price_min,
    rd.price_max,
    rd.market_value_mean,
    rd.historical_value_mean,
    COALESCE(rd.obs_count,       0) AS obs_count,
    rd.quantity_sum,
    rd.auctions_sum,
    COALESCE(rd.is_volume_proxy, 1) AS is_volume_proxy
FROM full_spine fs
LEFT JOIN raw_daily rd
    ON  fs.archetype_id = rd.archetype_id
    AND fs.obs_date     = rd.obs_date
ORDER BY fs.archetype_id, fs.obs_date;
"""


def fetch_daily_agg(
    conn: sqlite3.Connection,
    realm_slug: str,
    start_date: date,
    end_date: date,
) -> list[DailyAggRow]:
    """Query and return daily-aggregated price rows for one realm.

    The result is sorted by (archetype_id, obs_date) which is required by
    ``lag_rolling.compute_lag_rolling_features()``.

    The date range is clamped to the actual data extent: if the requested
    ``start_date`` is before the earliest observation, the CTE still works
    (it simply produces spine-only NULL rows at the beginning).  Rows outside
    the data extent have ``obs_count = 0`` and all price fields as None.

    Args:
        conn:       Open SQLite connection with Row factory set.
        realm_slug: Blizzard realm slug to aggregate (e.g. "area-52").
        start_date: First calendar date to include in the spine.
        end_date:   Last calendar date to include in the spine (inclusive).

    Returns:
        List of ``DailyAggRow`` objects, empty if the realm has no normalised
        observations.

    Note:
        Passing ``start_date > end_date`` returns an empty list (handled by the
        recursive CTE termination condition).
    """
    if start_date > end_date:
        return []

    # Step 1: compute actual data bounds so the date spine is anchored to real data.
    bounds_row = conn.execute(_BOUNDS_SQL, (realm_slug,)).fetchone()
    if bounds_row is None or bounds_row["min_d"] is None:
        # No normalised data exists for this realm.
        return []

    # Clamp spine: don't extend before the first actual observation.
    data_min = date.fromisoformat(bounds_row["min_d"])
    data_max = date.fromisoformat(bounds_row["max_d"])

    spine_start = max(start_date, data_min)
    spine_end   = min(end_date,   data_max)

    if spine_start > spine_end:
        return []

    rows = conn.execute(
        _DAILY_AGG_SQL,
        (
            spine_start.isoformat(),  # date_spine initial value
            spine_end.isoformat(),    # date_spine termination
            realm_slug,               # archetypes_in_realm filter
            realm_slug,               # raw_daily filter
            spine_start.isoformat(),  # raw_daily BETWEEN start
            spine_end.isoformat(),    # raw_daily BETWEEN end
            realm_slug,               # SELECT literal realm_slug
        ),
    ).fetchall()

    result: list[DailyAggRow] = []
    for r in rows:
        obs_date = (
            date.fromisoformat(r["obs_date"])
            if isinstance(r["obs_date"], str)
            else r["obs_date"]
        )
        result.append(
            DailyAggRow(
                archetype_id=int(r["archetype_id"]),
                realm_slug=realm_slug,
                obs_date=obs_date,
                price_mean=_float_or_none(r["price_mean"]),
                price_min=_float_or_none(r["price_min"]),
                price_max=_float_or_none(r["price_max"]),
                market_value_mean=_float_or_none(r["market_value_mean"]),
                historical_value_mean=_float_or_none(r["historical_value_mean"]),
                obs_count=int(r["obs_count"]),
                quantity_sum=_float_or_none(r["quantity_sum"]),
                auctions_sum=_float_or_none(r["auctions_sum"]),
                is_volume_proxy=bool(r["is_volume_proxy"]),
            )
        )
    return result


def _float_or_none(value: object) -> float | None:
    """Convert a DB value to float or None."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
