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
1.  **Archetype join through items table** — ``NormalizeStage`` has populated
    ``market_observations_normalized.archetype_id`` via ``_fetch_archetype_map()``
    since v1.3.4, but rows written before that release have ``NULL``.  We JOIN
    through ``items.archetype_id`` for backward compatibility and to exclude items
    with no archetype assignment; their count is surfaced in the quality report by
    ``quality.count_items_without_archetype()``.

2.  **Python-generated date spine** — prices are sparse (an item may not be
    listed every day).  After fetching only the actual aggregated observations
    from SQLite, Python generates the full calendar range and fills in NULL rows
    for days with no data via a dict lookup.  This avoids the recursive CTE and
    CROSS JOIN that previously forced SQLite to materialise N_dates × N_archetypes
    rows before the LEFT JOIN could execute.

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
from datetime import date, timedelta


@dataclass
class DailyAggRow:
    """One row in the daily-aggregated price series.

    All price fields are in gold.  NULL values (Python ``None``) indicate that
    no observations existed for that archetype/realm/date combination after the
    date-spine fill-in.

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

# All archetypes that have ever had non-outlier observations in this realm.
# No date filter — ensures spine rows are generated even for archetypes with
# no data in the requested window.
_ARCHETYPES_SQL = """
SELECT DISTINCT i.archetype_id
FROM market_observations_normalized mon
JOIN items i ON mon.item_id = i.item_id
WHERE i.archetype_id IS NOT NULL
  AND mon.is_outlier  = 0
  AND mon.realm_slug  = ?
"""

# Actual per-day aggregates within the requested window.  No spine or CROSS JOIN —
# SQLite only touches rows that exist.  The Python caller fills in NULL spine rows.
# Parameters: realm_slug, start_date, end_date
_RAW_DAILY_SQL = """
SELECT
    i.archetype_id,
    date(mon.observed_at)                                               AS obs_date,
    AVG(CASE WHEN mon.price_gold > 0 THEN mon.price_gold END)          AS price_mean,
    MIN(CASE WHEN mon.price_gold > 0 THEN mon.price_gold END)          AS price_min,
    MAX(CASE WHEN mon.price_gold > 0 THEN mon.price_gold END)          AS price_max,
    AVG(mon.market_value_gold)                                         AS market_value_mean,
    AVG(mon.historical_value_gold)                                     AS historical_value_mean,
    COUNT(*)                                                           AS obs_count,
    SUM(mon.quantity_listed)                                           AS quantity_sum,
    SUM(mon.num_auctions)                                              AS auctions_sum,
    CASE WHEN MAX(mon.quantity_listed) IS NULL THEN 1 ELSE 0 END       AS is_volume_proxy
FROM market_observations_normalized mon
JOIN items i ON mon.item_id = i.item_id
WHERE i.archetype_id IS NOT NULL
  AND mon.is_outlier  = 0
  AND mon.realm_slug  = ?
  AND date(mon.observed_at) BETWEEN ? AND ?
GROUP BY i.archetype_id, date(mon.observed_at)
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
    ``start_date`` is before the earliest observation, the Python spine still
    works (it simply produces NULL rows at the beginning).  Rows outside
    the data extent have ``obs_count = 0`` and all price fields as None.

    Args:
        conn:       Open SQLite connection with Row factory set.
        realm_slug: Blizzard realm slug to aggregate (e.g. "area-52").
        start_date: First calendar date to include in the spine.
        end_date:   Last calendar date to include in the spine (inclusive).

    Returns:
        List of ``DailyAggRow`` objects, empty if the realm has no normalised
        observations.
    """
    if start_date > end_date:
        return []

    # Step 1: compute actual data bounds so the date spine is anchored to real data.
    bounds_row = conn.execute(_BOUNDS_SQL, (realm_slug,)).fetchone()
    if bounds_row is None or bounds_row["min_d"] is None:
        return []

    # Clamp spine: don't extend before the first actual observation.
    data_min = date.fromisoformat(bounds_row["min_d"])
    data_max = date.fromisoformat(bounds_row["max_d"])

    spine_start = max(start_date, data_min)
    spine_end   = min(end_date,   data_max)

    if spine_start > spine_end:
        return []

    # Step 2: fetch all archetypes active in this realm (no date filter).
    arch_rows = conn.execute(_ARCHETYPES_SQL, (realm_slug,)).fetchall()
    arch_ids  = sorted(int(r["archetype_id"]) for r in arch_rows)
    if not arch_ids:
        return []

    # Step 3: fetch actual aggregated observations within the window.
    raw_rows = conn.execute(
        _RAW_DAILY_SQL,
        (realm_slug, spine_start.isoformat(), spine_end.isoformat()),
    ).fetchall()

    # Step 4: index actual data by (archetype_id, obs_date) for O(1) lookup.
    raw_lookup: dict[tuple[int, date], sqlite3.Row] = {}
    for r in raw_rows:
        obs_d = (
            date.fromisoformat(r["obs_date"])
            if isinstance(r["obs_date"], str)
            else r["obs_date"]
        )
        raw_lookup[(int(r["archetype_id"]), obs_d)] = r

    # Step 5: generate the date spine in Python — near-instant vs recursive CTE.
    n_days    = (spine_end - spine_start).days + 1
    date_spine = [spine_start + timedelta(days=i) for i in range(n_days)]

    # Step 6: cross-join archetypes × dates; fill spine-only rows with NULLs.
    result: list[DailyAggRow] = []
    for arch_id in arch_ids:
        for d in date_spine:
            r = raw_lookup.get((arch_id, d))
            if r is not None:
                result.append(DailyAggRow(
                    archetype_id=arch_id,
                    realm_slug=realm_slug,
                    obs_date=d,
                    price_mean=_float_or_none(r["price_mean"]),
                    price_min=_float_or_none(r["price_min"]),
                    price_max=_float_or_none(r["price_max"]),
                    market_value_mean=_float_or_none(r["market_value_mean"]),
                    historical_value_mean=_float_or_none(r["historical_value_mean"]),
                    obs_count=int(r["obs_count"]),
                    quantity_sum=_float_or_none(r["quantity_sum"]),
                    auctions_sum=_float_or_none(r["auctions_sum"]),
                    is_volume_proxy=bool(r["is_volume_proxy"]),
                ))
            else:
                result.append(DailyAggRow(
                    archetype_id=arch_id,
                    realm_slug=realm_slug,
                    obs_date=d,
                    price_mean=None,
                    price_min=None,
                    price_max=None,
                    market_value_mean=None,
                    historical_value_mean=None,
                    obs_count=0,
                    quantity_sum=None,
                    auctions_sum=None,
                    is_volume_proxy=True,
                ))
    return result


def _float_or_none(value: object) -> float | None:
    """Convert a DB value to float or None."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
