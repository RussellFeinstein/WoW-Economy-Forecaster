"""
Pre-aggregated rollup table management.

Two rollup tables cache daily aggregates from ``market_observations_normalized``:

- ``daily_rollup_archetype``: grain (archetype_id, realm_slug, obs_date)
- ``daily_rollup_item``:      grain (item_id, realm_slug, obs_date)

After each hourly ingest, the previous and current UTC dates are re-aggregated
via UPSERT (two dates so post-midnight runs complete the prior day's tail).
All downstream consumers (daily_agg, drift, viz, margins, ...) read from the
rollup instead of scanning the full 110M-row normalized table.

Both tables store zero-inclusive AND positive-only price aggregates because
consumers differ on whether they filter ``price_gold > 0``.

The WHERE predicates compare ``observed_at`` as a raw column against a
half-open date range so ``idx_obs_norm_realm_outlier_time`` can serve them;
wrapping the column in ``DATE()`` there would force a table scan (issue #65).
``DATE(mon.observed_at)`` stays in the SELECT list and GROUP BY only, where it
cannot affect index use, and with the range covering exactly one date it
produces the same single group the old equality predicate did.
"""

from __future__ import annotations

import logging
import sqlite3
from collections.abc import Callable
from datetime import date, timedelta

logger = logging.getLogger(__name__)


# ── Archetype rollup UPSERT ──────────────────────────────────────────────────

_UPSERT_ARCHETYPE_SQL = """
INSERT INTO daily_rollup_archetype (
    archetype_id, realm_slug, obs_date,
    obs_count, price_sum_all, price_sum_sq_all,
    price_obs_count, price_sum, price_sum_sq, price_min, price_max,
    qty_weighted_price_sum, qty_weight_sum,
    market_value_sum, market_value_count,
    historical_value_sum, historical_value_count,
    quantity_sum, auctions_sum, is_volume_proxy
)
SELECT
    i.archetype_id,
    mon.realm_slug,
    DATE(mon.observed_at)                                                       AS obs_date,
    COUNT(*)                                                                    AS obs_count,
    COALESCE(SUM(mon.price_gold), 0.0)                                         AS price_sum_all,
    COALESCE(SUM(mon.price_gold * mon.price_gold), 0.0)                        AS price_sum_sq_all,
    SUM(CASE WHEN mon.price_gold > 0 THEN 1 ELSE 0 END)                       AS price_obs_count,
    COALESCE(SUM(CASE WHEN mon.price_gold > 0
        THEN mon.price_gold END), 0.0)                                         AS price_sum,
    COALESCE(SUM(CASE WHEN mon.price_gold > 0
        THEN mon.price_gold * mon.price_gold END), 0.0)                        AS price_sum_sq,
    MIN(CASE WHEN mon.price_gold > 0 THEN mon.price_gold END)                  AS price_min,
    MAX(CASE WHEN mon.price_gold > 0 THEN mon.price_gold END)                  AS price_max,
    COALESCE(SUM(CASE WHEN mon.price_gold > 0
        THEN mon.price_gold * COALESCE(mon.quantity_listed, 1) END), 0.0)      AS qty_wp_sum,
    COALESCE(SUM(CASE WHEN mon.price_gold > 0
        THEN COALESCE(mon.quantity_listed, 1) END), 0.0)                       AS qty_w_sum,
    SUM(mon.market_value_gold)                                                  AS mv_sum,
    COUNT(mon.market_value_gold)                                                AS mv_count,
    SUM(mon.historical_value_gold)                                              AS hv_sum,
    COUNT(mon.historical_value_gold)                                            AS hv_count,
    SUM(mon.quantity_listed)                                                    AS qty_sum,
    SUM(mon.num_auctions)                                                       AS auc_sum,
    CASE WHEN MAX(mon.quantity_listed) IS NULL THEN 1 ELSE 0 END               AS vol_proxy
FROM market_observations_normalized mon
JOIN items i ON mon.item_id = i.item_id
WHERE i.archetype_id IS NOT NULL
  AND mon.is_outlier  = 0
  AND mon.realm_slug  = ?
  AND mon.observed_at >= ?
  AND mon.observed_at <  DATE(?, '+1 day')
GROUP BY i.archetype_id, mon.realm_slug, DATE(mon.observed_at)
ON CONFLICT(archetype_id, realm_slug, obs_date) DO UPDATE SET
    obs_count              = excluded.obs_count,
    price_sum_all          = excluded.price_sum_all,
    price_sum_sq_all       = excluded.price_sum_sq_all,
    price_obs_count        = excluded.price_obs_count,
    price_sum              = excluded.price_sum,
    price_sum_sq           = excluded.price_sum_sq,
    price_min              = excluded.price_min,
    price_max              = excluded.price_max,
    qty_weighted_price_sum = excluded.qty_weighted_price_sum,
    qty_weight_sum         = excluded.qty_weight_sum,
    market_value_sum       = excluded.market_value_sum,
    market_value_count     = excluded.market_value_count,
    historical_value_sum   = excluded.historical_value_sum,
    historical_value_count = excluded.historical_value_count,
    quantity_sum           = excluded.quantity_sum,
    auctions_sum           = excluded.auctions_sum,
    is_volume_proxy        = excluded.is_volume_proxy,
    updated_at             = strftime('%Y-%m-%dT%H:%M:%SZ', 'now')
"""


# ── Item rollup UPSERT ───────────────────────────────────────────────────────

_UPSERT_ITEM_SQL = """
INSERT INTO daily_rollup_item (
    item_id, realm_slug, obs_date,
    obs_count, price_sum, price_sum_sq, price_min, price_max,
    price_obs_count_pos, price_sum_pos,
    qty_weighted_price_sum, qty_weight_sum,
    qty_weighted_price_sum_pos, qty_weight_sum_pos,
    quantity_sum, auctions_sum
)
SELECT
    mon.item_id,
    mon.realm_slug,
    DATE(mon.observed_at)                                                       AS obs_date,
    COUNT(*)                                                                    AS obs_count,
    COALESCE(SUM(mon.price_gold), 0.0)                                         AS price_sum,
    COALESCE(SUM(mon.price_gold * mon.price_gold), 0.0)                        AS price_sum_sq,
    MIN(mon.price_gold)                                                         AS price_min,
    MAX(mon.price_gold)                                                         AS price_max,
    SUM(CASE WHEN mon.price_gold > 0 THEN 1 ELSE 0 END)                       AS poc_pos,
    COALESCE(SUM(CASE WHEN mon.price_gold > 0
        THEN mon.price_gold END), 0.0)                                         AS ps_pos,
    COALESCE(SUM(
        mon.price_gold * COALESCE(mon.quantity_listed, 1)), 0.0)               AS qwp_sum,
    COALESCE(SUM(COALESCE(mon.quantity_listed, 1)), 0.0)                       AS qw_sum,
    COALESCE(SUM(CASE WHEN mon.price_gold > 0
        THEN mon.price_gold * COALESCE(mon.quantity_listed, 1) END), 0.0)      AS qwp_pos,
    COALESCE(SUM(CASE WHEN mon.price_gold > 0
        THEN COALESCE(mon.quantity_listed, 1) END), 0.0)                       AS qw_pos,
    SUM(mon.quantity_listed)                                                    AS qty_sum,
    SUM(mon.num_auctions)                                                       AS auc_sum
FROM market_observations_normalized mon
WHERE mon.is_outlier  = 0
  AND mon.realm_slug  = ?
  AND mon.observed_at >= ?
  AND mon.observed_at <  DATE(?, '+1 day')
GROUP BY mon.item_id, mon.realm_slug, DATE(mon.observed_at)
ON CONFLICT(item_id, realm_slug, obs_date) DO UPDATE SET
    obs_count                  = excluded.obs_count,
    price_sum                  = excluded.price_sum,
    price_sum_sq               = excluded.price_sum_sq,
    price_min                  = excluded.price_min,
    price_max                  = excluded.price_max,
    price_obs_count_pos        = excluded.price_obs_count_pos,
    price_sum_pos              = excluded.price_sum_pos,
    qty_weighted_price_sum     = excluded.qty_weighted_price_sum,
    qty_weight_sum             = excluded.qty_weight_sum,
    qty_weighted_price_sum_pos = excluded.qty_weighted_price_sum_pos,
    qty_weight_sum_pos         = excluded.qty_weight_sum_pos,
    quantity_sum               = excluded.quantity_sum,
    auctions_sum               = excluded.auctions_sum,
    updated_at                 = strftime('%Y-%m-%dT%H:%M:%SZ', 'now')
"""


# ── Public API ────────────────────────────────────────────────────────────────


def upsert_archetype_rollup(
    conn: sqlite3.Connection,
    realm_slug: str,
    obs_date: str,
) -> int:
    """Recompute and upsert archetype-grain rollup rows for one (realm, date).

    Scans only ``market_observations_normalized`` rows matching the given
    realm + date.  Returns number of rows upserted.
    """
    cur = conn.execute(_UPSERT_ARCHETYPE_SQL, (realm_slug, obs_date, obs_date))
    conn.commit()
    return cur.rowcount


def upsert_item_rollup(
    conn: sqlite3.Connection,
    realm_slug: str,
    obs_date: str,
) -> int:
    """Recompute and upsert item-grain rollup rows for one (realm, date).

    Returns number of rows upserted.
    """
    cur = conn.execute(_UPSERT_ITEM_SQL, (realm_slug, obs_date, obs_date))
    conn.commit()
    return cur.rowcount


def upsert_rollups_for_date(
    conn: sqlite3.Connection,
    realm_slug: str,
    obs_date: str | date,
) -> tuple[int, int]:
    """Update both rollup tables for one (realm, date).

    Args:
        conn:       Open SQLite connection.
        realm_slug: Realm to update.
        obs_date:   Calendar date (str ``YYYY-MM-DD`` or ``date`` object).

    Returns:
        Tuple of (archetype_rows_upserted, item_rows_upserted).
    """
    d = str(obs_date)
    arch = upsert_archetype_rollup(conn, realm_slug, d)
    item = upsert_item_rollup(conn, realm_slug, d)
    return arch, item


def backfill_rollups(
    conn: sqlite3.Connection,
    realm_slug: str,
    batch_days: int = 7,
    progress_callback: Callable[[int, int], None] | None = None,
) -> tuple[int, int]:
    """Full backfill of both rollup tables from existing normalized data.

    Processes dates in batches to avoid holding large transactions.

    Args:
        conn:              Open SQLite connection.
        realm_slug:        Realm to backfill.
        batch_days:        Number of dates to process per commit.
        progress_callback: Optional ``fn(dates_done, total_dates)`` for progress.

    Returns:
        Tuple of (total_archetype_rows, total_item_rows).
    """
    # Find date range in normalized data
    row = conn.execute(
        """
        SELECT date(MIN(observed_at)) AS min_d, date(MAX(observed_at)) AS max_d
        FROM market_observations_normalized
        WHERE realm_slug = ? AND is_outlier = 0
        """,
        (realm_slug,),
    ).fetchone()

    if row is None or row[0] is None:
        logger.warning("No normalized data found for realm=%s", realm_slug)
        return 0, 0

    min_d = date.fromisoformat(row[0])
    max_d = date.fromisoformat(row[1])
    total_days = (max_d - min_d).days + 1

    logger.info(
        "Backfilling rollups for realm=%s: %s -> %s (%d days)",
        realm_slug, min_d, max_d, total_days,
    )

    total_arch = 0
    total_item = 0
    dates_done = 0

    current = min_d
    while current <= max_d:
        batch_end = min(current + timedelta(days=batch_days - 1), max_d)

        # Process each date in the batch
        d = current
        while d <= batch_end:
            d_str = d.isoformat()
            arch = conn.execute(_UPSERT_ARCHETYPE_SQL, (realm_slug, d_str, d_str)).rowcount
            item = conn.execute(_UPSERT_ITEM_SQL, (realm_slug, d_str, d_str)).rowcount
            total_arch += arch
            total_item += item
            dates_done += 1
            d += timedelta(days=1)

        conn.commit()

        if progress_callback:
            progress_callback(dates_done, total_days)

        logger.info(
            "  Backfill progress: %d/%d dates (arch=%d, item=%d)",
            dates_done, total_days, total_arch, total_item,
        )

        current = batch_end + timedelta(days=1)

    logger.info(
        "Backfill complete: %d archetype rows, %d item rows",
        total_arch, total_item,
    )
    return total_arch, total_item
