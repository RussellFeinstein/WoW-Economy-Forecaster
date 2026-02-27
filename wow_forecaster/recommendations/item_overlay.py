"""
Per-item discount scoring overlay for archetype-level recommendations.

Given a winning archetype and its current mean price, queries the most
recent normalized observations to find specific items whose price deviates
furthest from that archetype mean.

Discount formula
----------------
    discount_pct = (archetype_mean - item_price) / archetype_mean

    positive  -> item is cheaper than the archetype mean (underpriced; buy target)
    negative  -> item is more expensive than the archetype mean (overpriced; sell target)

Sort order by action
--------------------
    buy:   highest positive discount first (most underpriced)
    sell:  most negative discount first (most overpriced)
    other: highest |discount| first (most deviant regardless of direction)

Returns an empty list when no normalized observations exist for the archetype,
or when archetype_mean_gold <= 0.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass


@dataclass(frozen=True)
class ItemDiscountRow:
    """A single item ranked by its price deviation from the archetype mean.

    Attributes:
        item_id:         FK to items.item_id.
        name:            Item display name.
        item_price_gold: Recent average price in gold (from normalized obs).
        discount_pct:    Fractional deviation from archetype mean.
                         Positive = underpriced, negative = overpriced.
        obs_count:       Number of observations used to compute item_price_gold.
    """

    item_id:         int
    name:            str
    item_price_gold: float
    discount_pct:    float
    obs_count:       int


def fetch_item_discounts(
    conn:                sqlite3.Connection,
    archetype_id:        int,
    realm_slug:          str,
    archetype_mean_gold: float,
    action:              str = "buy",
    lookback_days:       int = 3,
    top_n:               int = 5,
    min_obs:             int = 1,
) -> list[ItemDiscountRow]:
    """Return top-N items within an archetype ranked by discount from archetype mean.

    Queries ``market_observations_normalized`` for observations within
    ``lookback_days`` days, averages price per item, then scores each item
    by its fractional deviation from ``archetype_mean_gold``.

    Args:
        conn:                Open DB connection (row_factory should be sqlite3.Row).
        archetype_id:        FK to economic_archetypes.
        realm_slug:          Realm to query.
        archetype_mean_gold: Archetype-level mean price in gold (from inference row).
        action:              Trading action — controls ranking direction.
        lookback_days:       Days of history to include (default 3).
        top_n:               Max items to return (default 5).
        min_obs:             Min observation count required to include an item.

    Returns:
        Ranked list of ItemDiscountRow.  Empty if no data or mean <= 0.
    """
    if archetype_mean_gold <= 0:
        return []

    sql = """
        SELECT
            i.item_id,
            i.name,
            AVG(n.price_gold)  AS item_price_gold,
            COUNT(*)           AS obs_count
        FROM market_observations_normalized n
        JOIN items i ON n.item_id = i.item_id
        WHERE i.archetype_id  = :archetype_id
          AND n.realm_slug    = :realm_slug
          AND n.is_outlier    = 0
          AND n.observed_at   >= datetime('now', :since)
        GROUP BY i.item_id, i.name
        HAVING COUNT(*) >= :min_obs
    """

    rows = conn.execute(
        sql,
        {
            "archetype_id": archetype_id,
            "realm_slug":   realm_slug,
            "since":        f"-{lookback_days} days",
            "min_obs":      min_obs,
        },
    ).fetchall()

    if not rows:
        return []

    results: list[ItemDiscountRow] = []
    for row in rows:
        item_price = float(row["item_price_gold"] or 0.0)
        discount   = (archetype_mean_gold - item_price) / archetype_mean_gold
        results.append(
            ItemDiscountRow(
                item_id         = row["item_id"],
                name            = row["name"],
                item_price_gold = item_price,
                discount_pct    = discount,
                obs_count       = row["obs_count"],
            )
        )

    if action == "buy":
        results.sort(key=lambda r: -r.discount_pct)   # most underpriced first
    elif action == "sell":
        results.sort(key=lambda r: r.discount_pct)    # most overpriced first
    else:
        results.sort(key=lambda r: -abs(r.discount_pct))

    return results[:top_n]
