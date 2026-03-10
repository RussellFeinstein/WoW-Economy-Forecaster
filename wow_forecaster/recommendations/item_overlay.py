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

import math
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
        price_z_score:   Standard deviations from archetype mean, using the
                         population std of all items in the archetype window.
                         Negative = overpriced; positive = underpriced.
                         0.0 when fewer than two items exist in the window.
    """

    item_id:         int
    name:            str
    item_price_gold: float
    discount_pct:    float
    obs_count:       int
    price_z_score:   float = 0.0


def fetch_item_discounts(
    conn:                sqlite3.Connection,
    archetype_id:        int,
    realm_slug:          str,
    archetype_mean_gold: float,
    action:              str = "buy",
    lookback_days:       int = 3,
    top_n:               int = 5,
    min_obs:             int = 1,
    expansion_slug:      str | None = None,
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
        expansion_slug:      If provided, only items with this expansion_slug are
                             returned.  Pass ``config.expansions.transfer_target``
                             (e.g. ``"midnight"``) to restrict the overlay to the
                             forecast-target expansion.  ``None`` = no filter.

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
          AND (:expansion_slug IS NULL OR i.expansion_slug = :expansion_slug)
        GROUP BY i.item_id, i.name
        HAVING COUNT(*) >= :min_obs
    """

    rows = conn.execute(
        sql,
        {
            "archetype_id":   archetype_id,
            "realm_slug":     realm_slug,
            "since":          f"-{lookback_days} days",
            "min_obs":        min_obs,
            "expansion_slug": expansion_slug,
        },
    ).fetchall()

    if not rows:
        return []

    # Pre-compute population std of all item prices for z-score normalisation.
    all_prices = [float(r["item_price_gold"] or 0.0) for r in rows]
    if len(all_prices) > 1:
        variance = sum((p - archetype_mean_gold) ** 2 for p in all_prices) / len(all_prices)
        price_std = math.sqrt(variance)
    else:
        price_std = 0.0

    results: list[ItemDiscountRow] = []
    for row in rows:
        item_price  = float(row["item_price_gold"] or 0.0)
        discount    = (archetype_mean_gold - item_price) / archetype_mean_gold
        z_score     = round((archetype_mean_gold - item_price) / price_std, 3) if price_std > 0 else 0.0
        results.append(
            ItemDiscountRow(
                item_id         = row["item_id"],
                name            = row["name"],
                item_price_gold = item_price,
                discount_pct    = discount,
                obs_count       = row["obs_count"],
                price_z_score   = z_score,
            )
        )

    if action == "buy":
        results.sort(key=lambda r: -r.discount_pct)   # most underpriced first
    elif action == "sell":
        results.sort(key=lambda r: r.discount_pct)    # most overpriced first
    else:
        results.sort(key=lambda r: -abs(r.discount_pct))

    return results[:top_n]


@dataclass(frozen=True)
class ItemForecastRoi:
    """A specific item with its forecast ROI relative to current market price.

    Attributes:
        item_id:       FK to items.item_id.
        name:          Item display name.
        current_price: Recent average price in gold (from normalized obs).
        forecast_price: Predicted price from the item-level forecast.
        roi_pct:       (forecast_price - current_price) / current_price.
                       Positive = forecast above current (buy opportunity).
                       Negative = forecast below current (sell pressure).
        horizon:       Forecast horizon label ("1d", "7d", "28d").
        obs_count:     Number of recent observations used for current_price.
    """

    item_id:       int
    name:          str
    current_price: float
    forecast_price: float
    roi_pct:       float
    horizon:       str
    obs_count:     int


def fetch_item_rois(
    conn:         sqlite3.Connection,
    archetype_id: int,
    realm_slug:   str,
    horizon:      str,
    action:       str = "buy",
    lookback_days: int = 3,
    top_n:        int = 5,
    min_obs:      int = 1,
) -> list[ItemForecastRoi]:
    """Return top-N items within an archetype ranked by forecast ROI.

    Requires item-level forecasts to have been generated by
    ForecastStage._generate_item_forecasts() (i.e. run-daily-forecast must
    have run at least once).  Items without a recent item-level forecast
    are silently omitted.

    Args:
        conn:         Open DB connection (row_factory should be sqlite3.Row).
        archetype_id: FK to economic_archetypes.
        realm_slug:   Realm to query.
        horizon:      Forecast horizon label ("1d", "7d", "28d").
        action:       Trading action — controls ranking direction.
                      "buy": highest ROI first (most bullish items).
                      "sell": lowest ROI first (most bearish items).
                      other:  highest |ROI| first.
        lookback_days: Days of history to compute current_price (default 3).
        top_n:        Max items to return (default 5).
        min_obs:      Min observation count required for current_price.

    Returns:
        Ranked list of ItemForecastRoi.  Empty if no item forecasts exist.
    """
    rows = conn.execute(
        """
        SELECT
            i.item_id,
            i.name,
            cp.current_price,
            fc.predicted_price_gold  AS forecast_price,
            cp.obs_count
        FROM items i
        JOIN (
            SELECT item_id,
                   AVG(price_gold)  AS current_price,
                   COUNT(*)         AS obs_count
            FROM market_observations_normalized
            WHERE realm_slug  = :realm_slug
              AND is_outlier  = 0
              AND observed_at >= datetime('now', :since)
            GROUP BY item_id
            HAVING COUNT(*) >= :min_obs
        ) cp ON i.item_id = cp.item_id
        JOIN (
            SELECT item_id, MAX(created_at) AS max_ts
            FROM forecast_outputs
            WHERE item_id IS NOT NULL
              AND archetype_id IS NULL
              AND realm_slug       = :realm_slug
              AND forecast_horizon = :horizon
            GROUP BY item_id
        ) latest ON i.item_id = latest.item_id
        JOIN forecast_outputs fc
             ON  fc.item_id          = latest.item_id
             AND fc.archetype_id     IS NULL
             AND fc.realm_slug       = :realm_slug
             AND fc.forecast_horizon = :horizon
             AND fc.created_at       = latest.max_ts
        WHERE i.archetype_id  = :archetype_id
          AND cp.current_price > 0
        """,
        {
            "archetype_id": archetype_id,
            "realm_slug":   realm_slug,
            "horizon":      horizon,
            "since":        f"-{lookback_days} days",
            "min_obs":      min_obs,
        },
    ).fetchall()

    if not rows:
        return []

    results: list[ItemForecastRoi] = []
    for row in rows:
        current  = float(row["current_price"] or 0.0)
        forecast = float(row["forecast_price"] or 0.0)
        if current <= 0:
            continue
        roi_pct = (forecast - current) / current
        results.append(
            ItemForecastRoi(
                item_id        = row["item_id"],
                name           = row["name"],
                current_price  = current,
                forecast_price = forecast,
                roi_pct        = roi_pct,
                horizon        = horizon,
                obs_count      = row["obs_count"],
            )
        )

    if action == "buy":
        results.sort(key=lambda r: -r.roi_pct)
    elif action == "sell":
        results.sort(key=lambda r: r.roi_pct)
    else:
        results.sort(key=lambda r: -abs(r.roi_pct))

    return results[:top_n]
