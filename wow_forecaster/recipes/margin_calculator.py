"""
Crafting margin calculator.

For each recipe × realm × date, computes:
  - output_price_gold    — latest price of the crafted output item
  - craft_cost_gold      — sum of (ingredient_price × quantity) for required reagents
  - margin_gold          — output_price - craft_cost
  - margin_pct           — margin / output_price  (None if output_price is 0)
  - ingredient_coverage  — fraction of required ingredients with price data

Prices are sourced from ``market_observations_normalized`` (non-outlier rows),
using the quantity-weighted mean price per item per day.

Results are written to ``crafting_margin_snapshots`` with an ON CONFLICT
update so re-running is idempotent.

Performance notes:
  - All computation is pushed into a single SQL CTE INSERT...SELECT so no
    Python loops over recipe × date combinations are needed.
  - Requires ``idx_obs_norm_realm_outlier_time`` on
    market_observations_normalized(realm_slug, is_outlier, observed_at).

Usage::

    from wow_forecaster.db.connection import get_connection
    from wow_forecaster.recipes.margin_calculator import MarginCalculator

    with get_connection(db_path) as conn:
        calc = MarginCalculator(conn)
        stats = calc.compute_margins(realm_slug="us", lookback_days=30)
        print(f"Wrote {stats.snapshots_written} snapshots")
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass
from datetime import date, timedelta

logger = logging.getLogger(__name__)

# Shared daily-price CTE used by both the main INSERT and the stats query.
# Filters with a direct observed_at range so the covering index is used.
_DAILY_PRICES_CTE = """
    daily_prices AS (
        SELECT
            item_id,
            DATE(observed_at) AS obs_date,
            SUM(price_gold * COALESCE(quantity_listed, 1))
                / NULLIF(SUM(COALESCE(quantity_listed, 1)), 0) AS avg_price
        FROM market_observations_normalized
        WHERE realm_slug = ?
          AND is_outlier  = 0
          AND observed_at >= ?
          AND observed_at <  ?
        GROUP BY item_id, DATE(observed_at)
    )
"""

_INSERT_CTE = f"""
    WITH {_DAILY_PRICES_CTE},
    output_prices AS (
        SELECT
            r.recipe_id,
            CASE WHEN r.output_quantity < 1 THEN 1 ELSE r.output_quantity END AS output_quantity,
            dp.obs_date,
            dp.avg_price AS output_price
        FROM recipes r
        JOIN daily_prices dp ON dp.item_id = r.output_item_id
        WHERE EXISTS (
            SELECT 1 FROM recipe_reagents rr
            WHERE rr.recipe_id = r.recipe_id AND rr.reagent_type = 'required'
        )
    ),
    reagent_agg AS (
        SELECT
            op.recipe_id,
            op.obs_date,
            op.output_price,
            op.output_quantity,
            COUNT(rr.id)                                                      AS total_ingredients,
            SUM(CASE WHEN dp.avg_price IS NOT NULL THEN 1 ELSE 0 END)         AS ingredients_with_price,
            SUM(COALESCE(dp.avg_price, 0.0) * rr.quantity)                    AS total_reagent_cost
        FROM output_prices op
        JOIN recipe_reagents rr
            ON rr.recipe_id = op.recipe_id AND rr.reagent_type = 'required'
        LEFT JOIN daily_prices dp
            ON dp.item_id = rr.ingredient_item_id AND dp.obs_date = op.obs_date
        GROUP BY op.recipe_id, op.obs_date, op.output_price, op.output_quantity
    )
    INSERT INTO crafting_margin_snapshots
        (recipe_id, realm_slug, obs_date, output_price_gold,
         craft_cost_gold, margin_gold, margin_pct, ingredient_coverage_pct)
    SELECT
        recipe_id,
        ? AS realm_slug,
        obs_date,
        output_price AS output_price_gold,
        CASE WHEN ingredients_with_price > 0
             THEN total_reagent_cost / output_quantity
             ELSE NULL END AS craft_cost_gold,
        CASE WHEN ingredients_with_price > 0
             THEN output_price - total_reagent_cost / output_quantity
             ELSE NULL END AS margin_gold,
        CASE WHEN ingredients_with_price > 0 AND output_price > 0
             THEN (output_price - total_reagent_cost / output_quantity) / output_price
             ELSE NULL END AS margin_pct,
        CAST(ingredients_with_price AS REAL) / total_ingredients AS ingredient_coverage_pct
    FROM reagent_agg
    WHERE CAST(ingredients_with_price AS REAL) / total_ingredients >= ?
    ON CONFLICT(recipe_id, realm_slug, obs_date) DO UPDATE SET
        output_price_gold       = excluded.output_price_gold,
        craft_cost_gold         = excluded.craft_cost_gold,
        margin_gold             = excluded.margin_gold,
        margin_pct              = excluded.margin_pct,
        ingredient_coverage_pct = excluded.ingredient_coverage_pct
"""


@dataclass
class MarginStats:
    """Summary of a margin computation run."""

    snapshots_written: int = 0
    snapshots_skipped: int = 0
    recipes_processed: int = 0
    recipes_no_output_price: int = 0
    recipes_low_coverage: int = 0


class MarginCalculator:
    """Computes crafting margin snapshots from market price data.

    Args:
        conn:                 Open sqlite3 connection.
        min_coverage:         Minimum fraction of ingredients with price data
                              required to write a snapshot (default: 0.5).
    """

    def __init__(
        self,
        conn: sqlite3.Connection,
        min_coverage: float = 0.5,
    ) -> None:
        self._conn = conn
        self._min_coverage = min_coverage

    def compute_margins(
        self,
        realm_slug: str,
        lookback_days: int = 30,
        end_date: date | None = None,
    ) -> MarginStats:
        """Compute and persist margin snapshots for all recipes.

        Processes all dates in [end_date - lookback_days, end_date].
        All computation is performed in a single SQL CTE INSERT...SELECT —
        no Python loops over recipe × date combinations.

        Args:
            realm_slug:    Realm/region to compute margins for (e.g. "us").
            lookback_days: Number of historical days to (re)compute.
            end_date:      Last date to compute (default: today).

        Returns:
            MarginStats summary.
        """
        if end_date is None:
            end_date = date.today()
        start_date = end_date - timedelta(days=lookback_days - 1)
        # Use direct timestamp bounds so the covering index on observed_at is used.
        start_ts = start_date.isoformat()
        end_ts = (end_date + timedelta(days=1)).isoformat()

        stats = MarginStats()
        stats.recipes_processed = self._conn.execute(
            """
            SELECT COUNT(*) FROM recipes r
            WHERE EXISTS (
                SELECT 1 FROM recipe_reagents rr
                WHERE rr.recipe_id = r.recipe_id AND rr.reagent_type = 'required'
            )
            """
        ).fetchone()[0]

        if stats.recipes_processed == 0:
            logger.info("No recipes found in DB — nothing to compute.")
            return stats

        self._conn.execute(
            _INSERT_CTE,
            (realm_slug, start_ts, end_ts, realm_slug, self._min_coverage),
        )
        stats.snapshots_written = self._conn.execute("SELECT changes()").fetchone()[0]
        self._conn.commit()

        logger.info(
            "Margin computation done: %d written/updated",
            stats.snapshots_written,
        )
        return stats
