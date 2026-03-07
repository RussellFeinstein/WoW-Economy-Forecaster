"""
Crafting margin calculator.

For each recipe × realm × date, computes:
  - output_price_gold    — latest price of the crafted output item
  - craft_cost_gold      — sum of (ingredient_price × quantity) for required reagents
  - margin_gold          — output_price - craft_cost
  - margin_pct           — margin / output_price  (None if output_price is 0)
  - ingredient_coverage  — fraction of required ingredients with price data

Prices are sourced from ``market_observations_normalized`` (non-outlier rows),
using the mean price per item per day (matching the daily aggregation approach).

Results are written to ``crafting_margin_snapshots`` with an ON CONFLICT
update so re-running is idempotent.

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
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


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

        stats = MarginStats()
        recipes = self._load_recipes_with_reagents()
        if not recipes:
            logger.info("No recipes found in DB — nothing to compute.")
            return stats

        # Build item_id -> {date: avg_price_gold} price lookup for the date range
        price_map = self._fetch_prices(realm_slug, start_date, end_date)
        logger.info(
            "Loaded prices for %d items over %d days",
            len(price_map), lookback_days,
        )

        current = start_date
        rows_to_write: list[tuple] = []

        while current <= end_date:
            date_str = current.isoformat()
            for recipe in recipes:
                snap = self._compute_snapshot(recipe, date_str, price_map)
                if snap is None:
                    stats.recipes_no_output_price += 1
                    continue
                if snap["ingredient_coverage_pct"] < self._min_coverage:
                    stats.recipes_low_coverage += 1
                    continue
                rows_to_write.append((
                    snap["recipe_id"],
                    realm_slug,
                    date_str,
                    snap["output_price_gold"],
                    snap["craft_cost_gold"],
                    snap["margin_gold"],
                    snap["margin_pct"],
                    snap["ingredient_coverage_pct"],
                ))
            current += timedelta(days=1)

        stats.recipes_processed = len(recipes)
        stats.snapshots_written, stats.snapshots_skipped = self._write_snapshots(
            rows_to_write
        )
        self._conn.commit()
        logger.info(
            "Margin computation done: %d written, %d skipped (already existed)",
            stats.snapshots_written, stats.snapshots_skipped,
        )
        return stats

    # ── Private helpers ────────────────────────────────────────────────────────

    def _load_recipes_with_reagents(self) -> list[dict]:
        """Load all recipes with their required reagents as a flat structure."""
        recipes_raw = self._conn.execute(
            """
            SELECT recipe_id, output_item_id, output_quantity
            FROM recipes
            """
        ).fetchall()

        if not recipes_raw:
            return []

        recipe_ids = [r[0] for r in recipes_raw]
        placeholders = ",".join("?" * len(recipe_ids))
        reagents_raw = self._conn.execute(
            f"""
            SELECT recipe_id, ingredient_item_id, quantity
            FROM recipe_reagents
            WHERE recipe_id IN ({placeholders}) AND reagent_type = 'required'
            """,
            recipe_ids,
        ).fetchall()

        reagents_by_recipe: dict[int, list[tuple[int, int]]] = {}
        for r in reagents_raw:
            reagents_by_recipe.setdefault(r[0], []).append((r[1], r[2]))

        return [
            {
                "recipe_id": r[0],
                "output_item_id": r[1],
                "output_quantity": r[2],
                "reagents": reagents_by_recipe.get(r[0], []),
            }
            for r in recipes_raw
        ]

    def _fetch_prices(
        self, realm_slug: str, start_date: date, end_date: date
    ) -> dict[int, dict[str, float]]:
        """Fetch mean daily prices for all items in the date range.

        Returns dict: item_id -> {date_str: avg_price_gold}.
        """
        rows = self._conn.execute(
            """
            SELECT
                item_id,
                DATE(observed_at) AS obs_date,
                AVG(price_gold)   AS avg_price
            FROM market_observations_normalized
            WHERE realm_slug = ?
              AND is_outlier  = 0
              AND DATE(observed_at) BETWEEN ? AND ?
            GROUP BY item_id, DATE(observed_at)
            """,
            (realm_slug, start_date.isoformat(), end_date.isoformat()),
        ).fetchall()

        result: dict[int, dict[str, float]] = {}
        for item_id, date_str, avg_price in rows:
            if avg_price is not None:
                result.setdefault(int(item_id), {})[date_str] = float(avg_price)
        return result

    def _compute_snapshot(
        self, recipe: dict, date_str: str, price_map: dict[int, dict[str, float]]
    ) -> dict | None:
        """Compute a single margin snapshot for one recipe on one date.

        Returns None if the output item has no price data on this date.
        """
        recipe_id = recipe["recipe_id"]
        output_item_id = recipe["output_item_id"]
        output_quantity = max(recipe["output_quantity"], 1)
        reagents: list[tuple[int, int]] = recipe["reagents"]  # (item_id, qty)

        # Output item price (per unit crafted)
        output_item_prices = price_map.get(output_item_id, {})
        output_price_raw = output_item_prices.get(date_str)
        if output_price_raw is None:
            return None

        output_price_gold = output_price_raw  # Already per-unit from normalization

        # Craft cost: sum of ingredient prices × quantities
        total_cost = 0.0
        ingredients_with_price = 0
        for ingredient_item_id, quantity in reagents:
            item_prices = price_map.get(ingredient_item_id, {})
            price = item_prices.get(date_str)
            if price is not None:
                total_cost += price * quantity
                ingredients_with_price += 1

        total_ingredients = len(reagents)
        coverage = (
            float(ingredients_with_price) / total_ingredients
            if total_ingredients > 0
            else 1.0  # Recipe with no reagents = 100% coverage
        )

        if total_ingredients > 0 and ingredients_with_price == 0:
            craft_cost_gold = None
            margin_gold = None
            margin_pct = None
        else:
            craft_cost_gold = total_cost / output_quantity
            margin_gold = output_price_gold - craft_cost_gold
            margin_pct = (
                margin_gold / output_price_gold
                if output_price_gold > 0
                else None
            )

        return {
            "recipe_id": recipe_id,
            "output_price_gold": output_price_gold,
            "craft_cost_gold": craft_cost_gold,
            "margin_gold": margin_gold,
            "margin_pct": margin_pct,
            "ingredient_coverage_pct": coverage,
        }

    def _write_snapshots(
        self, rows: list[tuple]
    ) -> tuple[int, int]:
        """Upsert snapshot rows. Returns (written, skipped)."""
        written = 0
        skipped = 0
        for row in rows:
            try:
                self._conn.execute(
                    """
                    INSERT INTO crafting_margin_snapshots
                        (recipe_id, realm_slug, obs_date, output_price_gold,
                         craft_cost_gold, margin_gold, margin_pct,
                         ingredient_coverage_pct)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(recipe_id, realm_slug, obs_date) DO UPDATE SET
                        output_price_gold       = excluded.output_price_gold,
                        craft_cost_gold         = excluded.craft_cost_gold,
                        margin_gold             = excluded.margin_gold,
                        margin_pct              = excluded.margin_pct,
                        ingredient_coverage_pct = excluded.ingredient_coverage_pct
                    """,
                    row,
                )
                written += 1
            except Exception as exc:
                logger.warning("Failed to write margin snapshot %s: %s", row[:3], exc)
                skipped += 1
        return written, skipped
