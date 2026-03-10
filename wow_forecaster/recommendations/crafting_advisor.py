"""
Crafting margin compression/expansion advisor.

Answers the question: for each craftable recipe, which (buy_mats, sell_output)
timing window yields the best expected margin — and is that margin currently
expanding or compressing?

Six temporal windows (buy_horizon -> sell_horizon):
  NOW_NOW   — craft and sell today
  NOW_7D    — buy mats now, sell in +7 days
  NOW_28D   — buy mats now, sell in +28 days
  _7D_7D    — buy mats at +7d, sell at +7d (craft same day as buying)
  _7D_28D   — buy mats at +7d, sell at +28d
  _28D_28D  — buy mats at +28d, sell at +28d

Ingredient price forecasts use existing ``forecast_outputs`` archetype-level
predictions.  Items without a forecast fall back to 7-day rolling mean from
``crafting_margin_snapshots``.

Compression/expansion is detected via a linear regression slope of
``margin_pct`` over the last ``margin_history_days`` days.

Volume gate: output items with ``quantity_sum_7d < min_volume_units`` are
excluded (no market depth = not actionable).

Final ranking score: best_window_margin_pct * volume_score, where
  volume_score = clamp(quantity_sum_7d / 500, 0, 1).
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass, field
from datetime import date, timedelta
from enum import Enum

logger = logging.getLogger(__name__)


class CraftingWindow(str, Enum):
    """(buy_horizon, sell_horizon) timing combinations."""

    NOW_NOW  = "now->now"
    NOW_7D   = "now->+7d"
    NOW_28D  = "now->+28d"
    _7D_7D   = "+7d->+7d"
    _7D_28D  = "+7d->+28d"
    _28D_28D = "+28d->+28d"


# Window definitions: (buy_horizon_days, sell_horizon_days)
_WINDOW_HORIZONS: dict[CraftingWindow, tuple[int, int]] = {
    CraftingWindow.NOW_NOW:  (0,  0),
    CraftingWindow.NOW_7D:   (0,  7),
    CraftingWindow.NOW_28D:  (0, 28),
    CraftingWindow._7D_7D:   (7,  7),
    CraftingWindow._7D_28D:  (7, 28),
    CraftingWindow._28D_28D: (28, 28),
}

# Forecast horizon label -> days (matches forecast_outputs.forecast_horizon)
_HORIZON_LABEL: dict[int, str] = {1: "1d", 7: "7d", 28: "28d"}


@dataclass
class CraftingOpportunity:
    """A ranked crafting opportunity across all temporal windows."""

    recipe_id: int
    recipe_name: str
    profession_slug: str
    output_item_id: int

    # Window margins: window -> margin_gold (None if not computable)
    windows: dict[CraftingWindow, float | None] = field(default_factory=dict)

    # Projected sell price per window (denominator for accurate margin %)
    window_sell_prices: dict[CraftingWindow, float | None] = field(default_factory=dict)

    # Best window (highest margin)
    best_window: CraftingWindow = CraftingWindow.NOW_NOW
    best_window_margin_gold: float | None = None
    best_window_margin_pct: float | None = None

    # Compression/expansion signal
    margin_status: str = "unknown"      # "expanding" | "compressing" | "stable" | "unknown"
    margin_slope_7d: float | None = None
    margin_pct_rank: float | None = None  # 0.0=lowest, 1.0=highest in 30d history

    # Current snapshot
    current_output_price_gold: float | None = None
    current_craft_cost_gold: float | None = None
    current_margin_gold: float | None = None
    ingredient_coverage_pct: float = 0.0

    # Volume (output item)
    quantity_sum_7d: float = 0.0
    volume_score: float = 0.0

    # Final ranking score
    opportunity_score: float = 0.0


def build_crafting_opportunities(
    conn: sqlite3.Connection,
    realm_slug: str,
    run_date: date,
    compression_slope_threshold: float = -0.02,
    expansion_slope_threshold: float = 0.02,
    margin_history_days: int = 30,
    min_volume_units: int = 50,
    min_ingredient_coverage: float = 0.5,
    allowed_expansions: list[str] | None = None,
) -> list[CraftingOpportunity]:
    """Build CraftingOpportunity objects for all recipes with margin data.

    Args:
        conn:                        Open DB connection.
        realm_slug:                  e.g. "us".
        run_date:                    The 'now' date for window calculations.
        compression_slope_threshold: margin_pct/day below which = compressing.
        expansion_slope_threshold:   margin_pct/day above which = expanding.
        margin_history_days:         Days of history for slope/percentile.
        min_volume_units:            Hard floor on output item volume (7-day sum).
        min_ingredient_coverage:     Minimum ingredient coverage to include.
        allowed_expansions:          If non-empty, only recipes from these expansion
                                     slugs are included.  None or [] = all expansions.

    Returns:
        List of CraftingOpportunity, unsorted.
    """
    # Load recipe metadata
    recipes = _load_recipes(conn, allowed_expansions=allowed_expansions or [])
    if not recipes:
        return []

    recipe_ids = [r["recipe_id"] for r in recipes]

    # Load most recent margin snapshot per recipe
    current_margins = _fetch_current_margins(conn, recipe_ids, realm_slug, run_date)

    # Load forecast prices: archetype-level and item-level (prefer item-level when available)
    forecasts = _fetch_forecasts(conn, realm_slug)
    item_forecasts = _fetch_item_forecasts(conn, realm_slug)

    # Load archetype_id for output items (to link to forecasts)
    item_archetype_map = _fetch_item_archetype_map(conn)

    # Load historical margin series for slope/percentile
    history = _fetch_margin_history(
        conn, recipe_ids, realm_slug, run_date, margin_history_days
    )

    # Batch-load reagents for all recipes (one query instead of one per recipe)
    reagents_map = _fetch_all_reagents(conn, recipe_ids)

    # Collect every item_id that needs a current price: output items + all reagents
    output_item_ids = list({r["output_item_id"] for r in recipes})
    all_price_item_ids = list(
        {r["output_item_id"] for r in recipes}
        | {iid for ids in reagents_map.values() for iid, _ in ids}
    )

    # Batch-load 7-day rolling prices for all needed items (one query)
    current_price_map = _fetch_item_prices_bulk(
        conn, all_price_item_ids, realm_slug, run_date
    )

    # Collect archetype IDs for all recipe-linked items (needed for trend-ratio scaling)
    all_archetype_ids = list({
        arch_id
        for item_id in all_price_item_ids
        if (arch_id := item_archetype_map.get(item_id)) is not None
    })

    # Batch-load 7-day rolling prices per archetype (denominator for trend-ratio scaling)
    archetype_current_prices = _fetch_archetype_prices_bulk(
        conn, all_archetype_ids, realm_slug, run_date
    )

    # Load volume (quantity_sum) for output items over last 7 days
    volume_map = _fetch_output_volumes(conn, output_item_ids, realm_slug, run_date)

    opportunities: list[CraftingOpportunity] = []
    for recipe in recipes:
        rid = recipe["recipe_id"]
        current = current_margins.get(rid)

        if current is None:
            continue  # No price data yet

        coverage = current.get("ingredient_coverage_pct", 0.0) or 0.0
        if coverage < min_ingredient_coverage:
            continue

        output_item_id = recipe["output_item_id"]
        vol_7d = volume_map.get(output_item_id, 0.0)

        # Hard volume gate
        if vol_7d < min_volume_units:
            continue

        volume_score = min(vol_7d / 500.0, 1.0)

        # Current prices
        current_output = current.get("output_price_gold")
        current_cost = current.get("craft_cost_gold")
        current_margin = current.get("margin_gold")
        current_margin_pct = current.get("margin_pct")

        # Build window projections
        windows, window_sell_prices = _compute_windows(
            recipe=recipe,
            reagents=reagents_map.get(recipe["recipe_id"], []),
            current_output_price=current_output,
            current_craft_cost=current_cost,
            item_archetype_map=item_archetype_map,
            forecasts=forecasts,
            current_price_map=current_price_map,
            archetype_current_prices=archetype_current_prices,
            item_forecasts=item_forecasts,
        )

        best_window, best_margin_gold = _find_best_window(windows, current_output)
        best_window_sell_price = window_sell_prices.get(best_window)
        best_margin_pct = (
            best_margin_gold / best_window_sell_price
            if best_margin_gold is not None and best_window_sell_price and best_window_sell_price > 0
            else None
        )

        # Compression/expansion detection.
        # Skip slope/percentile when output price is near zero — margin_pct is dominated
        # by the tiny denominator and produces meaningless signals.
        rec_history = history.get(rid, [])
        if current_output is not None and current_output < 0.1:
            slope, pct_rank, status = None, None, "unknown"
        else:
            slope, pct_rank, status = _compute_margin_status(
                rec_history,
                current_margin_pct,
                compression_slope_threshold,
                expansion_slope_threshold,
            )

        # Final score
        score = (
            (best_margin_pct or 0.0) * volume_score
            if best_margin_pct is not None
            else 0.0
        )

        opportunities.append(
            CraftingOpportunity(
                recipe_id=rid,
                recipe_name=recipe.get("recipe_name") or "",
                profession_slug=recipe["profession_slug"],
                output_item_id=output_item_id,
                windows=windows,
                window_sell_prices=window_sell_prices,
                best_window=best_window,
                best_window_margin_gold=best_margin_gold,
                best_window_margin_pct=best_margin_pct,
                margin_status=status,
                margin_slope_7d=slope,
                margin_pct_rank=pct_rank,
                current_output_price_gold=current_output,
                current_craft_cost_gold=current_cost,
                current_margin_gold=current_margin,
                ingredient_coverage_pct=coverage,
                quantity_sum_7d=vol_7d,
                volume_score=volume_score,
                opportunity_score=score,
            )
        )

    return opportunities


def rank_crafting_opportunities(
    opportunities: list[CraftingOpportunity],
    top_n: int = 10,
) -> list[CraftingOpportunity]:
    """Sort by opportunity_score descending and return top N."""
    return sorted(
        opportunities,
        key=lambda o: (o.opportunity_score, o.best_window_margin_gold or 0.0),
        reverse=True,
    )[:top_n]


# ── Data loading helpers ───────────────────────────────────────────────────────

def _load_recipes(
    conn: sqlite3.Connection,
    allowed_expansions: list[str] | None = None,
) -> list[dict]:
    """Load recipes that have at least one required reagent.

    Args:
        allowed_expansions: If non-empty, only recipes whose expansion_slug is
                            in this list are returned.  Empty / None = all.
    """
    if allowed_expansions:
        placeholders = ",".join("?" * len(allowed_expansions))
        rows = conn.execute(
            f"""
            SELECT r.recipe_id, r.profession_slug, r.output_item_id, r.output_quantity, r.recipe_name
            FROM recipes r
            WHERE r.expansion_slug IN ({placeholders})
              AND EXISTS (
                SELECT 1 FROM recipe_reagents rr
                WHERE rr.recipe_id = r.recipe_id AND rr.reagent_type = 'required'
              )
            ORDER BY r.profession_slug, r.recipe_id
            """,
            allowed_expansions,
        ).fetchall()
    else:
        rows = conn.execute(
            """
            SELECT r.recipe_id, r.profession_slug, r.output_item_id, r.output_quantity, r.recipe_name
            FROM recipes r
            WHERE EXISTS (
                SELECT 1 FROM recipe_reagents rr
                WHERE rr.recipe_id = r.recipe_id AND rr.reagent_type = 'required'
            )
            ORDER BY r.profession_slug, r.recipe_id
            """
        ).fetchall()
    return [
        {
            "recipe_id": row[0],
            "profession_slug": row[1],
            "output_item_id": row[2],
            "output_quantity": row[3],
            "recipe_name": row[4],
        }
        for row in rows
    ]


def _fetch_current_margins(
    conn: sqlite3.Connection,
    recipe_ids: list[int],
    realm_slug: str,
    run_date: date,
) -> dict[int, dict]:
    """Fetch the most recent margin snapshot per recipe on or before run_date."""
    if not recipe_ids:
        return {}

    placeholders = ",".join("?" * len(recipe_ids))
    rows = conn.execute(
        f"""
        SELECT cms.recipe_id,
               cms.output_price_gold,
               cms.craft_cost_gold,
               cms.margin_gold,
               cms.margin_pct,
               cms.ingredient_coverage_pct
        FROM crafting_margin_snapshots cms
        INNER JOIN (
            SELECT recipe_id, MAX(obs_date) AS latest_date
            FROM crafting_margin_snapshots
            WHERE realm_slug = ? AND obs_date <= ?
              AND recipe_id IN ({placeholders})
            GROUP BY recipe_id
        ) latest ON cms.recipe_id = latest.recipe_id
                 AND cms.obs_date  = latest.latest_date
                 AND cms.realm_slug = ?
        """,
        [realm_slug, run_date.isoformat()] + list(recipe_ids) + [realm_slug],
    ).fetchall()

    return {
        row[0]: {
            "output_price_gold": row[1],
            "craft_cost_gold": row[2],
            "margin_gold": row[3],
            "margin_pct": row[4],
            "ingredient_coverage_pct": row[5],
        }
        for row in rows
    }


def _fetch_forecasts(
    conn: sqlite3.Connection,
    realm_slug: str,
) -> dict[tuple[int, str], float]:
    """Fetch the most recent forecast per (archetype_id, horizon).

    Returns dict: (archetype_id, horizon_label) -> predicted_price_gold.
    """
    rows = conn.execute(
        """
        SELECT fo.archetype_id, fo.forecast_horizon, fo.predicted_price_gold
        FROM forecast_outputs fo
        INNER JOIN (
            SELECT archetype_id, forecast_horizon, MAX(created_at) AS latest
            FROM forecast_outputs
            WHERE realm_slug = ? AND archetype_id IS NOT NULL
            GROUP BY archetype_id, forecast_horizon
        ) latest ON fo.archetype_id    = latest.archetype_id
                 AND fo.forecast_horizon = latest.forecast_horizon
                 AND fo.created_at       = latest.latest
                 AND fo.realm_slug       = ?
        """,
        (realm_slug, realm_slug),
    ).fetchall()

    return {(row[0], row[1]): row[2] for row in rows}


def _fetch_item_forecasts(
    conn: sqlite3.Connection,
    realm_slug: str,
) -> dict[tuple[int, str], float]:
    """Fetch the most recent item-level forecast per (item_id, horizon).

    Item-level forecasts are written by ForecastStage._generate_item_forecasts()
    using trend-ratio scaling.  They are preferred over archetype-level forecasts
    in _compute_windows() because they preserve each item's specific price level.

    Returns dict: (item_id, horizon_label) -> predicted_price_gold.
    """
    rows = conn.execute(
        """
        SELECT fo.item_id, fo.forecast_horizon, fo.predicted_price_gold
        FROM forecast_outputs fo
        INNER JOIN (
            SELECT item_id, forecast_horizon, MAX(created_at) AS latest
            FROM forecast_outputs
            WHERE realm_slug = ? AND item_id IS NOT NULL
            GROUP BY item_id, forecast_horizon
        ) latest ON fo.item_id         = latest.item_id
                 AND fo.forecast_horizon = latest.forecast_horizon
                 AND fo.created_at       = latest.latest
                 AND fo.realm_slug       = ?
        """,
        (realm_slug, realm_slug),
    ).fetchall()

    return {(row[0], row[1]): row[2] for row in rows}


def _fetch_item_archetype_map(conn: sqlite3.Connection) -> dict[int, int]:
    """Return item_id -> archetype_id for all items with an archetype."""
    rows = conn.execute(
        "SELECT item_id, archetype_id FROM items WHERE archetype_id IS NOT NULL;"
    ).fetchall()
    return {row[0]: row[1] for row in rows}


def _fetch_margin_history(
    conn: sqlite3.Connection,
    recipe_ids: list[int],
    realm_slug: str,
    run_date: date,
    history_days: int,
) -> dict[int, list[tuple[str, float]]]:
    """Fetch margin_pct history per recipe for the trailing N days.

    Returns dict: recipe_id -> [(date_str, margin_pct), ...] sorted oldest first.
    """
    if not recipe_ids:
        return {}

    start = (run_date - timedelta(days=history_days - 1)).isoformat()
    end = run_date.isoformat()
    placeholders = ",".join("?" * len(recipe_ids))
    rows = conn.execute(
        f"""
        SELECT recipe_id, obs_date, margin_pct
        FROM crafting_margin_snapshots
        WHERE realm_slug = ?
          AND obs_date BETWEEN ? AND ?
          AND margin_pct IS NOT NULL
          AND recipe_id IN ({placeholders})
        ORDER BY recipe_id, obs_date
        """,
        [realm_slug, start, end] + list(recipe_ids),
    ).fetchall()

    result: dict[int, list[tuple[str, float]]] = {}
    for rid, d, mp in rows:
        result.setdefault(rid, []).append((d, float(mp)))
    return result


def _fetch_output_volumes(
    conn: sqlite3.Connection,
    output_item_ids: list[int],
    realm_slug: str,
    run_date: date,
) -> dict[int, float]:
    """Fetch 7-day quantity_sum for output items from market_observations_normalized.

    Returns dict: item_id -> total quantity_sum over last 7 days.
    """
    if not output_item_ids:
        return {}

    start = (run_date - timedelta(days=6)).isoformat()
    end = run_date.isoformat()
    placeholders = ",".join("?" * len(output_item_ids))
    rows = conn.execute(
        f"""
        SELECT item_id, SUM(quantity_listed)
        FROM market_observations_normalized
        WHERE realm_slug = ?
          AND is_outlier = 0
          AND DATE(observed_at) BETWEEN ? AND ?
          AND item_id IN ({placeholders})
          AND quantity_listed IS NOT NULL
        GROUP BY item_id
        """,
        [realm_slug, start, end] + list(output_item_ids),
    ).fetchall()

    return {row[0]: float(row[1]) for row in rows if row[1] is not None}


# ── Window computation ─────────────────────────────────────────────────────────

def _compute_windows(
    recipe: dict,
    reagents: list[tuple[int, int]],
    current_output_price: float | None,
    current_craft_cost: float | None,
    item_archetype_map: dict[int, int],
    forecasts: dict[tuple[int, str], float],
    current_price_map: dict[int, float],
    archetype_current_prices: dict[int, float] | None = None,
    item_forecasts: dict[tuple[int, str], float] | None = None,
) -> tuple[dict[CraftingWindow, float | None], dict[CraftingWindow, float | None]]:
    """Compute margin_gold and projected sell price for each of the 6 temporal windows.

    Price lookup priority for future horizons (horizon_days > 0):
      1. Item-level forecast (from forecast_outputs with item_id set) — most precise.
      2. Trend-ratio scaling: item_current × (archetype_forecast / archetype_current).
      3. Raw archetype forecast (loses item-specific price level).
      4. Item's current price (assumes no change).

    Returns:
        (windows, window_sell_prices) — both keyed by CraftingWindow.
    """
    output_item_id = recipe["output_item_id"]
    output_quantity = max(recipe.get("output_quantity", 1), 1)
    _arch_currents = archetype_current_prices or {}
    _item_fc = item_forecasts or {}

    def forecast_price(item_id: int, horizon_days: int) -> float | None:
        """Get forecasted price for an item at a given horizon.

        Priority: item-level forecast → trend-ratio → archetype forecast → current price.
        """
        if horizon_days == 0:
            return current_price_map.get(item_id)

        label = _HORIZON_LABEL.get(horizon_days)

        # Priority 1: item-level forecast (persisted by ForecastStage for recipe items)
        if label:
            item_fc = _item_fc.get((item_id, label))
            if item_fc is not None:
                return item_fc

        item_current = current_price_map.get(item_id)
        archetype_id = item_archetype_map.get(item_id)

        if archetype_id and label:
            archetype_forecast = forecasts.get((archetype_id, label))
            archetype_current = _arch_currents.get(archetype_id)

            # Priority 2: trend-ratio scaling — preserves intra-archetype differentiation.
            # e.g., rare herb at 80g vs common herb at 20g in the same archetype
            # project to different future prices, not both to the archetype mean.
            if (
                item_current is not None
                and archetype_forecast is not None
                and archetype_current is not None
                and archetype_current > 0
            ):
                return item_current * (archetype_forecast / archetype_current)

            # Priority 3: archetype forecast level (loses item-specific differentiation)
            if archetype_forecast is not None:
                return archetype_forecast

        # Priority 4: assume no price change from current
        return item_current

    def compute_craft_cost(buy_horizon: int) -> float | None:
        """Sum ingredient costs at the given buy horizon."""
        if not reagents:
            return 0.0
        total = 0.0
        covered = 0
        for item_id, qty in reagents:
            price = forecast_price(item_id, buy_horizon)
            if price is not None:
                total += price * qty
                covered += 1
        if covered == 0:
            return None
        return total / output_quantity

    def compute_output_price(sell_horizon: int) -> float | None:
        """Get expected output price at the given sell horizon.

        For sell_horizon == 0, returns the current margin snapshot price (which
        may differ from current_price_map if the snapshot is slightly dated).
        For future horizons, delegates to forecast_price() so trend-ratio scaling
        is applied consistently for both reagents and output items.
        """
        if sell_horizon == 0:
            return current_output_price
        return forecast_price(output_item_id, sell_horizon)

    windows: dict[CraftingWindow, float | None] = {}
    window_sell_prices: dict[CraftingWindow, float | None] = {}
    for window, (buy_h, sell_h) in _WINDOW_HORIZONS.items():
        sell_price = compute_output_price(sell_h)
        craft_cost = (
            current_craft_cost if buy_h == 0
            else compute_craft_cost(buy_h)
        )
        window_sell_prices[window] = sell_price
        if sell_price is not None and craft_cost is not None:
            windows[window] = sell_price - craft_cost
        else:
            windows[window] = None

    return windows, window_sell_prices


def _fetch_all_reagents(
    conn: sqlite3.Connection,
    recipe_ids: list[int],
) -> dict[int, list[tuple[int, int]]]:
    """Batch-load required reagents for all recipes in a single query.

    Returns dict: recipe_id -> [(ingredient_item_id, quantity), ...].
    """
    if not recipe_ids:
        return {}
    placeholders = ",".join("?" * len(recipe_ids))
    rows = conn.execute(
        f"""
        SELECT recipe_id, ingredient_item_id, quantity
        FROM recipe_reagents
        WHERE recipe_id IN ({placeholders}) AND reagent_type = 'required'
        """,
        recipe_ids,
    ).fetchall()
    result: dict[int, list[tuple[int, int]]] = {}
    for rid, iid, qty in rows:
        result.setdefault(rid, []).append((iid, qty))
    return result


def _fetch_item_prices_bulk(
    conn: sqlite3.Connection,
    item_ids: list[int],
    realm_slug: str,
    run_date: date,
) -> dict[int, float]:
    """Batch-load 7-day quantity-weighted mean prices for all items in a single query.

    Returns dict: item_id -> avg_price_gold.
    """
    if not item_ids:
        return {}
    start_ts = (run_date - timedelta(days=6)).isoformat()
    end_ts = (run_date + timedelta(days=1)).isoformat()
    placeholders = ",".join("?" * len(item_ids))
    rows = conn.execute(
        f"""
        SELECT item_id,
               SUM(price_gold * COALESCE(quantity_listed, 1))
                   / NULLIF(SUM(COALESCE(quantity_listed, 1)), 0)
        FROM market_observations_normalized
        WHERE realm_slug = ?
          AND is_outlier = 0
          AND observed_at >= ?
          AND observed_at <  ?
          AND item_id IN ({placeholders})
        GROUP BY item_id
        """,
        [realm_slug, start_ts, end_ts] + list(item_ids),
    ).fetchall()
    return {row[0]: float(row[1]) for row in rows if row[1] is not None}


def _fetch_archetype_prices_bulk(
    conn: sqlite3.Connection,
    archetype_ids: list[int],
    realm_slug: str,
    run_date: date,
) -> dict[int, float]:
    """Batch-load 7-day quantity-weighted mean prices for archetypes in a single query.

    Used as the denominator in trend-ratio scaling: archetype_forecast / archetype_current.
    Mirrors _fetch_item_prices_bulk() but groups by archetype_id via JOIN items.

    Returns dict: archetype_id -> avg_price_gold.
    """
    if not archetype_ids:
        return {}
    start_ts = (run_date - timedelta(days=6)).isoformat()
    end_ts = (run_date + timedelta(days=1)).isoformat()
    placeholders = ",".join("?" * len(archetype_ids))
    rows = conn.execute(
        f"""
        SELECT i.archetype_id,
               SUM(mon.price_gold * COALESCE(mon.quantity_listed, 1))
                   / NULLIF(SUM(COALESCE(mon.quantity_listed, 1)), 0)
        FROM market_observations_normalized mon
        JOIN items i ON mon.item_id = i.item_id
        WHERE mon.realm_slug = ?
          AND mon.is_outlier = 0
          AND mon.observed_at >= ?
          AND mon.observed_at <  ?
          AND mon.price_gold > 0
          AND i.archetype_id IN ({placeholders})
        GROUP BY i.archetype_id
        """,
        [realm_slug, start_ts, end_ts] + list(archetype_ids),
    ).fetchall()
    return {row[0]: float(row[1]) for row in rows if row[1] is not None}


def _find_best_window(
    windows: dict[CraftingWindow, float | None],
    current_output_price: float | None,
) -> tuple[CraftingWindow, float | None]:
    """Return the window with the highest non-None margin.

    Ties broken by earliest sell horizon (prefer immediate over deferred).
    """
    _TIE_ORDER: dict[CraftingWindow, int] = {
        CraftingWindow.NOW_NOW:  0,
        CraftingWindow.NOW_7D:   1,
        CraftingWindow._7D_7D:   2,
        CraftingWindow.NOW_28D:  3,
        CraftingWindow._7D_28D:  4,
        CraftingWindow._28D_28D: 5,
    }
    best_w = CraftingWindow.NOW_NOW
    best_m: float | None = windows.get(CraftingWindow.NOW_NOW)

    for w, m in windows.items():
        if m is None:
            continue
        if best_m is None or m > best_m:
            best_w, best_m = w, m
        elif m == best_m and _TIE_ORDER.get(w, 99) < _TIE_ORDER.get(best_w, 99):
            best_w = w

    return best_w, best_m


# ── Margin status ──────────────────────────────────────────────────────────────

def _compute_margin_status(
    history: list[tuple[str, float]],
    current_margin_pct: float | None,
    compression_threshold: float,
    expansion_threshold: float,
) -> tuple[float | None, float | None, str]:
    """Compute slope, percentile rank, and compression/expansion status.

    Args:
        history:               [(date_str, margin_pct), ...] oldest first.
        current_margin_pct:    Today's margin_pct.
        compression_threshold: Slope below this = compressing.
        expansion_threshold:   Slope above this = expanding.

    Returns:
        (slope_per_day, pct_rank_0_1, status_str)
    """
    if len(history) < 3:
        return None, None, "unknown"

    values = [mp for _, mp in history]

    # Linear regression slope (least squares over day indices)
    n = len(values)
    xs = list(range(n))
    mean_x = (n - 1) / 2.0
    mean_y = sum(values) / n
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, values))
    den = sum((x - mean_x) ** 2 for x in xs)
    slope = num / den if den > 0 else 0.0

    # Percentile rank of current value in history
    pct_rank: float | None = None
    if current_margin_pct is not None and values:
        below = sum(1 for v in values if v <= current_margin_pct)
        pct_rank = below / len(values)

    if slope > expansion_threshold:
        status = "expanding"
    elif slope < compression_threshold:
        status = "compressing"
    else:
        status = "stable"

    return slope, pct_rank, status
