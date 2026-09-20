"""
ForecastStage — generate price forecasts from trained LightGBM models.

Inference flow
--------------
For each realm:
  1. Load the latest model artifact (.pkl) for each configured horizon
     from config.model.artifact_dir.
  2. Load the latest inference Parquet from
     data/processed/features/inference/.
  3. Batch-predict prices for all archetypes in the inference Parquet.
  4. Compute heuristic CIs (rolling_std × z, widened for cold-start items).
  5. Generate item-level forecasts for recipe-linked items and items with
     history via trend-ratio scaling, on a read-only connection.
  6. Persist the archetype and item ForecastOutput rows to forecast_outputs
     in one short write transaction.

Item-level forecasts
--------------------
_generate_item_forecasts() computes item-specific predictions from the
in-memory archetype forecasts using the trend-ratio method:

    item_forecast = item_current × (archetype_forecast / archetype_current)

This preserves each item's current price level while applying the archetype's
directional trend.  Results are stored in forecast_outputs with item_id set
and archetype_id = None.

Coverage (union of two sets):
  • All recipe-linked items (output items and required reagents) — always
    included so the crafting advisor can price crafting windows precisely.
  • All items with ≥ 14 distinct observation days — broader coverage so the
    recommendation overlay can surface per-item ROIs within each archetype.

Items in both sets are de-duplicated.  Items without a current price
observation or without an archetype mapping are skipped.

Since issue #107 the history count and the 7-day current prices come from the
daily rollup tables (daily_rollup_item, daily_rollup_archetype), not from
market_observations_normalized: the rollups hold the same aggregates at the
day grain these queries reduce to, they survive retention and the durable
backup, and reading them takes well under a second where the normalized scan
took 23 to 38 minutes.  The item forecasts are built before the write
connection opens, so no write transaction ever spans a read of the
observation tables and the hourly ingest is never locked out by the daily.

Freshness gate
--------------
Before any inference, _execute() checks the age of the newest normalized
observation per realm.  If it exceeds config.forecast.max_data_age_hours
(default 26h; <= 0 disables), StaleDataError is raised and the run is
recorded as failed — forecasts are never silently generated from frozen
features (issue #12).

Look-ahead bias guard
---------------------
The inference Parquet was built by the dataset_builder with event features
filtered by announced_at <= obs_date.  This guarantee propagates through
inference — no future event information reaches the model.

Cold-start fallback
-------------------
Cold-start Midnight archetypes (is_cold_start=True) are scored by the same
global model — the model learned from cold-start training rows and the
is_cold_start_int feature.  The CI is widened proportionally to uncertainty.
The model_slug is suffixed "_transfer" or "_cold" for provenance.

Returns total number of ForecastOutput rows written to DB.
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

from wow_forecaster.models.forecast import ForecastOutput
from wow_forecaster.models.meta import RunMetadata
from wow_forecaster.pipeline.base import PipelineStage

logger = logging.getLogger(__name__)

# Horizon label → days offset for target_date computation
_HORIZON_DAYS: dict[str, int] = {"1d": 1, "7d": 7, "28d": 28}


# ── Freshness gate ─────────────────────────────────────────────────────────────


class StaleDataError(RuntimeError):
    """Raised when the newest observation is too old to forecast from.

    Guards against silently generating forecasts from frozen features when
    ingestion has stopped (issue #12 — the 2026-04-15 outage produced 90 days
    of forecasts from stale data before anyone noticed).
    """


def _fetch_max_observation_age_hours(
    conn: sqlite3.Connection,
    realm_slug: str,
    now: datetime | None = None,
) -> float | None:
    """Hours since the newest non-outlier normalized observation for a realm.

    Args:
        conn:       Open DB connection.
        realm_slug: Realm to check.
        now:        Reference time (default: current UTC). Injectable for
                    deterministic tests.

    Returns:
        Age in hours, or None when the realm has no observations at all.
    """
    if now is None:
        now = datetime.now(tz=UTC)
    row = conn.execute(
        """
        SELECT MAX(observed_at)
        FROM market_observations_normalized
        WHERE realm_slug = ? AND is_outlier = 0
        """,
        (realm_slug,),
    ).fetchone()
    if row is None or row[0] is None:
        return None
    newest = datetime.fromisoformat(str(row[0]).replace("Z", "+00:00"))
    if newest.tzinfo is None:
        newest = newest.replace(tzinfo=UTC)
    return (now - newest).total_seconds() / 3600.0


class ForecastStage(PipelineStage):
    """Run trained LightGBM models to produce point forecasts with CIs.

    Writes ForecastOutput rows to the forecast_outputs table and returns
    the total row count.
    """

    stage_name = "forecast"

    def _execute(
        self,
        run: RunMetadata,
        realm_slug: str | None = None,
        horizons: list[int] | None = None,
        now: datetime | None = None,
        **kwargs,
    ) -> int:
        """Generate and persist forecasts for configured realms.

        Args:
            run:        In-progress RunMetadata (mutable).
            realm_slug: Single realm to target. If None, uses config defaults.
            horizons:   Horizon list override (int days). If None, uses
                        config.features.target_horizons_days.
            now:        Reference time for the freshness gate and, when given,
                        the anchor date of the item-forecast price window
                        (default: current UTC, and today's date for the
                        window). Injectable for deterministic tests.

        Returns:
            Total ForecastOutput rows written to DB.

        Raises:
            ValueError: If run.run_id is not set after pre-persist.
            StaleDataError: If config.forecast.max_data_age_hours > 0 and any
                target realm's newest observation is older than that threshold
                (or the realm has no observations at all).
        """
        from wow_forecaster.db.connection import get_connection
        from wow_forecaster.db.repositories.forecast_repo import ForecastOutputRepository
        from wow_forecaster.ml.lgbm_model import LightGBMForecaster
        from wow_forecaster.ml.predictor import (
            find_latest_inference_parquet,
            run_inference,
        )
        from wow_forecaster.ml.trainer import find_latest_model_artifact
        from wow_forecaster.monitoring.reporter import get_latest_uncertainty_multiplier

        # Pre-persist to get run_id before run_inference() needs it
        self._persist_run(run)

        realms        = [realm_slug] if realm_slug else list(self.config.realms.defaults)
        horizons_int  = horizons or list(self.config.features.target_horizons_days)
        processed_dir = Path(self.config.data.processed_dir)
        artifact_dir  = Path(self.config.model.artifact_dir)
        total_outputs = 0

        # Freshness gate: refuse to forecast from stale data (issue #12).
        max_age_hours = self.config.forecast.max_data_age_hours
        if max_age_hours > 0:
            with get_connection(
                self.db_path,
                wal_mode=self.config.database.wal_mode,
                busy_timeout_ms=self.config.database.busy_timeout_ms,
            ) as conn:
                for realm in realms:
                    age = _fetch_max_observation_age_hours(conn, realm, now=now)
                    if age is None:
                        raise StaleDataError(
                            f"realm={realm}: no normalized observations exist; "
                            f"refusing to forecast. Run 'run-hourly-refresh' first."
                        )
                    if age > max_age_hours:
                        raise StaleDataError(
                            f"realm={realm}: newest observation is {age:.1f}h old "
                            f"(limit {max_age_hours:.1f}h); refusing to forecast "
                            f"from stale data. Run 'run-hourly-refresh' and see "
                            f"'check-data-health'."
                        )

        for realm in realms:
            # Load model artifacts for each horizon
            forecasters: dict[int, LightGBMForecaster] = {}
            for h in horizons_int:
                artifact_path = find_latest_model_artifact(artifact_dir, realm, h)
                if artifact_path is None:
                    logger.warning(
                        "No model artifact for realm=%s horizon=%dd. "
                        "Run 'train-model' first.",
                        realm, h,
                    )
                    continue
                try:
                    forecasters[h] = LightGBMForecaster.load(artifact_path)
                except Exception as exc:
                    logger.error(
                        "Failed to load model %s: %s", artifact_path, exc,
                        exc_info=True,
                    )

            if not forecasters:
                logger.warning(
                    "No valid model artifacts for realm=%s; skipping.", realm
                )
                continue

            inf_path = find_latest_inference_parquet(processed_dir, realm)
            if inf_path is None:
                logger.warning(
                    "No inference Parquet for realm=%s. Run 'build-datasets' first.",
                    realm,
                )
                continue

            logger.info(
                "Forecasting realm=%s  horizons=%s  parquet=%s",
                realm, list(forecasters.keys()), inf_path,
            )

            # Read drift-based uncertainty multiplier and cold-start blend data.
            with get_connection(
                self.db_path,
                wal_mode=self.config.database.wal_mode,
                busy_timeout_ms=self.config.database.busy_timeout_ms,
            ) as conn:
                uncertainty_mult = get_latest_uncertainty_multiplier(conn, realm)
                blend_data = _fetch_cold_start_blend_data(
                    conn,
                    realm_slug=realm,
                    source_expansion=self.config.expansions.active,
                    target_expansion=self.config.expansions.transfer_target,
                )

            if uncertainty_mult != 1.0:
                logger.info(
                    "realm=%s: applying drift CI multiplier=%.2f",
                    realm, uncertainty_mult,
                )
            if blend_data:
                logger.info(
                    "realm=%s: cold-start blend data available for %d archetypes.",
                    realm, len(blend_data),
                )

            try:
                outputs = run_inference(
                    config=self.config,
                    run=run,
                    forecasters=forecasters,
                    inference_parquet_path=inf_path,
                    realm_slug=realm,
                    uncertainty_multiplier=uncertainty_mult,
                    cold_start_blend=blend_data or None,
                )
            except Exception as exc:
                logger.error("Inference failed for realm=%s: %s", realm, exc, exc_info=True)
                continue

            # Item-level forecasts derive from the in-memory archetype outputs and
            # the rollup tables, so they are built on a read-only connection first:
            # the write transaction below holds inserts only and never spans a read
            # of the observation tables (issue #107, the 07:16 hourly lock-out).
            with get_connection(
                self.db_path,
                wal_mode=self.config.database.wal_mode,
                busy_timeout_ms=self.config.database.busy_timeout_ms,
            ) as conn:
                item_outputs = _generate_item_forecasts(
                    conn, run.run_id, outputs, realm,
                    run_date=now.date() if now is not None else None,
                )

            # Persist archetype-level forecasts, then the item-level rows
            with get_connection(
                self.db_path,
                wal_mode=self.config.database.wal_mode,
                busy_timeout_ms=self.config.database.busy_timeout_ms,
            ) as conn:
                repo = ForecastOutputRepository(conn)
                for fc in outputs:
                    fc_id = repo.insert_forecast(fc)
                    # Attach the DB-assigned forecast_id (needed by RecommendStage)
                    object.__setattr__(fc, "forecast_id", fc_id)
                for ifc in item_outputs:
                    repo.insert_forecast(ifc)

            total_outputs += len(outputs)
            logger.info(
                "realm=%s: %d archetype + %d item forecast rows persisted.",
                realm, len(outputs), len(item_outputs),
            )

        logger.info(
            "ForecastStage complete: %d archetype ForecastOutput rows across %d realm(s).",
            total_outputs, len(realms),
        )
        return total_outputs


# ── Item-level forecast generation ─────────────────────────────────────────────


def _generate_item_forecasts(
    conn: sqlite3.Connection,
    run_id: int,
    archetype_forecasts: list[ForecastOutput],
    realm_slug: str,
    min_history_days: int = 14,
    run_date: date | None = None,
) -> list[ForecastOutput]:
    """Generate item-level forecasts for recipe-linked items and items with history.

    Uses trend-ratio scaling: item_forecast = item_current × (archetype_forecast
    / archetype_current).  This preserves each item's specific price level while
    applying the archetype's directional trend for future horizons.

    Coverage is the union of:
      • Recipe-linked items (output items + required reagents) — always included.
      • Items with ≥ min_history_days distinct observation days — enables per-item
        ROI surfacing in recommendation reports.

    Items without a current price observation or without an archetype mapping
    are skipped.  Results are stored with item_id set and archetype_id = None so
    the crafting advisor can prefer them over archetype-level forecasts.

    Every read here is against the rollup tables and the small reference tables;
    nothing is written, and nothing reads the archetype forecasts back from the
    database, so the caller can (and does) run this on a read-only connection
    before its write transaction opens (issue #107).

    Args:
        conn:               Open DB connection, used read-only.
        run_id:             FK for provenance in forecast_outputs.
        archetype_forecasts: Archetype-level ForecastOutputs, in memory; they
                            need not have been persisted yet.
        realm_slug:         Realm to fetch current prices for.
        min_history_days:   Min distinct observation days required for non-recipe
                            items to be included (default 14).
        run_date:           Anchor date for current-price windows and target
                            dates (default: today).

    Returns:
        List of item-level ForecastOutput objects (not yet persisted by caller).
    """
    if run_date is None:
        run_date = date.today()
    if not archetype_forecasts:
        return []

    # Build archetype forecast lookup: (archetype_id, horizon_label) → ForecastOutput
    arch_fc_map: dict[tuple[int, str], ForecastOutput] = {}
    for fc in archetype_forecasts:
        if fc.archetype_id is not None:
            arch_fc_map[(fc.archetype_id, fc.forecast_horizon)] = fc

    if not arch_fc_map:
        return []

    # Determine the base model_slug from archetype forecasts for provenance
    base_model_slug = next(
        (fc.model_slug for fc in archetype_forecasts if fc.archetype_id is not None),
        "lgbm",
    )

    # Union of recipe-linked items and items with sufficient price history
    recipe_item_ids = _fetch_recipe_item_ids(conn)
    history_item_ids = _fetch_items_with_history(conn, realm_slug, min_history_days)
    all_item_ids = list(set(recipe_item_ids) | set(history_item_ids))
    if not all_item_ids:
        return []

    # Fetch item → archetype mapping for all candidate items
    item_archetype_map = _fetch_item_archetypes(conn, all_item_ids)
    if not item_archetype_map:
        return []

    # Fetch 7-day rolling mean prices per item and per archetype
    item_ids_with_arch = list(item_archetype_map.keys())
    archetype_ids = list(set(item_archetype_map.values()))

    item_current_prices = _fetch_item_prices(conn, item_ids_with_arch, realm_slug, run_date)
    archetype_current_prices = _fetch_archetype_prices(conn, archetype_ids, realm_slug, run_date)

    if not item_current_prices:
        # Every candidate is about to be skipped for want of a current price.
        # With the prices coming from the rollups, that is what a stalled rollup
        # step looks like from here, and it must not pass in silence while the
        # archetype forecasts keep flowing (the #123 rule).
        logger.warning(
            "realm=%s: daily_rollup_item holds no priced rows for any of the %d "
            "candidate items in the 7 days ending %s; no item-level forecasts this "
            "run. If the hourly rollup step has been failing, 'backfill-rollups' "
            "is the repair path.",
            realm_slug, len(item_ids_with_arch), run_date.isoformat(),
        )

    item_forecasts: list[ForecastOutput] = []
    for item_id, archetype_id in item_archetype_map.items():
        item_current = item_current_prices.get(item_id)
        if item_current is None:
            continue  # No recent price data — skip rather than invent a forecast

        archetype_current = archetype_current_prices.get(archetype_id)

        for horizon_label in ("1d", "7d", "28d"):
            arch_fc = arch_fc_map.get((archetype_id, horizon_label))
            if arch_fc is None:
                continue

            # Trend-ratio: scale item's current price by archetype trend direction
            if archetype_current is not None and archetype_current > 0:
                ratio = arch_fc.predicted_price_gold / archetype_current
                predicted = max(0.0, item_current * ratio)
                ci_lower = max(0.0, item_current * (arch_fc.confidence_lower / archetype_current))
                ci_upper = item_current * (arch_fc.confidence_upper / archetype_current)
            else:
                # Fallback: archetype forecast level (no item-level differentiation)
                predicted = arch_fc.predicted_price_gold
                ci_lower = arch_fc.confidence_lower
                ci_upper = arch_fc.confidence_upper

            # Ensure CI ordering is valid after any floating-point rounding
            ci_lower = min(ci_lower, predicted)
            ci_upper = max(ci_upper, predicted)

            target_date = run_date + timedelta(days=_HORIZON_DAYS[horizon_label])

            item_forecasts.append(
                ForecastOutput(
                    run_id=run_id,
                    archetype_id=None,
                    item_id=item_id,
                    realm_slug=realm_slug,
                    forecast_horizon=horizon_label,  # type: ignore[arg-type]
                    target_date=target_date,
                    predicted_price_gold=predicted,
                    confidence_lower=ci_lower,
                    confidence_upper=ci_upper,
                    confidence_pct=arch_fc.confidence_pct,
                    model_slug=f"item_ratio_{base_model_slug}",
                    features_hash=None,
                )
            )

    logger.info(
        "Generated %d item-level forecasts for %d items "
        "(%d recipe-linked, %d history-based) (realm=%s).",
        len(item_forecasts),
        len(item_archetype_map),
        len(recipe_item_ids),
        len(history_item_ids),
        realm_slug,
    )
    return item_forecasts


def _fetch_recipe_item_ids(conn: sqlite3.Connection) -> list[int]:
    """Return all item IDs that appear as recipe outputs or required reagents."""
    rows = conn.execute(
        """
        SELECT DISTINCT output_item_id AS item_id FROM recipes
        UNION
        SELECT DISTINCT ingredient_item_id AS item_id
        FROM recipe_reagents
        WHERE reagent_type = 'required'
        """
    ).fetchall()
    return [int(r[0]) for r in rows if r[0] is not None]


# ── Rollup queries ────────────────────────────────────────────────────────────
#
# Exposed through the *_sql() builders so tests can pin each query plan against
# the exact string production runs rather than a copy that can drift.  All three
# read the daily rollup tables (issue #107); see the module docstring.

_ITEMS_WITH_HISTORY_SQL = """
        SELECT item_id
        FROM daily_rollup_item
        WHERE realm_slug = ?
          AND price_obs_count_pos > 0
        GROUP BY item_id
        HAVING COUNT(*) >= ?
"""

_ITEM_PRICES_SQL = """
        SELECT item_id,
               SUM(qty_weighted_price_sum_pos) / NULLIF(SUM(qty_weight_sum_pos), 0)
        FROM daily_rollup_item
        WHERE realm_slug = ?
          AND obs_date >= ?
          AND obs_date <  ?
          AND item_id IN ({placeholders})
        GROUP BY item_id
"""

_ARCHETYPE_PRICES_SQL = """
        SELECT archetype_id,
               SUM(qty_weighted_price_sum) / NULLIF(SUM(qty_weight_sum), 0)
        FROM daily_rollup_archetype
        WHERE realm_slug = ?
          AND obs_date >= ?
          AND obs_date <  ?
          AND archetype_id IN ({placeholders})
        GROUP BY archetype_id
"""


def _items_with_history_sql() -> str:
    """Return the history query (no placeholder list to fill)."""
    return _ITEMS_WITH_HISTORY_SQL


def _item_prices_sql(n_items: int) -> str:
    """Return the item price query for an ``item_id IN (...)`` list of ``n_items``."""
    return _ITEM_PRICES_SQL.format(placeholders=",".join("?" * n_items))


def _archetype_prices_sql(n_archetypes: int) -> str:
    """Return the archetype price query for an ``IN (...)`` list of ``n_archetypes``."""
    return _ARCHETYPE_PRICES_SQL.format(placeholders=",".join("?" * n_archetypes))


def _price_window(run_date: date) -> tuple[str, str]:
    """The 7-day window as half-open ``obs_date`` bounds: [run_date - 6, run_date + 1)."""
    return (
        (run_date - timedelta(days=6)).isoformat(),
        (run_date + timedelta(days=1)).isoformat(),
    )


def _fetch_items_with_history(
    conn: sqlite3.Connection,
    realm_slug: str,
    min_days: int = 14,
) -> list[int]:
    """Return item IDs with at least min_days distinct observation days.

    Counts ``daily_rollup_item`` rows per item.  A rollup row exists for a
    (item, realm, day) only when that day had a non-outlier observation, and
    ``price_obs_count_pos > 0`` says at least one of those carried a positive
    price, so a counted row is exactly a day the old query's
    ``is_outlier = 0 AND price_gold > 0`` filter would have counted; the
    UNIQUE key makes ``COUNT(*)`` the distinct-day count.

    The count runs over every rollup day the item has.  That is the definition
    v1.12.0 wrote against the then-unbounded normalized table; between the
    retention prune (#149) and #107 it was silently a rolling 30-day count,
    because that is all the normalized table held.

    Args:
        conn:       Open DB connection.
        realm_slug: Realm to query.
        min_days:   Minimum number of distinct calendar days required.

    Returns:
        List of item_ids that satisfy the history threshold.
    """
    rows = conn.execute(_items_with_history_sql(), (realm_slug, min_days)).fetchall()
    return [int(r[0]) for r in rows if r[0] is not None]


def _fetch_item_archetypes(
    conn: sqlite3.Connection,
    item_ids: list[int],
) -> dict[int, int]:
    """Return item_id → archetype_id for items that have an archetype assigned."""
    if not item_ids:
        return {}
    placeholders = ",".join("?" * len(item_ids))
    rows = conn.execute(
        f"SELECT item_id, archetype_id FROM items "
        f"WHERE item_id IN ({placeholders}) AND archetype_id IS NOT NULL;",
        item_ids,
    ).fetchall()
    return {int(r[0]): int(r[1]) for r in rows}


def _fetch_item_prices(
    conn: sqlite3.Connection,
    item_ids: list[int],
    realm_slug: str,
    run_date: date,
) -> dict[int, float]:
    """7-day quantity-weighted mean price per item from ``daily_rollup_item``.

    ``qty_weighted_price_sum_pos`` and ``qty_weight_sum_pos`` are the per-day
    sums of ``price_gold * COALESCE(quantity_listed, 1)`` and of
    ``COALESCE(quantity_listed, 1)`` over non-outlier rows with
    ``price_gold > 0``, so their ratio across the window is exactly the mean
    the old query computed over the normalized rows under the same filter.
    The ``_pos`` pair is the right one here because this query always
    excluded zero prices; an item whose window carries no positive-price
    weight divides by NULL and is absent, as it was before.
    """
    if not item_ids:
        return {}
    start_date, end_date = _price_window(run_date)
    rows = conn.execute(
        _item_prices_sql(len(item_ids)),
        [realm_slug, start_date, end_date, *item_ids],
    ).fetchall()
    return {int(r[0]): float(r[1]) for r in rows if r[1] is not None}


def _fetch_archetype_prices(
    conn: sqlite3.Connection,
    archetype_ids: list[int],
    realm_slug: str,
    run_date: date,
) -> dict[int, float]:
    """7-day quantity-weighted mean price per archetype from ``daily_rollup_archetype``.

    The archetype rollup's ``qty_weighted_price_sum`` / ``qty_weight_sum`` are
    already positive-price-only and are grouped through ``items.archetype_id``
    at build time, which is the JOIN the old query made at read time.  An
    item reassigned to another archetype after a day was rolled up stays under
    the old archetype for that day, the same acceptance the training features
    (daily_agg) make.
    """
    if not archetype_ids:
        return {}
    start_date, end_date = _price_window(run_date)
    rows = conn.execute(
        _archetype_prices_sql(len(archetype_ids)),
        [realm_slug, start_date, end_date, *archetype_ids],
    ).fetchall()
    return {int(r[0]): float(r[1]) for r in rows if r[1] is not None}



def _fetch_cold_start_blend_data(
    conn: sqlite3.Connection,
    realm_slug: str,
    source_expansion: str,
    target_expansion: str,
    run_date: date | None = None,
) -> dict[int, tuple[float, float]]:
    """Fetch blend data for cold-start prediction anchoring.

    Queries archetype_mappings for TWW→Midnight archetype pairs, then fetches
    the 7-day rolling mean price for each source (TWW) archetype.  The result
    is keyed by target (Midnight) archetype_id so ``run_inference()`` can look
    up blend data by the archetype it is currently scoring.

    Args:
        conn:              Open DB connection.
        realm_slug:        Realm to fetch source archetype prices for.
        source_expansion:  Expansion slug of the source (e.g. ``"tww"``).
        target_expansion:  Expansion slug of the target (e.g. ``"midnight"``).
        run_date:          Anchor date for the rolling price window (default: today).

    Returns:
        Dict mapping target_archetype_id → (source_rolling_price, confidence).
        Entries are omitted when the source archetype has no recent price data.
    """
    # Fetch all mappings between expansions with confidence scores
    mapping_rows = conn.execute(
        """
        SELECT source_archetype_id, target_archetype_id, confidence_score
        FROM archetype_mappings
        WHERE source_expansion = ? AND target_expansion = ?
          AND confidence_score > 0
        """,
        [source_expansion, target_expansion],
    ).fetchall()

    if not mapping_rows:
        return {}

    # Build lookup: source_archetype_id → (target_archetype_id, confidence)
    source_to_target: dict[int, tuple[int, float]] = {
        int(r[0]): (int(r[1]), float(r[2])) for r in mapping_rows
    }
    source_ids = list(source_to_target.keys())

    # Fetch 7-day rolling prices for source archetypes
    if run_date is None:
        run_date = date.today()
    source_prices = _fetch_archetype_prices(conn, source_ids, realm_slug, run_date)

    # Build result: target_archetype_id → (source_price, confidence)
    result: dict[int, tuple[float, float]] = {}
    for source_id, (target_id, confidence) in source_to_target.items():
        source_price = source_prices.get(source_id)
        if source_price is not None and source_price > 0:
            result[target_id] = (source_price, confidence)

    return result
