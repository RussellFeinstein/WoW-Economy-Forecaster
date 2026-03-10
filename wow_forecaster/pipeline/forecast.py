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
  5. Persist ForecastOutput rows to forecast_outputs SQLite table.
  6. Generate and persist item-level forecasts for recipe-linked items
     (output items and required reagents) via trend-ratio scaling.

Item-level forecasts
--------------------
After archetype-level forecasts are written, _generate_item_forecasts()
computes item-specific predictions using the trend-ratio method:

    item_forecast = item_current × (archetype_forecast / archetype_current)

This preserves each item's current price level while applying the archetype's
directional trend.  Results are stored in forecast_outputs with item_id set
and archetype_id = None.  The crafting advisor prefers these over archetype
forecasts when pricing specific reagents and output items.

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
from datetime import date, timedelta
from pathlib import Path

from wow_forecaster.models.forecast import ForecastOutput
from wow_forecaster.models.meta import RunMetadata
from wow_forecaster.pipeline.base import PipelineStage

logger = logging.getLogger(__name__)

# Horizon label → days offset for target_date computation
_HORIZON_DAYS: dict[str, int] = {"1d": 1, "7d": 7, "28d": 28}


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
        **kwargs,
    ) -> int:
        """Generate and persist forecasts for configured realms.

        Args:
            run:        In-progress RunMetadata (mutable).
            realm_slug: Single realm to target. If None, uses config defaults.
            horizons:   Horizon list override (int days). If None, uses
                        config.features.target_horizons_days.

        Returns:
            Total ForecastOutput rows written to DB.

        Raises:
            ValueError: If run.run_id is not set after pre-persist.
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

            # Persist archetype-level forecasts and generate item-level forecasts
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

                # Generate and persist item-level forecasts for recipe-linked items
                item_outputs = _generate_item_forecasts(conn, run.run_id, outputs, realm)
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
) -> list[ForecastOutput]:
    """Generate item-level forecasts for all recipe-linked items.

    Uses trend-ratio scaling: item_forecast = item_current × (archetype_forecast
    / archetype_current).  This preserves each item's specific price level while
    applying the archetype's directional trend for future horizons.

    Items without a current price observation or without an archetype mapping
    are skipped.  Results are stored with item_id set and archetype_id = None so
    the crafting advisor can prefer them over archetype-level forecasts.

    Args:
        conn:               Open DB connection (within an active transaction).
        run_id:             FK for provenance in forecast_outputs.
        archetype_forecasts: Archetype-level ForecastOutputs just written to DB.
        realm_slug:         Realm to fetch current prices for.

    Returns:
        List of item-level ForecastOutput objects (not yet persisted by caller).
    """
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

    # Fetch recipe-linked item IDs (output items + required reagents)
    recipe_item_ids = _fetch_recipe_item_ids(conn)
    if not recipe_item_ids:
        return []

    # Fetch item → archetype mapping for recipe items
    item_archetype_map = _fetch_item_archetypes(conn, recipe_item_ids)
    if not item_archetype_map:
        return []

    # Fetch 7-day rolling mean prices per item and per archetype
    run_date = date.today()
    item_ids_with_arch = list(item_archetype_map.keys())
    archetype_ids = list(set(item_archetype_map.values()))

    item_current_prices = _fetch_item_prices(conn, item_ids_with_arch, realm_slug, run_date)
    archetype_current_prices = _fetch_archetype_prices(conn, archetype_ids, realm_slug, run_date)

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

            target_date = date.today() + timedelta(days=_HORIZON_DAYS[horizon_label])

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
        "Generated %d item-level forecasts for %d recipe-linked items (realm=%s).",
        len(item_forecasts),
        len(item_archetype_map),
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
    """7-day quantity-weighted mean price per item from market_observations_normalized."""
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
          AND price_gold > 0
          AND item_id IN ({placeholders})
        GROUP BY item_id
        """,
        [realm_slug, start_ts, end_ts] + list(item_ids),
    ).fetchall()
    return {int(r[0]): float(r[1]) for r in rows if r[1] is not None}


def _fetch_archetype_prices(
    conn: sqlite3.Connection,
    archetype_ids: list[int],
    realm_slug: str,
    run_date: date,
) -> dict[int, float]:
    """7-day quantity-weighted mean price per archetype from market_observations_normalized."""
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
    return {int(r[0]): float(r[1]) for r in rows if r[1] is not None}


def _fetch_cold_start_blend_data(
    conn: sqlite3.Connection,
    realm_slug: str,
    source_expansion: str,
    target_expansion: str,
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
    source_prices = _fetch_archetype_prices(conn, source_ids, realm_slug, date.today())

    # Build result: target_archetype_id → (source_price, confidence)
    result: dict[int, tuple[float, float]] = {}
    for source_id, (target_id, confidence) in source_to_target.items():
        source_price = source_prices.get(source_id)
        if source_price is not None and source_price > 0:
            result[target_id] = (source_price, confidence)

    return result
