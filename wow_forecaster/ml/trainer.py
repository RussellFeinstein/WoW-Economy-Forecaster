"""
Model training orchestrator.

train_models() loads a training Parquet, applies a time-based validation
split, trains one LightGBMForecaster per configured forecast horizon, and
writes model artifacts + metadata JSON to disk.

Design decisions
----------------
- One global model per horizon (not per-archetype).  Cross-archetype patterns
  (event-driven spikes, weekly seasonality) are best learned with many series.
- Time-based validation split: last ``validation_split_days`` calendar days.
  NEVER random — random splits on time-series create look-ahead bias.
- Target mapping: horizon=1 -> target_price_1d, horizon=7 -> target_price_7d,
  horizon=28 -> target_price_28d.
- Artifact naming: ``lgbm_{horizon}d_{realm}_{date}.{pkl,json}``
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import date
from pathlib import Path
from typing import Any

from wow_forecaster.config import AppConfig
from wow_forecaster.ml.feature_selector import (
    CATEGORICAL_FEATURE_COLS,
    TARGET_COL_MAP,
    TRAINING_FEATURE_COLS,
    encode_row,
)
from wow_forecaster.ml.lgbm_model import LightGBMForecaster
from wow_forecaster.models.meta import RunMetadata

logger = logging.getLogger(__name__)


def train_models(
    conn: sqlite3.Connection,
    config: AppConfig,
    run: RunMetadata,
    training_parquet_path: Path,
    realm_slug: str,
) -> dict[int, LightGBMForecaster]:
    """Train one LightGBMForecaster per forecast horizon.

    Args:
        conn:                  Open SQLite connection (for model_metadata).
        config:                Application config (hyperparameters + paths).
        run:                   In-progress RunMetadata (provenance only).
        training_parquet_path: Path to the training Parquet file.
        realm_slug:            Realm being trained for.

    Returns:
        Dict mapping horizon_days (int) to a fitted LightGBMForecaster.

    Raises:
        FileNotFoundError: If training_parquet_path does not exist.
        ValueError:         If the Parquet is empty or no models succeed.
    """
    import pyarrow.parquet as pq

    if not training_parquet_path.exists():
        raise FileNotFoundError(
            f"Training Parquet not found: {training_parquet_path}"
        )

    logger.info("Loading training data: %s", training_parquet_path)
    table = pq.read_table(str(training_parquet_path))
    raw_rows: list[dict[str, Any]] = table.to_pylist()

    if not raw_rows:
        raise ValueError(f"Training Parquet is empty: {training_parquet_path}")

    encoded_rows = [encode_row(r) for r in raw_rows]

    # ── Time-based validation split ───────────────────────────────────────────
    date_strs = sorted(
        {str(r.get("obs_date", "")) for r in raw_rows if r.get("obs_date") is not None}
    )

    val_days = config.model.validation_split_days
    val_split_date: str | None = None

    if len(date_strs) > val_days:
        # Hold out last val_days calendar dates as validation
        val_split_date = date_strs[-(val_days + 1)]
        train_rows = [r for r in encoded_rows if str(r.get("obs_date", "")) <= val_split_date]
        val_rows   = [r for r in encoded_rows if str(r.get("obs_date", "")) >  val_split_date]
    else:
        # Fallback: 80/20 row-count split when window is very short
        split_idx  = max(1, int(len(encoded_rows) * 0.8))
        train_rows = encoded_rows[:split_idx]
        val_rows   = encoded_rows[split_idx:]
        logger.warning(
            "Training window shorter than validation_split_days=%d; "
            "using 80/20 row-count split.",
            val_days,
        )

    logger.info(
        "Split: %d train rows, %d val rows (split_date=%s)",
        len(train_rows), len(val_rows), val_split_date or "80/20",
    )

    # ── Train one model per horizon ───────────────────────────────────────────
    horizons    = config.features.target_horizons_days
    artifact_dir = Path(config.model.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    today_str   = date.today().isoformat()

    trained: dict[int, LightGBMForecaster] = {}

    for horizon_days in horizons:
        target_col = TARGET_COL_MAP.get(horizon_days)
        if target_col is None:
            logger.warning("No target column for horizon=%dd; skipping.", horizon_days)
            continue

        logger.info(
            "Training LightGBM  horizon=%dd  target=%s", horizon_days, target_col
        )

        forecaster = LightGBMForecaster(
            horizon_days=horizon_days,
            num_leaves=config.model.num_leaves,
            learning_rate=config.model.learning_rate,
            n_estimators=config.model.n_estimators,
            min_child_samples=config.model.min_child_samples,
            feature_fraction=config.model.feature_fraction,
            bagging_fraction=config.model.bagging_fraction,
            bagging_freq=config.model.bagging_freq,
            early_stopping_rounds=config.model.early_stopping_rounds,
        )

        try:
            val_metrics = forecaster.fit(
                train_rows=train_rows,
                val_rows=val_rows,
                feature_cols=TRAINING_FEATURE_COLS,
                categorical_cols=CATEGORICAL_FEATURE_COLS,
                target_col=target_col,
            )
        except ValueError as exc:
            logger.error("Training failed  horizon=%dd: %s", horizon_days, exc)
            continue

        logger.info("horizon=%dd  val=%s", horizon_days, val_metrics)

        # Persist artifact
        artifact_path = (
            artifact_dir / f"lgbm_{horizon_days}d_{realm_slug}_{today_str}.pkl"
        )
        meta_path = (
            artifact_dir / f"lgbm_{horizon_days}d_{realm_slug}_{today_str}.json"
        )
        forecaster.save(artifact_path)
        forecaster.write_metadata(
            meta_path,
            realm_slug=realm_slug,
            dataset_version=training_parquet_path.name,
        )

        # Register in DB
        _register_model(
            conn=conn,
            horizon_days=horizon_days,
            realm_slug=realm_slug,
            artifact_path=str(artifact_path),
            val_metrics=val_metrics,
            training_data_start=date_strs[0] if date_strs else None,
            training_data_end=date_strs[-1] if date_strs else None,
        )

        trained[horizon_days] = forecaster

    if not trained:
        raise ValueError(
            f"No models successfully trained for realm '{realm_slug}'. "
            "Check training data has valid target labels."
        )

    logger.info("Training complete: %d model(s) for realm=%s", len(trained), realm_slug)
    return trained


def _register_model(
    conn: sqlite3.Connection,
    horizon_days: int,
    realm_slug: str,
    artifact_path: str,
    val_metrics: dict[str, float],
    training_data_start: str | None = None,
    training_data_end: str | None = None,
) -> None:
    """Deactivate previous models for this horizon+realm, insert new record."""
    slug_prefix = f"lgbm_{horizon_days}d"
    conn.execute(
        "UPDATE model_metadata SET is_active = 0 WHERE slug LIKE ? AND is_active = 1;",
        (f"{slug_prefix}%{realm_slug}%",),
    )
    new_slug = (
        f"{slug_prefix}_{LightGBMForecaster.MODEL_VERSION}_{realm_slug}"
    )
    conn.execute(
        """
        INSERT INTO model_metadata (
            slug, display_name, model_type, version, hyperparameters,
            training_data_start, training_data_end,
            validation_mae, validation_rmse, artifact_path, is_active
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
        ON CONFLICT(slug) DO UPDATE SET
            training_data_start = excluded.training_data_start,
            training_data_end   = excluded.training_data_end,
            validation_mae      = excluded.validation_mae,
            validation_rmse     = excluded.validation_rmse,
            artifact_path       = excluded.artifact_path,
            is_active           = 1;
        """,
        (
            new_slug,
            f"LightGBM {horizon_days}d — {realm_slug}",
            "lightgbm",
            LightGBMForecaster.MODEL_VERSION,
            json.dumps({"horizon_days": horizon_days}),
            training_data_start,
            training_data_end,
            val_metrics.get("mae"),
            val_metrics.get("rmse"),
            artifact_path,
        ),
    )
    conn.commit()


# ── Artifact discovery helpers ─────────────────────────────────────────────────


def find_latest_training_parquet(
    processed_dir: Path, realm_slug: str
) -> Path | None:
    """Return the most recently modified training Parquet for a realm.

    Args:
        processed_dir: ``config.data.processed_dir`` as a Path.
        realm_slug:    Realm to look up.

    Returns:
        Path to the .parquet file, or None if none found.
    """
    training_dir = processed_dir / "features" / "training"
    if not training_dir.exists():
        return None
    candidates = sorted(
        training_dir.glob(f"train_{realm_slug}_*.parquet"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def find_latest_model_artifact(
    artifact_dir: Path, realm_slug: str, horizon_days: int
) -> Path | None:
    """Return the most recently modified model artifact for a realm + horizon.

    Args:
        artifact_dir:  ``config.model.artifact_dir`` as a Path.
        realm_slug:    Realm slug.
        horizon_days:  Forecast horizon in days.

    Returns:
        Path to the .pkl file, or None if none found.
    """
    if not artifact_dir.exists():
        return None
    candidates = sorted(
        artifact_dir.glob(f"lgbm_{horizon_days}d_{realm_slug}_*.pkl"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None
