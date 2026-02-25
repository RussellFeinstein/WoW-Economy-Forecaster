"""
Inference module: run trained LightGBM models on the inference Parquet
to produce ForecastOutput objects.

Inference flow
--------------
1. Load inference Parquet (one row per archetype+realm — latest market state).
2. Encode string/bool features to integers via encode_row().
3. For each fitted horizon model:
   a. Batch-predict prices for all rows.
   b. Compute heuristic CI via cold_start.compute_confidence_interval().
   c. Annotate model_slug for cold-start/transfer archetypes.
   d. Hash the feature vector for reproducibility tracing.
4. Return list of ForecastOutput (caller persists to DB).

Horizon → ForecastHorizon string mapping
-----------------------------------------
  1  days -> "1d"
  7  days -> "7d"
  28 days -> "28d"   (added to ForecastHorizon Literal in v0.5.0)
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from wow_forecaster.ml.cold_start import cold_start_model_slug, compute_confidence_interval
from wow_forecaster.ml.feature_selector import TRAINING_FEATURE_COLS, encode_row
from wow_forecaster.ml.lgbm_model import LightGBMForecaster
from wow_forecaster.models.forecast import ForecastOutput
from wow_forecaster.models.meta import RunMetadata

logger = logging.getLogger(__name__)

_HORIZON_TO_STR: dict[int, str] = {1: "1d", 7: "7d", 14: "14d", 28: "28d", 30: "30d"}


def run_inference(
    config,
    run: RunMetadata,
    forecasters: dict[int, LightGBMForecaster],
    inference_parquet_path: Path,
    realm_slug: str,
    target_date: date | None = None,
) -> list[ForecastOutput]:
    """Generate ForecastOutput for all archetypes in the inference Parquet.

    Args:
        config:                  AppConfig instance.
        run:                     RunMetadata with run_id set (FK for outputs).
        forecasters:             Dict horizon_days -> fitted LightGBMForecaster.
        inference_parquet_path:  Path to the inference Parquet file.
        realm_slug:              Realm for this inference run.
        target_date:             Base date; forecast_date = target_date + horizon.
                                 Defaults to today.

    Returns:
        List of ForecastOutput objects (one per archetype × horizon).

    Raises:
        FileNotFoundError: If inference_parquet_path does not exist.
        ValueError:         If run.run_id is None.
    """
    import pyarrow.parquet as pq

    if not inference_parquet_path.exists():
        raise FileNotFoundError(
            f"Inference Parquet not found: {inference_parquet_path}"
        )

    if run.run_id is None:
        raise ValueError(
            "RunMetadata.run_id must be set before run_inference(). "
            "Call self._persist_run(run) first."
        )

    table = pq.read_table(str(inference_parquet_path))
    raw_rows: list[dict[str, Any]] = table.to_pylist()

    if not raw_rows:
        logger.warning("Inference Parquet is empty: %s", inference_parquet_path)
        return []

    if target_date is None:
        target_date = date.today()

    confidence_pct = config.forecast.confidence_pct
    encoded_rows   = [encode_row(r) for r in raw_rows]

    outputs: list[ForecastOutput] = []

    for horizon_days, forecaster in forecasters.items():
        horizon_str   = _HORIZON_TO_STR.get(horizon_days, f"{horizon_days}d")
        forecast_date = target_date + timedelta(days=horizon_days)

        predictions = forecaster.predict(encoded_rows)

        for raw_row, enc_row, pred in zip(raw_rows, encoded_rows, predictions):
            if pred is None:
                continue

            archetype_id = raw_row.get("archetype_id")
            if archetype_id is None:
                continue

            is_cold_start = bool(raw_row.get("is_cold_start", False))
            has_transfer  = bool(raw_row.get("has_transfer_mapping", False))
            rolling_std   = raw_row.get("price_roll_std_7d")
            xfer_conf     = raw_row.get("transfer_confidence")

            ci_lower, ci_upper = compute_confidence_interval(
                predicted=pred,
                rolling_std_7d=float(rolling_std) if rolling_std is not None else None,
                is_cold_start=is_cold_start,
                transfer_confidence=float(xfer_conf) if xfer_conf is not None else None,
                confidence_pct=confidence_pct,
            )

            base_slug  = f"lgbm_{horizon_days}d_{LightGBMForecaster.MODEL_VERSION}"
            model_slug = cold_start_model_slug(base_slug, is_cold_start, has_transfer)
            feat_hash  = _hash_features(enc_row, TRAINING_FEATURE_COLS)

            outputs.append(
                ForecastOutput(
                    run_id=run.run_id,
                    archetype_id=int(archetype_id),
                    realm_slug=realm_slug,
                    forecast_horizon=horizon_str,
                    target_date=forecast_date,
                    predicted_price_gold=round(pred, 4),
                    confidence_lower=round(ci_lower, 4),
                    confidence_upper=round(ci_upper, 4),
                    confidence_pct=confidence_pct,
                    model_slug=model_slug,
                    features_hash=feat_hash,
                )
            )

    logger.info(
        "Inference complete: %d forecasts (%d archetypes x %d horizons) for realm=%s",
        len(outputs), len(raw_rows), len(forecasters), realm_slug,
    )
    return outputs


def find_latest_inference_parquet(
    processed_dir: Path, realm_slug: str
) -> Path | None:
    """Return the most recently modified inference Parquet for a realm.

    Args:
        processed_dir: ``config.data.processed_dir`` as a Path.
        realm_slug:    Realm slug to look up.

    Returns:
        Path to the .parquet file, or None if none found.
    """
    inference_dir = processed_dir / "features" / "inference"
    if not inference_dir.exists():
        return None
    candidates = sorted(
        inference_dir.glob(f"inference_{realm_slug}_*.parquet"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def _hash_features(row: dict[str, Any], cols: list[str]) -> str:
    """Return a 16-char SHA-256 hex digest of the feature vector."""
    vec = {c: row.get(c) for c in cols}
    payload = json.dumps(vec, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]
