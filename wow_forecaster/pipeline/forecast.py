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
from pathlib import Path

from wow_forecaster.models.meta import RunMetadata
from wow_forecaster.pipeline.base import PipelineStage

logger = logging.getLogger(__name__)


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
                        "Failed to load model %s: %s", artifact_path, exc
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

            try:
                outputs = run_inference(
                    config=self.config,
                    run=run,
                    forecasters=forecasters,
                    inference_parquet_path=inf_path,
                    realm_slug=realm,
                )
            except Exception as exc:
                logger.error("Inference failed for realm=%s: %s", realm, exc)
                continue

            # Persist to DB
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

            total_outputs += len(outputs)
            logger.info(
                "realm=%s: %d forecast rows persisted.", realm, len(outputs)
            )

        logger.info(
            "ForecastStage complete: %d ForecastOutput rows across %d realm(s).",
            total_outputs, len(realms),
        )
        return total_outputs
