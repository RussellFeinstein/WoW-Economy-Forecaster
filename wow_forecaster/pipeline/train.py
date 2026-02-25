"""
TrainStage — train or update LightGBM forecasting models from feature data.

Training flow
-------------
For each realm in config.realms.defaults (or the caller-supplied list):
  1. Find the latest training Parquet in data/processed/features/training/.
  2. Apply time-based validation split (last validation_split_days days).
  3. Train one LightGBMForecaster per horizon in config.features.target_horizons_days.
  4. Write model artifacts (.pkl + .json) to config.model.artifact_dir.
  5. Register model in model_metadata DB table (marking old models inactive).

Output
------
Artifacts:   data/outputs/model_artifacts/lgbm_{horizon}d_{realm}_{date}.{pkl,json}
DB records:  model_metadata (one row per horizon trained)
run_metadata: one row per TrainStage.run() call

Returns number of models successfully trained across all realms.
"""

from __future__ import annotations

import logging
from pathlib import Path

from wow_forecaster.models.meta import RunMetadata
from wow_forecaster.pipeline.base import PipelineStage

logger = logging.getLogger(__name__)


class TrainStage(PipelineStage):
    """Train or update LightGBM price forecasting models.

    Implements the full training pipeline: load Parquet → split → fit
    LightGBM per horizon → save artifacts → register in DB.
    """

    stage_name = "train"

    def _execute(
        self,
        run: RunMetadata,
        realm_slugs: list[str] | None = None,
        **kwargs,
    ) -> int:
        """Train models for each configured realm.

        Args:
            run:         In-progress RunMetadata (mutable).
            realm_slugs: Realms to train for. Defaults to config.realms.defaults.

        Returns:
            Total number of model artifacts written (horizons × realms).
        """
        from wow_forecaster.db.connection import get_connection
        from wow_forecaster.ml.trainer import find_latest_training_parquet, train_models

        # Pre-persist to get a run_id for model_metadata FK use
        self._persist_run(run)

        realms        = realm_slugs or list(self.config.realms.defaults)
        processed_dir = Path(self.config.data.processed_dir)
        total_trained = 0

        for realm in realms:
            parquet_path = find_latest_training_parquet(processed_dir, realm)

            if parquet_path is None:
                logger.warning(
                    "No training Parquet found for realm '%s'. "
                    "Run 'build-datasets' first.",
                    realm,
                )
                continue

            logger.info("Training models for realm=%s  parquet=%s", realm, parquet_path)

            try:
                with get_connection(
                    self.db_path,
                    wal_mode=self.config.database.wal_mode,
                    busy_timeout_ms=self.config.database.busy_timeout_ms,
                ) as conn:
                    trained = train_models(
                        conn=conn,
                        config=self.config,
                        run=run,
                        training_parquet_path=parquet_path,
                        realm_slug=realm,
                    )
                total_trained += len(trained)
                logger.info(
                    "realm=%s trained %d model(s).", realm, len(trained)
                )
            except (FileNotFoundError, ValueError) as exc:
                logger.error("Training failed for realm=%s: %s", realm, exc)
                continue

        logger.info(
            "TrainStage complete: %d model artifact(s) across %d realm(s).",
            total_trained, len(realms),
        )
        return total_trained
