"""
FeatureBuildStage — engineer features from normalized observations for modeling.

Outputs training and inference Parquet files plus a JSON manifest to
``config.data.processed_dir/features/``.

Features engineered (per archetype × realm × date)
----------------------------------------------------
- **Price summary** — daily mean/min/max, market value, historical value, obs count.
- **Volume / velocity** — quantity_sum, auctions_sum, is_volume_proxy flag.
- **Lag features** — price_mean at N calendar days prior (1/3/7/14/28).
- **Rolling statistics** — mean and std over 7/14/28-day windows.
- **Momentum** — percentage price change vs N days ago.
- **Temporal** — day-of-week, day-of-month, week-of-year, days-since-expansion.
- **Event proximity** — days to next / since last event, active flag, severity,
  archetype-specific impact direction.  Uses strict ``is_known_at()`` leakage guard.
- **Archetype encoding** — category, sub-tag, cold-start flag, transfer mapping.
- **Target labels** (training Parquet only) — forward price at 1/7/28 days.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta

from wow_forecaster.models.meta import RunMetadata
from wow_forecaster.pipeline.base import PipelineStage

log = logging.getLogger(__name__)


class FeatureBuildStage(PipelineStage):
    """Build feature vectors from normalized market observations.

    Reads from ``market_observations_normalized`` (joined through ``items``
    for archetype_id — the normalized table's own archetype_id column is
    always NULL until the normalize TODO is filled).

    Writes:
    - ``<processed_dir>/features/training/train_<realm>_<start>_<end>.parquet``
    - ``<processed_dir>/features/inference/infer_<realm>_<start>_<end>.parquet``
    - ``<processed_dir>/features/manifests/manifest_<realm>_<start>_<end>.json``
    """

    stage_name = "feature_build"

    def _execute(
        self,
        run: RunMetadata,
        realm_slugs: list[str] | None = None,
        start_date: date | None = None,
        end_date: date | None = None,
        archetype_ids: list[int] | None = None,
        **kwargs,
    ) -> int:
        """Build and persist feature vectors + Parquet files.

        Args:
            run:           In-progress ``RunMetadata``.
            realm_slugs:   Realms to process.  Defaults to
                           ``config.realms.defaults``.
            start_date:    First obs_date to include.  Defaults to
                           ``end_date - training_lookback_days``.
            end_date:      Last obs_date to include.  Defaults to today.
            archetype_ids: Reserved for future filtering.  Currently unused;
                           all archetypes are processed.

        Returns:
            Total feature rows written across all realms and Parquet files.
        """
        from wow_forecaster.db.connection import get_connection
        from wow_forecaster.features.dataset_builder import build_datasets

        realms: list[str] = realm_slugs or list(self.config.realms.defaults)
        # Use today + 1 day so observations stored with a UTC timestamp that
        # crosses midnight (e.g. 00:22 UTC = evening prior day locally) are
        # always included in the window.
        end: date = end_date or (date.today() + timedelta(days=1))
        start: date = start_date or (
            end - timedelta(days=self.config.features.training_lookback_days)
        )

        log.info(
            "FeatureBuildStage: realms=%s  %s → %s",
            ", ".join(realms),
            start.isoformat(),
            end.isoformat(),
        )

        with get_connection(
            self.db_path,
            wal_mode=self.config.database.wal_mode,
            busy_timeout_ms=self.config.database.busy_timeout_ms,
        ) as conn:
            total = build_datasets(
                conn=conn,
                config=self.config,
                run=run,
                realm_slugs=realms,
                start_date=start,
                end_date=end,
            )

        log.info("FeatureBuildStage: wrote %d feature rows total.", total)
        return total
