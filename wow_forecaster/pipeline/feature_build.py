"""
FeatureBuildStage — engineer features from normalized observations for modeling.

Current status: **STUB** — raises ``NotImplementedError``.

Planned features to engineer:
  - Rolling statistics: mean, std, min, max over 7/14/30/90-day windows.
  - Trend features: linear slope over windows.
  - Seasonality indicators: day-of-week, days-since-expansion-launch.
  - Event distance features: days until/since known events (using is_known_at filter).
  - Cross-archetype features: supply pressure proxy (mat price / consumable price ratio).
  - Volume features: quantity_listed rolling stats, num_auctions trends.

Output: Parquet files written to ``config.data.processed_dir/features/``.
"""

from __future__ import annotations

from wow_forecaster.models.meta import RunMetadata
from wow_forecaster.pipeline.base import PipelineStage


class FeatureBuildStage(PipelineStage):
    """Build feature vectors from normalized market observations.

    Stub: not yet implemented. Raises ``NotImplementedError`` when run.
    """

    stage_name = "feature_build"

    def _execute(
        self,
        run: RunMetadata,
        archetype_ids: list[int] | None = None,
        **kwargs,
    ) -> int:
        """Build and persist feature vectors.

        Args:
            run: In-progress ``RunMetadata``.
            archetype_ids: If provided, only build features for these archetypes.
                           If ``None``, process all archetypes.

        Returns:
            Number of feature rows written.

        Raises:
            NotImplementedError: This stage is not yet implemented.
        """
        raise NotImplementedError(
            "FeatureBuildStage is not yet implemented. "
            "Implement rolling window statistics, event distance features, "
            "and Parquet output."
        )
