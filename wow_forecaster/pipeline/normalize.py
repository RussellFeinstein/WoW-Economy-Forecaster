"""
NormalizeStage — convert raw observations to ``market_observations_normalized``.

Current status: **STUB** — raises ``NotImplementedError``.

Future implementation:
  1. Fetch unprocessed raw observations in batches (``get_unprocessed_raw()``).
  2. Convert copper to gold (divide by 10_000).
  3. Compute rolling z-score within item+realm window.
  4. Flag outliers (``|z_score| > config.pipeline.outlier_z_threshold``).
  5. Write ``NormalizedMarketObservation`` records.
  6. Mark raw observations as processed (``mark_processed()``).
"""

from __future__ import annotations

from wow_forecaster.models.meta import RunMetadata
from wow_forecaster.pipeline.base import PipelineStage


class NormalizeStage(PipelineStage):
    """Transform raw copper-priced observations to gold-priced with outlier flags.

    Stub: not yet implemented. Raises ``NotImplementedError`` when run.
    """

    stage_name = "normalize"

    def _execute(self, run: RunMetadata, **kwargs) -> int:
        """Process unprocessed raw observations and write normalized records.

        Returns:
            Number of normalized rows written.

        Raises:
            NotImplementedError: This stage is not yet implemented.
        """
        raise NotImplementedError(
            "NormalizeStage is not yet implemented. "
            "Implement copper→gold conversion, z-score computation, and outlier detection."
        )
