"""
RecommendStage — derive trading recommendations from forecast outputs.

Current status: **STUB** — raises ``NotImplementedError``.

Planned recommendation logic:
  - For each archetype + horizon, evaluate:
      - Expected ROI: (predicted_price - current_price) / current_price
      - Confidence width: (CI_upper - CI_lower) / predicted_price (volatility proxy)
      - Event proximity: upcoming high-severity events within horizon window
  - Scoring rules (configurable thresholds):
      - BUY:   expected_roi > 20% AND confidence_width < 50% AND event spike incoming
      - SELL:  expected_roi < -15% OR event crash incoming AND holding position
      - HOLD:  |expected_roi| < 10% AND no major events
      - AVOID: confidence_width > 100% (high uncertainty)
  - Output: Top-N recommendations per category, ranked by priority score.
"""

from __future__ import annotations

from wow_forecaster.models.meta import RunMetadata
from wow_forecaster.pipeline.base import PipelineStage


class RecommendStage(PipelineStage):
    """Convert forecast outputs into ranked buy/sell/hold/avoid recommendations.

    Stub: not yet implemented. Raises ``NotImplementedError`` when run.
    """

    stage_name = "recommend"

    def _execute(
        self,
        run: RunMetadata,
        top_n_per_category: int = 3,
        **kwargs,
    ) -> int:
        """Generate and persist recommendation outputs.

        Args:
            run: In-progress ``RunMetadata``.
            top_n_per_category: Maximum recommendations per archetype category.

        Returns:
            Number of ``RecommendationOutput`` rows written.

        Raises:
            NotImplementedError: This stage is not yet implemented.
        """
        raise NotImplementedError(
            "RecommendStage is not yet implemented. "
            "Implement ROI scoring, event proximity weighting, and ranking logic."
        )
