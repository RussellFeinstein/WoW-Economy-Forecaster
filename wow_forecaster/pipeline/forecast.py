"""
ForecastStage — generate price forecasts from trained models.

Current status: **STUB** — raises ``NotImplementedError``.

Planned behavior:
  - Load the active model for each archetype from ``model_metadata``.
  - Fetch the latest feature vector for each (archetype, realm) pair.
  - Run model inference to produce ``(predicted_price, CI_lower, CI_upper)``.
  - Write ``ForecastOutput`` records for each configured horizon.
  - If no model is trained for a Midnight archetype, fall back to the mapped
    TWW archetype model scaled by transfer_confidence.

Look-ahead bias guard:
  - Event features must be filtered by ``WoWEvent.is_known_at(as_of=target_date)``
    when producing historical forecasts (backtesting mode).
"""

from __future__ import annotations

from wow_forecaster.models.meta import RunMetadata
from wow_forecaster.pipeline.base import PipelineStage


class ForecastStage(PipelineStage):
    """Run trained models to produce point forecasts with confidence intervals.

    Stub: not yet implemented. Raises ``NotImplementedError`` when run.
    """

    stage_name = "forecast"

    def _execute(
        self,
        run: RunMetadata,
        realm_slug: str | None = None,
        horizons: list[str] | None = None,
        **kwargs,
    ) -> int:
        """Generate and persist forecasts.

        Args:
            run: In-progress ``RunMetadata``.
            realm_slug: Target realm. If ``None``, uses config defaults.
            horizons: Forecast horizons. If ``None``, uses config defaults.

        Returns:
            Number of ``ForecastOutput`` rows written.

        Raises:
            NotImplementedError: This stage is not yet implemented.
        """
        raise NotImplementedError(
            "ForecastStage is not yet implemented. "
            "Implement model inference, CI generation, and transfer fallback logic."
        )
