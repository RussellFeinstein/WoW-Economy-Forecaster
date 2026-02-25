"""
TrainStage — train or update forecasting models from feature data.

Current status: **STUB** — raises ``NotImplementedError``.

Planned training approaches (choose per archetype):
  - Baseline: simple rolling average / linear trend (quick sanity check).
  - Prophet: handles seasonality and holiday events natively.
  - XGBoost: tabular feature approach; strong for event-feature interactions.
  - LSTM: sequence model; best for high-frequency patterns (future).

Transfer learning hook:
  - For Midnight archetypes with <30 observations, fall back to TWW archetype
    model weights scaled by ``transfer_confidence``.
  - This is the core value of the archetype mapping system.

Output: Serialized model artifacts written to ``config.data.processed_dir/../outputs/model_artifacts/``.
"""

from __future__ import annotations

from wow_forecaster.models.meta import RunMetadata
from wow_forecaster.pipeline.base import PipelineStage


class TrainStage(PipelineStage):
    """Train or update forecasting models from engineered feature data.

    Stub: not yet implemented. Raises ``NotImplementedError`` when run.
    """

    stage_name = "train"

    def _execute(
        self,
        run: RunMetadata,
        archetype_ids: list[int] | None = None,
        model_type: str = "stub",
        **kwargs,
    ) -> int:
        """Train models and persist artifacts.

        Args:
            run: In-progress ``RunMetadata``.
            archetype_ids: Archetypes to train models for. If ``None``, all.
            model_type: Model architecture to use.

        Returns:
            Number of models trained/updated.

        Raises:
            NotImplementedError: This stage is not yet implemented.
        """
        raise NotImplementedError(
            "TrainStage is not yet implemented. "
            "Implement model selection, cross-validation, and transfer learning hooks."
        )
