"""
Model and run metadata — the reproducibility backbone.

``ModelMetadata`` describes a trained forecasting model: what type it is,
what data it was trained on, and its validation performance metrics.

``RunMetadata`` is the pipeline execution audit log. Every pipeline run
records a complete ``config_snapshot`` (full AppConfig as a dict) so any
run can be exactly reproduced by restoring that config and re-running.

``RunMetadata`` is the **only** Pydantic model in the system that is NOT
frozen — its ``status``, ``rows_processed``, ``error_message``, and
``finished_at`` fields must be updated as the pipeline stage executes.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, field_validator

VALID_PIPELINE_STAGES = frozenset({
    "ingest", "normalize", "feature_build", "train", "forecast", "recommend",
    "backtest", "orchestrator", "drift_check",
})
VALID_RUN_STATUSES = frozenset({"started", "success", "failed", "skipped"})
VALID_MODEL_TYPES = frozenset({
    "stub", "linear", "prophet", "xgboost", "lightgbm", "lstm", "ensemble",
})


class ModelMetadata(BaseModel):
    """Describes a trained (or registered stub) forecasting model.

    Attributes:
        model_id: Auto-assigned DB PK; ``None`` before insertion.
        slug: Unique model identifier, e.g. ``"stub_linear_v0"``.
        display_name: Human-readable model name.
        model_type: Architecture type; one of the ``VALID_MODEL_TYPES``.
        version: Semantic version string, e.g. ``"0.1.0"``.
        hyperparameters: Dict of hyperparameter name → value.
        training_data_start: Start of training data window (UTC).
        training_data_end: End of training data window (UTC).
        validation_mae: Mean absolute error on hold-out validation set.
        validation_rmse: Root mean squared error on hold-out validation set.
        artifact_path: File system path to the serialized model object.
        is_active: Whether this model is the currently active production model.
    """

    model_config = ConfigDict(frozen=True)

    model_id: Optional[int] = None
    slug: str
    display_name: str
    model_type: str
    version: str = "0.1.0"
    hyperparameters: Optional[dict[str, Any]] = None
    training_data_start: Optional[datetime] = None
    training_data_end: Optional[datetime] = None
    validation_mae: Optional[float] = None
    validation_rmse: Optional[float] = None
    artifact_path: Optional[str] = None
    is_active: bool = False

    @field_validator("model_type")
    @classmethod
    def validate_model_type(cls, v: str) -> str:
        if v not in VALID_MODEL_TYPES:
            raise ValueError(
                f"Unknown model_type '{v}'. Must be one of {sorted(VALID_MODEL_TYPES)}."
            )
        return v


class RunMetadata(BaseModel):
    """Pipeline execution audit record.

    Mutable by design: ``status``, ``rows_processed``, ``error_message``,
    and ``finished_at`` are updated as the pipeline stage progresses.

    The ``config_snapshot`` field stores a complete serialization of the
    ``AppConfig`` used for this run, enabling exact reproduction.

    Attributes:
        run_id: Auto-assigned DB PK; ``None`` before insertion.
        run_slug: UUID4 string uniquely identifying this run.
        pipeline_stage: Which stage produced this run record.
        status: Current execution status.
        model_id: FK to ``model_metadata.model_id`` if applicable.
        realm_slug: Realm processed in this run, or ``None`` for global runs.
        expansion_slug: Expansion context for this run.
        config_snapshot: Full ``AppConfig.model_dump()`` at run start time.
        rows_processed: Running count of records processed (updated during run).
        error_message: Error description if ``status == "failed"``.
        started_at: UTC datetime when the run began.
        finished_at: UTC datetime when the run completed or failed.
    """

    # Not frozen — status, rows_processed, etc. are updated during execution
    model_config = ConfigDict(frozen=False)

    run_id: Optional[int] = None
    run_slug: str
    pipeline_stage: str
    status: str = "started"
    model_id: Optional[int] = None
    realm_slug: Optional[str] = None
    expansion_slug: Optional[str] = None
    config_snapshot: dict[str, Any]
    rows_processed: int = 0
    error_message: Optional[str] = None
    started_at: datetime
    finished_at: Optional[datetime] = None

    @field_validator("pipeline_stage")
    @classmethod
    def validate_pipeline_stage(cls, v: str) -> str:
        if v not in VALID_PIPELINE_STAGES:
            raise ValueError(
                f"Unknown pipeline_stage '{v}'. Must be one of {sorted(VALID_PIPELINE_STAGES)}."
            )
        return v

    @field_validator("status")
    @classmethod
    def validate_status(cls, v: str) -> str:
        if v not in VALID_RUN_STATUSES:
            raise ValueError(
                f"Unknown status '{v}'. Must be one of {sorted(VALID_RUN_STATUSES)}."
            )
        return v
