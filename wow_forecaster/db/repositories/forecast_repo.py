"""
Repositories for forecast outputs, recommendation outputs, and run metadata.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import date, datetime
from typing import Any, Optional

from wow_forecaster.db.repositories.base import BaseRepository
from wow_forecaster.models.forecast import ForecastOutput, RecommendationOutput
from wow_forecaster.models.meta import ModelMetadata, RunMetadata

logger = logging.getLogger(__name__)


class ForecastOutputRepository(BaseRepository):
    """Read/write access to ``forecast_outputs`` and ``recommendation_outputs``."""

    def insert_forecast(self, forecast: ForecastOutput) -> int:
        """Insert a forecast output and return its ``forecast_id``.

        Args:
            forecast: The ``ForecastOutput`` to persist.

        Returns:
            The newly assigned ``forecast_id``.
        """
        self.execute(
            """
            INSERT INTO forecast_outputs (
                run_id, archetype_id, item_id, realm_slug,
                forecast_horizon, target_date, predicted_price_gold,
                confidence_lower, confidence_upper, confidence_pct,
                model_slug, features_hash
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            (
                forecast.run_id,
                forecast.archetype_id,
                forecast.item_id,
                forecast.realm_slug,
                forecast.forecast_horizon,
                forecast.target_date.isoformat(),
                forecast.predicted_price_gold,
                forecast.confidence_lower,
                forecast.confidence_upper,
                forecast.confidence_pct,
                forecast.model_slug,
                forecast.features_hash,
            ),
        )
        return self.last_insert_rowid()

    def insert_recommendation(self, rec: RecommendationOutput) -> int:
        """Insert a recommendation output and return its ``rec_id``.

        Args:
            rec: The ``RecommendationOutput`` to persist.

        Returns:
            The newly assigned ``rec_id``.
        """
        self.execute(
            """
            INSERT INTO recommendation_outputs (
                forecast_id, action, reasoning, priority, expires_at
            ) VALUES (?, ?, ?, ?, ?);
            """,
            (
                rec.forecast_id,
                rec.action,
                rec.reasoning,
                rec.priority,
                rec.expires_at.isoformat() if rec.expires_at else None,
            ),
        )
        return self.last_insert_rowid()

    def get_forecasts_for_archetype(
        self,
        archetype_id: int,
        target_date: Optional[date] = None,
    ) -> list[ForecastOutput]:
        """Fetch forecasts for an archetype, optionally filtered by target date.

        Args:
            archetype_id: Archetype FK.
            target_date: If provided, filter to this specific target date.

        Returns:
            List of ``ForecastOutput`` objects.
        """
        if target_date:
            rows = self.fetchall(
                """
                SELECT * FROM forecast_outputs
                WHERE archetype_id = ? AND target_date = ?
                ORDER BY created_at DESC;
                """,
                (archetype_id, target_date.isoformat()),
            )
        else:
            rows = self.fetchall(
                """
                SELECT * FROM forecast_outputs
                WHERE archetype_id = ?
                ORDER BY target_date DESC;
                """,
                (archetype_id,),
            )
        return [_row_to_forecast(r) for r in rows]


class RunMetadataRepository(BaseRepository):
    """Read/write access to ``run_metadata`` and ``model_metadata``."""

    def insert_run(self, run: RunMetadata) -> int:
        """Insert a new run metadata record and return its ``run_id``.

        Args:
            run: The ``RunMetadata`` to persist.

        Returns:
            The newly assigned ``run_id``.
        """
        self.execute(
            """
            INSERT INTO run_metadata (
                run_slug, pipeline_stage, status, model_id,
                realm_slug, expansion_slug, config_snapshot,
                rows_processed, error_message, started_at, finished_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            (
                run.run_slug,
                run.pipeline_stage,
                run.status,
                run.model_id,
                run.realm_slug,
                run.expansion_slug,
                json.dumps(run.config_snapshot),
                run.rows_processed,
                run.error_message,
                run.started_at.isoformat(),
                run.finished_at.isoformat() if run.finished_at else None,
            ),
        )
        return self.last_insert_rowid()

    def update_run(self, run: RunMetadata) -> None:
        """Update the mutable fields of an existing run record.

        Args:
            run: The ``RunMetadata`` with updated fields. ``run.run_id`` must be set.

        Raises:
            ValueError: If ``run.run_id`` is ``None``.
        """
        if run.run_id is None:
            raise ValueError("Cannot update RunMetadata without a run_id.")
        self.execute(
            """
            UPDATE run_metadata SET
                status         = ?,
                rows_processed = ?,
                error_message  = ?,
                finished_at    = ?
            WHERE run_id = ?;
            """,
            (
                run.status,
                run.rows_processed,
                run.error_message,
                run.finished_at.isoformat() if run.finished_at else None,
                run.run_id,
            ),
        )

    def get_run_by_slug(self, run_slug: str) -> Optional[RunMetadata]:
        """Fetch a run by its UUID slug.

        Args:
            run_slug: The unique run slug.

        Returns:
            ``RunMetadata`` or ``None``.
        """
        row = self.fetchone(
            "SELECT * FROM run_metadata WHERE run_slug = ?;", (run_slug,)
        )
        return _row_to_run(row) if row else None

    def get_recent_runs(self, pipeline_stage: Optional[str] = None, limit: int = 20) -> list[RunMetadata]:
        """Fetch recent run records, optionally filtered by stage.

        Args:
            pipeline_stage: If provided, filter to this stage.
            limit: Maximum rows to return.

        Returns:
            List of ``RunMetadata``, most recent first.
        """
        if pipeline_stage:
            rows = self.fetchall(
                """
                SELECT * FROM run_metadata
                WHERE pipeline_stage = ?
                ORDER BY started_at DESC LIMIT ?;
                """,
                (pipeline_stage, limit),
            )
        else:
            rows = self.fetchall(
                "SELECT * FROM run_metadata ORDER BY started_at DESC LIMIT ?;",
                (limit,),
            )
        return [_row_to_run(r) for r in rows]

    def insert_model(self, model: ModelMetadata) -> int:
        """Insert a model metadata record and return its ``model_id``.

        Args:
            model: The ``ModelMetadata`` to persist.

        Returns:
            The newly assigned ``model_id``.
        """
        self.execute(
            """
            INSERT INTO model_metadata (
                slug, display_name, model_type, version, hyperparameters,
                training_data_start, training_data_end, validation_mae,
                validation_rmse, artifact_path, is_active
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            (
                model.slug,
                model.display_name,
                model.model_type,
                model.version,
                json.dumps(model.hyperparameters) if model.hyperparameters else None,
                model.training_data_start.isoformat() if model.training_data_start else None,
                model.training_data_end.isoformat() if model.training_data_end else None,
                model.validation_mae,
                model.validation_rmse,
                model.artifact_path,
                int(model.is_active),
            ),
        )
        return self.last_insert_rowid()


# ── Private helpers ────────────────────────────────────────────────────────────

def _row_to_forecast(row: sqlite3.Row) -> ForecastOutput:
    return ForecastOutput(
        forecast_id=row["forecast_id"],
        run_id=row["run_id"],
        archetype_id=row["archetype_id"],
        item_id=row["item_id"],
        realm_slug=row["realm_slug"],
        forecast_horizon=row["forecast_horizon"],
        target_date=date.fromisoformat(row["target_date"]),
        predicted_price_gold=row["predicted_price_gold"],
        confidence_lower=row["confidence_lower"],
        confidence_upper=row["confidence_upper"],
        confidence_pct=row["confidence_pct"],
        model_slug=row["model_slug"],
        features_hash=row["features_hash"],
    )


def _row_to_run(row: sqlite3.Row) -> RunMetadata:
    return RunMetadata(
        run_id=row["run_id"],
        run_slug=row["run_slug"],
        pipeline_stage=row["pipeline_stage"],
        status=row["status"],
        model_id=row["model_id"],
        realm_slug=row["realm_slug"],
        expansion_slug=row["expansion_slug"],
        config_snapshot=json.loads(row["config_snapshot"]),
        rows_processed=row["rows_processed"],
        error_message=row["error_message"],
        started_at=datetime.fromisoformat(row["started_at"]),
        finished_at=(
            datetime.fromisoformat(row["finished_at"]) if row["finished_at"] else None
        ),
    )
