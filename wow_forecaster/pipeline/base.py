"""
Abstract base class for all pipeline stages.

Every stage follows the same contract:
  1. Receive ``AppConfig`` at construction.
  2. ``run(**kwargs)`` is the sole public API.
  3. ``run()`` creates a ``RunMetadata`` record, calls ``_execute()``,
     and persists the run record with final status.
  4. ``_execute()`` is the stage-specific implementation (overridden by subclasses).

This design ensures:
  - Every run is auditable (run_metadata written to DB).
  - Status transitions (started → success/failed) are consistent.
  - Error handling is centralized — stages never swallow exceptions.
  - Config is always available to every stage.

Usage::

    class MyStage(PipelineStage):
        stage_name = "ingest"

        def _execute(self, run: RunMetadata, **kwargs) -> int:
            # Do work, return row count
            return 42

    stage = MyStage(config=app_config)
    result = stage.run(source_path="data/raw/snapshot.json")
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from datetime import timezone
from uuid import uuid4

from wow_forecaster.config import AppConfig
from wow_forecaster.models.meta import RunMetadata
from wow_forecaster.utils.time_utils import utcnow

logger = logging.getLogger(__name__)


class PipelineStage(ABC):
    """Abstract base for all pipeline stages.

    Subclasses must:
      1. Set ``stage_name`` class variable.
      2. Implement ``_execute(run, **kwargs) -> int``.

    Attributes:
        stage_name: String identifier matching a valid ``RunMetadata.pipeline_stage``.
        config: The application configuration for this run.
        db_path: Path to the SQLite database (defaults to ``config.database.db_path``).
    """

    stage_name: str  # Override in subclass

    def __init__(
        self,
        config: AppConfig,
        db_path: str | None = None,
    ) -> None:
        self.config = config
        self.db_path = db_path or config.database.db_path

    def run(self, **kwargs) -> RunMetadata:
        """Execute this pipeline stage.

        Creates a ``RunMetadata`` record, calls ``_execute()``, and returns
        the finalized run record with ``status='success'`` or ``status='failed'``.

        The run record is persisted to the database via ``_persist_run()``.

        Args:
            **kwargs: Stage-specific keyword arguments passed to ``_execute()``.

        Returns:
            ``RunMetadata`` with final ``status``, ``rows_processed``,
            and ``finished_at`` set.

        Raises:
            Exception: Re-raises any exception from ``_execute()`` after
                recording ``status='failed'`` in the run record.
        """
        run = RunMetadata(
            run_slug=str(uuid4()),
            pipeline_stage=self.stage_name,
            config_snapshot=self.config.model_dump(),
            started_at=utcnow(),
        )
        logger.info(
            "Stage [%s] starting | run_slug=%s", self.stage_name, run.run_slug
        )

        try:
            rows = self._execute(run=run, **kwargs)
            run.status = "success"
            run.rows_processed = rows
            run.finished_at = utcnow()
            logger.info(
                "Stage [%s] completed | rows=%d | run_slug=%s",
                self.stage_name, rows, run.run_slug,
            )

        except NotImplementedError:
            run.status = "failed"
            run.error_message = f"{self.__class__.__name__}._execute() is not yet implemented (stub)."
            run.finished_at = utcnow()
            logger.warning(
                "Stage [%s] is a stub | run_slug=%s", self.stage_name, run.run_slug
            )
            self._persist_run(run)
            raise

        except Exception as exc:
            run.status = "failed"
            run.error_message = str(exc)
            run.finished_at = utcnow()
            logger.error(
                "Stage [%s] FAILED: %s | run_slug=%s",
                self.stage_name, exc, run.run_slug,
            )
            self._persist_run(run)
            raise

        self._persist_run(run)
        return run

    @abstractmethod
    def _execute(self, run: RunMetadata, **kwargs) -> int:
        """Stage-specific implementation.

        Args:
            run: The in-progress ``RunMetadata`` record (mutable).
            **kwargs: Stage-specific parameters.

        Returns:
            Integer count of rows/records processed.

        Raises:
            NotImplementedError: For stub implementations.
        """
        ...

    def _persist_run(self, run: RunMetadata) -> None:
        """Persist the ``RunMetadata`` record to the database.

        Uses an import inside the method to avoid circular dependencies.
        Silently logs errors rather than raising — run persistence failure
        should not mask the original pipeline error.

        Args:
            run: The ``RunMetadata`` to write or update.
        """
        try:
            from wow_forecaster.db.connection import get_connection
            from wow_forecaster.db.repositories.forecast_repo import RunMetadataRepository

            with get_connection(self.db_path) as conn:
                repo = RunMetadataRepository(conn)
                if run.run_id is None:
                    run.run_id = repo.insert_run(run)
                else:
                    repo.update_run(run)
        except Exception as exc:
            logger.error(
                "Failed to persist RunMetadata for run_slug=%s: %s",
                run.run_slug, exc,
            )
