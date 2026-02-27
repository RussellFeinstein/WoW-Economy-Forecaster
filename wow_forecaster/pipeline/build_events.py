"""
BuildEventsStage: seed events and category impacts → SQLite + Parquet.

This stage is an **upstream dependency** of the feature build pipeline:

    build-events -> build-datasets -> train-model -> run-daily-forecast

It must be run (at least once) before ``build-datasets`` can produce non-null
event feature columns.

Execution sequence
------------------
1. Locate seed files from config (default: ``config/events/``).
2. Validate and upsert events into ``wow_events`` table.
3. Validate and upsert category impacts into ``event_category_impacts`` table.
4. Export both tables to Parquet under ``data/processed/events/``.
5. Update RunMetadata with counts.
"""

from __future__ import annotations

import logging
from pathlib import Path

from wow_forecaster.config import AppConfig
from wow_forecaster.events.seed_loader import build_events_table
from wow_forecaster.models.meta import RunMetadata
from wow_forecaster.pipeline.base import PipelineStage

log = logging.getLogger(__name__)

# Default paths relative to the project root.
_DEFAULT_EVENTS_FILE  = Path("config/events/tww_events.json")
_DEFAULT_IMPACTS_FILE = Path("config/events/tww_event_impacts.json")
_DEFAULT_OUTPUT_DIR   = Path("data/processed/events")


class BuildEventsStage(PipelineStage):
    """Pipeline stage that seeds the events and category-impacts tables.

    Attributes:
        events_path:  Path to the events JSON seed file.
        impacts_path: Path to the category impacts JSON seed file (optional).
        output_dir:   Directory to write ``events.parquet`` and
                      ``event_category_impacts.parquet``.
    """

    stage_name = "build_events"

    def __init__(
        self,
        events_path: Path = _DEFAULT_EVENTS_FILE,
        impacts_path: Path | None = _DEFAULT_IMPACTS_FILE,
        output_dir: Path = _DEFAULT_OUTPUT_DIR,
    ) -> None:
        self.events_path  = events_path
        self.impacts_path = impacts_path
        self.output_dir   = output_dir

    def _execute(
        self,
        conn,
        config: AppConfig,
        run: RunMetadata,
    ) -> int:
        """Load seed files, upsert to DB, export Parquet.

        Returns:
            Total records written (events + impacts combined).
        """
        events_path  = self.events_path
        impacts_path = self.impacts_path

        if not events_path.exists():
            raise FileNotFoundError(
                f"Events seed file not found: {events_path}. "
                "Run from the project root or pass an explicit path."
            )

        if impacts_path and not impacts_path.exists():
            log.warning(
                "Impacts seed file not found: %s — skipping impact seeding.",
                impacts_path,
            )
            impacts_path = None

        events_count, impacts_count = build_events_table(
            conn=conn,
            events_path=events_path,
            impacts_path=impacts_path,
            output_dir=self.output_dir,
        )

        total = events_count + impacts_count
        log.info(
            "BuildEventsStage complete: %d events, %d category impacts upserted.",
            events_count, impacts_count,
        )
        return total
