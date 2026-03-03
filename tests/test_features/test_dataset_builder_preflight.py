"""
Tests for dataset_builder.py preflight checks.

Verifies that build_datasets() raises RuntimeError with a clear message when
the wow_events table is empty (i.e. build-events has not been run first), and
that it proceeds normally when events are present.
"""

from __future__ import annotations

import sqlite3
from datetime import date, datetime, timezone

import pytest

from wow_forecaster.config import AppConfig
from wow_forecaster.db.schema import apply_schema
from wow_forecaster.features.dataset_builder import build_datasets
from wow_forecaster.models.meta import RunMetadata


def _make_run() -> RunMetadata:
    return RunMetadata(
        run_slug="preflight-test",
        pipeline_stage="feature_build",
        config_snapshot={},
        started_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
    )


class TestBuildDatasetsPreflight:
    def test_raises_when_wow_events_is_empty(self, in_memory_db: sqlite3.Connection):
        """build_datasets() must raise RuntimeError when wow_events has no rows."""
        # in_memory_db has full schema but no seed data — wow_events is empty.
        with pytest.raises(RuntimeError, match="build-events"):
            build_datasets(
                conn=in_memory_db,
                config=AppConfig(),
                run=_make_run(),
                realm_slugs=["us"],
                start_date=date(2025, 1, 1),
                end_date=date(2025, 1, 31),
            )

    def test_error_message_mentions_degraded_dataset(self, in_memory_db: sqlite3.Connection):
        """The error message must explain why this matters (not just say 'table empty')."""
        with pytest.raises(RuntimeError, match="degraded"):
            build_datasets(
                conn=in_memory_db,
                config=AppConfig(),
                run=_make_run(),
                realm_slugs=["us"],
                start_date=date(2025, 1, 1),
                end_date=date(2025, 1, 31),
            )

    def test_passes_preflight_when_events_seeded(self, feature_db: sqlite3.Connection):
        """build_datasets() must not raise when wow_events has rows.

        The call will proceed past the preflight and may return 0 rows if the
        normalised data is too thin — that is acceptable.  The absence of
        RuntimeError is the assertion.
        """
        # feature_db seeds 3 events; preflight should pass cleanly.
        try:
            build_datasets(
                conn=feature_db,
                config=AppConfig(),
                run=_make_run(),
                realm_slugs=["area-52"],
                start_date=date(2025, 1, 1),
                end_date=date(2025, 1, 31),
            )
        except RuntimeError as exc:
            if "build-events" in str(exc):
                pytest.fail(f"Preflight check incorrectly fired with seeded events: {exc}")
            # Other RuntimeErrors (e.g. from Parquet write path) are not our concern here.
