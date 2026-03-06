"""
Tests for dataset_builder.py preflight checks and schema invariants.

Verifies that build_datasets() raises RuntimeError with a clear message when
the wow_events table is empty (i.e. build-events has not been run first), and
that it proceeds normally when events are present.

Also verifies that the Parquet schema column counts match the feature registry.
"""

from __future__ import annotations

import sqlite3
from datetime import date, datetime, timezone

import pytest

from wow_forecaster.config import AppConfig
from wow_forecaster.db.schema import apply_schema
from wow_forecaster.features.dataset_builder import (
    _EXPECTED_INFERENCE_COLS,
    _EXPECTED_TRAINING_COLS,
    _add_temporal_features,
    _find_expansion_launch,
    build_datasets,
    build_parquet_schema,
)
from wow_forecaster.features.registry import inference_feature_names, training_feature_names
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


# ── Parquet schema column-count invariants ────────────────────────────────────

class TestParquetSchemaColumnCounts:
    """Guard against accidental FEATURE_REGISTRY changes breaking Parquet output."""

    def test_training_schema_matches_registry(self):
        """Training schema column count must match training_feature_names()."""
        schema = build_parquet_schema(include_targets=True)
        assert len(schema) == len(training_feature_names()), (
            f"Training schema has {len(schema)} columns but registry has "
            f"{len(training_feature_names())}.  Update _EXPECTED_TRAINING_COLS "
            f"and dataset_builder docstrings after an intentional change."
        )

    def test_inference_schema_matches_registry(self):
        """Inference schema column count must match inference_feature_names()."""
        schema = build_parquet_schema(include_targets=False)
        assert len(schema) == len(inference_feature_names()), (
            f"Inference schema has {len(schema)} columns but registry has "
            f"{len(inference_feature_names())}.  Update _EXPECTED_INFERENCE_COLS "
            f"after an intentional change."
        )

    def test_constants_agree_with_registry(self):
        """The module-level sentinels must equal the live registry counts."""
        assert _EXPECTED_TRAINING_COLS == len(training_feature_names())
        assert _EXPECTED_INFERENCE_COLS == len(inference_feature_names())

    def test_training_has_more_cols_than_inference(self):
        """Training set includes target columns; inference set does not."""
        training = build_parquet_schema(include_targets=True)
        inference = build_parquet_schema(include_targets=False)
        assert len(training) > len(inference)


# ── _find_expansion_launch and _add_temporal_features ────────────────────────

class TestFindExpansionLaunch:
    """_find_expansion_launch must return a sorted list of all launch dates."""

    def _make_event(self, event_type: str, start_date: date):
        from types import SimpleNamespace
        from wow_forecaster.taxonomy.event_taxonomy import EventType
        return SimpleNamespace(
            event_type=EventType(event_type),
            start_date=start_date,
        )

    def test_no_expansion_events_returns_empty(self):
        events = [self._make_event("major_patch", date(2025, 1, 1))]
        assert _find_expansion_launch(events) == []

    def test_single_expansion_launch(self):
        events = [self._make_event("expansion_launch", date(2024, 8, 26))]
        assert _find_expansion_launch(events) == [date(2024, 8, 26)]

    def test_two_expansion_launches_sorted(self):
        events = [
            self._make_event("expansion_launch", date(2026, 3, 2)),
            self._make_event("expansion_launch", date(2024, 8, 26)),
            self._make_event("major_patch", date(2025, 2, 25)),
        ]
        result = _find_expansion_launch(events)
        assert result == [date(2024, 8, 26), date(2026, 3, 2)]


class TestAddTemporalFeaturesDaysSinceExpansion:
    """days_since_expansion must anchor to the most recent launch <= obs_date."""

    def _row(self, obs_date: date) -> dict:
        return {"obs_date": obs_date, "other": 99}

    def test_no_launches_produces_none(self):
        rows = [self._row(date(2024, 1, 15))]
        result = _add_temporal_features(rows, [])
        assert result[0]["days_since_expansion"] is None

    def test_single_launch_before_obs(self):
        launch = date(2024, 8, 26)
        obs = date(2024, 9, 5)
        result = _add_temporal_features([self._row(obs)], [launch])
        assert result[0]["days_since_expansion"] == (obs - launch).days  # 10

    def test_obs_before_any_launch_produces_none(self):
        result = _add_temporal_features(
            [self._row(date(2024, 8, 25))],
            [date(2024, 8, 26)],
        )
        assert result[0]["days_since_expansion"] is None

    def test_obs_on_launch_day_is_zero(self):
        launch = date(2024, 8, 26)
        result = _add_temporal_features([self._row(launch)], [launch])
        assert result[0]["days_since_expansion"] == 0

    def test_two_expansions_anchors_to_most_recent(self):
        """After Midnight launches, rows must anchor to Midnight, not TWW."""
        tww = date(2024, 8, 26)
        midnight = date(2026, 3, 2)
        launches = [tww, midnight]

        tww_obs   = date(2025, 6, 1)   # during TWW — should anchor to TWW
        mid_obs   = date(2026, 3, 6)   # after Midnight — should anchor to Midnight

        rows = [self._row(tww_obs), self._row(mid_obs)]
        result = _add_temporal_features(rows, launches)

        by_date = {r["obs_date"]: r for r in result}
        assert by_date[tww_obs]["days_since_expansion"] == (tww_obs - tww).days
        assert by_date[mid_obs]["days_since_expansion"] == (mid_obs - midnight).days  # 4, not 557

    def test_between_expansions_anchors_to_first(self):
        """An obs between TWW and Midnight anchors to TWW."""
        tww = date(2024, 8, 26)
        midnight = date(2026, 3, 2)
        obs = date(2025, 12, 1)
        result = _add_temporal_features([self._row(obs)], [tww, midnight])
        assert result[0]["days_since_expansion"] == (obs - tww).days
