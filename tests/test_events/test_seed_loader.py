"""
Tests for wow_forecaster/events/seed_loader.py.

Covers:
  - Validation: duplicate slugs, end < start, invalid direction/category.
  - Upsert idempotency: running twice produces the same row count.
  - Impact validation: unknown event slug rejected.
  - build_events_table integration: end-to-end in-memory DB + Parquet.
  - Parquet output: correct schema, row count, and field values.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import date
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from wow_forecaster.db.schema import apply_schema
from wow_forecaster.events.seed_loader import (
    _validate_events,
    _validate_impacts,
    build_events_table,
    upsert_category_impacts,
    upsert_events,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def mem_db() -> sqlite3.Connection:
    """In-memory DB with full schema applied."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = OFF;")  # Avoid FK ordering issues in tests
    apply_schema(conn)
    return conn


def _minimal_event(slug: str = "test-event", **overrides) -> dict:
    base = {
        "slug": slug,
        "display_name": "Test Event",
        "event_type": "season_start",
        "scope": "global",
        "severity": "major",
        "expansion_slug": "tww",
        "start_date": "2025-01-10",
        "end_date": "2025-01-20",
        "announced_at": "2025-01-01T00:00:00+00:00",
        "is_recurring": False,
    }
    base.update(overrides)
    return base


def _minimal_impact(event_slug: str = "test-event", **overrides) -> dict:
    base = {
        "event_slug": event_slug,
        "archetype_category": "consumable",
        "impact_direction": "spike",
        "typical_magnitude": 0.25,
        "lag_days": 0,
        "duration_days": 7,
        "source": "seed",
        "notes": "Test impact.",
    }
    base.update(overrides)
    return base


# ── Validation: events ────────────────────────────────────────────────────────

class TestEventValidation:
    def test_valid_events_pass(self):
        records = [_minimal_event("event-a"), _minimal_event("event-b")]
        _validate_events(records)  # Should not raise

    def test_duplicate_slug_rejected(self):
        records = [_minimal_event("dup"), _minimal_event("dup")]
        with pytest.raises(ValueError, match="Duplicate event slug 'dup'"):
            _validate_events(records)

    def test_end_before_start_rejected(self):
        records = [_minimal_event(start_date="2025-01-20", end_date="2025-01-10")]
        with pytest.raises(ValueError, match="end_date.*before start_date"):
            _validate_events(records)

    def test_missing_slug_rejected(self):
        records = [{"display_name": "No Slug", "event_type": "rtwf"}]
        with pytest.raises(ValueError, match="missing 'slug'"):
            _validate_events(records)

    def test_same_start_and_end_valid(self):
        """An instantaneous event (end_date == start_date) is valid."""
        records = [_minimal_event(start_date="2025-01-10", end_date="2025-01-10")]
        _validate_events(records)  # Should not raise


# ── Validation: impacts ───────────────────────────────────────────────────────

class TestImpactValidation:
    def test_valid_impact_passes(self):
        known = {"test-event"}
        records = [_minimal_impact()]
        _validate_impacts(records, known)  # Should not raise

    def test_unknown_event_slug_rejected(self):
        known = {"other-event"}
        records = [_minimal_impact("test-event")]
        with pytest.raises(ValueError, match="unknown event slug 'test-event'"):
            _validate_impacts(records, known)

    def test_invalid_category_rejected(self):
        known = {"test-event"}
        records = [_minimal_impact(archetype_category="not_a_real_category")]
        with pytest.raises(ValueError, match="invalid archetype_category"):
            _validate_impacts(records, known)

    def test_invalid_direction_rejected(self):
        known = {"test-event"}
        records = [_minimal_impact(impact_direction="bullish")]
        with pytest.raises(ValueError, match="invalid impact_direction"):
            _validate_impacts(records, known)

    def test_duplicate_event_category_pair_rejected(self):
        known = {"test-event"}
        records = [_minimal_impact(), _minimal_impact()]  # same slug + category
        with pytest.raises(ValueError, match="Duplicate"):
            _validate_impacts(records, known)

    def test_multiple_categories_same_event_valid(self):
        known = {"test-event"}
        records = [
            _minimal_impact(archetype_category="consumable"),
            _minimal_impact(archetype_category="mat"),
        ]
        _validate_impacts(records, known)  # Should not raise


# ── Upsert ────────────────────────────────────────────────────────────────────

class TestUpsertEvents:
    def test_insert_and_retrieve(self, mem_db):
        records = [_minimal_event("evt-1")]
        count = upsert_events(mem_db, records)
        assert count == 1
        row = mem_db.execute("SELECT * FROM wow_events WHERE slug = 'evt-1'").fetchone()
        assert row is not None
        assert row["severity"] == "major"

    def test_idempotent_upsert(self, mem_db):
        records = [_minimal_event("evt-1")]
        upsert_events(mem_db, records)
        count = upsert_events(mem_db, records)  # Second upsert
        assert count == 1
        rows = mem_db.execute("SELECT COUNT(*) as n FROM wow_events WHERE slug = 'evt-1'").fetchone()
        assert rows["n"] == 1

    def test_upsert_updates_fields(self, mem_db):
        upsert_events(mem_db, [_minimal_event("evt-1", severity="minor")])
        upsert_events(mem_db, [_minimal_event("evt-1", severity="critical")])
        row = mem_db.execute("SELECT severity FROM wow_events WHERE slug = 'evt-1'").fetchone()
        assert row["severity"] == "critical"


class TestUpsertCategoryImpacts:
    def test_insert_impact(self, mem_db):
        upsert_events(mem_db, [_minimal_event("evt-1")])
        records = [_minimal_impact("evt-1")]
        count = upsert_category_impacts(mem_db, records)
        assert count == 1
        row = mem_db.execute(
            "SELECT * FROM event_category_impacts"
        ).fetchone()
        assert row["archetype_category"] == "consumable"
        assert row["impact_direction"] == "spike"

    def test_idempotent_impact_upsert(self, mem_db):
        upsert_events(mem_db, [_minimal_event("evt-1")])
        records = [_minimal_impact("evt-1")]
        upsert_category_impacts(mem_db, records)
        count = upsert_category_impacts(mem_db, records)
        assert count == 1
        n = mem_db.execute("SELECT COUNT(*) as n FROM event_category_impacts").fetchone()["n"]
        assert n == 1

    def test_unknown_slug_is_skipped_not_raised(self, mem_db):
        """Unknown event slug is warned but not fatal during upsert (validated before)."""
        records = [_minimal_impact("nonexistent-slug")]
        count = upsert_category_impacts(mem_db, records)
        assert count == 0


# ── build_events_table integration ────────────────────────────────────────────

class TestBuildEventsTable:
    def test_end_to_end(self, mem_db, tmp_path):
        """Full flow: JSON files -> DB upsert -> Parquet output."""
        events = [_minimal_event("e1"), _minimal_event("e2")]
        impacts = [_minimal_impact("e1"), _minimal_impact("e1", archetype_category="mat")]

        events_file  = tmp_path / "events.json"
        impacts_file = tmp_path / "impacts.json"
        events_file.write_text(json.dumps(events))
        impacts_file.write_text(json.dumps(impacts))

        ev_count, imp_count = build_events_table(
            conn=mem_db,
            events_path=events_file,
            impacts_path=impacts_file,
            output_dir=tmp_path / "out",
        )

        assert ev_count  == 2
        assert imp_count == 2

        # Parquet files written?
        assert (tmp_path / "out" / "events.parquet").exists()
        assert (tmp_path / "out" / "event_category_impacts.parquet").exists()

    def test_events_parquet_schema(self, mem_db, tmp_path):
        """events.parquet must contain all required columns."""
        events = [_minimal_event("schema-test")]
        events_file = tmp_path / "events.json"
        events_file.write_text(json.dumps(events))

        build_events_table(
            conn=mem_db,
            events_path=events_file,
            impacts_path=None,
            output_dir=tmp_path / "out",
        )

        pq_table = pq.read_table(tmp_path / "out" / "events.parquet")
        expected_cols = {
            "event_id", "event_name", "event_type", "scope", "severity",
            "expansion_slug", "start_ts", "end_ts", "announced_ts",
            "source", "metadata",
        }
        assert expected_cols.issubset(set(pq_table.schema.names))

    def test_events_parquet_row_count(self, mem_db, tmp_path):
        events = [_minimal_event(f"evt-{i}") for i in range(5)]
        events_file = tmp_path / "events.json"
        events_file.write_text(json.dumps(events))

        build_events_table(
            conn=mem_db,
            events_path=events_file,
            impacts_path=None,
            output_dir=tmp_path / "out",
        )

        pq_table = pq.read_table(tmp_path / "out" / "events.parquet")
        assert pq_table.num_rows == 5

    def test_events_parquet_start_ts_is_date(self, mem_db, tmp_path):
        """start_ts must be a date32 column containing the correct date."""
        events = [_minimal_event("date-test", start_date="2025-03-11", end_date="2025-03-20")]
        events_file = tmp_path / "events.json"
        events_file.write_text(json.dumps(events))

        build_events_table(
            conn=mem_db,
            events_path=events_file,
            impacts_path=None,
            output_dir=tmp_path / "out",
        )

        pq_table = pq.read_table(tmp_path / "out" / "events.parquet")
        start_ts = pq_table.column("start_ts")[0].as_py()
        assert start_ts == date(2025, 3, 11)

    def test_impacts_not_exported_when_none(self, mem_db, tmp_path):
        """event_category_impacts.parquet should NOT be written when there are no impacts."""
        events = [_minimal_event("no-impacts")]
        events_file = tmp_path / "events.json"
        events_file.write_text(json.dumps(events))

        build_events_table(
            conn=mem_db,
            events_path=events_file,
            impacts_path=None,
            output_dir=tmp_path / "out",
        )

        assert not (tmp_path / "out" / "event_category_impacts.parquet").exists()

    def test_comment_keys_stripped(self, mem_db, tmp_path):
        """JSON entries with only '_comment' keys (no 'slug') are ignored."""
        raw = [
            {"_comment": "This is a comment", "some_key": "value"},
            _minimal_event("real-event"),
        ]
        events_file = tmp_path / "events.json"
        events_file.write_text(json.dumps(raw))

        ev_count, _ = build_events_table(
            conn=mem_db,
            events_path=events_file,
            impacts_path=None,
            output_dir=tmp_path / "out",
        )

        assert ev_count == 1  # Only the real event

    def test_missing_events_file_raises(self, mem_db, tmp_path):
        with pytest.raises(FileNotFoundError):
            build_events_table(
                conn=mem_db,
                events_path=tmp_path / "nonexistent.json",
                impacts_path=None,
                output_dir=tmp_path / "out",
            )
