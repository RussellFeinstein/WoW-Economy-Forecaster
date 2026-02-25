"""
Tests for the ingestion persistence layer.

Covers:
  - IngestionSnapshotRepository: insert, get_by_source, get_latest_by_source,
    get_latest_successful_by_source, get_failed, count_by_source, count
  - ingestion_snapshots table present after apply_schema (via in_memory_db)
  - Schema roundtrip: insert → fetch → verify all fields
  - Snapshot file path convention (integration with build_snapshot_path)
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pytest

from wow_forecaster.db.repositories.ingestion_repo import (
    IngestionSnapshot,
    IngestionSnapshotRepository,
)
from wow_forecaster.db.schema import ALL_TABLE_NAMES
from wow_forecaster.ingestion.snapshot import build_snapshot_path


# ── Fixtures ───────────────────────────────────────────────────────────────────

_DT = datetime(2026, 2, 24, 15, 0, 0, tzinfo=timezone.utc)


def _make_snapshot(
    source: str = "undermine",
    success: bool = True,
    run_id: int | None = None,
    record_count: int = 3,
    endpoint: str = "realm_auctions/area-52/neutral",
    snapshot_path: str = "data/raw/snapshots/undermine/2026/02/24/area-52_20260224T150000Z.json",
    content_hash: str | None = "abc123",
    error_message: str | None = None,
    fetched_at: datetime = _DT,
) -> IngestionSnapshot:
    return IngestionSnapshot(
        snapshot_id=None,
        run_id=run_id,
        source=source,
        endpoint=endpoint,
        snapshot_path=snapshot_path,
        content_hash=content_hash,
        record_count=record_count,
        success=success,
        error_message=error_message,
        fetched_at=fetched_at,
    )


# ── Schema presence ────────────────────────────────────────────────────────────

def test_ingestion_snapshots_table_exists(in_memory_db):
    assert "ingestion_snapshots" in ALL_TABLE_NAMES


def test_ingestion_snapshots_in_memory_db(in_memory_db):
    tables = in_memory_db.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='ingestion_snapshots';"
    ).fetchall()
    assert len(tables) == 1


# ── insert ─────────────────────────────────────────────────────────────────────

class TestIngestionSnapshotInsert:
    def test_returns_integer_id(self, in_memory_db):
        repo = IngestionSnapshotRepository(in_memory_db)
        snap_id = repo.insert(_make_snapshot())
        assert isinstance(snap_id, int)
        assert snap_id >= 1

    def test_ids_auto_increment(self, in_memory_db):
        repo = IngestionSnapshotRepository(in_memory_db)
        id1 = repo.insert(_make_snapshot())
        id2 = repo.insert(_make_snapshot())
        assert id2 > id1

    def test_successful_snapshot_stored(self, in_memory_db):
        repo = IngestionSnapshotRepository(in_memory_db)
        snap = _make_snapshot(source="undermine", record_count=42)
        repo.insert(snap)
        fetched = repo.get_latest_by_source("undermine")
        assert fetched is not None
        assert fetched.record_count == 42
        assert fetched.success is True

    def test_failed_snapshot_stored(self, in_memory_db):
        repo = IngestionSnapshotRepository(in_memory_db)
        snap = _make_snapshot(
            source="blizzard_api",
            success=False,
            record_count=0,
            snapshot_path="",
            error_message="Connection timeout",
        )
        repo.insert(snap)
        fetched = repo.get_latest_by_source("blizzard_api")
        assert fetched is not None
        assert fetched.success is False
        assert fetched.error_message == "Connection timeout"

    def test_null_run_id_allowed(self, in_memory_db):
        repo = IngestionSnapshotRepository(in_memory_db)
        snap = _make_snapshot(run_id=None)
        snap_id = repo.insert(snap)
        assert snap_id >= 1

    def test_null_content_hash_allowed(self, in_memory_db):
        repo = IngestionSnapshotRepository(in_memory_db)
        snap = _make_snapshot(content_hash=None, success=False, error_message="err")
        snap_id = repo.insert(snap)
        assert snap_id >= 1


# ── get_by_source ──────────────────────────────────────────────────────────────

class TestGetBySource:
    def test_returns_only_matching_source(self, in_memory_db):
        repo = IngestionSnapshotRepository(in_memory_db)
        repo.insert(_make_snapshot(source="undermine"))
        repo.insert(_make_snapshot(source="blizzard_api"))
        repo.insert(_make_snapshot(source="undermine"))

        results = repo.get_by_source("undermine")
        assert len(results) == 2
        assert all(r.source == "undermine" for r in results)

    def test_empty_source_returns_empty_list(self, in_memory_db):
        repo = IngestionSnapshotRepository(in_memory_db)
        results = repo.get_by_source("nonexistent")
        assert results == []

    def test_respects_limit(self, in_memory_db):
        repo = IngestionSnapshotRepository(in_memory_db)
        for _ in range(5):
            repo.insert(_make_snapshot(source="undermine"))
        results = repo.get_by_source("undermine", limit=3)
        assert len(results) == 3

    def test_ordered_newest_first(self, in_memory_db):
        repo = IngestionSnapshotRepository(in_memory_db)
        dt1 = datetime(2026, 2, 24, 10, 0, 0, tzinfo=timezone.utc)
        dt2 = datetime(2026, 2, 24, 12, 0, 0, tzinfo=timezone.utc)
        repo.insert(_make_snapshot(fetched_at=dt1))
        repo.insert(_make_snapshot(fetched_at=dt2))
        results = repo.get_by_source("undermine")
        assert results[0].fetched_at > results[1].fetched_at


# ── get_latest_by_source ───────────────────────────────────────────────────────

class TestGetLatestBySource:
    def test_returns_most_recent(self, in_memory_db):
        repo = IngestionSnapshotRepository(in_memory_db)
        dt_old = datetime(2026, 2, 24, 10, 0, 0, tzinfo=timezone.utc)
        dt_new = datetime(2026, 2, 24, 15, 0, 0, tzinfo=timezone.utc)
        repo.insert(_make_snapshot(record_count=5, fetched_at=dt_old))
        repo.insert(_make_snapshot(record_count=10, fetched_at=dt_new))
        latest = repo.get_latest_by_source("undermine")
        assert latest is not None
        assert latest.record_count == 10

    def test_returns_none_for_unknown_source(self, in_memory_db):
        repo = IngestionSnapshotRepository(in_memory_db)
        assert repo.get_latest_by_source("unknown") is None


# ── get_latest_successful_by_source ───────────────────────────────────────────

class TestGetLatestSuccessfulBySource:
    def test_skips_failed_snapshots(self, in_memory_db):
        repo = IngestionSnapshotRepository(in_memory_db)
        dt_success = datetime(2026, 2, 24, 10, 0, 0, tzinfo=timezone.utc)
        dt_fail = datetime(2026, 2, 24, 15, 0, 0, tzinfo=timezone.utc)
        repo.insert(_make_snapshot(success=True, record_count=5, fetched_at=dt_success))
        repo.insert(_make_snapshot(success=False, record_count=0, fetched_at=dt_fail,
                                   error_message="err", snapshot_path=""))
        latest = repo.get_latest_successful_by_source("undermine")
        assert latest is not None
        assert latest.success is True
        assert latest.record_count == 5

    def test_returns_none_when_all_failed(self, in_memory_db):
        repo = IngestionSnapshotRepository(in_memory_db)
        repo.insert(_make_snapshot(success=False, record_count=0,
                                   error_message="err", snapshot_path=""))
        assert repo.get_latest_successful_by_source("undermine") is None


# ── get_failed ─────────────────────────────────────────────────────────────────

class TestGetFailed:
    def test_returns_only_failures(self, in_memory_db):
        repo = IngestionSnapshotRepository(in_memory_db)
        repo.insert(_make_snapshot(success=True))
        repo.insert(_make_snapshot(success=False, record_count=0,
                                   error_message="err1", snapshot_path=""))
        repo.insert(_make_snapshot(success=False, record_count=0,
                                   error_message="err2", snapshot_path="",
                                   source="blizzard_api"))
        failed = repo.get_failed()
        assert len(failed) == 2
        assert all(not f.success for f in failed)

    def test_empty_when_none_failed(self, in_memory_db):
        repo = IngestionSnapshotRepository(in_memory_db)
        repo.insert(_make_snapshot(success=True))
        assert repo.get_failed() == []


# ── count ──────────────────────────────────────────────────────────────────────

class TestCount:
    def test_count_by_source(self, in_memory_db):
        repo = IngestionSnapshotRepository(in_memory_db)
        repo.insert(_make_snapshot(source="undermine"))
        repo.insert(_make_snapshot(source="undermine"))
        repo.insert(_make_snapshot(source="blizzard_api"))
        assert repo.count_by_source("undermine") == 2
        assert repo.count_by_source("blizzard_api") == 1
        assert repo.count_by_source("blizzard_news") == 0

    def test_total_count(self, in_memory_db):
        repo = IngestionSnapshotRepository(in_memory_db)
        assert repo.count() == 0
        repo.insert(_make_snapshot())
        repo.insert(_make_snapshot(source="blizzard_api"))
        assert repo.count() == 2


# ── Snapshot path convention ───────────────────────────────────────────────────

class TestSnapshotPathConvention:
    """Verify the build_snapshot_path convention matches what ingestion uses."""

    def test_undermine_path_structure(self):
        dt = datetime(2026, 2, 24, 15, 30, 0, tzinfo=timezone.utc)
        path = build_snapshot_path("data/raw", "undermine", "area-52_neutral", dt)
        assert path.parent.name == "24"           # day
        assert path.parent.parent.name == "02"    # month
        assert path.parent.parent.parent.name == "2026"  # year
        assert "area-52_neutral" in path.name
        assert path.suffix == ".json"

    def test_blizzard_api_path_structure(self):
        dt = datetime(2026, 2, 24, 15, 30, 0, tzinfo=timezone.utc)
        path = build_snapshot_path("data/raw", "blizzard_api", "realm_area-52", dt)
        assert "blizzard_api" in str(path)
        assert path.suffix == ".json"

    def test_blizzard_news_path_structure(self):
        dt = datetime(2026, 2, 24, 15, 30, 0, tzinfo=timezone.utc)
        path = build_snapshot_path("data/raw", "blizzard_news", "news", dt)
        assert "blizzard_news" in str(path)

    def test_all_three_sources_have_unique_paths(self):
        dt = datetime(2026, 2, 24, 15, 0, 0, tzinfo=timezone.utc)
        p1 = build_snapshot_path("data/raw", "undermine", "area-52_neutral", dt)
        p2 = build_snapshot_path("data/raw", "blizzard_api", "realm_area-52", dt)
        p3 = build_snapshot_path("data/raw", "blizzard_news", "news", dt)
        assert p1 != p2
        assert p2 != p3
        assert p1 != p3
