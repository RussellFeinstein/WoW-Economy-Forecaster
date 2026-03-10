"""
Tests for governance/pruner.py — SnapshotPruner.

Covers:
  - Dry run on empty filesystem: no files deleted, result is empty
  - Missing snapshot directory is handled gracefully (no crash)
  - Stale files deleted; fresh files within retention window are kept
  - Empty day/month/year directories removed after pruning
  - DB: stale raw rows deleted; fresh rows are kept
  - DB: normalised FK children deleted before raw rows (FK constraint)
  - Dry run: counts reported but nothing deleted
  - PruneResult.__str__ includes key counts and [DRY RUN] label
"""

from __future__ import annotations

import sqlite3
import tempfile
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest

from wow_forecaster.db.schema import apply_schema
from wow_forecaster.governance.pruner import PruneResult, SnapshotPruner


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_pruner(
    tmp_path: Path,
    db_path: str | None = None,
    retention_days: int = 30,
) -> SnapshotPruner:
    if db_path is None:
        db_path = _make_file_db(tmp_path)
    return SnapshotPruner(
        raw_dir=str(tmp_path / "raw"),
        db_path=db_path,
        retention_days=retention_days,
    )


def _make_snapshot_file(raw_dir: Path, obs_date: date, filename: str = "realm_us_test.json") -> Path:
    """Create a dummy snapshot file at raw/snapshots/blizzard_api/YYYY/MM/DD/filename."""
    day_dir = (
        raw_dir
        / "snapshots"
        / "blizzard_api"
        / str(obs_date.year)
        / f"{obs_date.month:02d}"
        / f"{obs_date.day:02d}"
    )
    day_dir.mkdir(parents=True, exist_ok=True)
    p = day_dir / filename
    p.write_text('{"_meta": {}, "data": []}')
    return p


def _make_file_db(tmp_path: Path) -> str:
    """Create a file-based SQLite DB with the full schema applied. Returns path string."""
    db_path = str(tmp_path / "test.db")
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON;")
    apply_schema(conn)
    conn.commit()
    conn.close()
    return db_path


def _insert_raw_row(conn: sqlite3.Connection, item_id: int, observed_at: str) -> int:
    """Insert a row into market_observations_raw; item_id must already exist."""
    cursor = conn.execute(
        """
        INSERT INTO market_observations_raw
            (item_id, realm_slug, faction, observed_at, source, ingested_at, is_processed)
        VALUES (?, 'us', 'neutral', ?, 'blizzard_api', ?, 0)
        """,
        (item_id, observed_at, observed_at),
    )
    conn.commit()
    return cursor.lastrowid


def _insert_item(conn: sqlite3.Connection, item_id: int) -> None:
    """Insert a minimal item_category + item row so FK constraints are satisfied."""
    conn.execute(
        """
        INSERT OR IGNORE INTO item_categories (category_id, slug, display_name, archetype_tag)
        VALUES (1, 'test-materials', 'Test Materials', 'crafting_material')
        """
    )
    conn.execute(
        """
        INSERT OR IGNORE INTO items
            (item_id, name, category_id, expansion_slug, quality)
        VALUES (?, 'Test Item', 1, 'midnight', 'common')
        """,
        (item_id,),
    )
    conn.commit()


def _insert_norm_row(conn: sqlite3.Connection, obs_id: int) -> None:
    """Insert a minimal normalised observation referencing obs_id."""
    conn.execute(
        """
        INSERT INTO market_observations_normalized
            (obs_id, item_id, realm_slug, faction, observed_at, price_gold, is_outlier)
        VALUES (?, 1, 'us', 'neutral', '2024-01-01T00:00:00Z', 1.0, 0)
        """,
        (obs_id,),
    )
    conn.commit()


# ── Tests: file pruning ───────────────────────────────────────────────────────


def test_empty_filesystem_no_crash(tmp_path: Path) -> None:
    pruner = _make_pruner(tmp_path)
    result = pruner.prune(dry_run=False)
    assert result.files_deleted == 0
    assert result.dirs_removed == 0
    assert result.errors == []


def test_missing_snapshot_dir_no_crash(tmp_path: Path) -> None:
    """Snapshot dir doesn't exist yet — prune should be a no-op."""
    pruner = _make_pruner(tmp_path)
    result = pruner.prune(dry_run=True)
    assert result.files_deleted == 0


def test_stale_file_deleted(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    stale_date = date.today() - timedelta(days=45)
    stale_file = _make_snapshot_file(raw_dir, stale_date)

    pruner = _make_pruner(tmp_path, retention_days=30)
    result = pruner.prune(dry_run=False)

    assert result.files_deleted == 1
    assert not stale_file.exists()


def test_fresh_file_not_deleted(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    fresh_date = date.today() - timedelta(days=5)
    fresh_file = _make_snapshot_file(raw_dir, fresh_date)

    pruner = _make_pruner(tmp_path, retention_days=30)
    result = pruner.prune(dry_run=False)

    assert result.files_deleted == 0
    assert fresh_file.exists()


def test_boundary_day_not_deleted(tmp_path: Path) -> None:
    """A file exactly at retention_days boundary (cutoff date itself) is NOT deleted."""
    raw_dir = tmp_path / "raw"
    cutoff_date = date.today() - timedelta(days=30)
    boundary_file = _make_snapshot_file(raw_dir, cutoff_date)

    pruner = _make_pruner(tmp_path, retention_days=30)
    result = pruner.prune(dry_run=False)

    assert result.files_deleted == 0
    assert boundary_file.exists()


def test_multiple_files_some_stale(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    stale = date.today() - timedelta(days=60)
    fresh = date.today() - timedelta(days=10)
    _make_snapshot_file(raw_dir, stale, "realm_us_stale.json")
    fresh_file = _make_snapshot_file(raw_dir, fresh, "realm_us_fresh.json")

    pruner = _make_pruner(tmp_path, retention_days=30)
    result = pruner.prune(dry_run=False)

    assert result.files_deleted == 1
    assert fresh_file.exists()


def test_empty_dirs_removed_after_prune(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    stale_date = date.today() - timedelta(days=45)
    _make_snapshot_file(raw_dir, stale_date)

    stale_day_dir = (
        raw_dir / "snapshots" / "blizzard_api"
        / str(stale_date.year)
        / f"{stale_date.month:02d}"
        / f"{stale_date.day:02d}"
    )
    assert stale_day_dir.exists()

    pruner = _make_pruner(tmp_path, retention_days=30)
    pruner.prune(dry_run=False)

    assert not stale_day_dir.exists()


def test_dry_run_does_not_delete_files(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    stale_date = date.today() - timedelta(days=45)
    stale_file = _make_snapshot_file(raw_dir, stale_date)

    pruner = _make_pruner(tmp_path, retention_days=30)
    result = pruner.prune(dry_run=True)

    assert result.dry_run is True
    assert result.files_deleted == 1   # counted but not deleted
    assert stale_file.exists()         # still present


# ── Tests: DB pruning ─────────────────────────────────────────────────────────


def test_db_stale_raw_rows_deleted(tmp_path: Path) -> None:
    db_path = _make_file_db(tmp_path)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON;")
    _insert_item(conn, 1)
    stale_ts = (datetime.now(tz=timezone.utc) - timedelta(days=40)).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    _insert_raw_row(conn, 1, stale_ts)
    conn.close()

    pruner = _make_pruner(tmp_path, db_path=db_path, retention_days=30)
    result = pruner.prune(dry_run=False)

    assert result.raw_rows_deleted == 1
    conn2 = sqlite3.connect(db_path)
    count = conn2.execute("SELECT COUNT(*) FROM market_observations_raw").fetchone()[0]
    conn2.close()
    assert count == 0


def test_db_fresh_rows_not_deleted(tmp_path: Path) -> None:
    db_path = _make_file_db(tmp_path)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON;")
    _insert_item(conn, 1)
    fresh_ts = (datetime.now(tz=timezone.utc) - timedelta(days=5)).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    _insert_raw_row(conn, 1, fresh_ts)
    conn.close()

    pruner = _make_pruner(tmp_path, db_path=db_path, retention_days=30)
    result = pruner.prune(dry_run=False)

    assert result.raw_rows_deleted == 0
    conn2 = sqlite3.connect(db_path)
    count = conn2.execute("SELECT COUNT(*) FROM market_observations_raw").fetchone()[0]
    conn2.close()
    assert count == 1


def test_db_normalised_rows_deleted_before_raw(tmp_path: Path) -> None:
    """FK child (normalized) must be deleted first; then raw parent can be deleted."""
    db_path = _make_file_db(tmp_path)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON;")
    _insert_item(conn, 1)
    stale_ts = (datetime.now(tz=timezone.utc) - timedelta(days=40)).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    obs_id = _insert_raw_row(conn, 1, stale_ts)
    _insert_norm_row(conn, obs_id)
    conn.close()

    pruner = _make_pruner(tmp_path, db_path=db_path, retention_days=30)
    result = pruner.prune(dry_run=False)

    assert result.raw_rows_deleted == 1
    assert result.norm_rows_deleted == 1

    conn2 = sqlite3.connect(db_path)
    raw_count  = conn2.execute("SELECT COUNT(*) FROM market_observations_raw").fetchone()[0]
    norm_count = conn2.execute("SELECT COUNT(*) FROM market_observations_normalized").fetchone()[0]
    conn2.close()
    assert raw_count  == 0
    assert norm_count == 0


def test_db_dry_run_does_not_delete_rows(tmp_path: Path) -> None:
    db_path = _make_file_db(tmp_path)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON;")
    _insert_item(conn, 1)
    stale_ts = (datetime.now(tz=timezone.utc) - timedelta(days=40)).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    _insert_raw_row(conn, 1, stale_ts)
    conn.close()

    pruner = _make_pruner(tmp_path, db_path=db_path, retention_days=30)
    result = pruner.prune(dry_run=True)

    assert result.raw_rows_deleted == 1  # counted

    conn2 = sqlite3.connect(db_path)
    count = conn2.execute("SELECT COUNT(*) FROM market_observations_raw").fetchone()[0]
    conn2.close()
    assert count == 1  # still present


def test_db_no_stale_rows_no_error(tmp_path: Path) -> None:
    db_path = _make_file_db(tmp_path)
    pruner = _make_pruner(tmp_path, db_path=db_path, retention_days=30)
    result = pruner.prune(dry_run=False)
    assert result.raw_rows_deleted == 0
    assert result.errors == []


# ── Tests: PruneResult ────────────────────────────────────────────────────────


def test_prune_result_str_live() -> None:
    r = PruneResult(
        cutoff_date=date(2026, 2, 8),
        dry_run=False,
        files_deleted=5,
        dirs_removed=2,
        raw_rows_deleted=100,
        norm_rows_deleted=50,
    )
    s = str(r)
    assert "2026-02-08" in s
    assert "files=5" in s
    assert "raw_rows=100" in s
    assert "[DRY RUN]" not in s


def test_prune_result_str_dry_run() -> None:
    r = PruneResult(
        cutoff_date=date(2026, 2, 8),
        dry_run=True,
        files_deleted=3,
        dirs_removed=0,
        raw_rows_deleted=20,
        norm_rows_deleted=10,
    )
    s = str(r)
    assert "[DRY RUN]" in s
    assert "files=3" in s


def test_prune_result_errors_in_str() -> None:
    r = PruneResult(
        cutoff_date=date(2026, 2, 8),
        dry_run=False,
        errors=["something went wrong"],
    )
    assert "errors=1" in str(r)
