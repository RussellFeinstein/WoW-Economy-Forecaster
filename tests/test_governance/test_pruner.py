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
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

from wow_forecaster.db.schema import apply_schema
from wow_forecaster.governance.pruner import (
    MAX_SLICES_PER_RUN,
    PruneResult,
    SnapshotPruner,
    hour_slices,
)

# Injected into prune() so file-fixture dates and the pruner cutoff share one
# clock; fixtures built with local date.today() flaked whenever the local date
# and the pruner's UTC date disagreed.
FIXED_NOW = datetime(2026, 3, 15, 12, 0, tzinfo=UTC)

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


def _make_snapshot_file(
    raw_dir: Path, obs_date: date, filename: str = "realm_us_test.json"
) -> Path:
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
    stale_date = FIXED_NOW.date() - timedelta(days=45)
    stale_file = _make_snapshot_file(raw_dir, stale_date)

    pruner = _make_pruner(tmp_path, retention_days=30)
    result = pruner.prune(dry_run=False, now=FIXED_NOW)

    assert result.files_deleted == 1
    assert not stale_file.exists()


def test_fresh_file_not_deleted(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    fresh_date = FIXED_NOW.date() - timedelta(days=5)
    fresh_file = _make_snapshot_file(raw_dir, fresh_date)

    pruner = _make_pruner(tmp_path, retention_days=30)
    result = pruner.prune(dry_run=False, now=FIXED_NOW)

    assert result.files_deleted == 0
    assert fresh_file.exists()


def test_boundary_day_not_deleted(tmp_path: Path) -> None:
    """A file dated exactly retention_days before the reference clock is kept.

    Regression test for the time-of-day flake: the fixture date and the
    pruner cutoff must come from the same injected clock, not from local
    date.today() on one side and UTC on the other.
    """
    raw_dir = tmp_path / "raw"
    cutoff_date = FIXED_NOW.date() - timedelta(days=30)
    boundary_file = _make_snapshot_file(raw_dir, cutoff_date)

    pruner = _make_pruner(tmp_path, retention_days=30)
    result = pruner.prune(dry_run=False, now=FIXED_NOW)

    assert result.files_deleted == 0
    assert boundary_file.exists()


def test_day_past_boundary_deleted(tmp_path: Path) -> None:
    """A file dated one day past the retention boundary is deleted."""
    raw_dir = tmp_path / "raw"
    past_boundary = FIXED_NOW.date() - timedelta(days=31)
    stale_file = _make_snapshot_file(raw_dir, past_boundary)

    pruner = _make_pruner(tmp_path, retention_days=30)
    result = pruner.prune(dry_run=False, now=FIXED_NOW)

    assert result.files_deleted == 1
    assert not stale_file.exists()


def test_multiple_files_some_stale(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    stale = FIXED_NOW.date() - timedelta(days=60)
    fresh = FIXED_NOW.date() - timedelta(days=10)
    _make_snapshot_file(raw_dir, stale, "realm_us_stale.json")
    fresh_file = _make_snapshot_file(raw_dir, fresh, "realm_us_fresh.json")

    pruner = _make_pruner(tmp_path, retention_days=30)
    result = pruner.prune(dry_run=False, now=FIXED_NOW)

    assert result.files_deleted == 1
    assert fresh_file.exists()


def test_empty_dirs_removed_after_prune(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    stale_date = FIXED_NOW.date() - timedelta(days=45)
    _make_snapshot_file(raw_dir, stale_date)

    stale_day_dir = (
        raw_dir / "snapshots" / "blizzard_api"
        / str(stale_date.year)
        / f"{stale_date.month:02d}"
        / f"{stale_date.day:02d}"
    )
    assert stale_day_dir.exists()

    pruner = _make_pruner(tmp_path, retention_days=30)
    pruner.prune(dry_run=False, now=FIXED_NOW)

    assert not stale_day_dir.exists()


def test_dry_run_does_not_delete_files(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    stale_date = FIXED_NOW.date() - timedelta(days=45)
    stale_file = _make_snapshot_file(raw_dir, stale_date)

    pruner = _make_pruner(tmp_path, retention_days=30)
    result = pruner.prune(dry_run=True, now=FIXED_NOW)

    assert result.dry_run is True
    assert result.files_deleted == 1   # counted but not deleted
    assert stale_file.exists()         # still present


# ── Tests: DB pruning ─────────────────────────────────────────────────────────


def test_db_stale_raw_rows_deleted(tmp_path: Path) -> None:
    db_path = _make_file_db(tmp_path)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON;")
    _insert_item(conn, 1)
    stale_ts = (datetime.now(tz=UTC) - timedelta(days=40)).strftime(
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
    fresh_ts = (datetime.now(tz=UTC) - timedelta(days=5)).strftime(
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
    stale_ts = (datetime.now(tz=UTC) - timedelta(days=40)).strftime(
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
    stale_ts = (datetime.now(tz=UTC) - timedelta(days=40)).strftime(
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


# ── Tests: hour slicing (issue #149) ──────────────────────────────────────────
#
# The prune deletes in half-open hour slices with a commit per slice, so an
# interrupted run keeps the slices it finished instead of discarding the lot.
# hour_slices is pure over strings, so the boundary rules are testable without
# a database.


def test_hour_slices_covers_oldest_row_from_its_floored_hour() -> None:
    slices = hour_slices("2026-07-21T02:43:49.649565+00:00", "2026-07-21T05:00:00")
    assert slices[0][0] == "2026-07-21T02:00:00"


def test_hour_slices_are_half_open_and_contiguous() -> None:
    slices = hour_slices("2026-07-21T02:00:00", "2026-07-21T05:00:00")
    assert slices == [
        ("2026-07-21T02:00:00", "2026-07-21T03:00:00"),
        ("2026-07-21T03:00:00", "2026-07-21T04:00:00"),
        ("2026-07-21T04:00:00", "2026-07-21T05:00:00"),
    ]


def test_hour_slices_final_end_is_clamped_to_the_cutoff() -> None:
    """The union of the slices must equal the target set exactly.

    The cutoff is a bare calendar date, so the last hour boundary overshoots
    it and has to be clamped or the final partial hour is never pruned.
    """
    slices = hour_slices("2026-07-22T22:10:00Z", "2026-07-23")
    assert slices[-1][1] == "2026-07-23"
    assert slices[-1][0] == "2026-07-22T23:00:00"


def test_hour_slices_parses_both_stored_timestamp_formats() -> None:
    """Production rows carry '+00:00' with microseconds, fixtures carry 'Z'."""
    a = hour_slices("2026-07-21T02:43:49.649565+00:00", "2026-07-21T04:00:00")
    b = hour_slices("2026-07-21T02:43:49Z", "2026-07-21T04:00:00")
    assert a == b


def test_hour_slices_empty_when_nothing_precedes_cutoff() -> None:
    assert hour_slices("2026-07-23T00:00:00", "2026-07-23") == []


def test_hour_slices_capped_and_resumable() -> None:
    """One ancient row must not generate an unbounded slice list.

    The cap is not a silent truncation: the run deletes what it covers, the
    oldest row moves forward, and the next run continues from there.
    """
    slices = hour_slices("1970-01-01T00:00:00Z", "2026-07-23")
    assert len(slices) == MAX_SLICES_PER_RUN
    assert slices[0][0] == "1970-01-01T00:00:00"


# ── Tests: batched DB pruning (issue #149) ────────────────────────────────────


def _stale_ts(days: int, hour: int, minute: int = 30) -> str:
    """A timestamp `days` before FIXED_NOW at a given hour, in stored format."""
    d = (FIXED_NOW - timedelta(days=days)).date()
    return f"{d.isoformat()}T{hour:02d}:{minute:02d}:00Z"


def test_db_rows_across_many_hours_all_deleted(tmp_path: Path) -> None:
    """Rows spread over several hours are all pruned, not just the first hour."""
    db_path = _make_file_db(tmp_path)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON;")
    _insert_item(conn, 1)
    for hour in (1, 2, 5, 9, 23):
        _insert_raw_row(conn, 1, _stale_ts(40, hour))
    conn.close()

    pruner = _make_pruner(tmp_path, db_path=db_path, retention_days=30)
    result = pruner.prune(dry_run=False, now=FIXED_NOW)

    assert result.raw_rows_deleted == 5
    conn2 = sqlite3.connect(db_path)
    remaining = conn2.execute(
        "SELECT COUNT(*) FROM market_observations_raw"
    ).fetchone()[0]
    conn2.close()
    assert remaining == 0


def test_db_norm_children_deleted_across_slices(tmp_path: Path) -> None:
    """FK children in different hour slices are all removed with their parents."""
    db_path = _make_file_db(tmp_path)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON;")
    _insert_item(conn, 1)
    for hour in (3, 7, 14):
        obs_id = _insert_raw_row(conn, 1, _stale_ts(40, hour))
        _insert_norm_row(conn, obs_id)
    conn.close()

    pruner = _make_pruner(tmp_path, db_path=db_path, retention_days=30)
    result = pruner.prune(dry_run=False, now=FIXED_NOW)

    assert result.raw_rows_deleted == 3
    assert result.norm_rows_deleted == 3
    conn2 = sqlite3.connect(db_path)
    raw_left = conn2.execute(
        "SELECT COUNT(*) FROM market_observations_raw"
    ).fetchone()[0]
    norm_left = conn2.execute(
        "SELECT COUNT(*) FROM market_observations_normalized"
    ).fetchone()[0]
    conn2.close()
    assert raw_left == 0
    assert norm_left == 0


def test_db_fresh_rows_survive_a_batched_prune(tmp_path: Path) -> None:
    """Slicing must not walk past the cutoff into rows inside the window."""
    db_path = _make_file_db(tmp_path)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON;")
    _insert_item(conn, 1)
    for hour in (1, 6, 18):
        _insert_raw_row(conn, 1, _stale_ts(40, hour))
    fresh_ts = _stale_ts(5, 12)
    _insert_raw_row(conn, 1, fresh_ts)
    conn.close()

    pruner = _make_pruner(tmp_path, db_path=db_path, retention_days=30)
    result = pruner.prune(dry_run=False, now=FIXED_NOW)

    assert result.raw_rows_deleted == 3
    conn2 = sqlite3.connect(db_path)
    rows = conn2.execute("SELECT observed_at FROM market_observations_raw").fetchall()
    conn2.close()
    assert [r[0] for r in rows] == [fresh_ts]


def test_db_interrupted_prune_keeps_completed_slices(tmp_path: Path, monkeypatch) -> None:
    """A failure part-way through keeps the slices already committed.

    This is the whole point of batching. The monolithic version rolled back
    everything, so 24 hours of prune work was discarded on the kill that ended
    the incident in issue #149.
    """
    db_path = _make_file_db(tmp_path)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON;")
    _insert_item(conn, 1)
    hours = (1, 2, 3, 4, 5)
    for hour in hours:
        _insert_raw_row(conn, 1, _stale_ts(40, hour))
    conn.close()

    real_delete = SnapshotPruner._delete_slice
    calls: list[tuple[str, str]] = []

    def flaky(self, conn, start, end):  # noqa: ANN001
        calls.append((start, end))
        if len(calls) == 3:
            raise sqlite3.OperationalError("simulated failure mid-prune")
        return real_delete(self, conn, start, end)

    monkeypatch.setattr(SnapshotPruner, "_delete_slice", flaky)

    pruner = _make_pruner(tmp_path, db_path=db_path, retention_days=30)
    result = pruner.prune(dry_run=False, now=FIXED_NOW)

    assert result.errors, "the failure must be reported, not swallowed"

    conn2 = sqlite3.connect(db_path)
    remaining = [
        r[0]
        for r in conn2.execute(
            "SELECT observed_at FROM market_observations_raw ORDER BY observed_at"
        )
    ]
    conn2.close()

    # The two slices that completed before the failure stay deleted.
    assert _stale_ts(40, 1) not in remaining
    assert _stale_ts(40, 2) not in remaining
    # Everything from the failing slice onward is untouched and retried later.
    assert _stale_ts(40, 3) in remaining
    assert len(remaining) == 3


def test_db_interrupted_prune_reports_committed_counts(tmp_path: Path, monkeypatch) -> None:
    """Counts must reflect work that was actually committed.

    The old error path reset both counters to zero on any failure, which was
    right when one transaction covered everything and is a lie once slices
    commit independently.
    """
    db_path = _make_file_db(tmp_path)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON;")
    _insert_item(conn, 1)
    for hour in (1, 2, 3, 4):
        obs_id = _insert_raw_row(conn, 1, _stale_ts(40, hour))
        _insert_norm_row(conn, obs_id)
    conn.close()

    real_delete = SnapshotPruner._delete_slice
    calls: list[int] = []

    def flaky(self, conn, start, end):  # noqa: ANN001
        calls.append(1)
        if len(calls) == 3:
            raise sqlite3.OperationalError("simulated failure mid-prune")
        return real_delete(self, conn, start, end)

    monkeypatch.setattr(SnapshotPruner, "_delete_slice", flaky)

    pruner = _make_pruner(tmp_path, db_path=db_path, retention_days=30)
    result = pruner.prune(dry_run=False, now=FIXED_NOW)

    assert result.raw_rows_deleted == 2
    assert result.norm_rows_deleted == 2


def test_db_prune_stops_at_the_row_budget_and_resumes_next_run(
    tmp_path: Path, monkeypatch
) -> None:
    """Per-run delete work is bounded, and the remainder is not abandoned.

    The cutoff is a calendar date, so the first run after UTC midnight sees a
    whole day of rows become prunable at once. Deleting all of it in one hourly
    run would not fit the run budget. Empty hours cost nothing, so the budget
    counts rows rather than slices.
    """
    monkeypatch.setattr("wow_forecaster.governance.pruner.MAX_ROWS_PER_RUN", 2)

    db_path = _make_file_db(tmp_path)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON;")
    _insert_item(conn, 1)
    for hour in (1, 2, 3, 4, 5):
        _insert_raw_row(conn, 1, _stale_ts(40, hour))
    conn.close()

    pruner = _make_pruner(tmp_path, db_path=db_path, retention_days=30)

    first = pruner.prune(dry_run=False, now=FIXED_NOW)
    assert first.raw_rows_deleted == 2
    assert first.errors == []

    second = pruner.prune(dry_run=False, now=FIXED_NOW)
    assert second.raw_rows_deleted == 2

    conn2 = sqlite3.connect(db_path)
    remaining = conn2.execute(
        "SELECT COUNT(*) FROM market_observations_raw"
    ).fetchone()[0]
    conn2.close()
    assert remaining == 1


def test_db_dry_run_is_not_limited_by_the_row_budget(
    tmp_path: Path, monkeypatch
) -> None:
    """Dry run reports the whole backlog; it deletes nothing, so nothing to bound."""
    monkeypatch.setattr("wow_forecaster.governance.pruner.MAX_ROWS_PER_RUN", 2)

    db_path = _make_file_db(tmp_path)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON;")
    _insert_item(conn, 1)
    for hour in (1, 2, 3, 4, 5):
        _insert_raw_row(conn, 1, _stale_ts(40, hour))
    conn.close()

    pruner = _make_pruner(tmp_path, db_path=db_path, retention_days=30)
    result = pruner.prune(dry_run=True, now=FIXED_NOW)

    assert result.raw_rows_deleted == 5


def test_db_dry_run_counts_across_slices_and_deletes_nothing(tmp_path: Path) -> None:
    db_path = _make_file_db(tmp_path)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON;")
    _insert_item(conn, 1)
    for hour in (2, 8, 20):
        obs_id = _insert_raw_row(conn, 1, _stale_ts(40, hour))
        _insert_norm_row(conn, obs_id)
    conn.close()

    pruner = _make_pruner(tmp_path, db_path=db_path, retention_days=30)
    result = pruner.prune(dry_run=True, now=FIXED_NOW)

    assert result.raw_rows_deleted == 3
    assert result.norm_rows_deleted == 3

    conn2 = sqlite3.connect(db_path)
    raw_left = conn2.execute(
        "SELECT COUNT(*) FROM market_observations_raw"
    ).fetchone()[0]
    norm_left = conn2.execute(
        "SELECT COUNT(*) FROM market_observations_normalized"
    ).fetchone()[0]
    conn2.close()
    assert raw_left == 3
    assert norm_left == 3


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
