"""Tests for DB migration 0010 - normalized obs_id FK index (issue #149).

``market_observations_normalized.obs_id`` is the FK to
``market_observations_raw`` and had no index, so with ``PRAGMA foreign_keys =
ON`` every parent delete had to scan the whole child table to prove no child
referenced it. That made the retention prune unable to finish.

The plan assertions here check for a SEARCH rather than only naming the index:
the failure being pinned is a full table scan, and the scan is what makes the
prune impossible.
"""

from __future__ import annotations

import sqlite3

from wow_forecaster.db.migrations import MIGRATIONS, run_migrations
from wow_forecaster.db.schema import apply_schema, get_existing_indexes

NEW_INDEXES = ("idx_obs_norm_obs_id",)

# The FK parent-delete check SQLite runs for each deleted raw row.
FK_CHECK_SQL = "SELECT 1 FROM market_observations_normalized WHERE obs_id = 5"

# The pruner's child-delete probe, one hour slice.
CHILD_PROBE_SQL = (
    "SELECT 1 FROM market_observations_normalized WHERE obs_id IN ("
    "SELECT obs_id FROM market_observations_raw "
    "WHERE observed_at >= '2026-07-21T02:00:00' AND observed_at < '2026-07-21T03:00:00')"
)


def _plan(conn: sqlite3.Connection, sql: str) -> str:
    """Join the EXPLAIN QUERY PLAN detail strings for a query."""
    rows = conn.execute(f"EXPLAIN QUERY PLAN {sql}").fetchall()
    return " | ".join(str(row[-1]) for row in rows)


class TestIndexInSchema:
    def test_new_index_created_by_apply_schema(self, in_memory_db):
        indexes = get_existing_indexes(in_memory_db)
        for name in NEW_INDEXES:
            assert name in indexes, f"Expected index '{name}' not found. Found: {indexes}"


class TestMigration0010:
    def test_registered(self):
        assert "0010_norm_obs_id_index" in MIGRATIONS

    def test_upgrade_path_creates_index(self):
        """A pre-0010 DB (index absent) gains it from run_migrations()."""
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        apply_schema(conn)
        for name in NEW_INDEXES:
            conn.execute(f"DROP INDEX {name};")
        remaining = get_existing_indexes(conn)
        assert not any(name in remaining for name in NEW_INDEXES)

        run_migrations(conn)
        indexes = get_existing_indexes(conn)
        for name in NEW_INDEXES:
            assert name in indexes
        conn.close()

    def test_recorded_in_schema_versions(self, in_memory_db):
        run_migrations(in_memory_db)
        versions = {
            row[0]
            for row in in_memory_db.execute("SELECT version_id FROM schema_versions")
        }
        assert "0010_norm_obs_id_index" in versions

    def test_idempotent(self, in_memory_db):
        run_migrations(in_memory_db)
        run_migrations(in_memory_db)
        for name in NEW_INDEXES:
            assert name in get_existing_indexes(in_memory_db)


class TestQueryPlans:
    """The child table must be seekable by obs_id, not scanned.

    A scan here is the whole defect: 11.2M parent deletes against a 158M row
    child table is work that cannot complete.
    """

    def test_fk_check_seeks_instead_of_scanning(self, in_memory_db):
        plan = _plan(in_memory_db, FK_CHECK_SQL)
        assert "idx_obs_norm_obs_id" in plan
        assert "SCAN market_observations_normalized" not in plan

    def test_child_delete_probe_seeks_both_sides(self, in_memory_db):
        plan = _plan(in_memory_db, CHILD_PROBE_SQL)
        assert "idx_obs_norm_obs_id" in plan
        assert "SCAN market_observations_normalized" not in plan
        # The raw side was already seekable via migration 0009; assert the seek
        # term rather than the index name, per the 2026-07-22 lesson.
        assert "observed_at>" in plan
