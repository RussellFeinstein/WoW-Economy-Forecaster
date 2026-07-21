"""Tests for DB migration 0009 - check-data-health hot-path indexes (issue #59).

Covers the two new market_observations_raw indexes (schema and upgrade path)
and pins EXPLAIN QUERY PLAN index use for the health-check query shapes, which
is the acceptance criterion on the issue.
"""

from __future__ import annotations

import sqlite3

from wow_forecaster.db.migrations import MIGRATIONS, run_migrations
from wow_forecaster.db.schema import apply_schema, get_existing_indexes

NEW_INDEXES = ("idx_obs_raw_observed", "idx_obs_raw_realm_ingested")

# Mirrors of the health.py hot-path query shapes the new indexes must serve.
RETENTION_SQL = "SELECT MIN(observed_at) AS oldest FROM market_observations_raw"
LAST_INGEST_SQL = (
    "SELECT MAX(ingested_at) AS last_ingest FROM market_observations_raw "
    "WHERE realm_slug = 'us'"
)
COVERAGE_SQL = (
    "SELECT DISTINCT DATE(observed_at) AS obs_date "
    "FROM market_observations_normalized "
    "WHERE realm_slug = 'us' AND is_outlier = 0 AND observed_at >= '2026-07-07' "
    "ORDER BY obs_date"
)
FIRST_DATE_SQL = (
    "SELECT DATE(MIN(observed_at)) AS d FROM market_observations_normalized "
    "WHERE realm_slug = 'us' AND is_outlier = 0"
)
LAST_DATE_SQL = (
    "SELECT DATE(MAX(observed_at)) AS d FROM market_observations_normalized "
    "WHERE realm_slug = 'us' AND is_outlier = 0"
)


def _plan(conn: sqlite3.Connection, sql: str) -> str:
    """Join the EXPLAIN QUERY PLAN detail strings for a query."""
    rows = conn.execute(f"EXPLAIN QUERY PLAN {sql}").fetchall()
    return " | ".join(str(row[-1]) for row in rows)


class TestIndexesInSchema:
    def test_new_indexes_created_by_apply_schema(self, in_memory_db):
        indexes = get_existing_indexes(in_memory_db)
        for name in NEW_INDEXES:
            assert name in indexes, f"Expected index '{name}' not found. Found: {indexes}"


class TestMigration0009:
    def test_registered(self):
        assert "0009_health_check_indexes" in MIGRATIONS

    def test_upgrade_path_creates_indexes(self):
        """A pre-0009 DB (indexes absent) gains them from run_migrations()."""
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
        assert "0009_health_check_indexes" in versions

    def test_idempotent(self, in_memory_db):
        run_migrations(in_memory_db)
        run_migrations(in_memory_db)
        for name in NEW_INDEXES:
            assert name in get_existing_indexes(in_memory_db)


class TestQueryPlans:
    """EXPLAIN QUERY PLAN must show index use on the health-check hot paths.

    Only the index name is asserted: the surrounding plan wording varies
    across SQLite versions, the index choice does not.
    """

    def test_retention_sentinel_uses_observed_index(self, in_memory_db):
        assert "idx_obs_raw_observed" in _plan(in_memory_db, RETENTION_SQL)

    def test_last_ingest_uses_realm_ingested_index(self, in_memory_db):
        assert "idx_obs_raw_realm_ingested" in _plan(in_memory_db, LAST_INGEST_SQL)

    def test_coverage_range_uses_realm_outlier_time_index(self, in_memory_db):
        assert "idx_obs_norm_realm_outlier_time" in _plan(in_memory_db, COVERAGE_SQL)

    def test_date_range_probes_use_realm_outlier_time_index(self, in_memory_db):
        assert "idx_obs_norm_realm_outlier_time" in _plan(in_memory_db, FIRST_DATE_SQL)
        assert "idx_obs_norm_realm_outlier_time" in _plan(in_memory_db, LAST_DATE_SQL)
