"""Tests for DB migration 0012 - drop idx_obs_raw_realm_ingested (issue #155).

Two rows were missing from this index on the production database, and once
their hour slice entered the prune window the raw DELETE for that slice
raised "database disk image is malformed" on every hourly run for twenty
days, stalling the whole retention prune behind it. The index served exactly
one query, the health check's last-ingest lookup, and that query now reads
the newest normalized observation through idx_obs_norm_realm_outlier_time
instead. Dropping the index is the smallest-write repair available, the same
call #153 made for idx_obs_raw_item_time.

The pairing with schema.py is the part worth pinning. ``init-db`` runs
``apply_schema()`` before ``run_migrations()``, and the raw-index DDL constant
uses ``IF NOT EXISTS``, so leaving the DDL in place would rebuild the index
this migration had just dropped, on every init-db.
"""

from __future__ import annotations

import sqlite3

from wow_forecaster.db.migrations import MIGRATIONS, run_migrations
from wow_forecaster.db.schema import apply_schema, get_existing_indexes

DROPPED_INDEX = "idx_obs_raw_realm_ingested"

# The remaining raw-table indexes must survive: each one serves a live query
# (the pruner's observed_at range and the normalizer's is_processed scan).
SURVIVING_INDEXES = (
    "idx_obs_raw_observed",
    "idx_obs_raw_unprocessed",
)

_LEGACY_DDL = (
    "CREATE INDEX IF NOT EXISTS idx_obs_raw_realm_ingested "
    "ON market_observations_raw(realm_slug, ingested_at);"
)


class TestMigration0012:
    def test_registered(self):
        assert "0012_drop_raw_realm_ingested_index" in MIGRATIONS

    def test_upgrade_path_drops_a_legacy_index(self):
        """A database that already carries the index loses it."""
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        apply_schema(conn)
        conn.execute(_LEGACY_DDL)  # simulate a pre-0012 database
        assert DROPPED_INDEX in get_existing_indexes(conn)

        run_migrations(conn)

        assert DROPPED_INDEX not in get_existing_indexes(conn)
        conn.close()

    def test_survivors_untouched(self):
        """Dropping one index must not disturb the two that serve queries."""
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        apply_schema(conn)
        conn.execute(_LEGACY_DDL)

        run_migrations(conn)

        indexes = get_existing_indexes(conn)
        for name in SURVIVING_INDEXES:
            assert name in indexes, f"{name} should have survived. Found: {indexes}"
        conn.close()

    def test_no_op_on_a_fresh_database(self, in_memory_db):
        """apply_schema never creates it, so the migration has nothing to do."""
        assert DROPPED_INDEX not in get_existing_indexes(in_memory_db)
        run_migrations(in_memory_db)
        assert DROPPED_INDEX not in get_existing_indexes(in_memory_db)

    def test_recorded_in_schema_versions(self, in_memory_db):
        run_migrations(in_memory_db)
        versions = {
            row[0]
            for row in in_memory_db.execute("SELECT version_id FROM schema_versions")
        }
        assert "0012_drop_raw_realm_ingested_index" in versions

    def test_idempotent(self, in_memory_db):
        run_migrations(in_memory_db)
        run_migrations(in_memory_db)
        assert DROPPED_INDEX not in get_existing_indexes(in_memory_db)


class TestSchemaAndMigrationAgree:
    """The two halves must not fight each other on init-db.

    apply_schema runs first, so a surviving DDL line would recreate the index
    every time and the migration would be permanently undone.
    """

    def test_apply_schema_then_migrations_leaves_it_dropped(self):
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        apply_schema(conn)
        run_migrations(conn)
        apply_schema(conn)  # the second init-db
        run_migrations(conn)
        assert DROPPED_INDEX not in get_existing_indexes(conn)
        conn.close()
