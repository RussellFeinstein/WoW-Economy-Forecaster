"""Tests for DB migration 0011 - drop idx_obs_raw_item_time (issue #153).

The index was corrupt on the production database ("row missing from index",
100 reported occurrences, all naming this one index) and no query in the repo
uses it: every read of ``market_observations_raw`` filters on ``observed_at``,
``is_processed`` or ``realm_slug``. Dropping it is the smallest-write repair
available, which matters on a machine with a documented history of corrupting
large index builds.

The pairing with schema.py is the part worth pinning. ``init-db`` runs
``apply_schema()`` before ``run_migrations()``, and the raw-index DDL constant
uses ``IF NOT EXISTS``, so leaving the DDL in place would rebuild the index
this migration had just dropped, on every init-db.
"""

from __future__ import annotations

import sqlite3

from wow_forecaster.db.migrations import MIGRATIONS, run_migrations
from wow_forecaster.db.schema import apply_schema, get_existing_indexes

DROPPED_INDEX = "idx_obs_raw_item_time"

# The other raw-table indexes must survive: each one serves a live query.
SURVIVING_INDEXES = (
    "idx_obs_raw_observed",
    "idx_obs_raw_realm_ingested",
    "idx_obs_raw_unprocessed",
)

_LEGACY_DDL = (
    "CREATE INDEX IF NOT EXISTS idx_obs_raw_item_time "
    "ON market_observations_raw(item_id, observed_at);"
)


class TestMigration0011:
    def test_registered(self):
        assert "0011_drop_raw_item_time_index" in MIGRATIONS

    def test_upgrade_path_drops_a_legacy_index(self):
        """A database that already carries the index loses it."""
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        apply_schema(conn)
        conn.execute(_LEGACY_DDL)  # simulate a pre-0011 database
        assert DROPPED_INDEX in get_existing_indexes(conn)

        run_migrations(conn)

        assert DROPPED_INDEX not in get_existing_indexes(conn)
        conn.close()

    def test_survivors_untouched(self):
        """Dropping one index must not disturb the three that serve queries."""
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
        assert "0011_drop_raw_item_time_index" in versions

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
