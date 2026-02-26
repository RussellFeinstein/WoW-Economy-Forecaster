"""
Simple sequential schema migration bootstrap.

This is NOT a full migration framework (no Alembic, no down migrations).
For a local research tool, we use a lightweight approach:

  1. A ``schema_versions`` table tracks applied migration IDs.
  2. Each migration is a Python function taking a ``sqlite3.Connection``.
  3. ``run_migrations()`` applies any migrations not yet recorded.

Adding a new migration:
  1. Define a function ``migration_NNNN_description(conn)`` below.
  2. Add it to ``MIGRATIONS`` with a string key like ``"0001_initial"``.

Migrations are applied in dictionary insertion order (Python 3.7+ guarantees).
The initial schema is applied via ``apply_schema()`` in ``schema.py`` before
any migrations run — migrations are for incremental changes only.
"""

from __future__ import annotations

import logging
import sqlite3
from collections.abc import Callable

logger = logging.getLogger(__name__)

MigrationFn = Callable[[sqlite3.Connection], None]


def _ensure_version_table(conn: sqlite3.Connection) -> None:
    """Create the ``schema_versions`` tracking table if it does not exist."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS schema_versions (
            version_id  TEXT    NOT NULL PRIMARY KEY,
            applied_at  TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
            description TEXT
        );
    """)
    conn.commit()


def _get_applied_versions(conn: sqlite3.Connection) -> set[str]:
    """Return the set of already-applied migration version IDs."""
    rows = conn.execute("SELECT version_id FROM schema_versions;").fetchall()
    return {row["version_id"] for row in rows}


def _mark_applied(conn: sqlite3.Connection, version_id: str, description: str) -> None:
    """Record a migration as applied."""
    conn.execute(
        "INSERT INTO schema_versions(version_id, description) VALUES (?, ?);",
        (version_id, description),
    )
    conn.commit()


# ── Migration functions ────────────────────────────────────────────────────────
# Each function receives an open connection. Apply DDL changes here.
# The initial schema is handled by apply_schema() in schema.py,
# so migrations start from schema version 0001 onwards.

def migration_0001_add_schema_versions(conn: sqlite3.Connection) -> None:
    """Bootstrap: ensure schema_versions table exists (no-op after first run)."""
    # Already handled by _ensure_version_table above; this entry just anchors
    # the migration version baseline so future migrations have a reference point.
    pass


def migration_0002_add_backtest_tables(conn: sqlite3.Connection) -> None:
    """Add backtest_runs and backtest_fold_results tables for walk-forward evaluation."""
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS backtest_runs (
            backtest_run_id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id          INTEGER REFERENCES run_metadata(run_id),
            realm_slug      TEXT    NOT NULL,
            backtest_start  TEXT    NOT NULL,
            backtest_end    TEXT    NOT NULL,
            window_days     INTEGER NOT NULL,
            step_days       INTEGER NOT NULL,
            fold_count      INTEGER NOT NULL DEFAULT 0,
            models          TEXT    NOT NULL,
            config_snapshot TEXT    NOT NULL,
            created_at      TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
        );

        CREATE TABLE IF NOT EXISTS backtest_fold_results (
            result_id           INTEGER PRIMARY KEY AUTOINCREMENT,
            backtest_run_id     INTEGER NOT NULL REFERENCES backtest_runs(backtest_run_id),
            fold_index          INTEGER NOT NULL,
            train_end           TEXT    NOT NULL,
            test_date           TEXT    NOT NULL,
            horizon_days        INTEGER NOT NULL,
            archetype_id        INTEGER NOT NULL,
            realm_slug          TEXT    NOT NULL,
            category_tag        TEXT,
            model_name          TEXT    NOT NULL,
            actual_price        REAL,
            predicted_price     REAL,
            abs_error           REAL,
            pct_error           REAL,
            direction_actual    INTEGER,
            direction_predicted INTEGER,
            direction_correct   INTEGER,
            is_event_window     INTEGER NOT NULL DEFAULT 0,
            created_at          TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
        );

        CREATE INDEX IF NOT EXISTS idx_bt_results_run
            ON backtest_fold_results(backtest_run_id);
        CREATE INDEX IF NOT EXISTS idx_bt_results_archetype
            ON backtest_fold_results(archetype_id, model_name, horizon_days);
    """)
    conn.commit()


def migration_0003_add_recommendation_score(conn: sqlite3.Connection) -> None:
    """Add score, score_components, and category_tag to recommendation_outputs."""
    existing = {
        row[1]
        for row in conn.execute("PRAGMA table_info(recommendation_outputs);").fetchall()
    }
    if "score" not in existing:
        conn.execute(
            "ALTER TABLE recommendation_outputs ADD COLUMN score REAL DEFAULT 0.0;"
        )
    if "score_components" not in existing:
        conn.execute(
            "ALTER TABLE recommendation_outputs ADD COLUMN score_components TEXT;"
        )
    if "category_tag" not in existing:
        conn.execute(
            "ALTER TABLE recommendation_outputs ADD COLUMN category_tag TEXT;"
        )
    conn.commit()


def migration_0004_add_monitoring_tables(conn: sqlite3.Connection) -> None:
    """Add drift_check_results and model_health_snapshots tables for monitoring."""
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS drift_check_results (
            drift_id            INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id              INTEGER REFERENCES run_metadata(run_id),
            realm_slug          TEXT    NOT NULL,
            checked_at          TEXT    NOT NULL,
            data_drift_level    TEXT    NOT NULL DEFAULT 'none',
            error_drift_level   TEXT    NOT NULL DEFAULT 'none',
            event_shock_active  INTEGER NOT NULL DEFAULT 0,
            drift_details       TEXT,
            uncertainty_mult    REAL    NOT NULL DEFAULT 1.0,
            retrain_recommended INTEGER NOT NULL DEFAULT 0,
            created_at          TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
        );

        CREATE INDEX IF NOT EXISTS idx_drift_realm_time
            ON drift_check_results(realm_slug, checked_at DESC);

        CREATE INDEX IF NOT EXISTS idx_drift_run
            ON drift_check_results(run_id)
            WHERE run_id IS NOT NULL;

        CREATE TABLE IF NOT EXISTS model_health_snapshots (
            health_id        INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id           INTEGER REFERENCES run_metadata(run_id),
            realm_slug       TEXT    NOT NULL,
            horizon_days     INTEGER NOT NULL,
            n_evaluated      INTEGER NOT NULL DEFAULT 0,
            live_mae         REAL,
            baseline_mae     REAL,
            mae_ratio        REAL,
            live_dir_acc     REAL,
            baseline_dir_acc REAL,
            health_status    TEXT    NOT NULL DEFAULT 'unknown',
            checked_at       TEXT    NOT NULL,
            created_at       TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
        );

        CREATE INDEX IF NOT EXISTS idx_health_realm_horizon
            ON model_health_snapshots(realm_slug, horizon_days, checked_at DESC);
    """)
    conn.commit()


# ── Registry ──────────────────────────────────────────────────────────────────
# Add new migrations here. They will run once, in order.

MIGRATIONS: dict[str, tuple[MigrationFn, str]] = {
    "0001_bootstrap": (
        migration_0001_add_schema_versions,
        "Baseline: schema_versions table created",
    ),
    "0002_backtest_tables": (
        migration_0002_add_backtest_tables,
        "Add backtest_runs and backtest_fold_results tables",
    ),
    "0003_recommendation_score": (
        migration_0003_add_recommendation_score,
        "Add score, score_components, category_tag to recommendation_outputs",
    ),
    "0004_monitoring_tables": (
        migration_0004_add_monitoring_tables,
        "Add drift_check_results and model_health_snapshots tables",
    ),
}


def run_migrations(conn: sqlite3.Connection) -> int:
    """Apply all pending migrations.

    Args:
        conn: An open ``sqlite3.Connection`` with FK enforcement enabled.

    Returns:
        Number of migrations applied in this call.
    """
    _ensure_version_table(conn)
    applied = _get_applied_versions(conn)

    count = 0
    for version_id, (fn, description) in MIGRATIONS.items():
        if version_id in applied:
            logger.debug("Migration %s already applied; skipping.", version_id)
            continue

        logger.info("Applying migration %s: %s", version_id, description)
        try:
            fn(conn)
            _mark_applied(conn, version_id, description)
            count += 1
        except Exception as exc:
            conn.rollback()
            logger.error("Migration %s FAILED: %s", version_id, exc)
            raise

    if count:
        logger.info("Applied %d migration(s).", count)
    else:
        logger.debug("No pending migrations.")

    return count
