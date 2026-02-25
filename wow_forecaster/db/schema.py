"""
SQLite schema DDL — all CREATE TABLE and CREATE INDEX statements.

All statements use ``IF NOT EXISTS`` so ``apply_schema()`` is **idempotent**:
safe to call on an already-initialized database (e.g. after restart or in tests).

Table creation order respects foreign key dependencies:
  1. item_categories      (no FKs)
  2. economic_archetypes  (no FKs)
  3. items                (→ item_categories, economic_archetypes)
  4. market_observations_raw          (→ items)
  5. market_observations_normalized   (→ market_observations_raw, items, economic_archetypes)
  6. archetype_mappings  (→ economic_archetypes × 2)
  7. wow_events          (no FKs)
  8. event_archetype_impacts (→ wow_events, economic_archetypes)
  9. model_metadata      (no FKs)
  10. run_metadata        (→ model_metadata)
  11. forecast_outputs    (→ run_metadata, economic_archetypes, items)
  12. recommendation_outputs (→ forecast_outputs)
"""

from __future__ import annotations

import logging
import sqlite3

logger = logging.getLogger(__name__)

# ── DDL statements ─────────────────────────────────────────────────────────────

_DDL_ITEM_CATEGORIES = """
CREATE TABLE IF NOT EXISTS item_categories (
    category_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    slug            TEXT    NOT NULL UNIQUE,
    display_name    TEXT    NOT NULL,
    parent_slug     TEXT    REFERENCES item_categories(slug),
    archetype_tag   TEXT    NOT NULL,
    expansion_slug  TEXT,
    created_at      TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);
"""

_DDL_ECONOMIC_ARCHETYPES = """
CREATE TABLE IF NOT EXISTS economic_archetypes (
    archetype_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    slug                TEXT    NOT NULL UNIQUE,
    display_name        TEXT    NOT NULL,
    category_tag        TEXT    NOT NULL,
    sub_tag             TEXT,
    description         TEXT,
    is_transferable     INTEGER NOT NULL DEFAULT 1,
    transfer_confidence REAL    NOT NULL DEFAULT 0.5,
    transfer_notes      TEXT,
    created_at          TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);
"""

_DDL_ITEMS = """
CREATE TABLE IF NOT EXISTS items (
    item_id         INTEGER PRIMARY KEY,
    name            TEXT    NOT NULL,
    category_id     INTEGER NOT NULL REFERENCES item_categories(category_id),
    archetype_id    INTEGER REFERENCES economic_archetypes(archetype_id),
    expansion_slug  TEXT    NOT NULL,
    quality         TEXT    NOT NULL,
    is_crafted      INTEGER NOT NULL DEFAULT 0,
    is_boe          INTEGER NOT NULL DEFAULT 0,
    ilvl            INTEGER,
    notes           TEXT,
    created_at      TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    updated_at      TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);
"""

_DDL_MARKET_OBS_RAW = """
CREATE TABLE IF NOT EXISTS market_observations_raw (
    obs_id               INTEGER PRIMARY KEY AUTOINCREMENT,
    item_id              INTEGER NOT NULL REFERENCES items(item_id),
    realm_slug           TEXT    NOT NULL,
    faction              TEXT    NOT NULL DEFAULT 'neutral',
    observed_at          TEXT    NOT NULL,
    source               TEXT    NOT NULL,
    min_buyout_raw       INTEGER,
    market_value_raw     INTEGER,
    historical_value_raw INTEGER,
    quantity_listed      INTEGER,
    num_auctions         INTEGER,
    raw_json             TEXT,
    ingested_at          TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    is_processed         INTEGER NOT NULL DEFAULT 0
);
"""

_DDL_MARKET_OBS_RAW_INDEXES = """
CREATE INDEX IF NOT EXISTS idx_obs_raw_item_time
    ON market_observations_raw(item_id, observed_at);
CREATE INDEX IF NOT EXISTS idx_obs_raw_unprocessed
    ON market_observations_raw(is_processed)
    WHERE is_processed = 0;
"""

_DDL_MARKET_OBS_NORMALIZED = """
CREATE TABLE IF NOT EXISTS market_observations_normalized (
    norm_id               INTEGER PRIMARY KEY AUTOINCREMENT,
    obs_id                INTEGER NOT NULL REFERENCES market_observations_raw(obs_id),
    item_id               INTEGER NOT NULL REFERENCES items(item_id),
    archetype_id          INTEGER REFERENCES economic_archetypes(archetype_id),
    realm_slug            TEXT    NOT NULL,
    faction               TEXT    NOT NULL DEFAULT 'neutral',
    observed_at           TEXT    NOT NULL,
    price_gold            REAL    NOT NULL,
    market_value_gold     REAL,
    historical_value_gold REAL,
    quantity_listed       INTEGER,
    num_auctions          INTEGER,
    z_score               REAL,
    is_outlier            INTEGER NOT NULL DEFAULT 0,
    normalized_at         TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);
"""

_DDL_MARKET_OBS_NORMALIZED_INDEXES = """
CREATE INDEX IF NOT EXISTS idx_obs_norm_item_time
    ON market_observations_normalized(item_id, observed_at);
CREATE INDEX IF NOT EXISTS idx_obs_norm_archetype_time
    ON market_observations_normalized(archetype_id, observed_at);
"""

_DDL_ARCHETYPE_MAPPINGS = """
CREATE TABLE IF NOT EXISTS archetype_mappings (
    mapping_id          INTEGER PRIMARY KEY AUTOINCREMENT,
    source_archetype_id INTEGER NOT NULL REFERENCES economic_archetypes(archetype_id),
    target_archetype_id INTEGER NOT NULL REFERENCES economic_archetypes(archetype_id),
    source_expansion    TEXT    NOT NULL DEFAULT 'tww',
    target_expansion    TEXT    NOT NULL DEFAULT 'midnight',
    confidence_score    REAL    NOT NULL,
    mapping_rationale   TEXT    NOT NULL,
    created_by          TEXT    NOT NULL DEFAULT 'manual',
    created_at          TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    UNIQUE(source_archetype_id, target_archetype_id, source_expansion, target_expansion)
);
"""

_DDL_WOW_EVENTS = """
CREATE TABLE IF NOT EXISTS wow_events (
    event_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    slug            TEXT    NOT NULL UNIQUE,
    display_name    TEXT    NOT NULL,
    event_type      TEXT    NOT NULL,
    scope           TEXT    NOT NULL,
    severity        TEXT    NOT NULL,
    expansion_slug  TEXT    NOT NULL,
    patch_version   TEXT,
    start_date      TEXT    NOT NULL,
    end_date        TEXT,
    announced_at    TEXT,
    is_recurring    INTEGER NOT NULL DEFAULT 0,
    recurrence_rule TEXT,
    notes           TEXT,
    created_at      TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    updated_at      TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);
"""

_DDL_WOW_EVENTS_INDEXES = """
CREATE INDEX IF NOT EXISTS idx_events_type_date
    ON wow_events(event_type, start_date);
CREATE INDEX IF NOT EXISTS idx_events_expansion_date
    ON wow_events(expansion_slug, start_date);
"""

_DDL_EVENT_ARCHETYPE_IMPACTS = """
CREATE TABLE IF NOT EXISTS event_archetype_impacts (
    impact_id          INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id           INTEGER NOT NULL REFERENCES wow_events(event_id),
    archetype_id       INTEGER NOT NULL REFERENCES economic_archetypes(archetype_id),
    impact_direction   TEXT    NOT NULL,
    typical_magnitude  REAL,
    lag_days           INTEGER NOT NULL DEFAULT 0,
    duration_days      INTEGER,
    source             TEXT    NOT NULL DEFAULT 'manual',
    notes              TEXT,
    UNIQUE(event_id, archetype_id)
);
"""

_DDL_MODEL_METADATA = """
CREATE TABLE IF NOT EXISTS model_metadata (
    model_id             INTEGER PRIMARY KEY AUTOINCREMENT,
    slug                 TEXT    NOT NULL UNIQUE,
    display_name         TEXT    NOT NULL,
    model_type           TEXT    NOT NULL,
    version              TEXT    NOT NULL DEFAULT '0.1.0',
    hyperparameters      TEXT,
    training_data_start  TEXT,
    training_data_end    TEXT,
    validation_mae       REAL,
    validation_rmse      REAL,
    artifact_path        TEXT,
    is_active            INTEGER NOT NULL DEFAULT 0,
    created_at           TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);
"""

_DDL_RUN_METADATA = """
CREATE TABLE IF NOT EXISTS run_metadata (
    run_id          INTEGER PRIMARY KEY AUTOINCREMENT,
    run_slug        TEXT    NOT NULL UNIQUE,
    pipeline_stage  TEXT    NOT NULL,
    status          TEXT    NOT NULL DEFAULT 'started',
    model_id        INTEGER REFERENCES model_metadata(model_id),
    realm_slug      TEXT,
    expansion_slug  TEXT,
    config_snapshot TEXT    NOT NULL,
    rows_processed  INTEGER NOT NULL DEFAULT 0,
    error_message   TEXT,
    started_at      TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    finished_at     TEXT
);
"""

_DDL_FORECAST_OUTPUTS = """
CREATE TABLE IF NOT EXISTS forecast_outputs (
    forecast_id          INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id               INTEGER NOT NULL REFERENCES run_metadata(run_id),
    archetype_id         INTEGER REFERENCES economic_archetypes(archetype_id),
    item_id              INTEGER REFERENCES items(item_id),
    realm_slug           TEXT    NOT NULL,
    forecast_horizon     TEXT    NOT NULL,
    target_date          TEXT    NOT NULL,
    predicted_price_gold REAL    NOT NULL,
    confidence_lower     REAL    NOT NULL,
    confidence_upper     REAL    NOT NULL,
    confidence_pct       REAL    NOT NULL DEFAULT 0.80,
    model_slug           TEXT    NOT NULL,
    features_hash        TEXT,
    created_at           TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);
"""

_DDL_FORECAST_OUTPUTS_INDEXES = """
CREATE INDEX IF NOT EXISTS idx_forecast_archetype_date
    ON forecast_outputs(archetype_id, target_date);
CREATE INDEX IF NOT EXISTS idx_forecast_run
    ON forecast_outputs(run_id);
"""

_DDL_RECOMMENDATION_OUTPUTS = """
CREATE TABLE IF NOT EXISTS recommendation_outputs (
    rec_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    forecast_id INTEGER NOT NULL REFERENCES forecast_outputs(forecast_id),
    action      TEXT    NOT NULL,
    reasoning   TEXT    NOT NULL,
    priority    INTEGER NOT NULL DEFAULT 5,
    expires_at  TEXT,
    created_at  TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);
"""

# ── Ordered list of all DDL to apply ──────────────────────────────────────────

_ALL_DDL: list[str] = [
    _DDL_ITEM_CATEGORIES,
    _DDL_ECONOMIC_ARCHETYPES,
    _DDL_ITEMS,
    _DDL_MARKET_OBS_RAW,
    _DDL_MARKET_OBS_RAW_INDEXES,
    _DDL_MARKET_OBS_NORMALIZED,
    _DDL_MARKET_OBS_NORMALIZED_INDEXES,
    _DDL_ARCHETYPE_MAPPINGS,
    _DDL_WOW_EVENTS,
    _DDL_WOW_EVENTS_INDEXES,
    _DDL_EVENT_ARCHETYPE_IMPACTS,
    _DDL_MODEL_METADATA,
    _DDL_RUN_METADATA,
    _DDL_FORECAST_OUTPUTS,
    _DDL_FORECAST_OUTPUTS_INDEXES,
    _DDL_RECOMMENDATION_OUTPUTS,
]

# Table names for introspection / tests
ALL_TABLE_NAMES = [
    "item_categories",
    "economic_archetypes",
    "items",
    "market_observations_raw",
    "market_observations_normalized",
    "archetype_mappings",
    "wow_events",
    "event_archetype_impacts",
    "model_metadata",
    "run_metadata",
    "forecast_outputs",
    "recommendation_outputs",
]


def apply_schema(conn: sqlite3.Connection) -> None:
    """Apply all DDL statements to ``conn``.

    Idempotent — safe to call on an already-initialized database.
    Each statement uses ``IF NOT EXISTS`` guards.

    Args:
        conn: An open ``sqlite3.Connection`` (FK enforcement should be ON).
    """
    logger.debug("Applying schema to database...")

    for ddl in _ALL_DDL:
        # Each block may contain multiple semicolon-separated statements
        for statement in _split_ddl(ddl):
            if statement.strip():
                conn.execute(statement)

    conn.commit()
    logger.info("Schema applied: %d tables, indexes created/verified.", len(ALL_TABLE_NAMES))


def _split_ddl(ddl: str) -> list[str]:
    """Split a multi-statement DDL block on semicolons."""
    return [s.strip() for s in ddl.split(";") if s.strip()]


def get_existing_tables(conn: sqlite3.Connection) -> list[str]:
    """Return list of table names present in the database.

    Args:
        conn: An open ``sqlite3.Connection``.

    Returns:
        List of table name strings (sorted alphabetically).
    """
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;"
    ).fetchall()
    return [row["name"] for row in rows]


def get_existing_indexes(conn: sqlite3.Connection) -> list[str]:
    """Return list of index names present in the database.

    Args:
        conn: An open ``sqlite3.Connection``.

    Returns:
        List of index name strings (sorted alphabetically).
    """
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='index' ORDER BY name;"
    ).fetchall()
    return [row["name"] for row in rows]
