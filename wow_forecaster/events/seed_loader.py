"""
Seed event loader: JSON → SQLite → Parquet.

Responsibilities
----------------
1. Load ``config/events/tww_events.json`` (or any events JSON) and upsert into
   the ``wow_events`` table.
2. Load ``config/events/tww_event_impacts.json`` (or any impacts JSON) and upsert
   into the ``event_category_impacts`` table.
3. Export both tables to Parquet files under ``data/processed/events/``.

Parquet schema (events.parquet)
--------------------------------
Standardised column names (per spec):
  event_id      (string)  — stable slug (e.g. "tww-rtwf-nerubar-s1")
  event_name    (string)
  event_type    (string)
  scope         (string)
  start_ts      (date32)
  end_ts        (date32, nullable)
  announced_ts  (date32, nullable)
  source        (string)  — "seed", "blizzard_api", "manual", …
  severity      (string)
  expansion_slug (string)
  metadata      (string)  — JSON blob with patch_version, recurrence_rule, notes

Parquet schema (event_category_impacts.parquet)
------------------------------------------------
  event_id          (string)
  archetype_category (string)
  impact_direction  (string)
  typical_magnitude (float32, nullable)
  lag_days          (int32)
  duration_days     (int32, nullable)
  source            (string)
  notes             (string, nullable)

Validation rules
----------------
- Duplicate event slugs in the JSON input are rejected.
- ``end_date < start_date`` is rejected.
- ``impact_direction`` must be one of: spike, crash, mixed, neutral.
- ``archetype_category`` must be a valid ArchetypeCategory value.
- ``event_slug`` in impacts must reference a slug present in the events file.

Usage
-----
    from wow_forecaster.events.seed_loader import build_events_table

    rows_events, rows_impacts = build_events_table(
        conn=conn,
        events_path=Path("config/events/tww_events.json"),
        impacts_path=Path("config/events/tww_event_impacts.json"),
        output_dir=Path("data/processed/events"),
    )
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from wow_forecaster.taxonomy.archetype_taxonomy import ArchetypeCategory
from wow_forecaster.taxonomy.event_taxonomy import ImpactDirection

log = logging.getLogger(__name__)

# Valid impact_direction values
_VALID_DIRECTIONS: frozenset[str] = frozenset(d.value for d in ImpactDirection)

# Valid archetype_category values
_VALID_CATEGORIES: frozenset[str] = frozenset(c.value for c in ArchetypeCategory)


# ── Parquet schemas ────────────────────────────────────────────────────────────

_EVENTS_PA_SCHEMA = pa.schema([
    pa.field("event_id",       pa.string(),  nullable=False),
    pa.field("event_name",     pa.string(),  nullable=False),
    pa.field("event_type",     pa.string(),  nullable=False),
    pa.field("scope",          pa.string(),  nullable=False),
    pa.field("severity",       pa.string(),  nullable=False),
    pa.field("expansion_slug", pa.string(),  nullable=False),
    pa.field("start_ts",       pa.date32(),  nullable=False),
    pa.field("end_ts",         pa.date32(),  nullable=True),
    pa.field("announced_ts",   pa.date32(),  nullable=True),
    pa.field("source",         pa.string(),  nullable=False),
    pa.field("metadata",       pa.string(),  nullable=True),
])

_IMPACTS_PA_SCHEMA = pa.schema([
    pa.field("event_id",           pa.string(),  nullable=False),
    pa.field("archetype_category", pa.string(),  nullable=False),
    pa.field("impact_direction",   pa.string(),  nullable=False),
    pa.field("typical_magnitude",  pa.float32(), nullable=True),
    pa.field("lag_days",           pa.int32(),   nullable=False),
    pa.field("duration_days",      pa.int32(),   nullable=True),
    pa.field("source",             pa.string(),  nullable=False),
    pa.field("notes",              pa.string(),  nullable=True),
])


# ── Validation ────────────────────────────────────────────────────────────────

def _validate_events(records: list[dict[str, Any]]) -> None:
    """Raise ValueError for any schema violations in the events list."""
    seen_slugs: set[str] = set()
    for i, rec in enumerate(records):
        slug = rec.get("slug") or rec.get("event_id")
        if not slug:
            raise ValueError(f"Event at index {i} is missing 'slug' field.")
        if slug in seen_slugs:
            raise ValueError(f"Duplicate event slug '{slug}' at index {i}.")
        seen_slugs.add(slug)

        start_raw = rec.get("start_date")
        end_raw   = rec.get("end_date")
        if start_raw and end_raw:
            start = date.fromisoformat(start_raw)
            end   = date.fromisoformat(end_raw)
            if end < start:
                raise ValueError(
                    f"Event '{slug}': end_date {end} is before start_date {start}."
                )


def _validate_impacts(
    records: list[dict[str, Any]],
    known_slugs: set[str],
) -> None:
    """Raise ValueError for any schema violations in the impacts list."""
    seen: set[tuple[str, str]] = set()
    for i, rec in enumerate(records):
        slug = rec.get("event_slug")
        cat  = rec.get("archetype_category")

        if not slug:
            raise ValueError(f"Impact at index {i} is missing 'event_slug'.")
        if not cat:
            raise ValueError(f"Impact at index {i} is missing 'archetype_category'.")
        if slug not in known_slugs:
            raise ValueError(
                f"Impact at index {i} references unknown event slug '{slug}'."
            )
        if cat not in _VALID_CATEGORIES:
            raise ValueError(
                f"Impact at index {i} has invalid archetype_category '{cat}'. "
                f"Valid values: {sorted(_VALID_CATEGORIES)}"
            )
        direction = rec.get("impact_direction")
        if direction not in _VALID_DIRECTIONS:
            raise ValueError(
                f"Impact at index {i} has invalid impact_direction '{direction}'. "
                f"Valid values: {sorted(_VALID_DIRECTIONS)}"
            )
        key = (slug, cat)
        if key in seen:
            raise ValueError(
                f"Duplicate (event_slug, archetype_category) pair {key} at index {i}."
            )
        seen.add(key)


# ── DB upsert helpers ─────────────────────────────────────────────────────────

def upsert_events(
    conn: sqlite3.Connection,
    records: list[dict[str, Any]],
) -> int:
    """Insert or replace events into ``wow_events``. Returns count upserted."""
    upserted = 0
    for rec in records:
        # Support both 'slug' (seed file key) and 'event_id' (generic).
        slug = rec.get("slug") or rec.get("event_id")
        conn.execute(
            """
            INSERT INTO wow_events
                (slug, display_name, event_type, scope, severity,
                 expansion_slug, patch_version, start_date, end_date,
                 announced_at, is_recurring, recurrence_rule, notes,
                 updated_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,strftime('%Y-%m-%dT%H:%M:%SZ','now'))
            ON CONFLICT(slug) DO UPDATE SET
                display_name   = excluded.display_name,
                event_type     = excluded.event_type,
                scope          = excluded.scope,
                severity       = excluded.severity,
                expansion_slug = excluded.expansion_slug,
                patch_version  = excluded.patch_version,
                start_date     = excluded.start_date,
                end_date       = excluded.end_date,
                announced_at   = excluded.announced_at,
                is_recurring   = excluded.is_recurring,
                recurrence_rule = excluded.recurrence_rule,
                notes          = excluded.notes,
                updated_at     = strftime('%Y-%m-%dT%H:%M:%SZ','now')
            """,
            (
                slug,
                rec.get("display_name", slug),
                rec["event_type"],
                rec["scope"],
                rec["severity"],
                rec.get("expansion_slug", "tww"),
                rec.get("patch_version"),
                rec["start_date"],
                rec.get("end_date"),
                rec.get("announced_at"),
                int(rec.get("is_recurring", False)),
                rec.get("recurrence_rule"),
                rec.get("notes"),
            ),
        )
        upserted += 1
    conn.commit()
    log.info("Upserted %d events into wow_events.", upserted)
    return upserted


def upsert_category_impacts(
    conn: sqlite3.Connection,
    records: list[dict[str, Any]],
) -> int:
    """Insert or replace records into ``event_category_impacts``.

    Resolves ``event_slug`` → ``event_id`` via the ``wow_events`` table.
    Returns count upserted.
    """
    # Build slug → event_id lookup.
    slug_to_id: dict[str, int] = {}
    for row in conn.execute("SELECT event_id, slug FROM wow_events").fetchall():
        slug_to_id[row["slug"]] = row["event_id"]

    upserted = 0
    skipped  = 0
    for rec in records:
        slug = rec["event_slug"]
        event_id = slug_to_id.get(slug)
        if event_id is None:
            log.warning("Impact references unknown event slug '%s' — skipping.", slug)
            skipped += 1
            continue

        conn.execute(
            """
            INSERT INTO event_category_impacts
                (event_id, archetype_category, impact_direction,
                 typical_magnitude, lag_days, duration_days, source, notes)
            VALUES (?,?,?,?,?,?,?,?)
            ON CONFLICT(event_id, archetype_category) DO UPDATE SET
                impact_direction  = excluded.impact_direction,
                typical_magnitude = excluded.typical_magnitude,
                lag_days          = excluded.lag_days,
                duration_days     = excluded.duration_days,
                source            = excluded.source,
                notes             = excluded.notes
            """,
            (
                event_id,
                rec["archetype_category"],
                rec["impact_direction"],
                rec.get("typical_magnitude"),
                int(rec.get("lag_days", 0)),
                rec.get("duration_days"),
                rec.get("source", "seed"),
                rec.get("notes"),
            ),
        )
        upserted += 1
    conn.commit()
    if skipped:
        log.warning("Skipped %d impact records with unknown event slugs.", skipped)
    log.info("Upserted %d category impact records.", upserted)
    return upserted


# ── Parquet export ────────────────────────────────────────────────────────────

def export_events_parquet(
    conn: sqlite3.Connection,
    output_dir: Path,
) -> Path:
    """Export ``wow_events`` to ``events.parquet`` with the standard schema."""
    rows = conn.execute(
        """
        SELECT slug, display_name, event_type, scope, severity,
               expansion_slug, start_date, end_date, announced_at,
               patch_version, recurrence_rule, notes, is_recurring
        FROM wow_events
        ORDER BY start_date, slug
        """
    ).fetchall()

    event_ids:      list[str]             = []
    event_names:    list[str]             = []
    event_types:    list[str]             = []
    scopes:         list[str]             = []
    severities:     list[str]             = []
    expansions:     list[str]             = []
    start_tss:      list[date | None]     = []
    end_tss:        list[date | None]     = []
    announced_tss:  list[date | None]     = []
    sources:        list[str]             = []
    metadatas:      list[str | None]      = []

    for r in rows:
        event_ids.append(r["slug"])
        event_names.append(r["display_name"])
        event_types.append(r["event_type"])
        scopes.append(r["scope"])
        severities.append(r["severity"])
        expansions.append(r["expansion_slug"])
        start_tss.append(date.fromisoformat(r["start_date"]) if r["start_date"] else None)
        end_tss.append(date.fromisoformat(r["end_date"]) if r["end_date"] else None)
        # announced_at is a datetime ISO string; strip to date for Parquet
        if r["announced_at"]:
            ann_dt = datetime.fromisoformat(r["announced_at"].replace("Z", "+00:00"))
            announced_tss.append(ann_dt.date())
        else:
            announced_tss.append(None)
        sources.append("seed")
        meta: dict[str, Any] = {}
        if r["patch_version"]:
            meta["patch_version"] = r["patch_version"]
        if r["recurrence_rule"]:
            meta["recurrence_rule"] = r["recurrence_rule"]
        if r["notes"]:
            meta["notes"] = r["notes"]
        if r["is_recurring"]:
            meta["is_recurring"] = bool(r["is_recurring"])
        metadatas.append(json.dumps(meta) if meta else None)

    table = pa.table(
        {
            "event_id":       pa.array(event_ids,     type=pa.string()),
            "event_name":     pa.array(event_names,   type=pa.string()),
            "event_type":     pa.array(event_types,   type=pa.string()),
            "scope":          pa.array(scopes,         type=pa.string()),
            "severity":       pa.array(severities,    type=pa.string()),
            "expansion_slug": pa.array(expansions,    type=pa.string()),
            "start_ts":       pa.array(start_tss,     type=pa.date32()),
            "end_ts":         pa.array(end_tss,       type=pa.date32()),
            "announced_ts":   pa.array(announced_tss, type=pa.date32()),
            "source":         pa.array(sources,       type=pa.string()),
            "metadata":       pa.array(metadatas,     type=pa.string()),
        },
        schema=_EVENTS_PA_SCHEMA,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "events.parquet"
    pq.write_table(table, out_path, compression="snappy")
    log.info("Exported %d events to %s", len(rows), out_path)
    return out_path


def export_impacts_parquet(
    conn: sqlite3.Connection,
    output_dir: Path,
) -> Path:
    """Export ``event_category_impacts`` to ``event_category_impacts.parquet``."""
    rows = conn.execute(
        """
        SELECT we.slug AS event_slug,
               eci.archetype_category, eci.impact_direction,
               eci.typical_magnitude, eci.lag_days, eci.duration_days,
               eci.source, eci.notes
        FROM event_category_impacts eci
        JOIN wow_events we ON we.event_id = eci.event_id
        ORDER BY we.start_date, we.slug, eci.archetype_category
        """
    ).fetchall()

    table = pa.table(
        {
            "event_id":           pa.array([r["event_slug"]          for r in rows], type=pa.string()),
            "archetype_category": pa.array([r["archetype_category"]  for r in rows], type=pa.string()),
            "impact_direction":   pa.array([r["impact_direction"]    for r in rows], type=pa.string()),
            "typical_magnitude":  pa.array(
                [float(r["typical_magnitude"]) if r["typical_magnitude"] is not None else None for r in rows],
                type=pa.float32()
            ),
            "lag_days":           pa.array([int(r["lag_days"])       for r in rows], type=pa.int32()),
            "duration_days":      pa.array(
                [int(r["duration_days"]) if r["duration_days"] is not None else None for r in rows],
                type=pa.int32()
            ),
            "source":             pa.array([r["source"]              for r in rows], type=pa.string()),
            "notes":              pa.array([r["notes"]               for r in rows], type=pa.string()),
        },
        schema=_IMPACTS_PA_SCHEMA,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "event_category_impacts.parquet"
    pq.write_table(table, out_path, compression="snappy")
    log.info("Exported %d category impact records to %s", len(rows), out_path)
    return out_path


# ── Top-level entry point ─────────────────────────────────────────────────────

def build_events_table(
    conn: sqlite3.Connection,
    events_path: Path,
    impacts_path: Path | None,
    output_dir: Path,
) -> tuple[int, int]:
    """Load, validate, upsert, and export seed events and category impacts.

    Args:
        conn:         Open SQLite connection (row_factory set to sqlite3.Row).
        events_path:  Path to events JSON file (array of event dicts).
        impacts_path: Path to impacts JSON file (array of impact dicts), or None.
        output_dir:   Directory for Parquet output files.

    Returns:
        Tuple of (events_upserted, impacts_upserted).
    """
    # ── Load events ───────────────────────────────────────────────────────────
    log.info("Loading events from %s", events_path)
    raw_events: list[dict[str, Any]] = json.loads(events_path.read_text(encoding="utf-8"))
    # Strip comment-only entries (keys starting with "_comment").
    event_records = [r for r in raw_events if "slug" in r or "event_id" in r]
    _validate_events(event_records)
    events_count = upsert_events(conn, event_records)

    # ── Load impacts ──────────────────────────────────────────────────────────
    impacts_count = 0
    if impacts_path and impacts_path.exists():
        log.info("Loading category impacts from %s", impacts_path)
        raw_impacts: list[dict[str, Any]] = json.loads(impacts_path.read_text(encoding="utf-8"))
        impact_records = [r for r in raw_impacts if "event_slug" in r]
        known_slugs = {(r.get("slug") or r.get("event_id")) for r in event_records}
        _validate_impacts(impact_records, known_slugs)
        impacts_count = upsert_category_impacts(conn, impact_records)

    # ── Export Parquet ────────────────────────────────────────────────────────
    export_events_parquet(conn, output_dir)
    if impacts_count > 0:
        export_impacts_parquet(conn, output_dir)

    return events_count, impacts_count
