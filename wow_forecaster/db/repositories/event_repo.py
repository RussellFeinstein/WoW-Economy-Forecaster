"""
Repository for WoW events — insert, fetch, and update operations.
"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime
from typing import Optional

from wow_forecaster.db.repositories.base import BaseRepository
from wow_forecaster.models.event import WoWEvent
from wow_forecaster.taxonomy.event_taxonomy import EventScope, EventSeverity, EventType

logger = logging.getLogger(__name__)


class WoWEventRepository(BaseRepository):
    """Read/write access to the ``wow_events`` table."""

    def insert(self, event: WoWEvent) -> int:
        """Insert a new event and return its auto-assigned ``event_id``.

        Args:
            event: The ``WoWEvent`` to persist.

        Returns:
            The newly assigned ``event_id``.
        """
        self.execute(
            """
            INSERT INTO wow_events (
                slug, display_name, event_type, scope, severity,
                expansion_slug, patch_version, start_date, end_date,
                announced_at, is_recurring, recurrence_rule, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            (
                event.slug,
                event.display_name,
                event.event_type.value,
                event.scope.value,
                event.severity.value,
                event.expansion_slug,
                event.patch_version,
                event.start_date.isoformat(),
                event.end_date.isoformat() if event.end_date else None,
                event.announced_at.isoformat() if event.announced_at else None,
                int(event.is_recurring),
                event.recurrence_rule,
                event.notes,
            ),
        )
        return self.last_insert_rowid()

    def upsert(self, event: WoWEvent) -> int:
        """Insert or replace an event by slug.

        Args:
            event: The ``WoWEvent`` to persist.

        Returns:
            The ``event_id`` (existing or new).
        """
        self.execute(
            """
            INSERT INTO wow_events (
                slug, display_name, event_type, scope, severity,
                expansion_slug, patch_version, start_date, end_date,
                announced_at, is_recurring, recurrence_rule, notes,
                updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
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
                updated_at     = excluded.updated_at;
            """,
            (
                event.slug,
                event.display_name,
                event.event_type.value,
                event.scope.value,
                event.severity.value,
                event.expansion_slug,
                event.patch_version,
                event.start_date.isoformat(),
                event.end_date.isoformat() if event.end_date else None,
                event.announced_at.isoformat() if event.announced_at else None,
                int(event.is_recurring),
                event.recurrence_rule,
                event.notes,
            ),
        )
        row = self.fetchone("SELECT event_id FROM wow_events WHERE slug = ?;", (event.slug,))
        assert row is not None
        return int(row["event_id"])

    def get_by_id(self, event_id: int) -> Optional[WoWEvent]:
        """Fetch a single event by primary key.

        Args:
            event_id: The ``event_id`` to look up.

        Returns:
            ``WoWEvent`` or ``None`` if not found.
        """
        row = self.fetchone(
            "SELECT * FROM wow_events WHERE event_id = ?;", (event_id,)
        )
        return _row_to_event(row) if row else None

    def get_by_slug(self, slug: str) -> Optional[WoWEvent]:
        """Fetch a single event by slug.

        Args:
            slug: Unique event slug.

        Returns:
            ``WoWEvent`` or ``None`` if not found.
        """
        row = self.fetchone("SELECT * FROM wow_events WHERE slug = ?;", (slug,))
        return _row_to_event(row) if row else None

    def get_by_type(self, event_type: EventType) -> list[WoWEvent]:
        """Fetch all events of a given type.

        Args:
            event_type: The ``EventType`` to filter by.

        Returns:
            List of matching ``WoWEvent`` objects, ordered by ``start_date``.
        """
        rows = self.fetchall(
            "SELECT * FROM wow_events WHERE event_type = ? ORDER BY start_date;",
            (event_type.value,),
        )
        return [_row_to_event(r) for r in rows]

    def get_by_expansion(self, expansion_slug: str) -> list[WoWEvent]:
        """Fetch all events for a given expansion.

        Args:
            expansion_slug: Expansion identifier string.

        Returns:
            List of ``WoWEvent`` objects, ordered by ``start_date``.
        """
        rows = self.fetchall(
            "SELECT * FROM wow_events WHERE expansion_slug = ? ORDER BY start_date;",
            (expansion_slug,),
        )
        return [_row_to_event(r) for r in rows]

    def get_active_on(self, check_date: date) -> list[WoWEvent]:
        """Fetch all events active on a given date.

        Args:
            check_date: The date to check.

        Returns:
            List of events where ``start_date <= check_date <= end_date``
            (or ``end_date`` is NULL).
        """
        rows = self.fetchall(
            """
            SELECT * FROM wow_events
            WHERE start_date <= ?
              AND (end_date IS NULL OR end_date >= ?)
            ORDER BY severity, start_date;
            """,
            (check_date.isoformat(), check_date.isoformat()),
        )
        return [_row_to_event(r) for r in rows]

    def count(self) -> int:
        """Return total number of events in the table."""
        row = self.fetchone("SELECT COUNT(*) AS n FROM wow_events;")
        assert row is not None
        return int(row["n"])


# ── Private helper ────────────────────────────────────────────────────────────

import sqlite3  # noqa: E402  (needed for type annotation only)


def _row_to_event(row: sqlite3.Row) -> WoWEvent:
    """Convert a ``sqlite3.Row`` from ``wow_events`` to a ``WoWEvent``."""
    return WoWEvent(
        event_id=row["event_id"],
        slug=row["slug"],
        display_name=row["display_name"],
        event_type=EventType(row["event_type"]),
        scope=EventScope(row["scope"]),
        severity=EventSeverity(row["severity"]),
        expansion_slug=row["expansion_slug"],
        patch_version=row["patch_version"],
        start_date=date.fromisoformat(row["start_date"]),
        end_date=date.fromisoformat(row["end_date"]) if row["end_date"] else None,
        announced_at=(
            datetime.fromisoformat(row["announced_at"]) if row["announced_at"] else None
        ),
        is_recurring=bool(row["is_recurring"]),
        recurrence_rule=row["recurrence_rule"],
        notes=row["notes"],
    )
