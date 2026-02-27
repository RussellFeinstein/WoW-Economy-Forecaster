"""
Event-based feature engineering with strict look-ahead bias prevention.

Purpose
-------
For each (archetype_id, realm_slug, obs_date) row, compute proximity features
and severity indicators based on manually labelled WoW economy events.

Leakage prevention — three independent layers
----------------------------------------------
Layer 1 — **Database filter** (this module, ``load_known_events``):
    Only events with ``announced_at IS NOT NULL`` are loaded from the DB.
    An event with ``announced_at = None`` is never incorporated into any
    feature, regardless of how the Python logic is written.

Layer 2 — **Per-row Python guard** (this module, ``compute_event_features``):
    For each row with date D, we compute
    ``as_of = datetime(D, hour=23, minute=59, second=59, tzinfo=UTC)``.
    ``WoWEvent.is_known_at(as_of)`` returns True only when
    ``announced_at <= as_of``.  Using end-of-day (rather than midnight) ensures
    an event announced on the same calendar day as an observation is correctly
    included (e.g. announced at 17:00 UTC on the same day).

Layer 3 — **Quality heuristic** (in ``quality.build_quality_report``):
    Checks that ``event_days_to_next >= 0`` for every row, catching any logic
    error in the "next event" calculation that would imply an event in the past
    is treated as upcoming.

Assumptions & simplifications
------------------------------
- All seed events are ``scope = GLOBAL``.  Realm-specific scoping is not yet
  implemented (the archetype impact direction already varies by archetype_id,
  so realm filtering would be a future addition).
- ``event_archetype_impact`` is populated from ``event_archetype_impacts``,
  using the most recently *started* active event that has a record for this
  archetype.  If multiple active events have impact records, the one with the
  latest ``start_date`` wins.
- ``event_category_impact`` is populated from ``event_category_impacts``,
  using the same selection logic as above but matching on ``archetype_category``
  instead of ``archetype_id``.  This is used as a fallback when no specific
  archetype impact record exists.
- ``event_impact_magnitude``: taken from the category impact record of the most
  recently started active event for this archetype's category.
- ``days_until_major_event``: days until the next MAJOR or CRITICAL event that
  is known as of obs_date (leakage-safe).  Null if no such event is known.
- ``is_pre_event_window``: True if ``days_until_major_event <= 7``, signalling
  the 7-day pre-event demand run-up window.
- If no events were announced before obs_date, all event features are returned
  in their null/False state.

Input → Output
--------------
Input:  list[dict] with ``obs_date`` key (from lag_rolling output)
Output: same list augmented with event feature keys
"""

from __future__ import annotations

import sqlite3
from collections import defaultdict
from datetime import date, datetime, timezone
from typing import Any

from wow_forecaster.models.event import WoWEvent
from wow_forecaster.taxonomy.event_taxonomy import EventScope, EventSeverity, EventType


# Ordinal map used internally for max-severity computation.
_SEVERITY_ORDINAL: dict[str, int] = {
    EventSeverity.NEGLIGIBLE: 0,
    EventSeverity.MINOR:      1,
    EventSeverity.MODERATE:   2,
    EventSeverity.MAJOR:      3,
    EventSeverity.CRITICAL:   4,
}
_ORDINAL_SEVERITY: dict[int, str] = {v: k for k, v in _SEVERITY_ORDINAL.items()}

# Severities considered "major" for days_until_major_event / is_pre_event_window.
_MAJOR_SEVERITIES: frozenset[str] = frozenset({
    EventSeverity.MAJOR.value,
    EventSeverity.CRITICAL.value,
})

# Number of calendar days that defines the "pre-event window".
_PRE_EVENT_WINDOW_DAYS: int = 7


# ── DB helpers ─────────────────────────────────────────────────────────────────

def load_known_events(conn: sqlite3.Connection) -> list[WoWEvent]:
    """Load events that have a non-NULL announced_at from the database.

    Only events with ``announced_at IS NOT NULL`` are returned.  This is the
    first leakage guard layer: any event without an announcement timestamp is
    unconditionally excluded from all feature computation.

    Args:
        conn: Open SQLite connection with Row factory.

    Returns:
        List of ``WoWEvent`` objects sorted by ``start_date`` ascending.
    """
    rows = conn.execute(
        """
        SELECT event_id, slug, display_name, event_type, scope, severity,
               expansion_slug, patch_version, start_date, end_date,
               announced_at, is_recurring, recurrence_rule, notes
        FROM wow_events
        WHERE announced_at IS NOT NULL
        ORDER BY start_date ASC
        """
    ).fetchall()

    events: list[WoWEvent] = []
    for r in rows:
        events.append(
            WoWEvent(
                event_id=r["event_id"],
                slug=r["slug"],
                display_name=r["display_name"],
                event_type=EventType(r["event_type"]),
                scope=EventScope(r["scope"]),
                severity=EventSeverity(r["severity"]),
                expansion_slug=r["expansion_slug"],
                patch_version=r["patch_version"],
                start_date=date.fromisoformat(r["start_date"]),
                end_date=date.fromisoformat(r["end_date"]) if r["end_date"] else None,
                announced_at=datetime.fromisoformat(r["announced_at"]) if r["announced_at"] else None,
                is_recurring=bool(r["is_recurring"]),
                recurrence_rule=r["recurrence_rule"],
                notes=r["notes"],
            )
        )
    return events


def load_archetype_impacts(
    conn: sqlite3.Connection,
) -> dict[int, list[dict[str, Any]]]:
    """Load archetype-specific event impact records keyed by event_id.

    Returns:
        Dict mapping ``event_id → list[{archetype_id, impact_direction,
        lag_days, duration_days}]``.
    """
    rows = conn.execute(
        """
        SELECT event_id, archetype_id, impact_direction, lag_days, duration_days
        FROM event_archetype_impacts
        """
    ).fetchall()

    impacts: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        impacts[r["event_id"]].append(
            {
                "archetype_id":    r["archetype_id"],
                "impact_direction": r["impact_direction"],
                "lag_days":        r["lag_days"],
                "duration_days":   r["duration_days"],
            }
        )
    return dict(impacts)


def load_category_impacts(
    conn: sqlite3.Connection,
) -> dict[int, list[dict[str, Any]]]:
    """Load category-level event impact records keyed by event_id.

    Returns:
        Dict mapping ``event_id → list[{archetype_category, impact_direction,
        lag_days, duration_days, typical_magnitude}]``.
    """
    rows = conn.execute(
        """
        SELECT event_id, archetype_category, impact_direction,
               lag_days, duration_days, typical_magnitude
        FROM event_category_impacts
        """
    ).fetchall()

    impacts: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        impacts[r["event_id"]].append(
            {
                "archetype_category": r["archetype_category"],
                "impact_direction":   r["impact_direction"],
                "lag_days":           r["lag_days"],
                "duration_days":      r["duration_days"],
                "typical_magnitude":  r["typical_magnitude"],
            }
        )
    return dict(impacts)


# ── Feature computation ────────────────────────────────────────────────────────

def compute_event_features(
    rows: list[dict[str, Any]],
    events: list[WoWEvent],
    impacts: dict[int, list[dict[str, Any]]],
    archetype_id: int,
    category_impacts: dict[int, list[dict[str, Any]]] | None = None,
    archetype_category: str | None = None,
) -> list[dict[str, Any]]:
    """Add event features to a list of rows for one (archetype_id, realm_slug) series.

    For each row, filters ``events`` to those known at ``obs_date`` using
    ``WoWEvent.is_known_at()`` (layer 2 leakage guard), then computes:

    Original 5 columns:
    - ``event_active``: bool — any known event active on obs_date.
    - ``event_days_to_next``: float | None — days to next known future event.
    - ``event_days_since_last``: float | None — days since last completed event.
    - ``event_severity_max``: str | None — max severity of active events.
    - ``event_archetype_impact``: str | None — impact direction from
      ``event_archetype_impacts`` for this archetype (most-recently-started
      active event wins).

    New 3 columns:
    - ``event_impact_magnitude``: float | None — typical_magnitude from the
      category impact record of the most recently started active event for
      this archetype's category.  Falls back to None if no category impact
      record exists.
    - ``days_until_major_event``: float | None — days until the start of the
      next known MAJOR or CRITICAL event.  Leakage-safe: only counts events
      where ``announced_at <= obs_date``.  Null if no such event is known.
    - ``is_pre_event_window``: bool — True if ``days_until_major_event <= 7``.

    Args:
        rows:               Rows from ``compute_lag_rolling_features()``.  Must
                            share the same archetype_id and realm_slug.
        events:             Pre-loaded events from ``load_known_events()``; already
                            filtered to ``announced_at IS NOT NULL``.
        impacts:            Pre-loaded impacts from ``load_archetype_impacts()``.
        archetype_id:       Archetype being processed (for impact lookup).
        category_impacts:   Pre-loaded category impacts from
                            ``load_category_impacts()``.  Optional; when None,
                            ``event_impact_magnitude`` is always None.
        archetype_category: Category slug for this archetype (e.g. "consumable").
                            Required for category impact lookup.

    Returns:
        The same rows with event feature keys added in-place (new dict copy).
    """
    result: list[dict[str, Any]] = []
    for row in rows:
        obs_date: date = row["obs_date"]
        # Layer 2: per-row is_known_at guard using end-of-day boundary.
        as_of = datetime(
            obs_date.year, obs_date.month, obs_date.day,
            23, 59, 59, tzinfo=timezone.utc
        )
        known = [e for e in events if e.is_known_at(as_of)]

        # Partition known events relative to obs_date.
        active_events  = [e for e in known if _is_active(e, obs_date)]
        past_events    = [e for e in known if _is_past(e, obs_date)]
        future_events  = [e for e in known if _is_future(e, obs_date)]

        # ── event_active ──────────────────────────────────────────────────────
        event_active = len(active_events) > 0

        # ── event_days_to_next ────────────────────────────────────────────────
        if future_events:
            days_to_next: float | None = float(
                (min(e.start_date for e in future_events) - obs_date).days
            )
        else:
            days_to_next = None

        # ── event_days_since_last ─────────────────────────────────────────────
        if past_events:
            # Use end_date of the most recently ended event; default to start_date if no end.
            last_end = max(
                (e.end_date if e.end_date is not None else e.start_date)
                for e in past_events
            )
            days_since_last: float | None = float((obs_date - last_end).days)
        else:
            days_since_last = None

        # ── event_severity_max ────────────────────────────────────────────────
        if active_events:
            max_ord = max(_SEVERITY_ORDINAL.get(e.severity.value, 0) for e in active_events)
            severity_max: str | None = _ORDINAL_SEVERITY[max_ord]
        else:
            severity_max = None

        # ── event_archetype_impact — most recently started active event with a
        #    specific archetype impact record. ─────────────────────────────────
        archetype_impact: str | None = None
        if active_events:
            for e in sorted(active_events, key=lambda x: x.start_date, reverse=True):
                if e.event_id is None:
                    continue
                event_impacts = impacts.get(e.event_id, [])
                for imp in event_impacts:
                    if imp["archetype_id"] == archetype_id:
                        archetype_impact = imp["impact_direction"]
                        break
                if archetype_impact is not None:
                    break

        # ── event_impact_magnitude — from category impact record ──────────────
        impact_magnitude: float | None = None
        if active_events and category_impacts and archetype_category:
            for e in sorted(active_events, key=lambda x: x.start_date, reverse=True):
                if e.event_id is None:
                    continue
                cat_imps = category_impacts.get(e.event_id, [])
                for cimp in cat_imps:
                    if cimp["archetype_category"] == archetype_category:
                        mag = cimp.get("typical_magnitude")
                        if mag is not None:
                            impact_magnitude = float(mag)
                        break
                if impact_magnitude is not None:
                    break

        # ── days_until_major_event — next MAJOR/CRITICAL known future event ───
        major_future = [
            e for e in future_events
            if e.severity.value in _MAJOR_SEVERITIES
        ]
        if major_future:
            days_until_major: float | None = float(
                (min(e.start_date for e in major_future) - obs_date).days
            )
        else:
            days_until_major = None

        # ── is_pre_event_window ───────────────────────────────────────────────
        is_pre_window: bool = (
            days_until_major is not None
            and 0 < days_until_major <= _PRE_EVENT_WINDOW_DAYS
        )

        result.append({
            **row,
            "event_active":           event_active,
            "event_days_to_next":     days_to_next,
            "event_days_since_last":  days_since_last,
            "event_severity_max":     severity_max,
            "event_archetype_impact": archetype_impact,
            "event_impact_magnitude": impact_magnitude,
            "days_until_major_event": days_until_major,
            "is_pre_event_window":    is_pre_window,
        })
    return result


# ── Helpers ────────────────────────────────────────────────────────────────────

def _is_active(event: WoWEvent, d: date) -> bool:
    """True if ``d`` falls within the event's active window."""
    if d < event.start_date:
        return False
    if event.end_date is None:
        return True
    return d <= event.end_date


def _is_past(event: WoWEvent, d: date) -> bool:
    """True if the event ended before ``d``."""
    end = event.end_date if event.end_date is not None else event.start_date
    return end < d


def _is_future(event: WoWEvent, d: date) -> bool:
    """True if the event has not yet started as of ``d``."""
    return event.start_date > d
