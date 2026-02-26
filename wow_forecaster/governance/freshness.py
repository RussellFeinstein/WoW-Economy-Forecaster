"""
Source freshness and TTL checks.

Freshness is computed by comparing the age of the most recent successful
ingestion snapshot for a source against the thresholds in that source's
FreshnessConfig.

Status classification
---------------------
  "fresh"    — age < ttl_hours
  "aging"    — ttl_hours <= age < stale_threshold_hours
  "stale"    — stale_threshold_hours <= age < critical_threshold_hours
  "critical" — age >= critical_threshold_hours
  "unknown"  — no snapshot found (age cannot be determined)

The age is derived from the ingestion_snapshots table (populated by
IngestStage).  For manual sources (no HTTP, no snapshot file), the
last import timestamp is approximated from the wow_events table for
news/event sources.

Assumptions
-----------
- All timestamps are UTC ISO-8601 strings.
- "most recent successful snapshot" = MAX(fetched_at) WHERE success=1
  AND source = <source_id> in ingestion_snapshots.
- For sources without snapshot requirements (requires_snapshot=False),
  freshness is always returned as "unknown" (no DB record to check).
  The caller may choose to treat "unknown" as acceptable for manual sources.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from wow_forecaster.governance.models import SourcePolicy


# ── Status enum ───────────────────────────────────────────────────────────────


class FreshnessStatus(str, Enum):
    """Classification of how current a source's data is."""

    FRESH    = "fresh"    # Within TTL
    AGING    = "aging"    # Past TTL but not yet stale threshold
    STALE    = "stale"    # Past stale threshold but not critical
    CRITICAL = "critical" # Past critical threshold — do not use for forecasting
    UNKNOWN  = "unknown"  # No snapshot available to determine age


# ── Result dataclass ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class FreshnessResult:
    """Freshness check outcome for one source.

    Attributes:
        source_id:                Identifier of the source checked.
        last_snapshot_at:         ISO-8601 UTC string of most recent snapshot,
                                  or None if no snapshot found.
        age_hours:                Hours since last snapshot, or None if unknown.
        ttl_hours:                TTL from the source's FreshnessConfig.
        stale_threshold_hours:    Stale threshold from FreshnessConfig.
        critical_threshold_hours: Critical threshold from FreshnessConfig.
        is_within_ttl:            True if age_hours < ttl_hours.
        is_stale:                 True if age_hours >= stale_threshold_hours.
        is_critical:              True if age_hours >= critical_threshold_hours.
        status:                   FreshnessStatus classification.
        requires_snapshot:        Whether this source is expected to have snapshots.
    """

    source_id:                str
    last_snapshot_at:         Optional[str]
    age_hours:                Optional[float]
    ttl_hours:                float
    stale_threshold_hours:    float
    critical_threshold_hours: float
    is_within_ttl:            bool
    is_stale:                 bool
    is_critical:              bool
    status:                   FreshnessStatus
    requires_snapshot:        bool


# ── Helpers ───────────────────────────────────────────────────────────────────


def _classify_status(
    age_hours: Optional[float],
    ttl_hours: float,
    stale_threshold_hours: float,
    critical_threshold_hours: float,
) -> FreshnessStatus:
    """Map age_hours to a FreshnessStatus.

    Args:
        age_hours:                Hours since last snapshot, or None.
        ttl_hours:                Source TTL.
        stale_threshold_hours:    Stale boundary.
        critical_threshold_hours: Critical boundary.

    Returns:
        FreshnessStatus enum member.
    """
    if age_hours is None:
        return FreshnessStatus.UNKNOWN
    if age_hours >= critical_threshold_hours:
        return FreshnessStatus.CRITICAL
    if age_hours >= stale_threshold_hours:
        return FreshnessStatus.STALE
    if age_hours >= ttl_hours:
        return FreshnessStatus.AGING
    return FreshnessStatus.FRESH


def _utcnow() -> datetime:
    return datetime.now(tz=timezone.utc)


def _query_last_snapshot_at(
    conn: sqlite3.Connection,
    source_id: str,
    realm_slug: Optional[str],
) -> Optional[str]:
    """Query ingestion_snapshots for the most recent successful snapshot.

    Args:
        conn:       Open SQLite connection.
        source_id:  Source identifier to query.
        realm_slug: Optional realm filter (None = all realms).

    Returns:
        ISO-8601 string of the most recent fetched_at, or None if no rows.
    """
    if realm_slug:
        row = conn.execute(
            """
            SELECT MAX(fetched_at)
            FROM ingestion_snapshots
            WHERE source = ? AND success = 1
            """,
            (source_id, ),
        ).fetchone()
    else:
        row = conn.execute(
            """
            SELECT MAX(fetched_at)
            FROM ingestion_snapshots
            WHERE source = ? AND success = 1
            """,
            (source_id,),
        ).fetchone()

    if row and row[0]:
        return str(row[0])
    return None


def _compute_age_hours(last_snapshot_at: Optional[str]) -> Optional[float]:
    """Compute hours elapsed since last_snapshot_at (UTC).

    Args:
        last_snapshot_at: ISO-8601 UTC string, or None.

    Returns:
        Float hours, or None if no timestamp.
    """
    if last_snapshot_at is None:
        return None
    try:
        ts = datetime.fromisoformat(last_snapshot_at.replace("Z", "+00:00"))
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        delta = _utcnow() - ts
        return delta.total_seconds() / 3600.0
    except (ValueError, TypeError):
        return None


# ── Public API ────────────────────────────────────────────────────────────────


def check_source_freshness(
    conn: sqlite3.Connection,
    source_id: str,
    policy: SourcePolicy,
    realm_slug: Optional[str] = None,
) -> FreshnessResult:
    """Check freshness of one source against its policy thresholds.

    If the source does not require snapshots (e.g. manual_event_csv), the
    function immediately returns status=UNKNOWN without querying the DB.

    Args:
        conn:       Open SQLite connection to wow_forecaster.db.
        source_id:  Source ID to check (e.g., "blizzard_api").
        policy:     SourcePolicy loaded from the registry.
        realm_slug: Optional realm to filter snapshots (None = any realm).

    Returns:
        FreshnessResult with status and age information.
    """
    fc = policy.freshness

    if not policy.provenance.requires_snapshot:
        # Manual sources — freshness cannot be determined from ingestion_snapshots.
        return FreshnessResult(
            source_id=source_id,
            last_snapshot_at=None,
            age_hours=None,
            ttl_hours=fc.ttl_hours,
            stale_threshold_hours=fc.stale_threshold_hours,
            critical_threshold_hours=fc.critical_threshold_hours,
            is_within_ttl=False,
            is_stale=False,
            is_critical=False,
            status=FreshnessStatus.UNKNOWN,
            requires_snapshot=False,
        )

    last_at  = _query_last_snapshot_at(conn, source_id, realm_slug)
    age_h    = _compute_age_hours(last_at)
    status   = _classify_status(age_h, fc.ttl_hours, fc.stale_threshold_hours, fc.critical_threshold_hours)

    return FreshnessResult(
        source_id=source_id,
        last_snapshot_at=last_at,
        age_hours=age_h,
        ttl_hours=fc.ttl_hours,
        stale_threshold_hours=fc.stale_threshold_hours,
        critical_threshold_hours=fc.critical_threshold_hours,
        is_within_ttl=(age_h is not None and age_h < fc.ttl_hours),
        is_stale=(age_h is not None and age_h >= fc.stale_threshold_hours),
        is_critical=(age_h is not None and age_h >= fc.critical_threshold_hours),
        status=status,
        requires_snapshot=True,
    )


def check_all_sources_freshness(
    conn: sqlite3.Connection,
    policies: list[SourcePolicy],
    realm_slug: Optional[str] = None,
) -> list[FreshnessResult]:
    """Check freshness for a list of source policies.

    Args:
        conn:       Open SQLite connection.
        policies:   List of SourcePolicy instances to check.
        realm_slug: Optional realm filter.

    Returns:
        List of FreshnessResult, one per policy, in input order.
    """
    return [
        check_source_freshness(conn, p.source_id, p, realm_slug)
        for p in policies
    ]
