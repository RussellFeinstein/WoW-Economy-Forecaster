"""
Provenance tracking for the WoW Economy Forecaster hourly refresh.

Records which data sources contributed fresh data, how many records each
contributed in the last 24 hours, and whether any source is stale.

This is built from the ``ingestion_snapshots`` table, which is populated
by ``IngestStage`` (fixture mode or real API).

A source is considered "stale" if no successful snapshot was recorded in
the last ``stale_threshold_hours`` hours (default: 25h, slightly more than
one hourly cycle to account for scheduler jitter).
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

logger = logging.getLogger(__name__)

_KNOWN_SOURCES         = ("undermine", "blizzard_api", "blizzard_news")
_DEFAULT_STALE_HOURS   = 25
_DEFAULT_LOOKBACK_HOURS = 25


@dataclass(frozen=True)
class SourceProvenance:
    """Provenance record for a single data source.

    Attributes:
        source:              Source name ("undermine", "blizzard_api", etc.).
        last_snapshot_at:    ISO-8601 timestamp of the most recent snapshot
                             (None if no snapshot ever recorded).
        snapshot_count_24h:  Number of successful snapshots in lookback window.
        total_records_24h:   Sum of record_count across those snapshots.
        success_rate_24h:    Fraction of snapshots in window that succeeded.
        is_stale:            True if last_snapshot_at is None or older than
                             stale_threshold_hours.
    """

    source:              str
    last_snapshot_at:    Optional[str]
    snapshot_count_24h:  int
    total_records_24h:   int
    success_rate_24h:    float
    is_stale:            bool


@dataclass(frozen=True)
class ProvenanceSummary:
    """Provenance summary for the most recent hourly refresh.

    Attributes:
        realm_slug:       Realm this provenance summary covers.
        checked_at:       ISO-8601 UTC timestamp.
        sources:          Per-source provenance records.
        freshness_hours:  Hours since the most recent snapshot across all
                          sources.  None if no snapshots recorded.
        is_fresh:         True if at least one source has a non-stale snapshot.
    """

    realm_slug:      str
    checked_at:      str
    sources:         list[SourceProvenance]
    freshness_hours: Optional[float]
    is_fresh:        bool


def build_provenance_summary(
    conn: sqlite3.Connection,
    realm_slug: str,
    lookback_hours: int = _DEFAULT_LOOKBACK_HOURS,
    stale_threshold_hours: int = _DEFAULT_STALE_HOURS,
) -> ProvenanceSummary:
    """Build a provenance summary from the ingestion_snapshots table.

    Args:
        conn:                   Open SQLite connection.
        realm_slug:             Realm to report on (used for logging only;
                                snapshots may not be realm-keyed).
        lookback_hours:         How many hours back to look for snapshots.
        stale_threshold_hours:  Hours before a source is considered stale.

    Returns:
        ProvenanceSummary.
    """
    from wow_forecaster.monitoring.drift import _utc_now_iso

    now_str  = _utc_now_iso()
    now_dt   = datetime.now(tz=timezone.utc)
    cutoff   = (now_dt - timedelta(hours=lookback_hours)).strftime("%Y-%m-%dT%H:%M:%SZ")
    stale_dt = now_dt - timedelta(hours=stale_threshold_hours)

    sources: list[SourceProvenance] = []
    most_recent_dt: Optional[datetime] = None

    for source in _KNOWN_SOURCES:
        try:
            rows = conn.execute(
                """
                SELECT fetched_at, success, record_count
                FROM ingestion_snapshots
                WHERE source >= ?
                  AND source <= ?
                  AND fetched_at >= ?
                ORDER BY fetched_at DESC;
                """,
                # Use prefix range to match source names like "undermine", "blizzard_api"
                (source, source + "\xff", cutoff),
            ).fetchall()
        except Exception as exc:
            logger.warning("Provenance query failed for source=%s: %s", source, exc)
            rows = []

        # Fallback: also query by exact source name match
        try:
            rows_exact = conn.execute(
                """
                SELECT fetched_at, success, record_count
                FROM ingestion_snapshots
                WHERE source = ?
                  AND fetched_at >= ?
                ORDER BY fetched_at DESC;
                """,
                (source, cutoff),
            ).fetchall()
        except Exception:
            rows_exact = []

        # Merge and deduplicate by fetched_at
        seen: set[str] = set()
        merged: list = []
        for r in list(rows) + list(rows_exact):
            k = r["fetched_at"]
            if k not in seen:
                seen.add(k)
                merged.append(r)

        n_total   = len(merged)
        n_success = sum(1 for r in merged if r["success"])
        total_recs = sum(r["record_count"] or 0 for r in merged if r["success"])
        success_rate = n_success / n_total if n_total > 0 else 0.0

        # Most recent snapshot for this source
        last_ts: Optional[str] = None
        if merged:
            last_ts = merged[0]["fetched_at"]
            try:
                last_dt = datetime.strptime(last_ts, "%Y-%m-%dT%H:%M:%SZ").replace(
                    tzinfo=timezone.utc
                )
                if most_recent_dt is None or last_dt > most_recent_dt:
                    most_recent_dt = last_dt
            except ValueError:
                pass

        # Determine staleness
        is_stale = True
        if last_ts is not None:
            try:
                last_dt = datetime.strptime(last_ts, "%Y-%m-%dT%H:%M:%SZ").replace(
                    tzinfo=timezone.utc
                )
                is_stale = last_dt < stale_dt
            except ValueError:
                pass

        sources.append(SourceProvenance(
            source=source,
            last_snapshot_at=last_ts,
            snapshot_count_24h=n_success,
            total_records_24h=total_recs,
            success_rate_24h=round(success_rate, 4),
            is_stale=is_stale,
        ))

    # Freshness: hours since the most recent snapshot across any source
    freshness_hours: Optional[float] = None
    if most_recent_dt is not None:
        freshness_hours = round((now_dt - most_recent_dt).total_seconds() / 3600.0, 2)

    is_fresh = any(not s.is_stale for s in sources)

    logger.info(
        "Provenance | realm=%s | fresh=%s | freshness=%s",
        realm_slug, is_fresh,
        f"{freshness_hours:.1f}h ago" if freshness_hours is not None else "never",
    )

    return ProvenanceSummary(
        realm_slug=realm_slug,
        checked_at=now_str,
        sources=sources,
        freshness_hours=freshness_hours,
        is_fresh=is_fresh,
    )
