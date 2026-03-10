"""Data collection health reporting.

Queries the DB directly (not output JSON files) to give an authoritative
view of collection status and gap detection.

Exported functions
------------------
``collect_health_report(conn, realm_slugs, lookback_days)``
    Returns a :class:`HealthReport` with all findings.

``format_health_report(report)``
    Formats the report as an ASCII terminal summary.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Optional


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class RealmHealthStats:
    """Per-realm data collection statistics.

    Attributes:
        realm_slug:       Realm identifier.
        first_obs_date:   Earliest observation date in the DB for this realm.
        last_obs_date:    Most recent observation date.
        days_with_data:   Distinct calendar days that have at least one observation.
        days_checked:     Number of calendar days in the lookback window.
        coverage_pct:     ``days_with_data / days_checked * 100``.
        gap_dates:        UTC calendar dates within the lookback window that have
                          zero observations.  Sorted ascending.
        last_ingest_at:   Timestamp of the most recent successful ingest snapshot.
        last_ingest_age_hours: Hours since last_ingest_at (None if no ingests found).
    """

    realm_slug:            str
    first_obs_date:        Optional[str]   = None
    last_obs_date:         Optional[str]   = None
    days_with_data:        int             = 0
    days_checked:          int             = 0
    coverage_pct:          float           = 0.0
    gap_dates:             list[str]       = field(default_factory=list)
    last_ingest_at:        Optional[str]   = None
    last_ingest_age_hours: Optional[float] = None


@dataclass
class HealthReport:
    """Full data collection health report.

    Attributes:
        generated_at:     UTC ISO timestamp when this report was built.
        realms:           Per-realm statistics (one entry per queried realm).
        last_hourly_run:  Timestamp of the most recent orchestrator run with
                          status 'success' or 'partial'.
        last_hourly_status: Status value of that run.
        last_hourly_age_hours: Hours since last_hourly_run (None if no runs).
        last_forecast_run: Timestamp of the most recent successful daily forecast run.
        last_forecast_age_hours: Hours since last_forecast_run.
        item_forecast_count: Distinct item_ids with item-level forecasts in the DB.
        is_stale:         True when last successful ingest is older than
                          ``stale_threshold_hours`` for any realm.
        stale_threshold_hours: Threshold used to compute ``is_stale``.
    """

    generated_at:           str
    realms:                 list[RealmHealthStats] = field(default_factory=list)
    last_hourly_run:        Optional[str]          = None
    last_hourly_status:     Optional[str]          = None
    last_hourly_age_hours:  Optional[float]        = None
    last_forecast_run:      Optional[str]          = None
    last_forecast_age_hours: Optional[float]       = None
    item_forecast_count:    int                    = 0
    is_stale:               bool                   = False
    stale_threshold_hours:  float                  = 4.0


# ── Query helpers ─────────────────────────────────────────────────────────────

def _age_hours(ts_iso: str | None) -> float | None:
    """Return hours between an ISO timestamp and now (UTC).  None if ts_iso is None."""
    if not ts_iso:
        return None
    try:
        if "T" in ts_iso:
            dt = datetime.fromisoformat(ts_iso.replace("Z", "+00:00"))
        else:
            dt = datetime.fromisoformat(ts_iso + "T00:00:00+00:00")
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return (datetime.now(tz=timezone.utc) - dt).total_seconds() / 3600.0
    except (ValueError, OverflowError):
        return None


def _collect_realm_stats(
    conn:          sqlite3.Connection,
    realm_slug:    str,
    lookback_days: int,
) -> RealmHealthStats:
    """Build RealmHealthStats for one realm by querying the DB."""
    stats = RealmHealthStats(realm_slug=realm_slug)

    # ── Observation date range ────────────────────────────────────────────────
    row = conn.execute(
        """
        SELECT MIN(DATE(observed_at)) AS first,
               MAX(DATE(observed_at)) AS last
        FROM market_observations_normalized
        WHERE realm_slug = ? AND is_outlier = 0
        """,
        (realm_slug,),
    ).fetchone()
    if row:
        stats.first_obs_date = row["first"]
        stats.last_obs_date  = row["last"]

    # ── Days with data in lookback window ─────────────────────────────────────
    cutoff = (date.today() - timedelta(days=lookback_days)).isoformat()
    rows_with_data = conn.execute(
        """
        SELECT DISTINCT DATE(observed_at) AS obs_date
        FROM market_observations_normalized
        WHERE realm_slug  = ?
          AND is_outlier  = 0
          AND DATE(observed_at) >= ?
        ORDER BY obs_date
        """,
        (realm_slug, cutoff),
    ).fetchall()

    dates_with_data = {r["obs_date"] for r in rows_with_data}
    stats.days_with_data = len(dates_with_data)
    stats.days_checked   = lookback_days

    # ── Gap detection ─────────────────────────────────────────────────────────
    all_dates = {
        (date.today() - timedelta(days=i)).isoformat()
        for i in range(lookback_days)
    }
    stats.gap_dates = sorted(all_dates - dates_with_data)

    if lookback_days > 0:
        stats.coverage_pct = stats.days_with_data / lookback_days * 100.0

    # ── Last successful ingest snapshot ───────────────────────────────────────
    snap = conn.execute(
        """
        SELECT ingested_at
        FROM ingestion_snapshots
        WHERE realm_slug = ? AND success = 1
        ORDER BY ingested_at DESC
        LIMIT 1
        """,
        (realm_slug,),
    ).fetchone()
    if snap:
        stats.last_ingest_at        = snap["ingested_at"]
        stats.last_ingest_age_hours = _age_hours(snap["ingested_at"])

    return stats


def collect_health_report(
    conn:                  sqlite3.Connection,
    realm_slugs:           list[str],
    lookback_days:         int   = 14,
    stale_threshold_hours: float = 4.0,
) -> HealthReport:
    """Build a full data collection health report.

    Args:
        conn:                  Open DB connection (row_factory should be sqlite3.Row).
        realm_slugs:           Realms to report on.
        lookback_days:         Days of history to check for coverage gaps (default 14).
        stale_threshold_hours: Hours beyond which data is considered stale (default 4).

    Returns:
        :class:`HealthReport` with all findings populated.
    """
    now_iso = datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    report  = HealthReport(
        generated_at          = now_iso,
        stale_threshold_hours = stale_threshold_hours,
    )

    # ── Per-realm stats ───────────────────────────────────────────────────────
    for realm in realm_slugs:
        stats = _collect_realm_stats(conn, realm, lookback_days)
        report.realms.append(stats)

    # ── Last orchestrator (hourly) run ────────────────────────────────────────
    row = conn.execute(
        """
        SELECT started_at, status
        FROM run_metadata
        WHERE pipeline_stage = 'orchestrator'
          AND status IN ('success', 'partial')
        ORDER BY started_at DESC
        LIMIT 1
        """,
    ).fetchone()
    if row:
        report.last_hourly_run       = row["started_at"]
        report.last_hourly_status    = row["status"]
        report.last_hourly_age_hours = _age_hours(row["started_at"])

    # ── Last daily forecast run ───────────────────────────────────────────────
    row = conn.execute(
        """
        SELECT started_at
        FROM run_metadata
        WHERE pipeline_stage = 'recommend'
          AND status = 'success'
        ORDER BY started_at DESC
        LIMIT 1
        """,
    ).fetchone()
    if row:
        report.last_forecast_run        = row["started_at"]
        report.last_forecast_age_hours  = _age_hours(row["started_at"])

    # ── Item-level forecast coverage ──────────────────────────────────────────
    fc_row = conn.execute(
        """
        SELECT COUNT(DISTINCT item_id) AS cnt
        FROM forecast_outputs
        WHERE item_id IS NOT NULL
        """,
    ).fetchone()
    if fc_row:
        report.item_forecast_count = fc_row["cnt"] or 0

    # ── Staleness check ───────────────────────────────────────────────────────
    report.is_stale = any(
        (s.last_ingest_age_hours is None or
         s.last_ingest_age_hours > stale_threshold_hours)
        for s in report.realms
    )

    return report


# ── Formatter ─────────────────────────────────────────────────────────────────

def format_health_report(report: HealthReport) -> str:
    """Format a HealthReport as a human-readable ASCII table.

    Args:
        report: Report from :func:`collect_health_report`.

    Returns:
        Multi-line string suitable for ``typer.echo()``.
    """
    lines: list[str] = []
    lines.append("")
    lines.append("  Data Collection Health")
    lines.append(f"  Generated : {report.generated_at}")

    status_label = "[STALE]" if report.is_stale else "[HEALTHY]"
    lines.append(f"  Status    : {status_label}")
    lines.append("")

    # ── Pipeline runs ─────────────────────────────────────────────────────────
    def _age_str(hours: float | None) -> str:
        if hours is None:
            return "never"
        if hours < 1:
            return f"{hours * 60:.0f}m ago"
        return f"{hours:.1f}h ago"

    hourly_ts  = report.last_hourly_run or "none"
    hourly_age = _age_str(report.last_hourly_age_hours)
    hourly_tag = f"[{report.last_hourly_status}]" if report.last_hourly_status else ""
    lines.append(f"  Last hourly run  : {hourly_ts}  ({hourly_age}) {hourly_tag}")

    fc_ts  = report.last_forecast_run or "none"
    fc_age = _age_str(report.last_forecast_age_hours)
    lines.append(f"  Last forecast run: {fc_ts}  ({fc_age})")
    lines.append(f"  Item-level items : {report.item_forecast_count:,}")
    lines.append("")

    # ── Per-realm detail ──────────────────────────────────────────────────────
    for stats in report.realms:
        lines.append(f"  Realm: {stats.realm_slug}")
        lines.append(f"    First obs        : {stats.first_obs_date or 'none'}")
        lines.append(f"    Last obs         : {stats.last_obs_date or 'none'}")
        lines.append(
            f"    Coverage ({report.stale_threshold_hours:.0f}h check): "
            f"{stats.days_with_data} / {stats.days_checked} days "
            f"({stats.coverage_pct:.0f}%)"
        )
        ingest_ts  = stats.last_ingest_at or "none"
        ingest_age = _age_str(stats.last_ingest_age_hours)
        stale_mark = " [STALE]" if (
            stats.last_ingest_age_hours is None or
            stats.last_ingest_age_hours > report.stale_threshold_hours
        ) else ""
        lines.append(
            f"    Last ingest      : {ingest_ts}  ({ingest_age}){stale_mark}"
        )
        if stats.gap_dates:
            gap_preview = ", ".join(stats.gap_dates[:7])
            if len(stats.gap_dates) > 7:
                gap_preview += f" ... +{len(stats.gap_dates) - 7} more"
            lines.append(f"    Gaps (last {stats.days_checked}d)  : {gap_preview}")
        else:
            lines.append(f"    Gaps (last {stats.days_checked}d)  : none detected")
        lines.append("")

    return "\n".join(lines)
