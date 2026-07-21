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
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

# Days of slack past retention_days before the sentinel calls the pruner dead.
# The pruner runs as orchestrator step 7 after each successful ingest, so a
# healthy system never carries rows more than a few hours past the window;
# 2 days absorbs scheduling slack without hiding a real lapse.
RETENTION_GRACE_DAYS = 2

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
        last_ingest_at:   Timestamp when ingestion last landed rows for this
                          realm (MAX(ingested_at) over market_observations_raw;
                          only successful ingests insert raw rows).
        last_ingest_age_hours: Hours since last_ingest_at (None if no ingests found).
    """

    realm_slug:            str
    first_obs_date:        str | None   = None
    last_obs_date:         str | None   = None
    days_with_data:        int             = 0
    days_checked:          int             = 0
    coverage_pct:          float           = 0.0
    gap_dates:             list[str]       = field(default_factory=list)
    last_ingest_at:        str | None   = None
    last_ingest_age_hours: float | None = None


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
        lock_age_hours:   Age of the hourly lock file (None when no lock file
                          exists or no lock path was given).
        lock_is_stale:    True when the lock is older than ``lock_stale_minutes``
                          — the hourly pipeline is wedged or crashed mid-run.
        lock_stale_minutes: Threshold used to compute ``lock_is_stale``.
        oldest_raw_age_days: Age of the oldest market_observations_raw row by
                          observed_at (None when the table is empty).
        retention_violation: True when the oldest raw row is older than
                          ``retention_limit_days`` — the pruner has stopped
                          deleting rows the 30-day API ToS §2.r window requires.
        retention_limit_days: retention_days + grace; the age that flips
                          ``retention_violation``.
    """

    generated_at:           str
    realms:                 list[RealmHealthStats] = field(default_factory=list)
    last_hourly_run:        str | None          = None
    last_hourly_status:     str | None          = None
    last_hourly_age_hours:  float | None        = None
    last_forecast_run:      str | None          = None
    last_forecast_age_hours: float | None       = None
    item_forecast_count:    int                    = 0
    is_stale:               bool                   = False
    stale_threshold_hours:  float                  = 4.0
    lock_age_hours:         float | None        = None
    lock_is_stale:          bool                   = False
    lock_stale_minutes:     float                  = 180.0
    oldest_raw_age_days:    float | None        = None
    retention_violation:    bool                   = False
    retention_limit_days:   int                    = 30 + RETENTION_GRACE_DAYS

    @property
    def has_failures(self) -> bool:
        """True when any check failed — the single source for the exit code."""
        return self.is_stale or self.lock_is_stale or self.retention_violation


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
            dt = dt.replace(tzinfo=UTC)
        return (datetime.now(tz=UTC) - dt).total_seconds() / 3600.0
    except (ValueError, OverflowError):
        return None


def _collect_realm_stats(
    conn:          sqlite3.Connection,
    realm_slug:    str,
    lookback_days: int,
    as_of:         date,
) -> RealmHealthStats:
    """Build RealmHealthStats for one realm by querying the DB.

    ``as_of`` anchors the coverage window; observations carry UTC timestamps,
    so the anchor must be a UTC date rather than the local calendar date.
    """
    stats = RealmHealthStats(realm_slug=realm_slug)

    # ── Observation date range ────────────────────────────────────────────────
    # Two single-aggregate queries, DATE() outside the aggregate: SQLite's
    # one-probe min/max optimization needs a bare column and exactly one
    # MIN/MAX per query, and idx_obs_norm_realm_outlier_time then serves each
    # via a single seek instead of a scan (issue #59). observed_at is
    # zero-padded ISO-8601 text, so the lexicographic MIN/MAX is the
    # chronological one and DATE(MIN(x)) == MIN(DATE(x)).
    first_row = conn.execute(
        """
        SELECT DATE(MIN(observed_at)) AS d
        FROM market_observations_normalized
        WHERE realm_slug = ? AND is_outlier = 0
        """,
        (realm_slug,),
    ).fetchone()
    last_row = conn.execute(
        """
        SELECT DATE(MAX(observed_at)) AS d
        FROM market_observations_normalized
        WHERE realm_slug = ? AND is_outlier = 0
        """,
        (realm_slug,),
    ).fetchone()
    if first_row:
        stats.first_obs_date = first_row["d"]
    if last_row:
        stats.last_obs_date = last_row["d"]

    # ── Days with data in lookback window ─────────────────────────────────────
    # The predicate compares the raw column, not DATE(observed_at): a bare
    # "YYYY-MM-DD" cutoff sorts before every timestamp on that date and after
    # every timestamp on earlier dates (the zero-padded 10-char date prefix
    # dominates the comparison), so the row set is identical and
    # idx_obs_norm_realm_outlier_time can serve the range (issue #59).
    cutoff = (as_of - timedelta(days=lookback_days)).isoformat()
    rows_with_data = conn.execute(
        """
        SELECT DISTINCT DATE(observed_at) AS obs_date
        FROM market_observations_normalized
        WHERE realm_slug  = ?
          AND is_outlier  = 0
          AND observed_at >= ?
        ORDER BY obs_date
        """,
        (realm_slug, cutoff),
    ).fetchall()

    dates_with_data = {r["obs_date"] for r in rows_with_data}
    stats.days_with_data = len(dates_with_data)
    stats.days_checked   = lookback_days

    # ── Gap detection ─────────────────────────────────────────────────────────
    all_dates = {
        (as_of - timedelta(days=i)).isoformat()
        for i in range(lookback_days)
    }
    stats.gap_dates = sorted(all_dates - dates_with_data)

    if lookback_days > 0:
        stats.coverage_pct = stats.days_with_data / lookback_days * 100.0

    # ── Last successful ingest ────────────────────────────────────────────────
    # market_observations_raw.ingested_at records when each row landed; only
    # successful ingest runs insert rows, so MAX() is the last good ingest.
    # (ingestion_snapshots has neither a realm_slug nor an ingested_at column;
    # querying it here crashed check-data-health on real DBs until issue #12.)
    snap = conn.execute(
        """
        SELECT MAX(ingested_at) AS last_ingest
        FROM market_observations_raw
        WHERE realm_slug = ?
        """,
        (realm_slug,),
    ).fetchone()
    if snap and snap["last_ingest"] is not None:
        stats.last_ingest_at        = snap["last_ingest"]
        stats.last_ingest_age_hours = _age_hours(snap["last_ingest"])

    return stats


def collect_health_report(
    conn:                  sqlite3.Connection,
    realm_slugs:           list[str],
    lookback_days:         int   = 14,
    stale_threshold_hours: float = 4.0,
    as_of:                 date | None = None,
    lock_path:             Path | str | None = None,
    lock_stale_minutes:    float = 180.0,
    retention_days:        int   = 30,
) -> HealthReport:
    """Build a full data collection health report.

    Args:
        conn:                  Open DB connection (row_factory should be sqlite3.Row).
        realm_slugs:           Realms to report on.
        lookback_days:         Days of history to check for coverage gaps (default 14).
        stale_threshold_hours: Hours beyond which data is considered stale (default 4).
        as_of:                 Anchor date for the coverage window (default: the
                               current UTC date, matching the UTC timestamps on
                               observations). Injectable for deterministic tests.
        lock_path:             Path to the hourly lock file (``data/db/.hourly.lock``).
                               None skips the lock check.  The file is only ever
                               stat'ed, never modified.
        lock_stale_minutes:    Lock age beyond which the hourly pipeline is
                               considered wedged.  Must match STALE_MINUTES in
                               run_hourly.bat (180): the same threshold that
                               triggers the stale-lock takeover there marks the
                               lock stale here.
        retention_days:        Raw-data retention window in days (pass
                               ``config.retention.raw_snapshot_days``).  The
                               sentinel flags rows older than this plus
                               :data:`RETENTION_GRACE_DAYS`.

    Returns:
        :class:`HealthReport` with all findings populated.
    """
    if as_of is None:
        as_of = datetime.now(tz=UTC).date()
    now_iso = datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    report  = HealthReport(
        generated_at          = now_iso,
        stale_threshold_hours = stale_threshold_hours,
        lock_stale_minutes    = lock_stale_minutes,
        retention_limit_days  = retention_days + RETENTION_GRACE_DAYS,
    )

    # ── Per-realm stats ───────────────────────────────────────────────────────
    for realm in realm_slugs:
        stats = _collect_realm_stats(conn, realm, lookback_days, as_of)
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

    # ── Hourly lock check (read-only stat; issue #5) ──────────────────────────
    # A lock older than the takeover threshold means the hourly pipeline is
    # wedged or crashed mid-run — the condition behind the 96-day outage.
    if lock_path is not None:
        try:
            mtime = Path(lock_path).stat().st_mtime
            age_hours = (
                datetime.now(tz=UTC) - datetime.fromtimestamp(mtime, tz=UTC)
            ).total_seconds() / 3600.0
            report.lock_age_hours = age_hours
            report.lock_is_stale  = age_hours * 60.0 > lock_stale_minutes
        except OSError:
            # No lock (or it vanished mid-check) is the healthy between-runs
            # state, not an error.
            pass

    # ── Retention sentinel (issue #5) ─────────────────────────────────────────
    # observed_at, not ingested_at: the pruner deletes on observed_at, so this
    # asks "is there a row the pruner should have deleted?".  Catches a dead
    # pruner before it becomes an API ToS §2.r lapse.
    row = conn.execute(
        "SELECT MIN(observed_at) AS oldest FROM market_observations_raw",
    ).fetchone()
    if row and row["oldest"] is not None:
        oldest_hours = _age_hours(row["oldest"])
        if oldest_hours is not None:
            report.oldest_raw_age_days = oldest_hours / 24.0
            report.retention_violation = (
                report.oldest_raw_age_days > report.retention_limit_days
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

    # [STALE] keeps its original meaning (data too old); [UNHEALTHY] covers a
    # failing lock or retention check when the data itself is still fresh.
    if report.is_stale:
        status_label = "[STALE]"
    elif report.has_failures:
        status_label = "[UNHEALTHY]"
    else:
        status_label = "[HEALTHY]"
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

    if report.lock_age_hours is None:
        lines.append("  Hourly lock      : none")
    else:
        lock_mark = " [STALE LOCK]" if report.lock_is_stale else ""
        lines.append(
            f"  Hourly lock      : {report.lock_age_hours:.1f}h old{lock_mark}"
        )

    if report.oldest_raw_age_days is None:
        lines.append("  Oldest raw row   : none")
    else:
        ret_mark = " [RETENTION VIOLATION]" if report.retention_violation else ""
        lines.append(
            f"  Oldest raw row   : {report.oldest_raw_age_days:.1f}d old "
            f"(limit {report.retention_limit_days}d){ret_mark}"
        )
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
