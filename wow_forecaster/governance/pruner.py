"""
Raw data retention pruner — API ToS §2.r compliance.

Blizzard Developer API Terms of Service §2.r requires that data obtained
through the API be deleted within 30 days of acquisition.  This module
enforces that requirement by pruning:

  1. Raw JSON snapshot files on disk
     (``data/raw/snapshots/blizzard_api/YYYY/MM/DD/``)
  2. ``market_observations_raw`` rows in SQLite
     (pruning normalised rows first to satisfy the FK constraint)

Normalised observations are pruned together with their parent raw rows: they
are FK children of ``market_observations_raw``, and keeping per-observation
API data past the TTL would defeat the ToS requirement. Durable derived
artefacts (daily rollup tables, Parquet features, model weights) are NOT
pruned; they are the layer that survives the 30-day window.

The cutoff is a pure calendar date derived from UTC, the same clock that
names the snapshot path dates (YYYY/MM/DD).

Usage
-----
::

    from wow_forecaster.governance.pruner import SnapshotPruner

    pruner = SnapshotPruner(
        raw_dir="data/raw",
        db_path="data/db/wow_forecaster.db",
        retention_days=30,
    )
    result = pruner.prune(dry_run=False)
    print(result)
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)


# Rows deleted before a run stops and leaves the rest for the next one.
#
# The cutoff is a calendar date, so the first run after UTC midnight sees a
# whole day of rows become prunable at once (around 6M at current capture
# volume). Deleting that in a single hourly run does not fit the run budget, so
# the work spreads over the following runs instead. At 24 runs a day this has
# several times the throughput needed to keep up.
MAX_ROWS_PER_RUN = 1_500_000

# Hour slices walked in one run, whether or not they hold rows. Only a guard
# against an absurd oldest timestamp (a bad backfill, a clock error) turning
# the walk from the oldest row to the cutoff into an endless one. Empty slices
# are nearly free, which is why the real limit above counts rows.
MAX_SLICES_PER_RUN = 2160  # 90 days


def hour_slices(oldest_iso: str, cutoff_iso: str) -> list[tuple[str, str]]:
    """Half-open ``[start, end)`` hour ranges covering rows older than the cutoff.

    Pure over strings, so the boundary rules are testable without a database.
    Boundaries are bare ``YYYY-MM-DDTHH:00:00`` with no zone suffix, which is
    what makes them comparable to both stored timestamp formats: production
    rows carry ``+00:00`` with microseconds and older rows carry ``Z``, and
    both sort after the bare boundary for the same instant.

    Comparison is on the raw column so ``idx_obs_raw_observed`` can seek. Never
    wrap ``observed_at`` in ``DATE()`` here; that defeats the seek.

    Args:
        oldest_iso: ``MIN(observed_at)`` among rows older than the cutoff.
        cutoff_iso: Exclusive upper bound, the cutoff calendar date.

    Returns:
        Contiguous slices from the hour containing ``oldest_iso`` up to
        ``cutoff_iso``, the last one clamped so the union is exactly the
        target set. Empty when nothing precedes the cutoff.
    """
    # Parse only the YYYY-MM-DDTHH prefix: format-agnostic across both stored
    # shapes, and the boundaries are label strings rather than instants.
    start = datetime.strptime(oldest_iso[:13], "%Y-%m-%dT%H")

    slices: list[tuple[str, str]] = []
    while len(slices) < MAX_SLICES_PER_RUN:
        start_str = start.strftime("%Y-%m-%dT%H:00:00")
        if start_str >= cutoff_iso:
            break
        nxt = start + timedelta(hours=1)
        slices.append((start_str, min(nxt.strftime("%Y-%m-%dT%H:00:00"), cutoff_iso)))
        start = nxt

    return slices


# ── Result type ───────────────────────────────────────────────────────────────


@dataclass
class PruneResult:
    """Summary of one prune run.

    Attributes:
        cutoff_date:    Observations *older than* this date were targeted.
        dry_run:        If True, no files or rows were deleted.
        files_deleted:  Number of raw JSON snapshot files deleted (or that
                        would be deleted in dry-run mode).
        dirs_removed:   Number of now-empty YYYY/MM/DD directories removed.
        raw_rows_deleted:  Rows deleted from ``market_observations_raw``.
        norm_rows_deleted: Rows deleted from ``market_observations_normalized``
                           to satisfy the FK constraint before pruning raw.
        errors:         Non-fatal errors encountered during the run.
    """

    cutoff_date:       date
    dry_run:           bool
    files_deleted:     int = 0
    dirs_removed:      int = 0
    raw_rows_deleted:  int = 0
    norm_rows_deleted: int = 0
    errors:            list[str] = field(default_factory=list)

    def __str__(self) -> str:
        mode = "[DRY RUN] " if self.dry_run else ""
        return (
            f"{mode}Pruned data older than {self.cutoff_date}: "
            f"files={self.files_deleted}, dirs_removed={self.dirs_removed}, "
            f"raw_rows={self.raw_rows_deleted}, norm_rows={self.norm_rows_deleted}"
            + (f", errors={len(self.errors)}" if self.errors else "")
        )


# ── Pruner ────────────────────────────────────────────────────────────────────


class SnapshotPruner:
    """Prunes raw API data that has exceeded the retention window.

    Args:
        raw_dir:        Root of the raw snapshot directory tree
                        (e.g. ``data/raw``).  The pruner looks under
                        ``{raw_dir}/snapshots/blizzard_api/``.
        db_path:        Path to the SQLite database file.
        retention_days: Number of days to retain raw data (default 30).
                        Files/rows *older* than this are deleted.
        busy_timeout_ms: SQLite busy timeout passed to the connection.
        wal_mode:       Whether to use WAL journal mode.
    """

    def __init__(
        self,
        raw_dir: str,
        db_path: str,
        retention_days: int = 30,
        busy_timeout_ms: int = 5000,
        wal_mode: bool = True,
    ) -> None:
        self.snapshots_dir    = Path(raw_dir) / "snapshots" / "blizzard_api"
        self.db_path          = db_path
        self.retention_days   = retention_days
        self.busy_timeout_ms  = busy_timeout_ms
        self.wal_mode         = wal_mode

    # ── Public API ────────────────────────────────────────────────────────────

    def prune(self, dry_run: bool = False, now: datetime | None = None) -> PruneResult:
        """Execute the retention prune.

        Deletes raw snapshot files and ``market_observations_raw`` rows
        that are older than ``retention_days`` days.

        The cutoff is the UTC calendar date ``retention_days`` before ``now``,
        matching the UTC dates encoded in snapshot paths. A file dated exactly
        ``retention_days`` ago sits on the cutoff itself and is kept.

        Args:
            dry_run: If True, report what would be deleted without deleting.
            now:     Reference clock for the cutoff (default: current UTC
                     time). Injectable so tests are deterministic at any
                     wall-clock time in any timezone.

        Returns:
            PruneResult summarising the operation.
        """
        if now is None:
            now = datetime.now(tz=UTC)
        cutoff = (now - timedelta(days=self.retention_days)).date()
        result = PruneResult(cutoff_date=cutoff, dry_run=dry_run)

        self._prune_files(result, dry_run)
        self._prune_db_rows(result, cutoff, dry_run)

        return result

    def list_stale(self) -> tuple[list[Path], int]:
        """Return (stale_files, stale_db_rows) without deleting anything.

        Equivalent to ``prune(dry_run=True)`` but returns structured data
        instead of a PruneResult.  Useful for reporting.

        Returns:
            Tuple of (list of stale file paths, count of stale DB rows).
        """
        result = self.prune(dry_run=True)
        return [], result.raw_rows_deleted  # files aren't accumulated in list form

    # ── Private helpers ───────────────────────────────────────────────────────

    def _prune_files(self, result: PruneResult, dry_run: bool) -> None:
        """Walk snapshot directory tree and delete files older than cutoff."""
        if not self.snapshots_dir.exists():
            logger.debug("Snapshot dir does not exist; nothing to prune: %s", self.snapshots_dir)
            return

        cutoff_date = result.cutoff_date

        # Directory layout: blizzard_api/YYYY/MM/DD/
        for year_dir in sorted(self.snapshots_dir.iterdir()):
            if not year_dir.is_dir():
                continue
            try:
                year = int(year_dir.name)
            except ValueError:
                continue

            for month_dir in sorted(year_dir.iterdir()):
                if not month_dir.is_dir():
                    continue
                try:
                    month = int(month_dir.name)
                except ValueError:
                    continue

                for day_dir in sorted(month_dir.iterdir()):
                    if not day_dir.is_dir():
                        continue
                    try:
                        day = int(day_dir.name)
                        dir_date = date(year, month, day)
                    except (ValueError, OverflowError):
                        continue

                    if dir_date >= cutoff_date:
                        continue  # within retention window

                    # Delete all files in this day directory
                    for f in list(day_dir.iterdir()):
                        if f.is_file():
                            if not dry_run:
                                try:
                                    f.unlink()
                                except Exception as exc:
                                    err = f"Failed to delete {f}: {exc}"
                                    logger.warning(err)
                                    result.errors.append(err)
                                    continue
                            result.files_deleted += 1
                            logger.debug(
                                "%sDeleted snapshot file: %s", "[DRY] " if dry_run else "", f
                            )

                    # Remove the now-empty day directory
                    if not dry_run and day_dir.exists() and not any(day_dir.iterdir()):
                        try:
                            day_dir.rmdir()
                            result.dirs_removed += 1
                        except Exception as exc:
                            logger.debug("Could not remove empty dir %s: %s", day_dir, exc)

                # Remove empty month dir
                if not dry_run and month_dir.exists() and not any(month_dir.iterdir()):
                    try:
                        month_dir.rmdir()
                    except Exception:
                        pass

            # Remove empty year dir
            if not dry_run and year_dir.exists() and not any(year_dir.iterdir()):
                try:
                    year_dir.rmdir()
                except Exception:
                    pass

    def _delete_slice(
        self, conn: sqlite3.Connection, start: str, end: str
    ) -> tuple[int, int]:
        """Delete one half-open hour slice, FK children first.

        Args:
            conn:  Open connection. The caller commits.
            start: Inclusive lower bound, a bare hour boundary string.
            end:   Exclusive upper bound.

        Returns:
            ``(normalized_deleted, raw_deleted)`` for this slice.
        """
        norm_deleted = conn.execute(
            """
            DELETE FROM market_observations_normalized
            WHERE obs_id IN (
                SELECT obs_id FROM market_observations_raw
                WHERE observed_at >= ? AND observed_at < ?
            );
            """,
            (start, end),
        ).rowcount

        raw_deleted = conn.execute(
            """
            DELETE FROM market_observations_raw
            WHERE observed_at >= ? AND observed_at < ?;
            """,
            (start, end),
        ).rowcount

        return norm_deleted, raw_deleted

    def _prune_db_rows(
        self, result: PruneResult, cutoff: date, dry_run: bool
    ) -> None:
        """Delete stale rows from market_observations_raw (and normalized FK children).

        Deletes in half-open hour slices with a commit per slice. Interruptibility
        is the point: a killed prune keeps the slices it finished and the next run
        resumes from the new oldest row. The single-transaction version this
        replaced discarded a full day of work when it was killed (issue #149).

        Both tables are seekable for this: ``idx_obs_raw_observed`` for the range
        and ``idx_obs_norm_obs_id`` for the child lookup and the FK parent check.
        Without the latter each parent delete scans the whole child table, which
        is what made the prune unable to finish at all.
        """
        cutoff_iso = cutoff.isoformat()

        try:
            from wow_forecaster.db.connection import get_connection

            with get_connection(
                self.db_path,
                wal_mode=self.wal_mode,
                busy_timeout_ms=self.busy_timeout_ms,
            ) as conn:
                (oldest,) = conn.execute(
                    """
                    SELECT MIN(observed_at) FROM market_observations_raw
                    WHERE observed_at < ?;
                    """,
                    (cutoff_iso,),
                ).fetchone()

                if oldest is None:
                    logger.debug("No stale market_observations_raw rows to prune.")
                    return

                slices = hour_slices(oldest, cutoff_iso)
                if len(slices) == MAX_SLICES_PER_RUN:
                    logger.warning(
                        "Prune slice walk hit its %d slice guard. Oldest row %s is "
                        "far behind cutoff %s; check for a bad timestamp. The next "
                        "run resumes from whatever is oldest then.",
                        MAX_SLICES_PER_RUN, oldest, cutoff_iso,
                    )

                if dry_run:
                    for start, end in slices:
                        (raw_n,) = conn.execute(
                            """
                            SELECT COUNT(*) FROM market_observations_raw
                            WHERE observed_at >= ? AND observed_at < ?;
                            """,
                            (start, end),
                        ).fetchone()
                        (norm_n,) = conn.execute(
                            """
                            SELECT COUNT(*) FROM market_observations_normalized
                            WHERE obs_id IN (
                                SELECT obs_id FROM market_observations_raw
                                WHERE observed_at >= ? AND observed_at < ?
                            );
                            """,
                            (start, end),
                        ).fetchone()
                        result.raw_rows_deleted  += raw_n
                        result.norm_rows_deleted += norm_n

                    logger.info(
                        "[DRY RUN] Would delete %d raw + %d normalized rows older "
                        "than %s across %d hour slices",
                        result.raw_rows_deleted, result.norm_rows_deleted,
                        cutoff_iso, len(slices),
                    )
                    return

                slices_done = 0
                for start, end in slices:
                    if result.raw_rows_deleted >= MAX_ROWS_PER_RUN:
                        logger.info(
                            "Prune stopped at the %d row budget after %d of %d hour "
                            "slices. Rows before %s remain and the next run continues "
                            "from there.",
                            MAX_ROWS_PER_RUN, slices_done, len(slices), start,
                        )
                        break

                    norm_n, raw_n = self._delete_slice(conn, start, end)
                    conn.commit()
                    result.norm_rows_deleted += norm_n
                    result.raw_rows_deleted  += raw_n
                    slices_done += 1

                logger.info(
                    "Pruned %d raw + %d normalized rows older than %s across %d "
                    "hour slices",
                    result.raw_rows_deleted, result.norm_rows_deleted,
                    cutoff_iso, slices_done,
                )

        except Exception as exc:
            # Counts are not reset: slices commit independently, so whatever is
            # already counted is already deleted and durable.
            err = f"DB prune failed: {exc}"
            logger.error(err, exc_info=True)
            result.errors.append(err)
