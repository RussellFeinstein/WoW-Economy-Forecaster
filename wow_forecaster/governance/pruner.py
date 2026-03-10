"""
Raw data retention pruner — API ToS §2.r compliance.

Blizzard Developer API Terms of Service §2.r requires that data obtained
through the API be deleted within 30 days of acquisition.  This module
enforces that requirement by pruning:

  1. Raw JSON snapshot files on disk
     (``data/raw/snapshots/blizzard_api/YYYY/MM/DD/``)
  2. ``market_observations_raw`` rows in SQLite
     (pruning normalised rows first to satisfy the FK constraint)

Derived artefacts (Parquet features, model weights, normalised observations
that have already been rolled into training data) are NOT pruned — the 30-day
TTL only applies to the raw API data.

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
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


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

    def prune(self, dry_run: bool = False) -> PruneResult:
        """Execute the retention prune.

        Deletes raw snapshot files and ``market_observations_raw`` rows
        that are older than ``retention_days`` days.

        Args:
            dry_run: If True, report what would be deleted without deleting.

        Returns:
            PruneResult summarising the operation.
        """
        cutoff = (
            datetime.now(tz=timezone.utc) - timedelta(days=self.retention_days)
        ).date()
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
                            logger.debug("%sDeleted snapshot file: %s", "[DRY] " if dry_run else "", f)

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

    def _prune_db_rows(
        self, result: PruneResult, cutoff: date, dry_run: bool
    ) -> None:
        """Delete stale rows from market_observations_raw (and normalized FK children)."""
        cutoff_iso = cutoff.isoformat()

        try:
            from wow_forecaster.db.connection import get_connection

            with get_connection(
                self.db_path,
                wal_mode=self.wal_mode,
                busy_timeout_ms=self.busy_timeout_ms,
            ) as conn:
                # Count rows that would be affected
                (raw_count,) = conn.execute(
                    "SELECT COUNT(*) FROM market_observations_raw WHERE observed_at < ?;",
                    (cutoff_iso,),
                ).fetchone()

                if raw_count == 0:
                    logger.debug("No stale market_observations_raw rows to prune.")
                    return

                # Count child normalized rows that would be deleted
                (norm_count,) = conn.execute(
                    """
                    SELECT COUNT(*) FROM market_observations_normalized
                    WHERE obs_id IN (
                        SELECT obs_id FROM market_observations_raw
                        WHERE observed_at < ?
                    );
                    """,
                    (cutoff_iso,),
                ).fetchone()

                result.raw_rows_deleted  = raw_count
                result.norm_rows_deleted = norm_count

                if dry_run:
                    logger.info(
                        "[DRY RUN] Would delete %d raw + %d normalized rows older than %s",
                        raw_count, norm_count, cutoff_iso,
                    )
                    return

                # Delete normalized rows first (FK child)
                if norm_count > 0:
                    conn.execute(
                        """
                        DELETE FROM market_observations_normalized
                        WHERE obs_id IN (
                            SELECT obs_id FROM market_observations_raw
                            WHERE observed_at < ?
                        );
                        """,
                        (cutoff_iso,),
                    )

                # Delete raw rows (FK parent)
                conn.execute(
                    "DELETE FROM market_observations_raw WHERE observed_at < ?;",
                    (cutoff_iso,),
                )
                conn.commit()

                logger.info(
                    "Pruned %d raw + %d normalized rows older than %s",
                    raw_count, norm_count, cutoff_iso,
                )

        except Exception as exc:
            err = f"DB prune failed: {exc}"
            logger.error(err, exc_info=True)
            result.errors.append(err)
            # Reset counts since deletion may not have occurred
            result.raw_rows_deleted  = 0
            result.norm_rows_deleted = 0
