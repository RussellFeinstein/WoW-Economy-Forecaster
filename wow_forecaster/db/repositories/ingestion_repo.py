"""
Repository for ingestion snapshot metadata.

``IngestionSnapshot`` tracks every raw API fetch attempt — successful or failed —
with the path to the saved JSON file, a content hash for deduplication, and a
link back to the pipeline ``RunMetadata`` record.

Deduplication note:
  ``content_hash`` allows callers to detect when a fetch returned identical data
  to the previous run (AH data often doesn't change between hourly fetches).
  Check ``get_latest_by_source()`` before processing if you want to skip no-ops.
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from wow_forecaster.db.repositories.base import BaseRepository

logger = logging.getLogger(__name__)


@dataclass
class IngestionSnapshot:
    """Metadata for a single raw API snapshot saved to disk.

    Attributes:
        snapshot_id: Auto-assigned DB PK; ``None`` before insertion.
        run_id: FK to ``run_metadata.run_id``. ``None`` if run hadn't been
            persisted yet when the snapshot was recorded.
        source: Provider name (``"undermine"``, ``"blizzard_api"``,
            ``"blizzard_news"``).
        endpoint: Specific API endpoint or method called.
        snapshot_path: Relative or absolute path to the JSON file on disk.
            Empty string for failed fetches where no file was written.
        content_hash: SHA-256 of the envelope JSON for deduplication.
        record_count: Number of records in the payload (``len(data)``).
        success: ``True`` if the fetch and save succeeded.
        error_message: Exception message for failed fetches; ``None`` on success.
        fetched_at: UTC timestamp of the fetch attempt.
    """

    snapshot_id: Optional[int]
    run_id: Optional[int]
    source: str
    endpoint: str
    snapshot_path: str
    content_hash: Optional[str]
    record_count: int
    success: bool
    error_message: Optional[str]
    fetched_at: datetime


class IngestionSnapshotRepository(BaseRepository):
    """Read/write access to the ``ingestion_snapshots`` table."""

    def insert(self, snapshot: IngestionSnapshot) -> int:
        """Insert an ingestion snapshot record.

        Args:
            snapshot: The :class:`IngestionSnapshot` to persist.

        Returns:
            The newly assigned ``snapshot_id``.
        """
        self.execute(
            """
            INSERT INTO ingestion_snapshots (
                run_id, source, endpoint, snapshot_path,
                content_hash, record_count, success, error_message, fetched_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            (
                snapshot.run_id,
                snapshot.source,
                snapshot.endpoint,
                snapshot.snapshot_path,
                snapshot.content_hash,
                snapshot.record_count,
                int(snapshot.success),
                snapshot.error_message,
                snapshot.fetched_at.isoformat(),
            ),
        )
        return self.last_insert_rowid()

    def get_by_source(self, source: str, limit: int = 50) -> list[IngestionSnapshot]:
        """Fetch recent snapshots for a given source, newest first.

        Args:
            source: Provider name (``"undermine"``, ``"blizzard_api"``, etc.).
            limit: Maximum rows to return.

        Returns:
            List of :class:`IngestionSnapshot` objects ordered by ``fetched_at`` DESC.
        """
        rows = self.fetchall(
            """
            SELECT * FROM ingestion_snapshots
            WHERE source = ?
            ORDER BY fetched_at DESC
            LIMIT ?;
            """,
            (source, limit),
        )
        return [_row_to_snapshot(r) for r in rows]

    def get_latest_by_source(self, source: str) -> Optional[IngestionSnapshot]:
        """Fetch the most recent snapshot for a given source.

        Useful for deduplication — compare ``content_hash`` against the new
        fetch before deciding to process.

        Args:
            source: Provider name.

        Returns:
            Most recent :class:`IngestionSnapshot`, or ``None`` if no records exist.
        """
        row = self.fetchone(
            """
            SELECT * FROM ingestion_snapshots
            WHERE source = ?
            ORDER BY fetched_at DESC
            LIMIT 1;
            """,
            (source,),
        )
        return _row_to_snapshot(row) if row else None

    def get_latest_successful_by_source(self, source: str) -> Optional[IngestionSnapshot]:
        """Fetch the most recent *successful* snapshot for a given source.

        Args:
            source: Provider name.

        Returns:
            Most recent successful :class:`IngestionSnapshot`, or ``None``.
        """
        row = self.fetchone(
            """
            SELECT * FROM ingestion_snapshots
            WHERE source = ? AND success = 1
            ORDER BY fetched_at DESC
            LIMIT 1;
            """,
            (source,),
        )
        return _row_to_snapshot(row) if row else None

    def get_failed(self, limit: int = 20) -> list[IngestionSnapshot]:
        """Fetch recent failed ingestion attempts for debugging.

        Args:
            limit: Maximum rows to return.

        Returns:
            List of failed :class:`IngestionSnapshot` objects.
        """
        rows = self.fetchall(
            """
            SELECT * FROM ingestion_snapshots
            WHERE success = 0
            ORDER BY fetched_at DESC
            LIMIT ?;
            """,
            (limit,),
        )
        return [_row_to_snapshot(r) for r in rows]

    def count_by_source(self, source: str) -> int:
        """Return total snapshot count for a given source.

        Args:
            source: Provider name.

        Returns:
            Integer row count.
        """
        row = self.fetchone(
            "SELECT COUNT(*) AS n FROM ingestion_snapshots WHERE source = ?;",
            (source,),
        )
        assert row is not None
        return int(row["n"])

    def count(self) -> int:
        """Return total number of snapshot records."""
        row = self.fetchone("SELECT COUNT(*) AS n FROM ingestion_snapshots;")
        assert row is not None
        return int(row["n"])


# ── Private helper ─────────────────────────────────────────────────────────────

def _row_to_snapshot(row: sqlite3.Row) -> IngestionSnapshot:
    """Convert a ``sqlite3.Row`` from ``ingestion_snapshots`` to a dataclass."""
    return IngestionSnapshot(
        snapshot_id=row["snapshot_id"],
        run_id=row["run_id"],
        source=row["source"],
        endpoint=row["endpoint"],
        snapshot_path=row["snapshot_path"],
        content_hash=row["content_hash"],
        record_count=row["record_count"],
        success=bool(row["success"]),
        error_message=row["error_message"],
        fetched_at=datetime.fromisoformat(row["fetched_at"]),
    )
