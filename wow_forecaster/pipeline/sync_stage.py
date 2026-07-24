"""
SyncSnapshotsStage — drain cloud-captured snapshots into the local database.

The desktop half of issue #43.  ``cloud_fetch`` uploads an hourly commodities
snapshot to R2 from GitHub Actions; this stage lists what is there, skips
everything already covered locally, and runs the rest through the same ingest
path a live fetch uses, then normalizes and rolls up.

Why it exists: the commodities endpoint serves only the *current* snapshot, so
an hour missed while the machine slept is unrecoverable from the API.  Draining
the bucket is the only way those hours reach the database.

Design notes:

- **Object selection is pure and lives in** :mod:`wow_forecaster.ingestion.cloud_sync`.
  This module holds the database work only.
- **Idempotent.**  Re-running inserts nothing: already-ingested objects are
  filtered by local snapshot path, and hours the desktop captured itself are
  filtered by UTC hour.
- **Three-phase connections**, as in ``IngestStage``: a short read connection for
  the item FK guard, no connection held during downloads, and a short write
  connection per object.
- **Partial failure is per object.**  A corrupt or unreadable object is logged,
  recorded as a failure, and skipped; it is never marked ingested, so the next
  run retries it.  One bad object does not abandon the rest of the backlog.
- **Holds the hourly lock** for the write phase so a catch-up run and an
  overrunning hourly run never write concurrently.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any

from wow_forecaster.ingestion.cloud_sync import (
    SyncResult,
    download_snapshot,
    hourly_lock,
    list_objects_since,
    make_s3_client,
    resolve_s3_env,
    select_objects_to_ingest,
)
from wow_forecaster.models.meta import RunMetadata
from wow_forecaster.pipeline.base import PipelineStage
from wow_forecaster.pipeline.ingest import parse_blizzard_records

logger = logging.getLogger(__name__)

SOURCE = "blizzard_api"


class SyncSnapshotsStage(PipelineStage):
    """Ingest cloud-captured snapshots that the local pipeline has not seen.

    Returns the number of raw observations inserted.  The full breakdown is left
    on ``self.result`` for the CLI to report.
    """

    stage_name = "sync_snapshots"

    def __init__(self, config, db_path: str | None = None) -> None:
        super().__init__(config, db_path)
        self.result = SyncResult()
        self._s3: Any | None = None
        self._bucket: str = ""

    def _execute(
        self,
        run: RunMetadata,
        *,
        since: datetime | None = None,
        dry_run: bool = False,
        limit: int | None = None,
        now: datetime | None = None,
        s3: Any | None = None,
        env: dict[str, str] | None = None,
        **kwargs,
    ) -> int:
        """Drain the bucket into the database.

        Args:
            run:     In-progress :class:`RunMetadata` (mutable).
            since:   Oldest capture to consider.  Defaults to
                ``cloud_sync.max_backfill_days`` before ``now``, and is always
                clamped forward to the retention cutoff.
            dry_run: Report what would be ingested and touch nothing.
            limit:   Cap on objects ingested this run; overrides the config
                default.  ``0`` or negative means no cap.
            now:     Injectable clock (naive UTC) for tests.
            s3:      Injectable S3 client for tests.
            env:     Injectable resolved credentials for tests.

        Returns:
            Number of raw market observations inserted.
        """
        from wow_forecaster.utils.time_utils import utcnow

        cfg = self.config.cloud_sync
        result = self.result
        result.dry_run = dry_run

        now = now or utcnow()
        retention_cutoff = now - timedelta(
            days=self.config.retention.raw_snapshot_days
        )
        if since is None:
            since = now - timedelta(days=cfg.max_backfill_days)
        # Never look past the retention horizon: the pruner deletes those rows
        # by observed_at on the next hourly run (API ToS section 2.r).
        since = max(since, retention_cutoff)

        max_objects = cfg.max_objects_per_run if limit is None else limit
        if max_objects is not None and max_objects <= 0:
            max_objects = None

        env = env or resolve_s3_env()
        self._bucket = env["bucket"]
        self._s3 = s3 if s3 is not None else make_s3_client(env)

        logger.info(
            "Listing cloud snapshots from %s to %s (bucket=%s)",
            since.isoformat(timespec="seconds"),
            now.isoformat(timespec="seconds"),
            self._bucket,
        )
        keys = list_objects_since(self._s3, self._bucket, since, now)
        result.listed = len(keys)

        if dry_run:
            selected = self._select(keys, since, now, retention_cutoff, max_objects)
            self._log_selection(selected, since)
            return 0

        # The lock is taken before coverage is read, not after: an hourly run
        # finishing in between would otherwise cover an hour this run already
        # decided to ingest.
        lock_path = self._lock_path()
        with hourly_lock(lock_path, wait_seconds=cfg.lock_wait_seconds):
            selected = self._select(keys, since, now, retention_cutoff, max_objects)
            self._log_selection(selected, since)
            if not selected:
                return 0

            known_item_ids = self._load_known_item_ids()
            realms, dates = self._ingest_objects(selected, run, known_item_ids)

            if result.ingested == 0:
                return 0

            if self._normalize():
                self._rollup(realms, dates)

        return result.observations_inserted

    # ── Steps ─────────────────────────────────────────────────────────────────

    def _select(self, keys, since, now, retention_cutoff, max_objects):
        """Read local coverage and pick the objects to ingest."""
        from wow_forecaster.db.connection import get_connection
        from wow_forecaster.db.repositories.ingestion_repo import (
            IngestionSnapshotRepository,
        )
        from wow_forecaster.db.repositories.market_repo import (
            MarketObservationRepository,
        )

        with self._connect(get_connection) as conn:
            already = IngestionSnapshotRepository(conn).get_ingested_paths_since(
                SOURCE, since
            )
            hours = MarketObservationRepository(conn).get_covered_hours(
                SOURCE, since, now
            )

        selected, skips = select_objects_to_ingest(
            keys,
            raw_dir=self.config.data.raw_dir,
            already_ingested=already,
            hours_covered=hours,
            retention_cutoff=retention_cutoff,
            max_objects=max_objects,
        )
        self.result.selected = len(selected)
        self.result.skips = skips
        self.result.truncated = skips.over_limit > 0
        return selected

    def _load_known_item_ids(self) -> set[int]:
        """Phase 1: short read connection for the FK guard set."""
        from wow_forecaster.db.connection import get_connection
        from wow_forecaster.db.repositories.item_repo import ItemRepository

        with self._connect(get_connection) as conn:
            known = ItemRepository(conn).get_all_item_ids()
        logger.info("SyncSnapshotsStage: %d known items in registry", len(known))
        return known

    def _ingest_objects(
        self,
        selected,
        run: RunMetadata,
        known_item_ids: set[int],
    ) -> tuple[set[str], set[str]]:
        """Download and persist each object; return realms and dates touched."""
        from wow_forecaster.db.connection import get_connection
        from wow_forecaster.db.repositories.ingestion_repo import (
            IngestionSnapshot,
            IngestionSnapshotRepository,
        )
        from wow_forecaster.db.repositories.market_repo import (
            MarketObservationRepository,
        )
        from wow_forecaster.ingestion.snapshot import compute_hash

        env_bucket = self._bucket
        realms: set[str] = set()
        dates: set[str] = set()
        result = self.result

        for obj in selected:
            try:
                # Phase 2: network, no connection held.
                envelope, raw_bytes = download_snapshot(self._s3, env_bucket, obj.key)
                records = envelope["data"]

                obj.local_path.parent.mkdir(parents=True, exist_ok=True)
                obj.local_path.write_bytes(raw_bytes)

                observations, skipped_fk = parse_blizzard_records(
                    records, obj.captured_at, known_item_ids
                )

                # Phase 3: short write connection for this object's rows.
                with self._connect(get_connection) as conn:
                    IngestionSnapshotRepository(conn).insert(
                        IngestionSnapshot(
                            snapshot_id=None,
                            run_id=run.run_id,
                            source=SOURCE,
                            endpoint=f"r2://{env_bucket}/{obj.key}",
                            snapshot_path=str(obj.local_path),
                            content_hash=compute_hash(envelope),
                            record_count=len(records),
                            success=True,
                            error_message=None,
                            fetched_at=obj.captured_at,
                        )
                    )
                    inserted = MarketObservationRepository(conn).insert_raw_batch(
                        observations
                    )
                    conn.commit()

                result.ingested += 1
                result.observations_inserted += inserted
                result.items_skipped_fk += skipped_fk
                realms.update(o.realm_slug for o in observations)
                dates.add(obj.captured_at.date().isoformat())
                logger.info(
                    "Ingested %s | %d records | inserted=%d | skipped_missing_items=%d",
                    obj.key, len(records), inserted, skipped_fk,
                )

            except Exception as exc:
                # Not recorded as ingested, so the next run retries this object.
                result.failures.append((obj.key, str(exc)))
                logger.error("Failed to ingest %s: %s", obj.key, exc, exc_info=True)

        result.dates_touched = sorted(dates)
        return realms, dates

    def _normalize(self) -> bool:
        """Run NormalizeStage over the newly inserted rows.  True on success."""
        from wow_forecaster.pipeline.normalize import NormalizeStage

        try:
            norm_run = NormalizeStage(
                config=self.config, db_path=self.db_path
            ).run()
            self.result.normalized_rows = norm_run.rows_processed or 0
            logger.info("Normalized %d observations", self.result.normalized_rows)
            return True
        except Exception as exc:
            # Rollups read normalized rows, so there is nothing to roll up if
            # this failed.  Recorded as a failure so the CLI exits nonzero.
            self.result.failures.append(("normalize", str(exc)))
            logger.error("NormalizeStage failed: %s", exc, exc_info=True)
            return False

    def _rollup(self, realms: set[str], dates: set[str]) -> None:
        """Upsert daily rollups for every (realm, UTC date) the drain touched."""
        from wow_forecaster.db.connection import get_connection
        from wow_forecaster.db.rollup import upsert_rollups_for_date

        targets = sorted(realms) or list(self.config.realms.defaults)
        try:
            with self._connect(get_connection) as conn:
                for realm in targets:
                    for obs_date in sorted(dates):
                        arch_n, item_n = upsert_rollups_for_date(conn, realm, obs_date)
                        logger.info(
                            "Rollup updated for realm=%s date=%s: arch=%d item=%d",
                            realm, obs_date, arch_n, item_n,
                        )
        except Exception as exc:
            self.result.failures.append(("rollup", str(exc)))
            logger.error("Rollup step failed: %s", exc, exc_info=True)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _connect(self, get_connection):
        return get_connection(
            self.db_path,
            wal_mode=self.config.database.wal_mode,
            busy_timeout_ms=self.config.database.busy_timeout_ms,
        )

    def _lock_path(self):
        from pathlib import Path

        return Path(self.db_path).parent / ".hourly.lock"

    def _log_selection(self, selected, since) -> None:
        result = self.result
        logger.info(
            "Cloud snapshots: listed=%d selected=%d since=%s | skipped: %s",
            result.listed, len(selected), since.date().isoformat(),
            result.skips.summary(),
        )
        if result.truncated:
            logger.warning(
                "Capped at %d objects this run; %d more are waiting and will be "
                "picked up by the next run",
                len(selected), result.skips.over_limit,
            )


def sync_snapshots(
    config,
    *,
    since: datetime | None = None,
    dry_run: bool = False,
    limit: int | None = None,
    db_path: str | None = None,
    now: datetime | None = None,
    s3: Any | None = None,
    env: dict[str, str] | None = None,
) -> SyncResult:
    """Run one catch-up sync and return its :class:`SyncResult`.

    The CLI entry point, mirroring ``durable_backup.run_backup``.
    """
    stage = SyncSnapshotsStage(config=config, db_path=db_path)
    stage.run(since=since, dry_run=dry_run, limit=limit, now=now, s3=s3, env=env)
    return stage.result
