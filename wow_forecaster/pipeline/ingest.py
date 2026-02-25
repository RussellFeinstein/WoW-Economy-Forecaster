"""
IngestStage — fetch raw AH data and save JSON snapshots to disk.

Current behaviour (stub / fixture mode):
  - When no API credentials are set, all three clients return fixture data.
  - Snapshot JSON files are written to ``data/raw/snapshots/{source}/YYYY/MM/DD/``.
  - Snapshot metadata is recorded in the ``ingestion_snapshots`` SQLite table.
  - NO rows are written to ``market_observations_raw`` yet — that requires
    real item data with canonical WoW item IDs. See TODO below.

Switching to real API data:
  Set credentials in ``.env`` (gitignored):
    UNDERMINE_API_KEY=...
    BLIZZARD_CLIENT_ID=...
    BLIZZARD_CLIENT_SECRET=...

  Then implement the ``TODO: parse records`` section below to convert snapshot
  records into ``RawMarketObservation`` objects and bulk-insert via the repo.

Extension points:
  - Override ``_ingest_source()`` per source for custom error handling.
  - Add per-realm deduplication by comparing ``content_hash`` against
    ``IngestionSnapshotRepository.get_latest_successful_by_source()``.
"""

from __future__ import annotations

import logging
import os

from wow_forecaster.models.meta import RunMetadata
from wow_forecaster.pipeline.base import PipelineStage

logger = logging.getLogger(__name__)


class IngestStage(PipelineStage):
    """Fetch raw AH data from all configured providers and persist to disk.

    Writes timestamped JSON snapshots and records metadata in SQLite.
    Returns the number of snapshot files successfully written.
    """

    stage_name = "ingest"

    def _execute(
        self,
        run: RunMetadata,
        realm_slugs: list[str] | None = None,
        **kwargs,
    ) -> int:
        """Fetch snapshots from Undermine, Blizzard API, and Blizzard news.

        Args:
            run: In-progress :class:`RunMetadata` (mutable).
            realm_slugs: Realms to ingest. Defaults to
                ``config.realms.defaults``.

        Returns:
            Number of snapshot files successfully written.
        """
        from wow_forecaster.db.connection import get_connection
        from wow_forecaster.db.repositories.ingestion_repo import (
            IngestionSnapshot,
            IngestionSnapshotRepository,
        )
        from wow_forecaster.ingestion.blizzard_client import BlizzardClient
        from wow_forecaster.ingestion.blizzard_news_client import BlizzardNewsClient
        from wow_forecaster.ingestion.snapshot import build_snapshot_path, save_snapshot
        from wow_forecaster.ingestion.undermine_client import UndermineClient
        from wow_forecaster.utils.time_utils import utcnow

        # Pre-persist run to get a run_id for FK use in ingestion_snapshots.
        # PipelineStage.run() will UPDATE (not re-insert) when called again.
        self._persist_run(run)

        realms = realm_slugs or list(self.config.realms.defaults)
        raw_dir = self.config.data.raw_dir

        # Read credentials from env — None → fixture mode
        undermine_key = os.environ.get("UNDERMINE_API_KEY")
        blizzard_id = os.environ.get("BLIZZARD_CLIENT_ID")
        blizzard_secret = os.environ.get("BLIZZARD_CLIENT_SECRET")

        undermine = UndermineClient(api_key=undermine_key)
        blizzard = BlizzardClient(client_id=blizzard_id, client_secret=blizzard_secret)
        news = BlizzardNewsClient()

        total_snapshots = 0

        with get_connection(self.db_path) as conn:
            snap_repo = IngestionSnapshotRepository(conn)

            # ── Undermine Exchange ─────────────────────────────────────────────
            for realm in realms:
                faction = self.config.realms.default_faction
                snap = self._fetch_undermine(
                    undermine=undermine,
                    realm=realm,
                    faction=faction,
                    raw_dir=raw_dir,
                    run=run,
                    undermine_key=undermine_key,
                )
                snap_repo.insert(snap)
                if snap.success:
                    total_snapshots += 1

            # ── Blizzard Game Data API ─────────────────────────────────────────
            for realm in realms:
                snap = self._fetch_blizzard(
                    blizzard=blizzard,
                    realm=realm,
                    raw_dir=raw_dir,
                    run=run,
                    blizzard_id=blizzard_id,
                )
                snap_repo.insert(snap)
                if snap.success:
                    total_snapshots += 1

            # ── Blizzard News ──────────────────────────────────────────────────
            snap = self._fetch_news(news=news, raw_dir=raw_dir, run=run)
            snap_repo.insert(snap)
            if snap.success:
                total_snapshots += 1

            conn.commit()

        mode = "fixture" if not (undermine_key or blizzard_id) else "live"
        logger.info(
            "IngestStage complete | mode=%s | snapshots=%d | "
            "market_observations_raw: 0 (TODO: implement record parsing)",
            mode, total_snapshots,
        )
        return 0  # TODO: return len(inserted raw obs) once record parsing is implemented

    # ── Per-source helpers ─────────────────────────────────────────────────────

    def _fetch_undermine(
        self,
        undermine,
        realm: str,
        faction: str,
        raw_dir: str,
        run: RunMetadata,
        undermine_key: str | None,
    ):
        from wow_forecaster.db.repositories.ingestion_repo import IngestionSnapshot
        from wow_forecaster.ingestion.snapshot import build_snapshot_path, save_snapshot
        from wow_forecaster.utils.time_utils import utcnow

        try:
            if undermine_key:
                response = undermine.fetch_realm_auctions(realm, faction)
            else:
                logger.warning(
                    "UNDERMINE_API_KEY not set — using fixture data for %s/%s",
                    realm, faction,
                )
                response = undermine.get_fixture_response(realm, faction)

            records_data = [
                {
                    "item_id": r.item_id,
                    "realm_slug": r.realm_slug,
                    "faction": r.faction,
                    "min_buyout": r.min_buyout,
                    "market_value": r.market_value,
                    "historical_value": r.historical_value,
                    "quantity": r.quantity,
                    "num_auctions": r.num_auctions,
                }
                for r in response.records
            ]

            path = build_snapshot_path(
                raw_dir, "undermine", f"{realm}_{faction}", response.fetched_at
            )
            content_hash, record_count = save_snapshot(
                path,
                records_data,
                metadata={
                    "source": "undermine",
                    "realm": realm,
                    "faction": faction,
                    "is_fixture": response.is_fixture,
                    "run_slug": run.run_slug,
                },
            )
            logger.info(
                "Undermine snapshot: %s | %d records | fixture=%s",
                path.name, record_count, response.is_fixture,
            )
            # TODO: parse records_data → RawMarketObservation → market_observations_raw
            return IngestionSnapshot(
                snapshot_id=None,
                run_id=run.run_id,
                source="undermine",
                endpoint=response.endpoint,
                snapshot_path=str(path),
                content_hash=content_hash,
                record_count=record_count,
                success=True,
                error_message=None,
                fetched_at=response.fetched_at,
            )

        except Exception as exc:
            logger.error("Undermine fetch failed for %s/%s: %s", realm, faction, exc)
            return IngestionSnapshot(
                snapshot_id=None,
                run_id=run.run_id,
                source="undermine",
                endpoint=f"realm_auctions/{realm}/{faction}",
                snapshot_path="",
                content_hash=None,
                record_count=0,
                success=False,
                error_message=str(exc),
                fetched_at=utcnow(),
            )

    def _fetch_blizzard(
        self,
        blizzard,
        realm: str,
        raw_dir: str,
        run: RunMetadata,
        blizzard_id: str | None,
    ):
        from wow_forecaster.db.repositories.ingestion_repo import IngestionSnapshot
        from wow_forecaster.ingestion.snapshot import build_snapshot_path, save_snapshot
        from wow_forecaster.utils.time_utils import utcnow

        try:
            if blizzard_id:
                response = blizzard.fetch_connected_realm_auctions(realm)
            else:
                logger.warning(
                    "BLIZZARD_CLIENT_ID not set — using fixture data for %s", realm
                )
                response = blizzard.get_fixture_response(realm)

            records_data = [
                {
                    "item_id": r.item_id,
                    "realm_slug": r.realm_slug,
                    "buyout": r.buyout,
                    "bid": r.bid,
                    "unit_price": r.unit_price,
                    "quantity": r.quantity,
                    "time_left": r.time_left,
                }
                for r in response.records
            ]

            path = build_snapshot_path(
                raw_dir, "blizzard_api", f"realm_{realm}", response.fetched_at
            )
            content_hash, record_count = save_snapshot(
                path,
                records_data,
                metadata={
                    "source": "blizzard_api",
                    "realm": realm,
                    "is_fixture": response.is_fixture,
                    "run_slug": run.run_slug,
                },
            )
            logger.info(
                "Blizzard API snapshot: %s | %d records | fixture=%s",
                path.name, record_count, response.is_fixture,
            )
            # TODO: parse records_data → RawMarketObservation → market_observations_raw
            return IngestionSnapshot(
                snapshot_id=None,
                run_id=run.run_id,
                source="blizzard_api",
                endpoint=response.endpoint,
                snapshot_path=str(path),
                content_hash=content_hash,
                record_count=record_count,
                success=True,
                error_message=None,
                fetched_at=response.fetched_at,
            )

        except Exception as exc:
            logger.error("Blizzard API fetch failed for %s: %s", realm, exc)
            return IngestionSnapshot(
                snapshot_id=None,
                run_id=run.run_id,
                source="blizzard_api",
                endpoint=f"connected-realm/{realm}/auctions",
                snapshot_path="",
                content_hash=None,
                record_count=0,
                success=False,
                error_message=str(exc),
                fetched_at=utcnow(),
            )

    def _fetch_news(self, news, raw_dir: str, run: RunMetadata):
        from wow_forecaster.db.repositories.ingestion_repo import IngestionSnapshot
        from wow_forecaster.ingestion.snapshot import build_snapshot_path, save_snapshot
        from wow_forecaster.utils.time_utils import utcnow

        try:
            response = news.get_fixture_response()

            news_data = [
                {
                    "title": item.title,
                    "url": item.url,
                    "published_at": item.published_at.isoformat(),
                    "category": item.category,
                    "summary": item.summary,
                    "is_patch_notes": item.is_patch_notes,
                    "patch_version": item.patch_version,
                }
                for item in response.items
            ]

            path = build_snapshot_path(
                raw_dir, "blizzard_news", "news", response.fetched_at
            )
            content_hash, record_count = save_snapshot(
                path,
                news_data,
                metadata={
                    "source": "blizzard_news",
                    "is_fixture": response.is_fixture,
                    "run_slug": run.run_slug,
                },
            )
            logger.info(
                "Blizzard news snapshot: %s | %d items", path.name, record_count
            )
            return IngestionSnapshot(
                snapshot_id=None,
                run_id=run.run_id,
                source="blizzard_news",
                endpoint=response.endpoint,
                snapshot_path=str(path),
                content_hash=content_hash,
                record_count=record_count,
                success=True,
                error_message=None,
                fetched_at=response.fetched_at,
            )

        except Exception as exc:
            logger.error("News fetch failed: %s", exc)
            return IngestionSnapshot(
                snapshot_id=None,
                run_id=run.run_id,
                source="blizzard_news",
                endpoint="recent-news",
                snapshot_path="",
                content_hash=None,
                record_count=0,
                success=False,
                error_message=str(exc),
                fetched_at=utcnow(),
            )
