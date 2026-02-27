"""
IngestStage — fetch raw AH data, save JSON snapshots, and parse records into
market_observations_raw.

Fixture mode (no credentials set):
  All three clients return canned fixture records.  Snapshot JSON files are
  written to ``data/raw/snapshots/{source}/YYYY/MM/DD/``, snapshot metadata is
  recorded in ``ingestion_snapshots``, and any fixture items that exist in the
  ``items`` table are inserted into ``market_observations_raw``.

FK safety:
  Observations for item IDs not present in ``items`` are silently skipped.
  This means an empty item registry (e.g. in unit tests) results in 0 inserts
  without any FK violation.

Switching to real API data:
  Set credentials in ``.env`` (gitignored):
    UNDERMINE_API_KEY=...
    BLIZZARD_CLIENT_ID=...
    BLIZZARD_CLIENT_SECRET=...

Extension points:
  - Override ``_ingest_source()`` per source for custom error handling.
  - Add per-realm deduplication by comparing ``content_hash`` against
    ``IngestionSnapshotRepository.get_latest_successful_by_source()``.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from typing import TYPE_CHECKING

from wow_forecaster.models.meta import RunMetadata
from wow_forecaster.pipeline.base import PipelineStage

if TYPE_CHECKING:
    from wow_forecaster.models.market import RawMarketObservation

logger = logging.getLogger(__name__)


class IngestStage(PipelineStage):
    """Fetch raw AH data from all configured providers and persist to disk.

    Writes timestamped JSON snapshots, records metadata in SQLite, and parses
    each snapshot's records into ``market_observations_raw`` (skipping any
    item IDs absent from the ``items`` registry).

    Returns the number of raw observations successfully inserted.
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
            Number of raw market observations inserted into
            ``market_observations_raw``.
        """
        from wow_forecaster.db.connection import get_connection
        from wow_forecaster.db.repositories.ingestion_repo import (
            IngestionSnapshotRepository,
        )
        from wow_forecaster.db.repositories.item_repo import ItemRepository
        from wow_forecaster.db.repositories.market_repo import MarketObservationRepository
        from wow_forecaster.ingestion.blizzard_client import BlizzardClient
        from wow_forecaster.ingestion.blizzard_news_client import BlizzardNewsClient
        from wow_forecaster.ingestion.undermine_client import UndermineClient

        # Pre-persist run to get a run_id for FK use in ingestion_snapshots.
        self._persist_run(run)

        realms = realm_slugs or list(self.config.realms.defaults)
        raw_dir = self.config.data.raw_dir

        # Read credentials — None → fixture mode
        undermine_key = os.environ.get("UNDERMINE_API_KEY")
        blizzard_id = os.environ.get("BLIZZARD_CLIENT_ID")
        blizzard_secret = os.environ.get("BLIZZARD_CLIENT_SECRET")

        undermine = UndermineClient(api_key=undermine_key)
        blizzard = BlizzardClient(client_id=blizzard_id, client_secret=blizzard_secret)
        news = BlizzardNewsClient()

        total_snapshots = 0
        total_inserted_raw = 0

        with get_connection(self.db_path) as conn:
            snap_repo = IngestionSnapshotRepository(conn)
            market_repo = MarketObservationRepository(conn)
            item_repo = ItemRepository(conn)

            # Load known item IDs once — FK guard for all sources this run.
            known_item_ids: set[int] = item_repo.get_all_item_ids()
            logger.info(
                "IngestStage: %d known items in registry", len(known_item_ids)
            )

            # ── Undermine Exchange ─────────────────────────────────────────────
            for realm in realms:
                faction = self.config.realms.default_faction
                snap, records_data = self._fetch_undermine(
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
                    observations, skipped = self._parse_undermine_records(
                        records_data, snap.fetched_at, known_item_ids
                    )
                    inserted = market_repo.insert_raw_batch(observations)
                    total_inserted_raw += inserted
                    logger.info(
                        "Undermine %s/%s: %d records | inserted=%d | "
                        "skipped_missing_items=%d",
                        realm, faction, len(records_data), inserted, skipped,
                    )

            # ── Blizzard Game Data API ─────────────────────────────────────────
            # Live mode: fetch commodities ONCE for the whole US region.
            # Commodities are region-wide since patch 9.2.7 — one call covers all realms.
            # Fixture mode keeps the old per-realm loop so existing tests still pass.
            if blizzard_id:
                snap, records_data = self._fetch_blizzard_commodities(
                    blizzard=blizzard,
                    raw_dir=raw_dir,
                    run=run,
                )
                snap_repo.insert(snap)
                if snap.success:
                    total_snapshots += 1
                    observations, skipped = self._parse_blizzard_records(
                        records_data, snap.fetched_at, known_item_ids
                    )
                    inserted = market_repo.insert_raw_batch(observations)
                    total_inserted_raw += inserted
                    logger.info(
                        "Blizzard commodities (US region-wide): %d records | "
                        "inserted=%d | skipped_missing_items=%d",
                        len(records_data), inserted, skipped,
                    )
            else:
                for realm in realms:
                    snap, records_data = self._fetch_blizzard(
                        blizzard=blizzard,
                        realm=realm,
                        raw_dir=raw_dir,
                        run=run,
                        blizzard_id=blizzard_id,
                    )
                    snap_repo.insert(snap)
                    if snap.success:
                        total_snapshots += 1
                        observations, skipped = self._parse_blizzard_records(
                            records_data, snap.fetched_at, known_item_ids
                        )
                        inserted = market_repo.insert_raw_batch(observations)
                        total_inserted_raw += inserted
                        logger.info(
                            "Blizzard API %s: %d records | inserted=%d | "
                            "skipped_missing_items=%d",
                            realm, len(records_data), inserted, skipped,
                        )

            # ── Blizzard News (content only — no market table rows) ────────────
            snap = self._fetch_news(news=news, raw_dir=raw_dir, run=run)
            snap_repo.insert(snap)
            if snap.success:
                total_snapshots += 1

            conn.commit()

        mode = "fixture" if not (undermine_key or blizzard_id) else "live"
        logger.info(
            "IngestStage complete | mode=%s | snapshots=%d | "
            "market_observations_raw inserted=%d",
            mode, total_snapshots, total_inserted_raw,
        )
        return total_inserted_raw

    # ── Record parsing helpers ─────────────────────────────────────────────────

    def _parse_undermine_records(
        self,
        records_data: list[dict],
        observed_at: datetime,
        known_item_ids: set[int],
    ) -> tuple[list[RawMarketObservation], int]:
        """Convert Undermine Exchange records to :class:`RawMarketObservation` objects.

        Args:
            records_data: List of serialised record dicts from the snapshot.
            observed_at: UTC timestamp from the API response (``fetched_at``).
            known_item_ids: Set of item IDs present in ``items`` table.
                Records for absent IDs are skipped to avoid FK violations.

        Returns:
            Tuple of ``(observations, skipped_count)`` where ``skipped_count``
            is the number of records dropped because item_id was unknown.
        """
        from wow_forecaster.models.market import RawMarketObservation

        observations: list[RawMarketObservation] = []
        skipped = 0

        for rec in records_data:
            item_id = rec["item_id"]
            if item_id not in known_item_ids:
                skipped += 1
                continue

            observations.append(
                RawMarketObservation(
                    item_id=item_id,
                    realm_slug=rec["realm_slug"],
                    faction=rec["faction"],
                    observed_at=observed_at,
                    source="undermine_api",
                    min_buyout_raw=rec.get("min_buyout"),
                    market_value_raw=rec.get("market_value"),
                    historical_value_raw=rec.get("historical_value"),
                    quantity_listed=rec.get("quantity"),
                    num_auctions=rec.get("num_auctions"),
                    raw_json=json.dumps(rec, separators=(",", ":")),
                )
            )

        return observations, skipped

    def _parse_blizzard_records(
        self,
        records_data: list[dict],
        observed_at: datetime,
        known_item_ids: set[int],
    ) -> tuple[list[RawMarketObservation], int]:
        """Convert Blizzard AH records to :class:`RawMarketObservation` objects.

        Blizzard auction records contain per-auction listings without TSM-style
        market values.  The min-buyout field is derived as follows:

        - ``unit_price > 0`` (commodity): use ``unit_price`` as ``min_buyout_raw``
        - ``buyout > 0`` (non-commodity with buyout): use ``buyout``
        - Otherwise (bid-only listing): ``min_buyout_raw = None``

        Faction is always ``"neutral"`` — Blizzard connected-realm auctions are
        not faction-segregated for our modelling purposes.

        Args:
            records_data: List of serialised record dicts from the snapshot.
            observed_at: UTC timestamp from the API response (``fetched_at``).
            known_item_ids: Set of item IDs present in ``items`` table.

        Returns:
            Tuple of ``(observations, skipped_count)``.
        """
        from wow_forecaster.models.market import RawMarketObservation

        observations: list[RawMarketObservation] = []
        skipped = 0

        for rec in records_data:
            item_id = rec["item_id"]
            if item_id not in known_item_ids:
                skipped += 1
                continue

            unit_price = rec.get("unit_price") or 0
            buyout = rec.get("buyout") or 0
            if unit_price > 0:
                min_buyout_raw: int | None = unit_price
            elif buyout > 0:
                min_buyout_raw = buyout
            else:
                min_buyout_raw = None

            observations.append(
                RawMarketObservation(
                    item_id=item_id,
                    realm_slug=rec["realm_slug"],
                    faction="neutral",
                    observed_at=observed_at,
                    source="blizzard_api",
                    min_buyout_raw=min_buyout_raw,
                    market_value_raw=None,
                    historical_value_raw=None,
                    quantity_listed=rec.get("quantity"),
                    num_auctions=1,
                    raw_json=json.dumps(rec, separators=(",", ":")),
                )
            )

        return observations, skipped

    # ── Per-source fetch helpers ───────────────────────────────────────────────

    def _fetch_undermine(
        self,
        undermine,
        realm: str,
        faction: str,
        raw_dir: str,
        run: RunMetadata,
        undermine_key: str | None,
    ) -> tuple[object, list[dict]]:
        """Fetch and snapshot Undermine Exchange data for one realm/faction.

        Returns:
            Tuple of ``(IngestionSnapshot, records_data)`` where
            ``records_data`` is empty on failure.
        """
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
            snap = IngestionSnapshot(
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
            return snap, records_data

        except Exception as exc:
            logger.error("Undermine fetch failed for %s/%s: %s", realm, faction, exc)
            snap = IngestionSnapshot(
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
            return snap, []

    def _fetch_blizzard(
        self,
        blizzard,
        realm: str,
        raw_dir: str,
        run: RunMetadata,
        blizzard_id: str | None,
    ) -> tuple[object, list[dict]]:
        """Fetch and snapshot Blizzard AH data for one realm.

        Returns:
            Tuple of ``(IngestionSnapshot, records_data)`` where
            ``records_data`` is empty on failure.
        """
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
            snap = IngestionSnapshot(
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
            return snap, records_data

        except Exception as exc:
            logger.error("Blizzard API fetch failed for %s: %s", realm, exc)
            snap = IngestionSnapshot(
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
            return snap, []

    def _fetch_blizzard_commodities(
        self,
        blizzard,
        raw_dir: str,
        run: RunMetadata,
    ) -> tuple[object, list[dict]]:
        """Fetch and snapshot Blizzard US-region commodity AH data (one call).

        Commodities are region-wide since patch 9.2.7.  Records are tagged
        ``realm_slug=blizzard.region`` ("us") and ``faction="neutral"``.

        Returns:
            Tuple of ``(IngestionSnapshot, records_data)`` where
            ``records_data`` is empty on failure.
        """
        from wow_forecaster.db.repositories.ingestion_repo import IngestionSnapshot
        from wow_forecaster.ingestion.snapshot import build_snapshot_path, save_snapshot
        from wow_forecaster.utils.time_utils import utcnow

        try:
            response = blizzard.fetch_commodities()

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
                raw_dir, "blizzard_api", f"commodities_{blizzard.region}",
                response.fetched_at,
            )
            content_hash, record_count = save_snapshot(
                path,
                records_data,
                metadata={
                    "source": "blizzard_api",
                    "type": "commodities",
                    "region": blizzard.region,
                    "is_fixture": response.is_fixture,
                    "run_slug": run.run_slug,
                },
            )
            logger.info(
                "Blizzard commodities snapshot: %s | %d records",
                path.name, record_count,
            )
            snap = IngestionSnapshot(
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
            return snap, records_data

        except Exception as exc:
            logger.error("Blizzard commodities fetch failed: %s", exc)
            snap = IngestionSnapshot(
                snapshot_id=None,
                run_id=run.run_id,
                source="blizzard_api",
                endpoint="data/wow/auctions/commodities",
                snapshot_path="",
                content_hash=None,
                record_count=0,
                success=False,
                error_message=str(exc),
                fetched_at=utcnow(),
            )
            return snap, []

    def _fetch_news(self, news, raw_dir: str, run: RunMetadata):
        """Fetch and snapshot Blizzard news items.

        News content is never written to the market observations table.

        Returns:
            An :class:`IngestionSnapshot` (no records_data needed).
        """
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
