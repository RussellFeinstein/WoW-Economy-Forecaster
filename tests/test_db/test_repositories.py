"""Tests for repository round-trip operations using in-memory SQLite."""

from __future__ import annotations

from datetime import date, datetime, timezone

import pytest

from wow_forecaster.db.repositories.archetype_repo import (
    ArchetypeMappingRepository,
    ArchetypeRepository,
)
from wow_forecaster.db.repositories.event_repo import WoWEventRepository
from wow_forecaster.db.repositories.forecast_repo import (
    ForecastOutputRepository,
    RunMetadataRepository,
)
from wow_forecaster.db.repositories.item_repo import ItemCategoryRepository, ItemRepository
from wow_forecaster.db.repositories.market_repo import MarketObservationRepository
from wow_forecaster.models.archetype import ArchetypeMapping, EconomicArchetype
from wow_forecaster.models.event import WoWEvent
from wow_forecaster.models.forecast import ForecastOutput
from wow_forecaster.models.item import Item, ItemCategory
from wow_forecaster.models.market import RawMarketObservation
from wow_forecaster.models.meta import RunMetadata
from wow_forecaster.taxonomy.archetype_taxonomy import ArchetypeCategory, ArchetypeTag
from wow_forecaster.taxonomy.event_taxonomy import EventScope, EventSeverity, EventType


# ── Helpers ────────────────────────────────────────────────────────────────────

def _insert_category(conn) -> int:
    repo = ItemCategoryRepository(conn)
    cat = ItemCategory(
        slug="consumable.flask",
        display_name="Flasks",
        archetype_tag="consumable.flask.stat",
        expansion_slug="tww",
    )
    return repo.insert(cat)


def _insert_archetype(conn) -> int:
    repo = ArchetypeRepository(conn)
    arch = EconomicArchetype(
        slug="consumable.flask.stat",
        display_name="Stat Flask",
        category_tag=ArchetypeCategory.CONSUMABLE,
        sub_tag=ArchetypeTag.CONSUMABLE_FLASK_STAT,
        transfer_confidence=0.90,
    )
    return repo.insert(arch)


def _insert_item(conn, category_id: int, archetype_id: int) -> int:
    repo = ItemRepository(conn)
    item = Item(
        item_id=191528,
        name="Test Flask",
        category_id=category_id,
        archetype_id=archetype_id,
        expansion_slug="tww",
        quality="rare",
        is_crafted=True,
    )
    return repo.insert(item)


# ── Event repository tests ─────────────────────────────────────────────────────

class TestWoWEventRepository:
    def test_insert_and_fetch_by_id(self, in_memory_db, sample_event):
        repo = WoWEventRepository(in_memory_db)
        event_id = repo.insert(sample_event)
        assert event_id > 0
        fetched = repo.get_by_id(event_id)
        assert fetched is not None
        assert fetched.slug == sample_event.slug
        assert fetched.event_type == EventType.RTWF

    def test_insert_and_fetch_by_slug(self, in_memory_db, sample_event):
        repo = WoWEventRepository(in_memory_db)
        repo.insert(sample_event)
        fetched = repo.get_by_slug(sample_event.slug)
        assert fetched is not None
        assert fetched.display_name == sample_event.display_name

    def test_fetch_by_type(self, in_memory_db, sample_event):
        repo = WoWEventRepository(in_memory_db)
        repo.insert(sample_event)
        results = repo.get_by_type(EventType.RTWF)
        assert len(results) == 1
        assert results[0].slug == sample_event.slug

    def test_fetch_by_type_returns_empty_list(self, in_memory_db):
        repo = WoWEventRepository(in_memory_db)
        results = repo.get_by_type(EventType.BLIZZCON)
        assert results == []

    def test_upsert_updates_existing(self, in_memory_db, sample_event):
        repo = WoWEventRepository(in_memory_db)
        repo.insert(sample_event)
        updated = WoWEvent(
            slug=sample_event.slug,
            display_name="Updated Name",
            event_type=sample_event.event_type,
            scope=sample_event.scope,
            severity=EventSeverity.CRITICAL,  # changed
            expansion_slug=sample_event.expansion_slug,
            start_date=sample_event.start_date,
        )
        repo.upsert(updated)
        fetched = repo.get_by_slug(sample_event.slug)
        assert fetched is not None
        assert fetched.severity == EventSeverity.CRITICAL
        assert fetched.display_name == "Updated Name"

    def test_count(self, in_memory_db, sample_event):
        repo = WoWEventRepository(in_memory_db)
        assert repo.count() == 0
        repo.insert(sample_event)
        assert repo.count() == 1

    def test_get_active_on(self, in_memory_db, sample_event):
        repo = WoWEventRepository(in_memory_db)
        repo.insert(sample_event)
        # Active in middle of range
        active = repo.get_active_on(date(2024, 9, 17))
        assert len(active) == 1
        # Not active before range
        not_active = repo.get_active_on(date(2024, 9, 9))
        assert len(not_active) == 0

    def test_announced_at_roundtrip(self, in_memory_db, sample_event):
        repo = WoWEventRepository(in_memory_db)
        event_id = repo.insert(sample_event)
        fetched = repo.get_by_id(event_id)
        assert fetched is not None
        assert fetched.announced_at is not None
        assert fetched.announced_at == sample_event.announced_at


# ── Market observation repository tests ───────────────────────────────────────

class TestMarketObservationRepository:
    def _make_raw(self, item_id: int = 191528) -> RawMarketObservation:
        return RawMarketObservation(
            item_id=item_id,
            realm_slug="area-52",
            faction="neutral",
            observed_at=datetime(2024, 9, 15, 12, 0, tzinfo=timezone.utc),
            source="tsm_export",
            min_buyout_raw=5_000_000,
            quantity_listed=10,
        )

    def test_insert_raw_and_fetch_unprocessed(self, in_memory_db):
        cat_id = _insert_category(in_memory_db)
        arch_id = _insert_archetype(in_memory_db)
        _insert_item(in_memory_db, cat_id, arch_id)

        repo = MarketObservationRepository(in_memory_db)
        obs = self._make_raw()
        obs_id = repo.insert_raw(obs)
        assert obs_id > 0

        unprocessed = repo.get_unprocessed_raw(limit=10)
        assert len(unprocessed) == 1
        assert unprocessed[0].item_id == 191528

    def test_mark_processed(self, in_memory_db):
        cat_id = _insert_category(in_memory_db)
        arch_id = _insert_archetype(in_memory_db)
        _insert_item(in_memory_db, cat_id, arch_id)

        repo = MarketObservationRepository(in_memory_db)
        obs_id = repo.insert_raw(self._make_raw())

        # Before mark
        assert len(repo.get_unprocessed_raw()) == 1

        # Mark as processed
        updated = repo.mark_processed([obs_id])
        assert updated == 1

        # After mark
        assert len(repo.get_unprocessed_raw()) == 0

    def test_insert_raw_batch(self, in_memory_db):
        cat_id = _insert_category(in_memory_db)
        arch_id = _insert_archetype(in_memory_db)
        _insert_item(in_memory_db, cat_id, arch_id)

        repo = MarketObservationRepository(in_memory_db)
        obs_list = [self._make_raw() for _ in range(5)]
        count = repo.insert_raw_batch(obs_list)
        assert count == 5
        assert len(repo.get_unprocessed_raw(limit=10)) == 5


# ── Run metadata repository tests ─────────────────────────────────────────────

class TestRunMetadataRepository:
    def test_insert_and_fetch(self, in_memory_db, sample_run_metadata):
        repo = RunMetadataRepository(in_memory_db)
        run_id = repo.insert_run(sample_run_metadata)
        assert run_id > 0

        fetched = repo.get_run_by_slug(sample_run_metadata.run_slug)
        assert fetched is not None
        assert fetched.pipeline_stage == "ingest"
        assert fetched.status == "started"

    def test_update_status(self, in_memory_db, sample_run_metadata):
        repo = RunMetadataRepository(in_memory_db)
        run_id = repo.insert_run(sample_run_metadata)

        sample_run_metadata.run_id = run_id
        sample_run_metadata.status = "success"
        sample_run_metadata.rows_processed = 42
        sample_run_metadata.finished_at = datetime(2024, 9, 15, 13, 0, tzinfo=timezone.utc)

        repo.update_run(sample_run_metadata)

        updated = repo.get_run_by_slug(sample_run_metadata.run_slug)
        assert updated is not None
        assert updated.status == "success"
        assert updated.rows_processed == 42
        assert updated.finished_at is not None

    def test_get_recent_runs(self, in_memory_db, sample_run_metadata):
        repo = RunMetadataRepository(in_memory_db)
        repo.insert_run(sample_run_metadata)
        runs = repo.get_recent_runs(limit=10)
        assert len(runs) == 1

    def test_update_without_run_id_raises(self, in_memory_db, sample_run_metadata):
        repo = RunMetadataRepository(in_memory_db)
        with pytest.raises(ValueError, match="run_id"):
            repo.update_run(sample_run_metadata)  # run_id is None
