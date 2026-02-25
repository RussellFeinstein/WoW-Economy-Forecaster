"""
Shared pytest fixtures for the WoW Economy Forecaster test suite.

Provides:
  - ``in_memory_db``: A fresh in-memory SQLite connection with the full
    schema applied. Created anew for each test that requests it.
  - Sample domain object factories for use in multiple test modules.
"""

from __future__ import annotations

import sqlite3
from datetime import date, datetime, timezone
from typing import Generator

import pytest

from wow_forecaster.db.connection import get_connection
from wow_forecaster.db.schema import apply_schema
from wow_forecaster.models.archetype import ArchetypeMapping, EconomicArchetype
from wow_forecaster.models.event import WoWEvent
from wow_forecaster.models.forecast import ForecastOutput, RecommendationOutput
from wow_forecaster.models.item import Item, ItemCategory
from wow_forecaster.models.market import NormalizedMarketObservation, RawMarketObservation
from wow_forecaster.models.meta import ModelMetadata, RunMetadata
from wow_forecaster.taxonomy.archetype_taxonomy import ArchetypeCategory, ArchetypeTag
from wow_forecaster.taxonomy.event_taxonomy import EventScope, EventSeverity, EventType


# ── Database fixture ──────────────────────────────────────────────────────────

@pytest.fixture
def in_memory_db() -> Generator[sqlite3.Connection, None, None]:
    """Yield a fresh in-memory SQLite connection with the full schema applied.

    Foreign key enforcement is ON. Schema is applied idempotently.
    Connection is closed after the test.
    """
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    apply_schema(conn)
    yield conn
    conn.close()


# ── Sample domain object factories ────────────────────────────────────────────

@pytest.fixture
def sample_raw_observation() -> RawMarketObservation:
    """A valid ``RawMarketObservation`` for testing."""
    return RawMarketObservation(
        item_id=12345,
        realm_slug="area-52",
        faction="neutral",
        observed_at=datetime(2024, 9, 15, 12, 0, 0, tzinfo=timezone.utc),
        source="tsm_export",
        min_buyout_raw=50_000_00,   # 500 gold in copper
        market_value_raw=55_000_00,
        historical_value_raw=48_000_00,
        quantity_listed=42,
        num_auctions=8,
        raw_json='{"source": "tsm"}',
    )


@pytest.fixture
def sample_normalized_observation() -> NormalizedMarketObservation:
    """A valid ``NormalizedMarketObservation`` for testing."""
    return NormalizedMarketObservation(
        obs_id=1,
        item_id=12345,
        archetype_id=1,
        realm_slug="area-52",
        faction="neutral",
        observed_at=datetime(2024, 9, 15, 12, 0, 0, tzinfo=timezone.utc),
        price_gold=500.0,
        market_value_gold=550.0,
        historical_value_gold=480.0,
        quantity_listed=42,
        num_auctions=8,
        z_score=0.42,
        is_outlier=False,
    )


@pytest.fixture
def sample_event() -> WoWEvent:
    """A valid ``WoWEvent`` (RTWF) for testing."""
    return WoWEvent(
        slug="test-rtwf-s1",
        display_name="Test RTWF Season 1",
        event_type=EventType.RTWF,
        scope=EventScope.GLOBAL,
        severity=EventSeverity.MAJOR,
        expansion_slug="tww",
        start_date=date(2024, 9, 10),
        end_date=date(2024, 9, 24),
        announced_at=datetime(2024, 8, 19, 17, 0, 0, tzinfo=timezone.utc),
        notes="Test RTWF event fixture.",
    )


@pytest.fixture
def sample_archetype() -> EconomicArchetype:
    """A valid ``EconomicArchetype`` for testing."""
    return EconomicArchetype(
        slug="consumable.flask.stat",
        display_name="Stat Flask",
        category_tag=ArchetypeCategory.CONSUMABLE,
        sub_tag=ArchetypeTag.CONSUMABLE_FLASK_STAT,
        description="Primary stat flasks consumed on raid nights.",
        is_transferable=True,
        transfer_confidence=0.90,
        transfer_notes="Stat flasks exist in every expansion; pattern highly transferable.",
    )


@pytest.fixture
def sample_archetype_mapping(sample_archetype: EconomicArchetype) -> ArchetypeMapping:
    """A valid ``ArchetypeMapping`` for testing (self-mapping as placeholder)."""
    return ArchetypeMapping(
        source_archetype_id=1,
        target_archetype_id=2,
        source_expansion="tww",
        target_expansion="midnight",
        confidence_score=0.85,
        mapping_rationale=(
            "Midnight will have equivalent stat flasks for primary stats. "
            "Economic behavior (RTWF spike, season-end crash) expected to transfer directly."
        ),
        created_by="manual",
    )


@pytest.fixture
def sample_item_category() -> ItemCategory:
    """A valid ``ItemCategory`` for testing."""
    return ItemCategory(
        slug="consumable.flask",
        display_name="Flasks",
        parent_slug="consumable",
        archetype_tag="consumable.flask.stat",
        expansion_slug="tww",
    )


@pytest.fixture
def sample_item() -> Item:
    """A valid ``Item`` for testing."""
    return Item(
        item_id=191528,
        name="Phial of Tepid Versatility",
        category_id=1,
        archetype_id=1,
        expansion_slug="tww",
        quality="rare",
        is_crafted=True,
        is_boe=False,
        ilvl=None,
        notes="Example TWW flask for testing.",
    )


@pytest.fixture
def sample_run_metadata() -> RunMetadata:
    """A valid mutable ``RunMetadata`` for testing."""
    return RunMetadata(
        run_slug="test-run-uuid-0001",
        pipeline_stage="ingest",
        status="started",
        config_snapshot={"database": {"db_path": ":memory:"}, "debug": True},
        started_at=datetime(2024, 9, 15, 12, 0, 0, tzinfo=timezone.utc),
    )


@pytest.fixture
def sample_model_metadata() -> ModelMetadata:
    """A valid ``ModelMetadata`` for testing."""
    return ModelMetadata(
        slug="stub_linear_v0",
        display_name="Stub Linear Model v0",
        model_type="stub",
        version="0.1.0",
        is_active=True,
    )


@pytest.fixture
def sample_forecast(sample_run_metadata: RunMetadata) -> ForecastOutput:
    """A valid ``ForecastOutput`` for testing."""
    return ForecastOutput(
        run_id=1,
        archetype_id=1,
        realm_slug="area-52",
        forecast_horizon="7d",
        target_date=date(2024, 9, 22),
        predicted_price_gold=520.0,
        confidence_lower=480.0,
        confidence_upper=580.0,
        confidence_pct=0.80,
        model_slug="stub_linear_v0",
    )
