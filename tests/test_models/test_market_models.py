"""Tests for market observation models (raw and normalized)."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from wow_forecaster.models.market import NormalizedMarketObservation, RawMarketObservation


class TestRawMarketObservation:
    def test_valid_construction(self, sample_raw_observation):
        obs = sample_raw_observation
        assert obs.item_id == 12345
        assert obs.realm_slug == "area-52"
        assert obs.faction == "neutral"
        assert obs.source == "tsm_export"

    def test_invalid_faction_raises(self):
        with pytest.raises(ValidationError, match="faction"):
            RawMarketObservation(
                item_id=1,
                realm_slug="area-52",
                faction="nightelf",
                observed_at=datetime(2024, 9, 15, tzinfo=timezone.utc),
                source="tsm_export",
            )

    def test_invalid_source_raises(self):
        with pytest.raises(ValidationError, match="source"):
            RawMarketObservation(
                item_id=1,
                realm_slug="area-52",
                faction="neutral",
                observed_at=datetime(2024, 9, 15, tzinfo=timezone.utc),
                source="unknown_source",
            )

    def test_negative_copper_raises(self):
        with pytest.raises(ValidationError):
            RawMarketObservation(
                item_id=1,
                realm_slug="area-52",
                faction="neutral",
                observed_at=datetime(2024, 9, 15, tzinfo=timezone.utc),
                source="tsm_export",
                min_buyout_raw=-100,
            )

    def test_negative_quantity_raises(self):
        with pytest.raises(ValidationError):
            RawMarketObservation(
                item_id=1,
                realm_slug="area-52",
                faction="neutral",
                observed_at=datetime(2024, 9, 15, tzinfo=timezone.utc),
                source="tsm_export",
                quantity_listed=-5,
            )

    def test_zero_copper_is_valid(self):
        obs = RawMarketObservation(
            item_id=1,
            realm_slug="area-52",
            faction="neutral",
            observed_at=datetime(2024, 9, 15, tzinfo=timezone.utc),
            source="tsm_export",
            min_buyout_raw=0,
        )
        assert obs.min_buyout_raw == 0

    def test_none_copper_is_valid(self):
        obs = RawMarketObservation(
            item_id=1,
            realm_slug="area-52",
            faction="neutral",
            observed_at=datetime(2024, 9, 15, tzinfo=timezone.utc),
            source="tsm_export",
        )
        assert obs.min_buyout_raw is None

    def test_frozen_immutable(self, sample_raw_observation):
        with pytest.raises(Exception):  # ValidationError or TypeError (frozen)
            sample_raw_observation.item_id = 99999

    def test_all_factions_valid(self):
        for faction in ("alliance", "horde", "neutral"):
            obs = RawMarketObservation(
                item_id=1,
                realm_slug="area-52",
                faction=faction,
                observed_at=datetime(2024, 9, 15, tzinfo=timezone.utc),
                source="tsm_export",
            )
            assert obs.faction == faction


class TestNormalizedMarketObservation:
    def test_valid_construction(self, sample_normalized_observation):
        norm = sample_normalized_observation
        assert norm.price_gold == 500.0
        assert norm.is_outlier is False

    def test_negative_gold_raises(self):
        with pytest.raises(ValidationError):
            NormalizedMarketObservation(
                obs_id=1,
                item_id=1,
                realm_slug="area-52",
                faction="neutral",
                observed_at=datetime(2024, 9, 15, tzinfo=timezone.utc),
                price_gold=-1.0,
            )

    def test_frozen_immutable(self, sample_normalized_observation):
        with pytest.raises(Exception):
            sample_normalized_observation.price_gold = 999.0

    def test_optional_fields_default_none(self):
        norm = NormalizedMarketObservation(
            obs_id=1,
            item_id=1,
            realm_slug="area-52",
            faction="neutral",
            observed_at=datetime(2024, 9, 15, tzinfo=timezone.utc),
            price_gold=100.0,
        )
        assert norm.archetype_id is None
        assert norm.z_score is None
        assert norm.is_outlier is False
