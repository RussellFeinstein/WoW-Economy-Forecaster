"""Tests for EconomicArchetype and ArchetypeMapping models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from wow_forecaster.models.archetype import ArchetypeMapping, EconomicArchetype
from wow_forecaster.taxonomy.archetype_taxonomy import ArchetypeCategory, ArchetypeTag


class TestEconomicArchetype:
    def test_valid_construction(self, sample_archetype):
        a = sample_archetype
        assert a.slug == "consumable.flask.stat"
        assert a.category_tag == ArchetypeCategory.CONSUMABLE
        assert a.transfer_confidence == 0.90

    def test_confidence_below_zero_raises(self):
        with pytest.raises(ValidationError, match="transfer_confidence"):
            EconomicArchetype(
                slug="bad.conf",
                display_name="Bad",
                category_tag=ArchetypeCategory.CONSUMABLE,
                transfer_confidence=-0.1,
            )

    def test_confidence_above_one_raises(self):
        with pytest.raises(ValidationError, match="transfer_confidence"):
            EconomicArchetype(
                slug="bad.conf2",
                display_name="Bad",
                category_tag=ArchetypeCategory.CONSUMABLE,
                transfer_confidence=1.1,
            )

    def test_confidence_zero_is_valid(self):
        a = EconomicArchetype(
            slug="zero.conf",
            display_name="Zero Confidence",
            category_tag=ArchetypeCategory.SERVICE,
            transfer_confidence=0.0,
        )
        assert a.transfer_confidence == 0.0

    def test_confidence_one_is_valid(self):
        a = EconomicArchetype(
            slug="full.conf",
            display_name="Full Confidence",
            category_tag=ArchetypeCategory.CONSUMABLE,
            sub_tag=ArchetypeTag.CONSUMABLE_FLASK_STAT,
            transfer_confidence=1.0,
        )
        assert a.transfer_confidence == 1.0

    def test_slug_with_spaces_raises(self):
        with pytest.raises(ValidationError, match="slug"):
            EconomicArchetype(
                slug="bad slug",
                display_name="Bad",
                category_tag=ArchetypeCategory.CONSUMABLE,
            )

    def test_slug_uppercase_raises(self):
        with pytest.raises(ValidationError, match="slug"):
            EconomicArchetype(
                slug="Consumable.Flask",
                display_name="Bad",
                category_tag=ArchetypeCategory.CONSUMABLE,
            )

    def test_frozen_immutable(self, sample_archetype):
        with pytest.raises(Exception):
            sample_archetype.transfer_confidence = 0.5

    def test_optional_sub_tag_defaults_none(self):
        a = EconomicArchetype(
            slug="no.subtag",
            display_name="No SubTag",
            category_tag=ArchetypeCategory.TRADE_GOOD,
        )
        assert a.sub_tag is None


class TestArchetypeMapping:
    def test_valid_construction(self, sample_archetype_mapping):
        m = sample_archetype_mapping
        assert m.confidence_score == 0.85
        assert m.source_expansion == "tww"
        assert m.target_expansion == "midnight"

    def test_invalid_confidence_below_zero_raises(self):
        with pytest.raises(ValidationError, match="confidence_score"):
            ArchetypeMapping(
                source_archetype_id=1,
                target_archetype_id=2,
                confidence_score=-0.1,
                mapping_rationale="Test mapping.",
            )

    def test_invalid_confidence_above_one_raises(self):
        with pytest.raises(ValidationError, match="confidence_score"):
            ArchetypeMapping(
                source_archetype_id=1,
                target_archetype_id=2,
                confidence_score=1.5,
                mapping_rationale="Test mapping.",
            )

    def test_empty_rationale_raises(self):
        with pytest.raises(ValidationError, match="rationale"):
            ArchetypeMapping(
                source_archetype_id=1,
                target_archetype_id=2,
                confidence_score=0.8,
                mapping_rationale="",
            )

    def test_whitespace_only_rationale_raises(self):
        with pytest.raises(ValidationError, match="rationale"):
            ArchetypeMapping(
                source_archetype_id=1,
                target_archetype_id=2,
                confidence_score=0.8,
                mapping_rationale="   ",
            )

    def test_rationale_is_stripped(self):
        m = ArchetypeMapping(
            source_archetype_id=1,
            target_archetype_id=2,
            confidence_score=0.8,
            mapping_rationale="  Leading spaces trimmed.  ",
        )
        assert m.mapping_rationale == "Leading spaces trimmed."

    def test_frozen_immutable(self, sample_archetype_mapping):
        with pytest.raises(Exception):
            sample_archetype_mapping.confidence_score = 0.5
