"""Tests for event taxonomy integrity — enums, completeness, uniqueness."""

from __future__ import annotations

import pytest

from wow_forecaster.taxonomy.event_taxonomy import (
    EventScope,
    EventSeverity,
    EventType,
    ImpactDirection,
)


class TestEventTypeEnum:
    def test_all_values_are_strings(self):
        for member in EventType:
            assert isinstance(member.value, str), f"{member} value should be a string"

    def test_no_duplicate_values(self):
        values = [m.value for m in EventType]
        assert len(values) == len(set(values)), "EventType has duplicate values"

    def test_minimum_event_types(self):
        # Ensure key event types for forecast features are present
        required = {
            "expansion_launch", "rtwf", "season_start", "season_end",
            "major_patch", "holiday_event", "content_drought",
        }
        actual = {m.value for m in EventType}
        missing = required - actual
        assert not missing, f"Required EventType values missing: {missing}"

    def test_slug_format(self):
        for member in EventType:
            assert " " not in member.value, f"EventType.{member.name} contains spaces"
            assert member.value == member.value.lower(), f"EventType.{member.name} not lowercase"

    def test_rtwf_exists(self):
        assert EventType.RTWF == "rtwf"

    def test_expansion_launch_exists(self):
        assert EventType.EXPANSION_LAUNCH == "expansion_launch"


class TestEventScopeEnum:
    def test_all_four_scopes_present(self):
        scope_values = {m.value for m in EventScope}
        assert "global" in scope_values
        assert "region" in scope_values
        assert "realm_cluster" in scope_values
        assert "faction" in scope_values

    def test_no_duplicate_values(self):
        values = [m.value for m in EventScope]
        assert len(values) == len(set(values))

    def test_all_values_lowercase_strings(self):
        for member in EventScope:
            assert isinstance(member.value, str)
            assert member.value == member.value.lower()


class TestEventSeverityEnum:
    def test_all_five_severities_present(self):
        severity_values = {m.value for m in EventSeverity}
        assert "critical" in severity_values
        assert "major" in severity_values
        assert "moderate" in severity_values
        assert "minor" in severity_values
        assert "negligible" in severity_values

    def test_no_duplicate_values(self):
        values = [m.value for m in EventSeverity]
        assert len(values) == len(set(values))

    def test_count_is_five(self):
        assert len(list(EventSeverity)) == 5


class TestImpactDirectionEnum:
    def test_all_four_directions_present(self):
        direction_values = {m.value for m in ImpactDirection}
        assert "spike" in direction_values
        assert "crash" in direction_values
        assert "mixed" in direction_values
        assert "neutral" in direction_values

    def test_count_is_four(self):
        assert len(list(ImpactDirection)) == 4

    def test_no_duplicate_values(self):
        values = [m.value for m in ImpactDirection]
        assert len(values) == len(set(values))
