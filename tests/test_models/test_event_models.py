"""Tests for WoWEvent model — validation and is_known_at() behavior."""

from __future__ import annotations

from datetime import date, datetime, timezone

import pytest
from pydantic import ValidationError

from wow_forecaster.models.event import WoWEvent
from wow_forecaster.taxonomy.event_taxonomy import EventScope, EventSeverity, EventType


class TestWoWEventConstruction:
    def test_valid_full_construction(self, sample_event):
        ev = sample_event
        assert ev.slug == "test-rtwf-s1"
        assert ev.event_type == EventType.RTWF
        assert ev.scope == EventScope.GLOBAL
        assert ev.severity == EventSeverity.MAJOR
        assert ev.expansion_slug == "tww"

    def test_end_before_start_raises(self):
        with pytest.raises(ValidationError, match="end_date"):
            WoWEvent(
                slug="bad-dates",
                display_name="Bad Dates",
                event_type=EventType.MAINTENANCE_WINDOW,
                scope=EventScope.GLOBAL,
                severity=EventSeverity.NEGLIGIBLE,
                expansion_slug="tww",
                start_date=date(2024, 9, 20),
                end_date=date(2024, 9, 10),  # before start!
            )

    def test_same_start_end_is_valid(self):
        ev = WoWEvent(
            slug="single-day",
            display_name="Single Day Event",
            event_type=EventType.MAINTENANCE_WINDOW,
            scope=EventScope.GLOBAL,
            severity=EventSeverity.NEGLIGIBLE,
            expansion_slug="tww",
            start_date=date(2024, 9, 10),
            end_date=date(2024, 9, 10),
        )
        assert ev.end_date == ev.start_date

    def test_null_end_date_is_valid(self):
        ev = WoWEvent(
            slug="no-end",
            display_name="No End Date",
            event_type=EventType.EXPANSION_LAUNCH,
            scope=EventScope.GLOBAL,
            severity=EventSeverity.CRITICAL,
            expansion_slug="tww",
            start_date=date(2024, 8, 26),
        )
        assert ev.end_date is None

    def test_invalid_expansion_raises(self):
        with pytest.raises(ValidationError):
            WoWEvent(
                slug="bad-exp",
                display_name="Bad Expansion",
                event_type=EventType.SEASON_START,
                scope=EventScope.GLOBAL,
                severity=EventSeverity.MAJOR,
                expansion_slug="not_a_real_expansion",
                start_date=date(2024, 9, 10),
            )

    def test_frozen_immutable(self, sample_event):
        with pytest.raises(Exception):
            sample_event.severity = EventSeverity.CRITICAL


class TestIsKnownAt:
    def test_returns_false_when_announced_at_none(self):
        ev = WoWEvent(
            slug="unknown",
            display_name="Unknown Announce",
            event_type=EventType.HOLIDAY_EVENT,
            scope=EventScope.GLOBAL,
            severity=EventSeverity.MINOR,
            expansion_slug="tww",
            start_date=date(2024, 9, 10),
            announced_at=None,
        )
        as_of = datetime(2024, 9, 15, tzinfo=timezone.utc)
        assert ev.is_known_at(as_of) is False

    def test_returns_false_before_announced_at(self, sample_event):
        # sample_event announced 2024-08-19
        before = datetime(2024, 8, 1, tzinfo=timezone.utc)
        assert sample_event.is_known_at(before) is False

    def test_returns_true_at_announced_at(self, sample_event):
        # exactly at announced_at timestamp
        exactly_at = datetime(2024, 8, 19, 17, 0, 0, tzinfo=timezone.utc)
        assert sample_event.is_known_at(exactly_at) is True

    def test_returns_true_after_announced_at(self, sample_event):
        after = datetime(2024, 9, 5, tzinfo=timezone.utc)
        assert sample_event.is_known_at(after) is True


class TestIsActiveOn:
    def test_active_on_start_date(self, sample_event):
        assert sample_event.is_active_on(date(2024, 9, 10)) is True

    def test_active_on_end_date(self, sample_event):
        assert sample_event.is_active_on(date(2024, 9, 24)) is True

    def test_active_in_middle(self, sample_event):
        assert sample_event.is_active_on(date(2024, 9, 17)) is True

    def test_not_active_before_start(self, sample_event):
        assert sample_event.is_active_on(date(2024, 9, 9)) is False

    def test_not_active_after_end(self, sample_event):
        assert sample_event.is_active_on(date(2024, 9, 25)) is False

    def test_active_with_no_end_date(self):
        ev = WoWEvent(
            slug="ongoing",
            display_name="Ongoing",
            event_type=EventType.EXPANSION_LAUNCH,
            scope=EventScope.GLOBAL,
            severity=EventSeverity.CRITICAL,
            expansion_slug="tww",
            start_date=date(2024, 8, 26),
        )
        assert ev.is_active_on(date(2025, 1, 1)) is True
