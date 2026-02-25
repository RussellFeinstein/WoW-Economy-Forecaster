"""
Tests for look-ahead bias prevention in event features.

These are the most critical correctness tests in the feature engineering suite.
A failure here means the model could be trained on information that was not
publicly available at the time of the observation — producing optimistically
biased backtest results.

The three leakage prevention layers tested here:
  Layer 1 — DB filter: events with announced_at=None are excluded by load_known_events()
  Layer 2 — Python guard: is_known_at(as_of=end_of_day) per row in compute_event_features()
  Layer 3 — Quality heuristic: event_days_to_next < 0 triggers a leakage warning
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from typing import Any

import pytest

from wow_forecaster.features.event_features import (
    _is_active,
    _is_future,
    _is_past,
    compute_event_features,
)
from wow_forecaster.models.event import WoWEvent
from wow_forecaster.taxonomy.event_taxonomy import EventScope, EventSeverity, EventType


def _make_event(
    slug: str,
    start: date,
    announced_at: datetime | None,
    end: date | None = None,
    severity: EventSeverity = EventSeverity.MINOR,
) -> WoWEvent:
    return WoWEvent(
        slug=slug,
        display_name=slug.replace("-", " ").title(),
        event_type=EventType.HOLIDAY_EVENT,
        scope=EventScope.GLOBAL,
        severity=severity,
        expansion_slug="tww",
        start_date=start,
        end_date=end,
        announced_at=announced_at,
    )


def _make_row(obs_date: date) -> dict[str, Any]:
    """Minimal feature row containing obs_date and archetype_id."""
    return {
        "archetype_id":          1,
        "realm_slug":            "area-52",
        "obs_date":              obs_date,
        "price_mean":            100.0,
        "price_min":             95.0,
        "price_max":             105.0,
        "market_value_mean":     None,
        "historical_value_mean": None,
        "obs_count":             3,
        "quantity_sum":          None,
        "auctions_sum":          None,
        "is_volume_proxy":       True,
        "price_lag_1d":          None,
        "price_lag_7d":          None,
        "price_roll_mean_7d":    None,
        "price_roll_std_7d":     None,
        "price_pct_change_7d":   None,
        "target_price_1d":       None,
        "target_price_7d":       None,
    }


class TestLeakageGuard:
    def test_event_announced_after_obs_date_is_excluded(self):
        """An event announced AFTER obs_date must NOT appear in event features."""
        obs_date = date(2025, 1, 15)
        future_event = _make_event(
            "future-event",
            start=date(2025, 1, 20),
            announced_at=datetime(2025, 1, 16, 12, 0, 0, tzinfo=timezone.utc),  # after obs_date
        )
        rows = [_make_row(obs_date)]
        result = compute_event_features(rows, [future_event], {}, archetype_id=1)

        assert result[0]["event_active"] is False
        assert result[0]["event_days_to_next"] is None   # event not known yet

    def test_event_announced_before_obs_date_is_included(self):
        """An event announced BEFORE obs_date must appear in event features."""
        obs_date = date(2025, 1, 15)
        known_event = _make_event(
            "known-event",
            start=date(2025, 1, 20),
            announced_at=datetime(2025, 1, 10, 12, 0, 0, tzinfo=timezone.utc),  # before obs_date
        )
        rows = [_make_row(obs_date)]
        result = compute_event_features(rows, [known_event], {}, archetype_id=1)

        # Event starts 5 days in the future and was already announced.
        assert result[0]["event_active"] is False
        assert result[0]["event_days_to_next"] == pytest.approx(5.0)

    def test_event_with_none_announced_at_is_always_excluded(self):
        """Layer 1 guard: events with announced_at=None must NEVER appear in outputs.

        Note: load_known_events() filters these at the DB level.  Here we test
        that compute_event_features() also handles them correctly when called
        directly (is_known_at returns False for announced_at=None).
        """
        obs_date = date(2025, 1, 15)
        unknown_event = _make_event(
            "unknown-event",
            start=date(2025, 1, 10),
            announced_at=None,    # no announcement date
        )
        rows = [_make_row(obs_date)]
        result = compute_event_features(rows, [unknown_event], {}, archetype_id=1)

        assert result[0]["event_active"] is False
        assert result[0]["event_days_to_next"] is None

    def test_event_announced_same_day_as_obs_is_included(self):
        """An event announced at 17:00 on obs_date should be counted (as_of = 23:59:59)."""
        obs_date = date(2025, 1, 15)
        same_day_event = _make_event(
            "same-day-event",
            start=date(2025, 1, 20),
            announced_at=datetime(2025, 1, 15, 17, 0, 0, tzinfo=timezone.utc),
        )
        rows = [_make_row(obs_date)]
        result = compute_event_features(rows, [same_day_event], {}, archetype_id=1)

        # Announced on the same day: should be included (as_of = end-of-day 23:59:59)
        assert result[0]["event_days_to_next"] == pytest.approx(5.0)

    def test_event_days_to_next_never_negative(self):
        """event_days_to_next must be >= 0 for all future events.

        A negative value would indicate a past event is incorrectly labelled as
        'next upcoming', which is a logic error caught by the quality heuristic.
        """
        obs_date = date(2025, 1, 15)
        past_event = _make_event(
            "past-event",
            start=date(2025, 1, 5),
            end=date(2025, 1, 12),
            announced_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
        )
        future_event = _make_event(
            "future-event",
            start=date(2025, 1, 20),
            announced_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
        )
        rows = [_make_row(obs_date)]
        result = compute_event_features(rows, [past_event, future_event], {}, archetype_id=1)

        days_to_next = result[0]["event_days_to_next"]
        # Should be 5.0 (future_event starts 5 days ahead), not negative.
        assert days_to_next is not None and days_to_next >= 0.0

    def test_target_price_columns_are_forward_looking_by_design(self):
        """Verify that target_price_Nd correctly references future prices.

        This test documents that targets ARE intentionally forward-looking.
        They are excluded from the inference Parquet to prevent them from
        being used as model inputs.
        """
        from wow_forecaster.features.registry import target_feature_names, inference_feature_names

        targets = set(target_feature_names())
        inference_cols = set(inference_feature_names())

        # No target column should appear in the inference feature set.
        assert targets.isdisjoint(inference_cols), (
            f"Target columns found in inference features: {targets & inference_cols}"
        )


class TestEventHelpers:
    """Tests for the internal _is_active, _is_past, _is_future helpers."""

    def test_is_active_within_window(self):
        e = _make_event("e", start=date(2025, 1, 10), end=date(2025, 1, 20),
                         announced_at=datetime(2025, 1, 1, tzinfo=timezone.utc))
        assert _is_active(e, date(2025, 1, 15)) is True

    def test_is_active_before_start(self):
        e = _make_event("e", start=date(2025, 1, 10), end=date(2025, 1, 20),
                         announced_at=datetime(2025, 1, 1, tzinfo=timezone.utc))
        assert _is_active(e, date(2025, 1, 9)) is False

    def test_is_future_event(self):
        e = _make_event("e", start=date(2025, 1, 20),
                         announced_at=datetime(2025, 1, 1, tzinfo=timezone.utc))
        assert _is_future(e, date(2025, 1, 15)) is True

    def test_is_past_event(self):
        e = _make_event("e", start=date(2025, 1, 5), end=date(2025, 1, 10),
                         announced_at=datetime(2025, 1, 1, tzinfo=timezone.utc))
        assert _is_past(e, date(2025, 1, 15)) is True
