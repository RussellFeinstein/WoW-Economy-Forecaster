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


class TestNewEventFeatures:
    """Tests for event_impact_magnitude, days_until_major_event, is_pre_event_window."""

    def test_event_impact_magnitude_from_category_impacts(self):
        """event_impact_magnitude is populated from category impacts for the active event."""
        obs_date = date(2025, 1, 15)
        active_event = _make_event(
            "rtwf",
            start=date(2025, 1, 10),
            end=date(2025, 1, 24),
            announced_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
            severity=EventSeverity.MAJOR,
        )
        # event_id must be set on the model; patch via object to simulate DB lookup
        # Since _make_event sets event_id=None, we need a real event_id for impacts dict.
        # Use event_id=99 by constructing manually.
        from wow_forecaster.models.event import WoWEvent
        from wow_forecaster.taxonomy.event_taxonomy import EventScope, EventType
        evt = WoWEvent(
            event_id=99,
            slug="rtwf",
            display_name="RTWF",
            event_type=EventType.RTWF,
            scope=EventScope.GLOBAL,
            severity=EventSeverity.MAJOR,
            expansion_slug="tww",
            start_date=date(2025, 1, 10),
            end_date=date(2025, 1, 24),
            announced_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
        )
        category_impacts = {
            99: [{"archetype_category": "consumable", "impact_direction": "spike",
                  "typical_magnitude": 0.40, "lag_days": 0, "duration_days": 14}]
        }
        rows = [_make_row(obs_date)]
        result = compute_event_features(
            rows, [evt], {}, archetype_id=1,
            category_impacts=category_impacts,
            archetype_category="consumable",
        )
        assert result[0]["event_impact_magnitude"] == pytest.approx(0.40)

    def test_event_impact_magnitude_null_when_no_category_impacts(self):
        """event_impact_magnitude is None when category_impacts is not provided."""
        obs_date = date(2025, 1, 15)
        evt = _make_event(
            "active", start=date(2025, 1, 10), end=date(2025, 1, 24),
            announced_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
        )
        rows = [_make_row(obs_date)]
        result = compute_event_features(rows, [evt], {}, archetype_id=1)
        assert result[0]["event_impact_magnitude"] is None

    def test_event_impact_magnitude_null_when_wrong_category(self):
        """event_impact_magnitude is None when the category doesn't match."""
        from wow_forecaster.models.event import WoWEvent
        from wow_forecaster.taxonomy.event_taxonomy import EventScope, EventType
        evt = WoWEvent(
            event_id=77,
            slug="rtwf",
            display_name="RTWF",
            event_type=EventType.RTWF,
            scope=EventScope.GLOBAL,
            severity=EventSeverity.MAJOR,
            expansion_slug="tww",
            start_date=date(2025, 1, 10),
            end_date=date(2025, 1, 24),
            announced_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
        )
        category_impacts = {
            77: [{"archetype_category": "gear", "impact_direction": "spike",
                  "typical_magnitude": 0.35, "lag_days": 0, "duration_days": 14}]
        }
        rows = [_make_row(date(2025, 1, 15))]
        result = compute_event_features(
            rows, [evt], {}, archetype_id=1,
            category_impacts=category_impacts,
            archetype_category="consumable",  # Different from impact category
        )
        assert result[0]["event_impact_magnitude"] is None

    def test_days_until_major_event_major_severity(self):
        """days_until_major_event populated for MAJOR events known as of obs_date."""
        obs_date = date(2025, 1, 15)
        major_event = _make_event(
            "major-future",
            start=date(2025, 1, 22),
            announced_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
            severity=EventSeverity.MAJOR,
        )
        rows = [_make_row(obs_date)]
        result = compute_event_features(rows, [major_event], {}, archetype_id=1)
        assert result[0]["days_until_major_event"] == pytest.approx(7.0)

    def test_days_until_major_event_critical_severity(self):
        """days_until_major_event is populated for CRITICAL events too."""
        obs_date = date(2025, 1, 15)
        critical_event = _make_event(
            "critical-future",
            start=date(2025, 1, 20),
            announced_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
            severity=EventSeverity.CRITICAL,
        )
        rows = [_make_row(obs_date)]
        result = compute_event_features(rows, [critical_event], {}, archetype_id=1)
        assert result[0]["days_until_major_event"] == pytest.approx(5.0)

    def test_days_until_major_event_ignores_minor_severity(self):
        """days_until_major_event is None when only MINOR/MODERATE events are upcoming."""
        obs_date = date(2025, 1, 15)
        minor_event = _make_event(
            "minor-future",
            start=date(2025, 1, 20),
            announced_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
            severity=EventSeverity.MINOR,
        )
        rows = [_make_row(obs_date)]
        result = compute_event_features(rows, [minor_event], {}, archetype_id=1)
        assert result[0]["days_until_major_event"] is None

    def test_days_until_major_event_leakage_safe(self):
        """days_until_major_event is None if the major event is not announced yet."""
        obs_date = date(2025, 1, 15)
        future_unannounced = _make_event(
            "unannounced-major",
            start=date(2025, 1, 20),
            announced_at=datetime(2025, 1, 16, tzinfo=timezone.utc),  # Announced AFTER obs_date
            severity=EventSeverity.MAJOR,
        )
        rows = [_make_row(obs_date)]
        result = compute_event_features(rows, [future_unannounced], {}, archetype_id=1)
        assert result[0]["days_until_major_event"] is None

    def test_is_pre_event_window_true_within_7_days(self):
        """is_pre_event_window is True when days_until_major_event is 1–7."""
        obs_date = date(2025, 1, 15)
        major_event = _make_event(
            "major-soon",
            start=date(2025, 1, 20),  # 5 days away
            announced_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
            severity=EventSeverity.MAJOR,
        )
        rows = [_make_row(obs_date)]
        result = compute_event_features(rows, [major_event], {}, archetype_id=1)
        assert result[0]["is_pre_event_window"] is True

    def test_is_pre_event_window_false_beyond_7_days(self):
        """is_pre_event_window is False when event is more than 7 days away."""
        obs_date = date(2025, 1, 15)
        major_event = _make_event(
            "major-far",
            start=date(2025, 1, 30),  # 15 days away
            announced_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
            severity=EventSeverity.MAJOR,
        )
        rows = [_make_row(obs_date)]
        result = compute_event_features(rows, [major_event], {}, archetype_id=1)
        assert result[0]["is_pre_event_window"] is False

    def test_is_pre_event_window_false_on_event_day(self):
        """is_pre_event_window is False on the event start day (days_until = 0, window is 1-7)."""
        obs_date = date(2025, 1, 15)
        major_event = _make_event(
            "major-today",
            start=date(2025, 1, 15),  # Today
            end=date(2025, 1, 22),
            announced_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
            severity=EventSeverity.MAJOR,
        )
        rows = [_make_row(obs_date)]
        result = compute_event_features(rows, [major_event], {}, archetype_id=1)
        assert result[0]["is_pre_event_window"] is False

    def test_is_pre_event_window_false_no_major_events(self):
        """is_pre_event_window is False when no major events are known."""
        obs_date = date(2025, 1, 15)
        rows = [_make_row(obs_date)]
        result = compute_event_features(rows, [], {}, archetype_id=1)
        assert result[0]["is_pre_event_window"] is False

    def test_new_columns_present_in_output(self):
        """All 8 event feature columns must appear in every output row."""
        obs_date = date(2025, 1, 15)
        rows = [_make_row(obs_date)]
        result = compute_event_features(rows, [], {}, archetype_id=1)
        expected_keys = {
            "event_active",
            "event_days_to_next",
            "event_days_since_last",
            "event_severity_max",
            "event_archetype_impact",
            "event_impact_magnitude",
            "days_until_major_event",
            "is_pre_event_window",
        }
        assert expected_keys.issubset(result[0].keys())


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
