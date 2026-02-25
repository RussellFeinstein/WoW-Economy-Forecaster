"""
Tests for the data quality report.

All tests use synthetic feature row dicts — no DB dependency.
"""

from __future__ import annotations

from datetime import date
from typing import Any

import pytest

from wow_forecaster.features.quality import DataQualityReport, build_quality_report


def _make_clean_rows(n: int = 10) -> list[dict[str, Any]]:
    """Return n clean, non-duplicate rows with all required quality fields."""
    return [
        {
            "archetype_id":           1,
            "realm_slug":             "area-52",
            "obs_date":               date(2025, 1, i + 1),
            "price_mean":             100.0 + i,
            "price_min":              95.0,
            "price_max":              105.0,
            "market_value_mean":      None,
            "historical_value_mean":  None,
            "obs_count":              3,
            "quantity_sum":           None,
            "auctions_sum":           None,
            "is_volume_proxy":        False,
            "price_lag_1d":           None if i == 0 else 100.0 + i - 1,
            "price_lag_3d":           None,
            "price_lag_7d":           None,
            "price_lag_14d":          None,
            "price_lag_28d":          None,
            "price_roll_mean_7d":     100.0 + i,
            "price_roll_std_7d":      0.5,
            "price_roll_mean_14d":    None,
            "price_roll_std_14d":     None,
            "price_roll_mean_28d":    None,
            "price_roll_std_28d":     None,
            "price_pct_change_7d":    None,
            "price_pct_change_14d":   None,
            "price_pct_change_28d":   None,
            "day_of_week":            (i % 7) + 1,
            "day_of_month":           i + 1,
            "week_of_year":           1,
            "days_since_expansion":   150 + i,
            "event_active":           False,
            "event_days_to_next":     float(30 - i),
            "event_days_since_last":  None,
            "event_severity_max":     None,
            "event_archetype_impact": None,
            "archetype_category":     "consumable",
            "archetype_sub_tag":      "consumable.flask.stat",
            "is_transferable":        True,
            "is_cold_start":          False,
            "item_count_in_archetype": 1,
            "has_transfer_mapping":   True,
            "transfer_confidence":    0.85,
            "target_price_1d":        101.0 + i,
            "target_price_7d":        107.0 + i,
            "target_price_28d":       128.0 + i,
        }
        for i in range(n)
    ]


class TestEmptyInput:
    def test_empty_rows_returns_empty_report(self):
        report = build_quality_report([])
        assert report.total_rows == 0
        assert report.is_clean is True
        assert report.missingness == {}
        assert report.duplicate_key_count == 0

    def test_empty_has_none_date_range(self):
        report = build_quality_report([])
        assert report.date_range_start is None
        assert report.date_range_end is None


class TestCleanData:
    def test_clean_rows_produce_clean_report(self):
        rows = _make_clean_rows(10)
        report = build_quality_report(rows)
        assert report.is_clean is True
        assert report.duplicate_key_count == 0
        assert report.leakage_warnings == []

    def test_total_rows_correct(self):
        rows = _make_clean_rows(7)
        report = build_quality_report(rows)
        assert report.total_rows == 7

    def test_date_range_computed_correctly(self):
        rows = _make_clean_rows(5)
        report = build_quality_report(rows)
        assert report.date_range_start == date(2025, 1, 1)
        assert report.date_range_end == date(2025, 1, 5)


class TestDuplicateDetection:
    def test_duplicate_row_detected(self):
        rows = _make_clean_rows(3)
        # Add a duplicate of the first row (same archetype_id, realm_slug, obs_date).
        rows.append(rows[0].copy())
        report = build_quality_report(rows)
        assert report.duplicate_key_count == 1
        assert report.is_clean is False


class TestMissingness:
    def test_missingness_fraction_computed(self):
        rows = _make_clean_rows(10)
        for r in rows[:4]:
            r["price_mean"] = None   # 40% null
        report = build_quality_report(rows)
        assert report.missingness["price_mean"] == pytest.approx(0.4)

    def test_high_missingness_flagged(self):
        rows = _make_clean_rows(10)
        for r in rows[:4]:
            r["price_mean"] = None   # 40% > default 30% threshold
        report = build_quality_report(rows, missingness_threshold=0.30)
        assert "price_mean" in report.high_missingness_cols

    def test_missingness_below_threshold_not_flagged(self):
        rows = _make_clean_rows(10)
        rows[0]["price_mean"] = None  # only 10% null
        report = build_quality_report(rows, missingness_threshold=0.30)
        assert "price_mean" not in report.high_missingness_cols


class TestTimeContinuity:
    def test_continuous_series_no_gap_count(self):
        rows = _make_clean_rows(10)   # dates 2025-01-01 to 2025-01-10, no gaps
        report = build_quality_report(rows)
        assert report.date_gap_series_count == 0

    def test_gap_in_series_detected(self):
        rows = _make_clean_rows(5)
        # Skip 2025-01-03 (index 2) by manually overriding dates.
        rows[2]["obs_date"] = date(2025, 1, 5)   # jump from Jan 2 to Jan 5 = gap of 3
        rows[3]["obs_date"] = date(2025, 1, 6)
        rows[4]["obs_date"] = date(2025, 1, 7)
        report = build_quality_report(rows)
        assert report.date_gap_series_count >= 1


class TestLeakageHeuristic:
    def test_negative_days_to_next_triggers_warning(self):
        """event_days_to_next < 0 should produce a leakage warning."""
        rows = _make_clean_rows(3)
        rows[1]["event_days_to_next"] = -2.0   # would mean event already started but labelled "next"
        report = build_quality_report(rows)
        assert len(report.leakage_warnings) >= 1
        assert report.is_clean is False

    def test_non_negative_days_to_next_no_warning(self):
        rows = _make_clean_rows(5)
        report = build_quality_report(rows)
        assert report.leakage_warnings == []


class TestProxyAndColdStart:
    def test_volume_proxy_pct_computed(self):
        rows = _make_clean_rows(10)
        for r in rows[:3]:
            r["is_volume_proxy"] = True
        report = build_quality_report(rows)
        assert report.volume_proxy_pct == pytest.approx(0.3)

    def test_cold_start_pct_computed(self):
        rows = _make_clean_rows(10)
        for r in rows[:2]:
            r["is_cold_start"] = True
        report = build_quality_report(rows)
        assert report.cold_start_pct == pytest.approx(0.2)

    def test_items_excluded_passed_through(self):
        rows = _make_clean_rows(5)
        report = build_quality_report(rows, items_excluded=12)
        assert report.items_excluded_no_archetype == 12
