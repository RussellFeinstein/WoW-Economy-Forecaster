"""
Tests for drift detection logic.

What we test
------------
1. _classify_data_drift — drift_fraction thresholds map to correct DriftLevel.
2. _classify_error_drift — mae_ratio thresholds map to correct DriftLevel.
3. _overall_drift_level — max rule + event shock bump.
4. DriftChecker.check_data_drift — correct behaviour with empty DB.
5. DriftChecker.check_data_drift — correct behaviour with populated DB.
6. DriftChecker.check_error_drift — returns NONE when no forecast-vs-actual.
7. DriftChecker.check_event_shocks — detects active events, flags shock_active.
8. DriftChecker.run_all — composite result structure is coherent.
9. DriftCheckResult uncertainty_multiplier matches adaptive policy.
"""

from __future__ import annotations

import sqlite3
from datetime import date, datetime, timedelta, timezone

import pytest

from wow_forecaster.db.schema import apply_schema
from wow_forecaster.monitoring.drift import (
    DriftChecker,
    DriftLevel,
    _classify_data_drift,
    _classify_error_drift,
    _overall_drift_level,
)


# ── Fixtures ───────────────────────────────────────────────────────────────────

_obs_counter: list[int] = [0]


@pytest.fixture
def drift_db() -> sqlite3.Connection:
    """In-memory DB with full schema applied.

    FK enforcement is OFF so tests can insert normalized obs directly
    without building the full raw-obs/items FK chain.
    """
    _obs_counter[0] = 0
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = OFF;")
    apply_schema(conn)
    return conn


def _insert_normalized_obs(
    conn: sqlite3.Connection,
    archetype_id: int,
    realm_slug: str,
    observed_at: str,
    price_gold: float,
    is_outlier: int = 0,
) -> None:
    """Insert a minimal normalized observation row (FK enforcement OFF)."""
    _obs_counter[0] += 1
    conn.execute(
        """
        INSERT INTO market_observations_normalized
            (obs_id, item_id, archetype_id, realm_slug, faction, observed_at,
             price_gold, is_outlier)
        VALUES (?, 1, ?, ?, 'neutral', ?, ?, ?)
        """,
        (_obs_counter[0], archetype_id, realm_slug, observed_at, price_gold, is_outlier),
    )
    conn.commit()


def _insert_wow_event(
    conn: sqlite3.Connection,
    slug: str,
    severity: str,
    start_date: str,
    end_date: str | None = None,
) -> None:
    conn.execute(
        """
        INSERT INTO wow_events
            (slug, display_name, event_type, scope, severity,
             expansion_slug, start_date, end_date)
        VALUES (?, ?, 'patch_day', 'global', ?, 'tww', ?, ?)
        """,
        (slug, slug, severity, start_date, end_date),
    )
    conn.commit()


# ── _classify_data_drift ───────────────────────────────────────────────────────

class TestClassifyDataDrift:
    def test_none_level_at_zero(self):
        assert _classify_data_drift(0.0) == DriftLevel.NONE

    def test_none_level_below_low_threshold(self):
        assert _classify_data_drift(0.09) == DriftLevel.NONE

    def test_low_level_at_boundary(self):
        assert _classify_data_drift(0.10) == DriftLevel.LOW

    def test_low_level_midpoint(self):
        assert _classify_data_drift(0.20) == DriftLevel.LOW

    def test_medium_level_at_boundary(self):
        assert _classify_data_drift(0.25) == DriftLevel.MEDIUM

    def test_medium_level_midpoint(self):
        assert _classify_data_drift(0.30) == DriftLevel.MEDIUM

    def test_high_level_at_boundary(self):
        assert _classify_data_drift(0.40) == DriftLevel.HIGH

    def test_critical_level_at_boundary(self):
        assert _classify_data_drift(0.60) == DriftLevel.CRITICAL

    def test_critical_level_at_one(self):
        assert _classify_data_drift(1.0) == DriftLevel.CRITICAL

    def test_custom_thresholds(self):
        # Double the thresholds — should require twice the fraction
        assert _classify_data_drift(0.10, thresholds=(0.20, 0.50, 0.80, 1.20)) == DriftLevel.NONE
        assert _classify_data_drift(0.20, thresholds=(0.20, 0.50, 0.80, 1.20)) == DriftLevel.LOW


# ── _classify_error_drift ──────────────────────────────────────────────────────

class TestClassifyErrorDrift:
    def test_none_when_ratio_is_none(self):
        assert _classify_error_drift(None) == DriftLevel.NONE

    def test_none_below_low_threshold(self):
        assert _classify_error_drift(1.0)  == DriftLevel.NONE
        assert _classify_error_drift(1.19) == DriftLevel.NONE

    def test_low_at_boundary(self):
        assert _classify_error_drift(1.2) == DriftLevel.LOW

    def test_medium_at_boundary(self):
        assert _classify_error_drift(1.5) == DriftLevel.MEDIUM

    def test_high_at_boundary(self):
        assert _classify_error_drift(2.0) == DriftLevel.HIGH

    def test_critical_at_boundary(self):
        assert _classify_error_drift(3.0) == DriftLevel.CRITICAL

    def test_critical_above_boundary(self):
        assert _classify_error_drift(10.0) == DriftLevel.CRITICAL

    def test_custom_thresholds(self):
        # Very tight thresholds
        assert _classify_error_drift(1.05, thresholds=(1.01, 1.02, 1.03, 1.04)) == DriftLevel.CRITICAL


# ── _overall_drift_level ───────────────────────────────────────────────────────

class TestOverallDriftLevel:
    def test_takes_max_of_data_and_error(self):
        assert _overall_drift_level(DriftLevel.LOW, DriftLevel.HIGH, False) == DriftLevel.HIGH
        assert _overall_drift_level(DriftLevel.HIGH, DriftLevel.LOW, False) == DriftLevel.HIGH

    def test_shock_bumps_level_by_one(self):
        # NONE + shock -> LOW
        assert _overall_drift_level(DriftLevel.NONE, DriftLevel.NONE, True) == DriftLevel.LOW

    def test_shock_does_not_exceed_critical(self):
        # HIGH + shock -> CRITICAL (not beyond)
        assert _overall_drift_level(DriftLevel.HIGH, DriftLevel.NONE, True) == DriftLevel.CRITICAL
        # CRITICAL + shock -> CRITICAL (clamped)
        assert _overall_drift_level(DriftLevel.CRITICAL, DriftLevel.NONE, True) == DriftLevel.CRITICAL

    def test_no_shock_no_bump(self):
        assert _overall_drift_level(DriftLevel.MEDIUM, DriftLevel.NONE, False) == DriftLevel.MEDIUM

    def test_shock_and_high_error_drift(self):
        # HIGH error drift + shock -> CRITICAL
        result = _overall_drift_level(DriftLevel.NONE, DriftLevel.HIGH, True)
        assert result == DriftLevel.CRITICAL


# ── DriftChecker.check_data_drift ─────────────────────────────────────────────

class TestCheckDataDrift:
    def test_empty_db_returns_none_level(self, drift_db: sqlite3.Connection):
        checker = DriftChecker(drift_db)
        report  = checker.check_data_drift("area-52")
        assert report.drift_level == DriftLevel.NONE
        assert report.n_series_checked == 0
        assert report.n_series_drifted == 0
        assert report.drift_fraction   == 0.0
        assert report.realm_slug       == "area-52"

    def test_baseline_only_no_recent_returns_none(self, drift_db: sqlite3.Connection):
        # Insert obs that are in the baseline window but not recent
        for i in range(5):
            past_date = (date.today() - timedelta(days=10 + i)).isoformat()
            _insert_normalized_obs(drift_db, 1, "area-52", f"{past_date}T10:00:00Z", 100.0)
        checker = DriftChecker(drift_db, drift_window_hours=1, baseline_days=30)
        report  = checker.check_data_drift("area-52")
        # No recent obs → n_series_checked may be 0 or 1 depending on baseline; drift=NONE
        assert report.drift_level == DriftLevel.NONE

    def test_stable_prices_no_drift(self, drift_db: sqlite3.Connection):
        """Recent mean == baseline mean -> no drift."""
        # Baseline: 20 days of 100g
        for d in range(2, 22):
            obs_date = (date.today() - timedelta(days=d)).isoformat()
            _insert_normalized_obs(drift_db, 1, "area-52", f"{obs_date}T12:00:00Z", 100.0)
        # Recent: today at 100g (same mean)
        today = date.today().isoformat()
        _insert_normalized_obs(drift_db, 1, "area-52", f"{today}T10:00:00Z", 100.0)

        checker = DriftChecker(drift_db, drift_window_hours=25, baseline_days=20, z_threshold=2.0)
        report  = checker.check_data_drift("area-52")
        # z_mean_shift ≈ 0 — should not be drifted
        assert report.n_series_drifted == 0
        assert report.drift_level == DriftLevel.NONE

    def test_large_price_shift_flagged_as_drifted(self, drift_db: sqlite3.Connection):
        """A 10x price spike in recent window should be flagged as drifted."""
        # Baseline: 20 days at 100g with std ~0 (constant)
        for d in range(2, 22):
            obs_date = (date.today() - timedelta(days=d)).isoformat()
            # Vary slightly to avoid zero std
            price = 100.0 + (d % 3)
            _insert_normalized_obs(drift_db, 1, "area-52", f"{obs_date}T12:00:00Z", price)
        # Recent: today at 1000g (10x baseline mean)
        today = date.today().isoformat()
        _insert_normalized_obs(drift_db, 1, "area-52", f"{today}T10:00:00Z", 1000.0)

        checker = DriftChecker(drift_db, drift_window_hours=25, baseline_days=20, z_threshold=2.0)
        report  = checker.check_data_drift("area-52")
        assert report.n_series_drifted >= 1
        assert report.drift_level in (
            DriftLevel.LOW, DriftLevel.MEDIUM, DriftLevel.HIGH, DriftLevel.CRITICAL
        )

    def test_outlier_obs_excluded(self, drift_db: sqlite3.Connection):
        """Outlier observations (is_outlier=1) should be excluded from stats."""
        today = date.today().isoformat()
        # Normal baseline
        for d in range(2, 12):
            obs_date = (date.today() - timedelta(days=d)).isoformat()
            _insert_normalized_obs(drift_db, 1, "area-52", f"{obs_date}T12:00:00Z", 100.0)
        # Recent obs marked as outlier — should not trigger drift
        _insert_normalized_obs(drift_db, 1, "area-52", f"{today}T10:00:00Z", 9999.0, is_outlier=1)

        checker = DriftChecker(drift_db, drift_window_hours=25, baseline_days=10, z_threshold=2.0)
        report  = checker.check_data_drift("area-52")
        # With outlier excluded, no recent data -> n_series_drifted == 0
        assert report.n_series_drifted == 0


# ── DriftChecker.check_error_drift ────────────────────────────────────────────

class TestCheckErrorDrift:
    def test_empty_db_returns_none_level(self, drift_db: sqlite3.Connection):
        checker = DriftChecker(drift_db)
        report  = checker.check_error_drift("area-52", horizon_days=1)
        assert report.drift_level  == DriftLevel.NONE
        assert report.n_evaluated  == 0
        assert report.live_mae     is None
        assert report.baseline_mae is None
        assert report.mae_ratio    is None

    def test_no_backtest_baseline_returns_none_ratio(self, drift_db: sqlite3.Connection):
        """If there's no backtest run to compare against, ratio is None."""
        # No backtest_runs row exists
        checker = DriftChecker(drift_db)
        report  = checker.check_error_drift("area-52")
        assert report.mae_ratio is None
        assert report.drift_level == DriftLevel.NONE


# ── DriftChecker.check_event_shocks ───────────────────────────────────────────

class TestCheckEventShocks:
    def test_empty_db_no_shock(self, drift_db: sqlite3.Connection):
        checker = DriftChecker(drift_db)
        report  = checker.check_event_shocks()
        assert report.shock_active       == False
        assert report.active_events      == []
        assert report.upcoming_events    == []

    def test_active_minor_event_no_shock(self, drift_db: sqlite3.Connection):
        """Minor active events do not trigger shock flag."""
        today  = date.today().isoformat()
        future = (date.today() + timedelta(days=7)).isoformat()
        _insert_wow_event(drift_db, "minor-event", "minor", today, future)

        checker = DriftChecker(drift_db, shock_window_days=7)
        report  = checker.check_event_shocks()
        assert len(report.active_events) == 1
        assert report.shock_active == False  # minor doesn't trigger

    def test_active_major_event_triggers_shock(self, drift_db: sqlite3.Connection):
        """Active MAJOR event triggers shock_active=True."""
        today  = date.today().isoformat()
        future = (date.today() + timedelta(days=3)).isoformat()
        _insert_wow_event(drift_db, "major-event", "major", today, future)

        checker = DriftChecker(drift_db, shock_window_days=7)
        report  = checker.check_event_shocks()
        assert report.shock_active == True

    def test_upcoming_critical_event_triggers_shock(self, drift_db: sqlite3.Connection):
        """Upcoming CRITICAL event within window triggers shock flag."""
        soon = (date.today() + timedelta(days=3)).isoformat()
        _insert_wow_event(drift_db, "critical-event", "critical", soon)

        checker = DriftChecker(drift_db, shock_window_days=7)
        report  = checker.check_event_shocks()
        assert len(report.upcoming_events) == 1
        assert report.shock_active == True

    def test_past_event_not_active(self, drift_db: sqlite3.Connection):
        """Events that ended yesterday are not in active list."""
        past_start = (date.today() - timedelta(days=10)).isoformat()
        past_end   = (date.today() - timedelta(days=1)).isoformat()
        _insert_wow_event(drift_db, "past-event", "major", past_start, past_end)

        checker = DriftChecker(drift_db, shock_window_days=7)
        report  = checker.check_event_shocks()
        assert len(report.active_events) == 0
        assert report.shock_active == False

    def test_far_future_event_not_upcoming(self, drift_db: sqlite3.Connection):
        """Events starting beyond shock_window_days are not in upcoming list."""
        far_future = (date.today() + timedelta(days=30)).isoformat()
        _insert_wow_event(drift_db, "far-event", "critical", far_future)

        checker = DriftChecker(drift_db, shock_window_days=7)
        report  = checker.check_event_shocks()
        assert len(report.upcoming_events) == 0
        assert report.shock_active == False


# ── DriftChecker.run_all ──────────────────────────────────────────────────────

class TestRunAll:
    def test_run_all_returns_coherent_result(self, drift_db: sqlite3.Connection):
        """run_all returns a DriftCheckResult with all three sub-reports."""
        checker = DriftChecker(drift_db)
        result  = checker.run_all("area-52")

        assert result.realm_slug         == "area-52"
        assert result.data_drift         is not None
        assert result.error_drift        is not None
        assert result.event_shock        is not None
        assert result.overall_drift_level is not None
        assert result.uncertainty_multiplier >= 1.0
        assert isinstance(result.retrain_recommended, bool)

    def test_empty_db_gives_none_overall(self, drift_db: sqlite3.Connection):
        """Empty DB -> overall drift is NONE, multiplier is 1.0."""
        checker = DriftChecker(drift_db)
        result  = checker.run_all("area-52")

        assert result.overall_drift_level    == DriftLevel.NONE
        assert result.uncertainty_multiplier == 1.0
        assert result.retrain_recommended    == False

    def test_uncertainty_multiplier_matches_policy(self, drift_db: sqlite3.Connection):
        """Verify that the uncertainty_multiplier in the result matches adaptive policy."""
        from wow_forecaster.monitoring.adaptive import evaluate_policy
        checker = DriftChecker(drift_db)
        result  = checker.run_all("area-52")
        policy  = evaluate_policy(result.overall_drift_level)
        assert result.uncertainty_multiplier == policy.uncertainty_multiplier
