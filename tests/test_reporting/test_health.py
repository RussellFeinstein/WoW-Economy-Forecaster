"""
Tests for wow_forecaster/reporting/health.py.

What we test
------------
collect_health_report():
  - Returns correct generated_at timestamp.
  - Per-realm: days_with_data counts distinct calendar days only.
  - Per-realm: gaps correctly identified when days are missing.
  - Per-realm: no gaps when all days in window have data.
  - Per-realm: coverage_pct computed correctly.
  - Per-realm: last_ingest_at populated from market_observations_raw.
  - Per-realm: last_ingest_age_hours reflects age of last ingest.
  - Realm isolation: stats only include rows for the queried realm.
  - is_stale=True when last ingest is older than stale_threshold_hours.
  - is_stale=False when last ingest is within threshold.
  - is_stale=True when no ingest snapshots exist.
  - last_hourly_run populated from run_metadata (orchestrator stage).
  - last_forecast_run populated from run_metadata (recommend stage).
  - item_forecast_count counts distinct item_ids.

format_health_report():
  - Output contains realm slug.
  - Output contains [HEALTHY] when not stale.
  - Output contains [STALE] when stale.
  - Output shows gap dates when gaps exist.
  - Output shows "none detected" when no gaps.
"""

from __future__ import annotations

import itertools
import sqlite3
from datetime import UTC, datetime, timedelta

import pytest

from wow_forecaster.db.schema import apply_schema
from wow_forecaster.reporting.health import (
    HealthReport,
    collect_health_report,
    format_health_report,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def conn() -> sqlite3.Connection:
    """In-memory DB with the real production schema.

    apply_schema() rather than hand-rolled DDL: a hand-rolled fixture once
    invented an ingestion_snapshots shape (realm_slug + ingested_at columns)
    that production never had, which let a crashing query pass its tests
    (issue #12).  Foreign keys are off so helpers can insert minimal rows.
    """
    db = sqlite3.connect(":memory:")
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA foreign_keys = OFF;")
    apply_schema(db)
    db.commit()
    return db


# ── Helpers ───────────────────────────────────────────────────────────────────

_SLUG_COUNTER = itertools.count()


def _insert_obs(
    conn, realm: str, days_ago: float, count: int = 1,
    anchor: datetime | None = None,
) -> None:
    if anchor is None:
        anchor = datetime.now(tz=UTC)
    ts = (anchor - timedelta(days=days_ago)).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    for _ in range(count):
        conn.execute(
            "INSERT INTO market_observations_normalized "
            "(obs_id, item_id, realm_slug, observed_at, price_gold) "
            "VALUES (1, 1, ?, ?, 10.0)",
            (realm, ts),
        )
    conn.commit()


def _insert_ingested_row(conn, realm: str, hours_ago: float) -> None:
    """Insert one raw observation whose ingested_at is hours_ago old.

    Only successful ingest runs write market_observations_raw rows, so
    inserting a row IS the record of a successful ingest.  A failed ingest
    leaves no row, which the tests model by simply not calling this helper.
    """
    ts = (datetime.now(tz=UTC) - timedelta(hours=hours_ago)).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    conn.execute(
        "INSERT INTO market_observations_raw "
        "(item_id, realm_slug, observed_at, source, ingested_at) "
        "VALUES (1, ?, ?, 'test', ?)",
        (realm, ts, ts),
    )
    conn.commit()


def _insert_run(
    conn,
    stage: str,
    status: str = "success",
    hours_ago: float = 1.0,
) -> None:
    ts = (datetime.now(tz=UTC) - timedelta(hours=hours_ago)).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    conn.execute(
        "INSERT INTO run_metadata "
        "(run_slug, pipeline_stage, status, started_at, config_snapshot) "
        "VALUES (?, ?, ?, ?, '{}')",
        (f"health-test-{next(_SLUG_COUNTER)}", stage, status, ts),
    )
    conn.commit()


def _insert_item_forecast(conn, item_id: int) -> None:
    conn.execute(
        "INSERT INTO forecast_outputs "
        "(run_id, item_id, archetype_id, realm_slug, forecast_horizon, "
        " target_date, predicted_price_gold, confidence_lower, "
        " confidence_upper, model_slug) "
        "VALUES (1, ?, NULL, 'us', '1d', '2026-03-11', 100.0, 90.0, 110.0, 'lgbm')",
        (item_id,),
    )
    conn.commit()


# ── Tests: collect_health_report ──────────────────────────────────────────────

class TestCollectHealthReport:
    def test_generated_at_is_populated(self, conn):
        report = collect_health_report(conn, ["us"])
        assert report.generated_at != ""
        assert "T" in report.generated_at

    def test_days_with_data_counts_distinct_days(self, conn):
        # 3 obs on 1 day, 1 obs on another day → 2 distinct days
        _insert_obs(conn, "us", days_ago=0.5, count=3)
        _insert_obs(conn, "us", days_ago=1.5)
        report = collect_health_report(conn, ["us"], lookback_days=7)
        assert report.realms[0].days_with_data == 2

    def test_gaps_detected_when_days_missing(self, conn):
        # Only today has data → yesterday and before are gaps
        _insert_obs(conn, "us", days_ago=0.5)
        report = collect_health_report(conn, ["us"], lookback_days=3)
        stats = report.realms[0]
        # 3 days checked, 1 has data → 2 gaps
        assert stats.days_with_data == 1
        assert len(stats.gap_dates) == 2

    def test_no_gaps_when_all_days_covered(self, conn):
        # A midday anchor puts each (d + 0.5)-days-ago observation at 00:00 of
        # a distinct date, covering exactly as_of .. as_of-6. A wall-clock
        # anchor flips the day-0 slot to yesterday whenever the suite runs
        # before 12:00 UTC.
        anchor = datetime(2026, 3, 9, 12, 0, tzinfo=UTC)
        for d in range(7):
            _insert_obs(conn, "us", days_ago=d + 0.5, anchor=anchor)
        report = collect_health_report(
            conn, ["us"], lookback_days=7, as_of=anchor.date()
        )
        assert report.realms[0].gap_dates == []

    def test_coverage_pct_computed(self, conn):
        # 1 day of data in a 4-day window → 25%
        _insert_obs(conn, "us", days_ago=0.5)
        report = collect_health_report(conn, ["us"], lookback_days=4)
        assert abs(report.realms[0].coverage_pct - 25.0) < 1.0

    def test_realm_isolation(self, conn):
        _insert_obs(conn, "us", days_ago=0.5)
        _insert_obs(conn, "eu", days_ago=0.5)
        report = collect_health_report(conn, ["us"], lookback_days=7)
        # Only 1 realm in the result
        assert len(report.realms) == 1
        assert report.realms[0].realm_slug == "us"

    def test_last_ingest_at_from_raw_observations(self, conn):
        _insert_ingested_row(conn, "us", hours_ago=1.5)
        report = collect_health_report(conn, ["us"])
        assert report.realms[0].last_ingest_at is not None
        assert report.realms[0].last_ingest_age_hours is not None
        assert report.realms[0].last_ingest_age_hours < 3.0

    def test_newest_ingested_row_wins(self, conn):
        _insert_ingested_row(conn, "us", hours_ago=50.0)
        _insert_ingested_row(conn, "us", hours_ago=2.0)
        report = collect_health_report(conn, ["us"])
        assert report.realms[0].last_ingest_age_hours < 3.0

    def test_ingest_realm_isolation(self, conn):
        _insert_ingested_row(conn, "eu", hours_ago=0.5)
        report = collect_health_report(conn, ["us"])
        # Another realm's ingest is not this realm's last ingest
        assert report.realms[0].last_ingest_at is None

    def test_is_stale_when_no_ingests(self, conn):
        report = collect_health_report(conn, ["us"], stale_threshold_hours=4.0)
        assert report.is_stale is True

    def test_is_stale_when_ingest_too_old(self, conn):
        _insert_ingested_row(conn, "us", hours_ago=6.0)
        report = collect_health_report(conn, ["us"], stale_threshold_hours=4.0)
        assert report.is_stale is True

    def test_not_stale_when_fresh_ingest(self, conn):
        _insert_ingested_row(conn, "us", hours_ago=1.0)
        report = collect_health_report(conn, ["us"], stale_threshold_hours=4.0)
        assert report.is_stale is False

    def test_last_hourly_run_from_run_metadata(self, conn):
        _insert_run(conn, stage="orchestrator", status="success", hours_ago=2.0)
        report = collect_health_report(conn, ["us"])
        assert report.last_hourly_run is not None
        assert report.last_hourly_status == "success"
        assert report.last_hourly_age_hours is not None
        assert abs(report.last_hourly_age_hours - 2.0) < 0.1

    def test_partial_hourly_run_counts(self, conn):
        _insert_run(conn, stage="orchestrator", status="partial", hours_ago=1.0)
        report = collect_health_report(conn, ["us"])
        assert report.last_hourly_run is not None
        assert report.last_hourly_status == "partial"

    def test_failed_hourly_run_not_counted(self, conn):
        _insert_run(conn, stage="orchestrator", status="failed", hours_ago=0.5)
        report = collect_health_report(conn, ["us"])
        assert report.last_hourly_run is None

    def test_last_forecast_run_from_run_metadata(self, conn):
        _insert_run(conn, stage="recommend", status="success", hours_ago=12.0)
        report = collect_health_report(conn, ["us"])
        assert report.last_forecast_run is not None
        assert abs(report.last_forecast_age_hours - 12.0) < 0.1

    def test_item_forecast_count(self, conn):
        _insert_item_forecast(conn, 100)
        _insert_item_forecast(conn, 200)
        _insert_item_forecast(conn, 100)  # duplicate item_id
        report = collect_health_report(conn, ["us"])
        assert report.item_forecast_count == 2


# ── Tests: format_health_report ───────────────────────────────────────────────

class TestFormatHealthReport:
    def _base_report(self, is_stale: bool = False) -> HealthReport:
        from wow_forecaster.reporting.health import RealmHealthStats
        stats = RealmHealthStats(
            realm_slug="us",
            first_obs_date="2026-02-25",
            last_obs_date="2026-03-09",
            days_with_data=13,
            days_checked=14,
            coverage_pct=92.9,
            gap_dates=[],
        )
        return HealthReport(
            generated_at="2026-03-10T10:00:00Z",
            realms=[stats],
            is_stale=is_stale,
        )

    def test_contains_realm_slug(self):
        report = self._base_report()
        output = format_health_report(report)
        assert "us" in output

    def test_healthy_label_when_not_stale(self):
        report = self._base_report(is_stale=False)
        output = format_health_report(report)
        assert "[HEALTHY]" in output

    def test_stale_label_when_stale(self):
        report = self._base_report(is_stale=True)
        output = format_health_report(report)
        assert "[STALE]" in output

    def test_gap_dates_shown(self):
        from wow_forecaster.reporting.health import RealmHealthStats
        stats = RealmHealthStats(
            realm_slug="us",
            days_with_data=12,
            days_checked=14,
            coverage_pct=85.7,
            gap_dates=["2026-03-05", "2026-03-07"],
        )
        report = HealthReport(
            generated_at="2026-03-10T10:00:00Z",
            realms=[stats],
        )
        output = format_health_report(report)
        assert "2026-03-05" in output
        assert "2026-03-07" in output

    def test_no_gaps_message_when_clean(self):
        report = self._base_report()
        output = format_health_report(report)
        assert "none detected" in output
