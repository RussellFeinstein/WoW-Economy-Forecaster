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
  - Per-realm: last_ingest_at populated from ingestion_snapshots.
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

import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

from wow_forecaster.reporting.health import (
    HealthReport,
    collect_health_report,
    format_health_report,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def conn() -> sqlite3.Connection:
    """In-memory DB with minimal health-check schema."""
    db = sqlite3.connect(":memory:")
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA foreign_keys = OFF;")
    db.executescript("""
        CREATE TABLE market_observations_normalized (
            norm_id      INTEGER PRIMARY KEY AUTOINCREMENT,
            item_id      INTEGER NOT NULL DEFAULT 1,
            archetype_id INTEGER,
            realm_slug   TEXT    NOT NULL,
            faction      TEXT    NOT NULL DEFAULT 'neutral',
            observed_at  TEXT    NOT NULL,
            price_gold   REAL    NOT NULL DEFAULT 10.0,
            is_outlier   INTEGER NOT NULL DEFAULT 0
        );
        CREATE TABLE ingestion_snapshots (
            snapshot_id  INTEGER PRIMARY KEY AUTOINCREMENT,
            realm_slug   TEXT    NOT NULL,
            source_id    TEXT    NOT NULL DEFAULT 'blizzard_api',
            success      INTEGER NOT NULL DEFAULT 1,
            record_count INTEGER NOT NULL DEFAULT 100,
            ingested_at  TEXT    NOT NULL
        );
        CREATE TABLE run_metadata (
            run_id         INTEGER PRIMARY KEY AUTOINCREMENT,
            realm_slug     TEXT    NOT NULL DEFAULT 'us',
            run_date       TEXT    NOT NULL DEFAULT '2026-03-10',
            status         TEXT    NOT NULL DEFAULT 'success',
            pipeline_stage TEXT    NOT NULL DEFAULT 'orchestrator',
            started_at     TEXT    NOT NULL,
            rows_processed INTEGER NOT NULL DEFAULT 0,
            error_message  TEXT,
            config_snapshot TEXT   NOT NULL DEFAULT '{}'
        );
        CREATE TABLE forecast_outputs (
            forecast_id          INTEGER PRIMARY KEY AUTOINCREMENT,
            item_id              INTEGER,
            archetype_id         INTEGER,
            realm_slug           TEXT    NOT NULL DEFAULT 'us',
            forecast_horizon     TEXT    NOT NULL DEFAULT '1d',
            target_date          TEXT    NOT NULL DEFAULT '2026-03-11',
            predicted_price_gold REAL    NOT NULL DEFAULT 100.0,
            confidence_lower     REAL    NOT NULL DEFAULT 90.0,
            confidence_upper     REAL    NOT NULL DEFAULT 110.0,
            confidence_pct       REAL    NOT NULL DEFAULT 0.80,
            model_slug           TEXT    NOT NULL DEFAULT 'lgbm',
            ci_quality           TEXT    NOT NULL DEFAULT 'good',
            run_id               INTEGER NOT NULL DEFAULT 1,
            created_at           TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now'))
        );
    """)
    db.commit()
    return db


# ── Helpers ───────────────────────────────────────────────────────────────────

def _insert_obs(conn, realm: str, days_ago: float, count: int = 1) -> None:
    ts = (datetime.now(tz=timezone.utc) - timedelta(days=days_ago)).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    for _ in range(count):
        conn.execute(
            "INSERT INTO market_observations_normalized "
            "(realm_slug, observed_at) VALUES (?, ?)",
            (realm, ts),
        )
    conn.commit()


def _insert_ingest_snap(conn, realm: str, hours_ago: float, success: int = 1) -> None:
    ts = (datetime.now(tz=timezone.utc) - timedelta(hours=hours_ago)).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    conn.execute(
        "INSERT INTO ingestion_snapshots (realm_slug, success, ingested_at) VALUES (?, ?, ?)",
        (realm, success, ts),
    )
    conn.commit()


def _insert_run(
    conn,
    stage: str,
    status: str = "success",
    hours_ago: float = 1.0,
) -> None:
    ts = (datetime.now(tz=timezone.utc) - timedelta(hours=hours_ago)).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    conn.execute(
        "INSERT INTO run_metadata (pipeline_stage, status, started_at) VALUES (?, ?, ?)",
        (stage, status, ts),
    )
    conn.commit()


def _insert_item_forecast(conn, item_id: int) -> None:
    conn.execute(
        "INSERT INTO forecast_outputs (item_id, archetype_id) VALUES (?, NULL)",
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
        for d in range(7):
            _insert_obs(conn, "us", days_ago=d + 0.5)
        report = collect_health_report(conn, ["us"], lookback_days=7)
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

    def test_last_ingest_at_from_snapshots(self, conn):
        _insert_ingest_snap(conn, "us", hours_ago=1.5)
        report = collect_health_report(conn, ["us"])
        assert report.realms[0].last_ingest_at is not None
        assert report.realms[0].last_ingest_age_hours is not None
        assert report.realms[0].last_ingest_age_hours < 3.0

    def test_only_successful_ingests_count(self, conn):
        _insert_ingest_snap(conn, "us", hours_ago=0.5, success=0)  # failed
        report = collect_health_report(conn, ["us"])
        # Failed snapshot should not be reported as last ingest
        assert report.realms[0].last_ingest_at is None

    def test_is_stale_when_no_ingests(self, conn):
        report = collect_health_report(conn, ["us"], stale_threshold_hours=4.0)
        assert report.is_stale is True

    def test_is_stale_when_ingest_too_old(self, conn):
        _insert_ingest_snap(conn, "us", hours_ago=6.0)
        report = collect_health_report(conn, ["us"], stale_threshold_hours=4.0)
        assert report.is_stale is True

    def test_not_stale_when_fresh_ingest(self, conn):
        _insert_ingest_snap(conn, "us", hours_ago=1.0)
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
