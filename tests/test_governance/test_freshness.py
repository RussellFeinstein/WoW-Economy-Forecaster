"""
Tests for governance/freshness.py — FreshnessResult, check_source_freshness().

Covers:
  - TTL classification: fresh / aging / stale / critical / unknown
  - Manual sources (requires_snapshot=False) always return UNKNOWN
  - Age calculation from snapshot timestamps
  - check_all_sources_freshness batches correctly
  - DB query: most recent successful snapshot is used
"""

import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

from wow_forecaster.governance.freshness import (
    FreshnessStatus,
    check_all_sources_freshness,
    check_source_freshness,
    _classify_status,
    _compute_age_hours,
)
from wow_forecaster.governance.models import (
    BackoffConfig,
    FreshnessConfig,
    PolicyNotes,
    ProvenanceRequirements,
    RateLimitConfig,
    RetentionConfig,
    SourcePolicy,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────


def _make_freshness_policy(
    source_id: str = "test_api",
    ttl_hours: float = 1.0,
    stale_threshold_hours: float = 3.0,
    critical_threshold_hours: float = 25.0,
    requires_snapshot: bool = True,
) -> SourcePolicy:
    return SourcePolicy(
        source_id=source_id,
        display_name="Test",
        source_type="auction_data",
        access_method="api",
        requires_auth=True,
        enabled=True,
        rate_limit=RateLimitConfig(),
        backoff=BackoffConfig(),
        freshness=FreshnessConfig(
            ttl_hours=ttl_hours,
            refresh_cadence_hours=ttl_hours,
            stale_threshold_hours=stale_threshold_hours,
            critical_threshold_hours=critical_threshold_hours,
        ),
        provenance=ProvenanceRequirements(requires_snapshot=requires_snapshot),
        retention=RetentionConfig(),
        policy_notes=PolicyNotes(access_type="authorized_api"),
    )


@pytest.fixture()
def fresh_db() -> sqlite3.Connection:
    """In-memory SQLite DB with ingestion_snapshots table."""
    conn = sqlite3.connect(":memory:")
    conn.execute("PRAGMA foreign_keys = OFF")
    conn.execute(
        """
        CREATE TABLE ingestion_snapshots (
            snapshot_id  INTEGER PRIMARY KEY,
            run_id       INTEGER,
            source       TEXT NOT NULL,
            endpoint     TEXT,
            snapshot_path TEXT,
            content_hash TEXT,
            record_count INTEGER DEFAULT 0,
            success      INTEGER NOT NULL DEFAULT 1,
            error_message TEXT,
            fetched_at   TEXT NOT NULL
        )
        """
    )
    conn.commit()
    return conn


def _insert_snapshot(
    conn: sqlite3.Connection,
    source: str,
    fetched_at: str,
    success: int = 1,
) -> None:
    conn.execute(
        "INSERT INTO ingestion_snapshots (source, fetched_at, success) VALUES (?,?,?)",
        (source, fetched_at, success),
    )
    conn.commit()


def _utcnow() -> datetime:
    return datetime.now(tz=timezone.utc)


def _ts(hours_ago: float) -> str:
    return (_utcnow() - timedelta(hours=hours_ago)).isoformat()


# ── _classify_status ──────────────────────────────────────────────────────────


class TestClassifyStatus:
    def test_none_age_returns_unknown(self):
        assert _classify_status(None, 1.0, 3.0, 25.0) == FreshnessStatus.UNKNOWN

    def test_within_ttl_is_fresh(self):
        assert _classify_status(0.5, 1.0, 3.0, 25.0) == FreshnessStatus.FRESH

    def test_at_ttl_boundary_is_aging(self):
        # Exactly at ttl_hours: not within TTL (< used), so AGING
        assert _classify_status(1.0, 1.0, 3.0, 25.0) == FreshnessStatus.AGING

    def test_between_ttl_and_stale_is_aging(self):
        assert _classify_status(2.0, 1.0, 3.0, 25.0) == FreshnessStatus.AGING

    def test_at_stale_boundary_is_stale(self):
        assert _classify_status(3.0, 1.0, 3.0, 25.0) == FreshnessStatus.STALE

    def test_between_stale_and_critical_is_stale(self):
        assert _classify_status(10.0, 1.0, 3.0, 25.0) == FreshnessStatus.STALE

    def test_at_critical_is_critical(self):
        assert _classify_status(25.0, 1.0, 3.0, 25.0) == FreshnessStatus.CRITICAL

    def test_beyond_critical_is_critical(self):
        assert _classify_status(100.0, 1.0, 3.0, 25.0) == FreshnessStatus.CRITICAL


# ── _compute_age_hours ────────────────────────────────────────────────────────


class TestComputeAgeHours:
    def test_none_returns_none(self):
        assert _compute_age_hours(None) is None

    def test_recent_timestamp(self):
        ts = _ts(2.0)  # 2 hours ago
        age = _compute_age_hours(ts)
        assert age is not None
        assert 1.9 < age < 2.1  # within 6-minute tolerance

    def test_z_suffix_parsed(self):
        ts = (_utcnow() - timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
        age = _compute_age_hours(ts)
        assert age is not None
        assert 0.9 < age < 1.1

    def test_malformed_returns_none(self):
        assert _compute_age_hours("not-a-timestamp") is None


# ── check_source_freshness ────────────────────────────────────────────────────


class TestCheckSourceFreshness:
    def test_fresh_snapshot(self, fresh_db):
        _insert_snapshot(fresh_db, "test_api", _ts(0.5))
        policy = _make_freshness_policy(source_id="test_api", ttl_hours=1.0, stale_threshold_hours=3.0)
        result = check_source_freshness(fresh_db, "test_api", policy)

        assert result.source_id == "test_api"
        assert result.status == FreshnessStatus.FRESH
        assert result.is_within_ttl is True
        assert result.is_stale is False
        assert result.is_critical is False
        assert result.requires_snapshot is True

    def test_stale_snapshot(self, fresh_db):
        _insert_snapshot(fresh_db, "test_api", _ts(5.0))
        policy = _make_freshness_policy(source_id="test_api", ttl_hours=1.0, stale_threshold_hours=3.0)
        result = check_source_freshness(fresh_db, "test_api", policy)

        assert result.status == FreshnessStatus.STALE
        assert result.is_stale is True
        assert result.is_critical is False

    def test_critical_snapshot(self, fresh_db):
        _insert_snapshot(fresh_db, "test_api", _ts(30.0))
        policy = _make_freshness_policy(
            source_id="test_api", ttl_hours=1.0, stale_threshold_hours=3.0, critical_threshold_hours=25.0
        )
        result = check_source_freshness(fresh_db, "test_api", policy)

        assert result.status == FreshnessStatus.CRITICAL
        assert result.is_critical is True

    def test_no_snapshot_returns_unknown(self, fresh_db):
        policy = _make_freshness_policy(source_id="missing_source")
        result = check_source_freshness(fresh_db, "missing_source", policy)

        assert result.status == FreshnessStatus.UNKNOWN
        assert result.last_snapshot_at is None
        assert result.age_hours is None

    def test_failed_snapshots_ignored(self, fresh_db):
        # Inserts only failed snapshots — should behave like no snapshot
        _insert_snapshot(fresh_db, "test_api", _ts(0.5), success=0)
        policy = _make_freshness_policy(source_id="test_api")
        result = check_source_freshness(fresh_db, "test_api", policy)

        assert result.status == FreshnessStatus.UNKNOWN

    def test_most_recent_snapshot_used(self, fresh_db):
        # Insert old and new — result should reflect the most recent
        _insert_snapshot(fresh_db, "test_api", _ts(24.0))
        _insert_snapshot(fresh_db, "test_api", _ts(0.3))  # most recent
        policy = _make_freshness_policy(source_id="test_api", ttl_hours=1.0)
        result = check_source_freshness(fresh_db, "test_api", policy)

        assert result.status == FreshnessStatus.FRESH

    def test_manual_source_returns_unknown(self, fresh_db):
        # Even if there are snapshots, manual sources skip the query
        _insert_snapshot(fresh_db, "manual_csv", _ts(0.1))
        policy = _make_freshness_policy(
            source_id="manual_csv",
            requires_snapshot=False,
        )
        result = check_source_freshness(fresh_db, "manual_csv", policy)

        assert result.status == FreshnessStatus.UNKNOWN
        assert result.requires_snapshot is False

    def test_result_fields_populated(self, fresh_db):
        _insert_snapshot(fresh_db, "test_api", _ts(0.5))
        policy = _make_freshness_policy(
            source_id="test_api",
            ttl_hours=1.0,
            stale_threshold_hours=3.0,
            critical_threshold_hours=25.0,
        )
        r = check_source_freshness(fresh_db, "test_api", policy)

        assert r.ttl_hours == 1.0
        assert r.stale_threshold_hours == 3.0
        assert r.critical_threshold_hours == 25.0
        assert r.last_snapshot_at is not None


# ── check_all_sources_freshness ───────────────────────────────────────────────


class TestCheckAllSourcesFreshness:
    def test_empty_policies_returns_empty(self, fresh_db):
        results = check_all_sources_freshness(fresh_db, [])
        assert results == []

    def test_multiple_sources(self, fresh_db):
        _insert_snapshot(fresh_db, "src_a", _ts(0.5))
        _insert_snapshot(fresh_db, "src_b", _ts(10.0))

        policies = [
            _make_freshness_policy("src_a", ttl_hours=1.0, stale_threshold_hours=3.0),
            _make_freshness_policy("src_b", ttl_hours=1.0, stale_threshold_hours=3.0),
        ]
        results = check_all_sources_freshness(fresh_db, policies)

        assert len(results) == 2
        assert results[0].source_id == "src_a"
        assert results[0].status == FreshnessStatus.FRESH
        assert results[1].source_id == "src_b"
        assert results[1].status == FreshnessStatus.STALE

    def test_preserves_order(self, fresh_db):
        policies = [
            _make_freshness_policy("z_source"),
            _make_freshness_policy("a_source"),
        ]
        results = check_all_sources_freshness(fresh_db, policies)
        assert results[0].source_id == "z_source"
        assert results[1].source_id == "a_source"
