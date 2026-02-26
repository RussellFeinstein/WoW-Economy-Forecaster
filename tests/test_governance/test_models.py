"""
Tests for governance/models.py — SourcePolicy and sub-model validation.

Covers:
  - Valid construction of all sub-models and SourcePolicy
  - Field-level validation: non-negative rates, valid strategy, positive hours
  - Model-level validation: freshness thresholds must be non-decreasing
  - Invalid values raise pydantic.ValidationError
  - Frozen models reject mutation
"""

import pytest
from pydantic import ValidationError

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


def _make_policy(**overrides) -> SourcePolicy:
    """Helper: build a minimal valid SourcePolicy with optional overrides."""
    defaults = dict(
        source_id="test_source",
        display_name="Test Source",
        source_type="auction_data",
        access_method="api",
        requires_auth=True,
        enabled=False,
        rate_limit=RateLimitConfig(requests_per_minute=10, cooldown_seconds=5.0),
        backoff=BackoffConfig(strategy="exponential", base_seconds=1.0),
        freshness=FreshnessConfig(
            ttl_hours=1.0,
            refresh_cadence_hours=1.0,
            stale_threshold_hours=3.0,
            critical_threshold_hours=25.0,
        ),
        provenance=ProvenanceRequirements(requires_snapshot=True),
        retention=RetentionConfig(raw_snapshot_days=30),
        policy_notes=PolicyNotes(access_type="authorized_api"),
    )
    defaults.update(overrides)
    return SourcePolicy(**defaults)


# ── RateLimitConfig ───────────────────────────────────────────────────────────


class TestRateLimitConfig:
    def test_defaults(self):
        r = RateLimitConfig()
        assert r.requests_per_minute == 0
        assert r.cooldown_seconds == 0.0

    def test_valid_values(self):
        r = RateLimitConfig(requests_per_minute=20, requests_per_hour=500, burst_limit=5, cooldown_seconds=3.0)
        assert r.requests_per_minute == 20
        assert r.cooldown_seconds == 3.0

    def test_zero_means_unlimited(self):
        # 0 is explicitly valid (means "no limit enforced")
        r = RateLimitConfig(requests_per_minute=0)
        assert r.requests_per_minute == 0

    def test_negative_requests_per_minute_raises(self):
        with pytest.raises(ValidationError, match="must be >= 0"):
            RateLimitConfig(requests_per_minute=-1)

    def test_negative_burst_limit_raises(self):
        with pytest.raises(ValidationError, match="must be >= 0"):
            RateLimitConfig(burst_limit=-5)

    def test_negative_cooldown_raises(self):
        with pytest.raises(ValidationError, match="must be >= 0"):
            RateLimitConfig(cooldown_seconds=-0.5)

    def test_frozen(self):
        r = RateLimitConfig(cooldown_seconds=1.0)
        with pytest.raises(Exception):
            r.cooldown_seconds = 2.0  # type: ignore[misc]


# ── BackoffConfig ─────────────────────────────────────────────────────────────


class TestBackoffConfig:
    def test_defaults(self):
        b = BackoffConfig()
        assert b.strategy == "exponential"
        assert b.max_retries == 5

    def test_valid_strategies(self):
        for strategy in ("exponential", "linear", "fixed"):
            b = BackoffConfig(strategy=strategy)
            assert b.strategy == strategy

    def test_invalid_strategy_raises(self):
        with pytest.raises(ValidationError, match="strategy must be one of"):
            BackoffConfig(strategy="random")

    def test_negative_base_seconds_raises(self):
        with pytest.raises(ValidationError, match="must be >= 0"):
            BackoffConfig(base_seconds=-1.0)

    def test_negative_max_seconds_raises(self):
        with pytest.raises(ValidationError, match="must be >= 0"):
            BackoffConfig(max_seconds=-1.0)

    def test_negative_max_retries_raises(self):
        with pytest.raises(ValidationError, match="must be >= 0"):
            BackoffConfig(max_retries=-1)

    def test_zero_retries_valid(self):
        b = BackoffConfig(max_retries=0)
        assert b.max_retries == 0


# ── FreshnessConfig ───────────────────────────────────────────────────────────


class TestFreshnessConfig:
    def test_valid_construction(self):
        f = FreshnessConfig(
            ttl_hours=1.0,
            refresh_cadence_hours=1.0,
            stale_threshold_hours=3.0,
            critical_threshold_hours=25.0,
        )
        assert f.ttl_hours == 1.0
        assert f.critical_threshold_hours == 25.0

    def test_stale_less_than_ttl_raises(self):
        with pytest.raises(ValidationError, match="stale_threshold_hours"):
            FreshnessConfig(
                ttl_hours=10.0,
                refresh_cadence_hours=10.0,
                stale_threshold_hours=5.0,   # less than ttl
                critical_threshold_hours=25.0,
            )

    def test_critical_less_than_stale_raises(self):
        with pytest.raises(ValidationError, match="critical_threshold_hours"):
            FreshnessConfig(
                ttl_hours=1.0,
                refresh_cadence_hours=1.0,
                stale_threshold_hours=10.0,
                critical_threshold_hours=5.0,  # less than stale
            )

    def test_equal_thresholds_valid(self):
        # stale == ttl and critical == stale are both acceptable
        f = FreshnessConfig(
            ttl_hours=5.0,
            refresh_cadence_hours=5.0,
            stale_threshold_hours=5.0,
            critical_threshold_hours=5.0,
        )
        assert f.stale_threshold_hours == 5.0

    def test_zero_ttl_raises(self):
        with pytest.raises(ValidationError, match="must be > 0"):
            FreshnessConfig(ttl_hours=0.0, refresh_cadence_hours=1.0,
                            stale_threshold_hours=2.0, critical_threshold_hours=25.0)


# ── PolicyNotes ───────────────────────────────────────────────────────────────


class TestPolicyNotes:
    def test_valid_access_types(self):
        for at in ("authorized_api", "export", "manual"):
            n = PolicyNotes(access_type=at)
            assert n.access_type == at

    def test_invalid_access_type_raises(self):
        with pytest.raises(ValidationError, match="access_type must be one of"):
            PolicyNotes(access_type="public_scrape")

    def test_defaults(self):
        n = PolicyNotes(access_type="manual")
        assert n.personal_research_only is True
        assert n.requires_registered_account is False


# ── SourcePolicy ──────────────────────────────────────────────────────────────


class TestSourcePolicy:
    def test_valid_construction(self):
        p = _make_policy()
        assert p.source_id == "test_source"
        assert p.enabled is False

    def test_invalid_source_type_raises(self):
        with pytest.raises(ValidationError, match="source_type must be one of"):
            _make_policy(source_type="price_feed")

    def test_invalid_access_method_raises(self):
        with pytest.raises(ValidationError, match="access_method must be one of"):
            _make_policy(access_method="scrape")

    def test_all_source_types_valid(self):
        for st in ("auction_data", "news_event", "manual_event", "other"):
            p = _make_policy(source_type=st)
            assert p.source_type == st

    def test_all_access_methods_valid(self):
        for am in ("api", "export", "manual"):
            p = _make_policy(access_method=am)
            assert p.access_method == am

    def test_frozen(self):
        p = _make_policy()
        with pytest.raises(Exception):
            p.enabled = True  # type: ignore[misc]

    def test_enabled_true(self):
        p = _make_policy(enabled=True)
        assert p.enabled is True

    def test_manual_source_no_auth(self):
        p = _make_policy(
            access_method="manual",
            requires_auth=False,
            source_type="manual_event",
            rate_limit=RateLimitConfig(),
            backoff=BackoffConfig(strategy="fixed", base_seconds=0.0, max_seconds=0.0, max_retries=0),
            freshness=FreshnessConfig(
                ttl_hours=720.0,
                refresh_cadence_hours=720.0,
                stale_threshold_hours=1440.0,
                critical_threshold_hours=2160.0,
            ),
            provenance=ProvenanceRequirements(requires_snapshot=False),
            retention=RetentionConfig(raw_snapshot_days=0),
            policy_notes=PolicyNotes(access_type="manual"),
        )
        assert p.requires_auth is False
        assert p.provenance.requires_snapshot is False
        assert p.retention.raw_snapshot_days == 0
