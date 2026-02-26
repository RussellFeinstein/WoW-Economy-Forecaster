"""
Tests for governance/preflight.py — run_preflight_checks(), assert_source_enabled().

Covers:
  - Enabled source passes all checks
  - Disabled source is blocked (checks["enabled"]=False, blocked_reason set)
  - Cooldown active: checks["cooldown"]=False, warning emitted
  - Cooldown elapsed: checks["cooldown"]=True, passes
  - No last_call_at: cooldown check skipped (treated as passed)
  - assert_source_enabled raises SourceDisabledError for disabled source
  - assert_source_enabled passes silently for enabled source
  - PreflightCheckResult.passed is True only when ALL checks pass
"""

from datetime import datetime, timedelta, timezone

import pytest

from wow_forecaster.governance.models import (
    BackoffConfig,
    FreshnessConfig,
    PolicyNotes,
    ProvenanceRequirements,
    RateLimitConfig,
    RetentionConfig,
    SourcePolicy,
)
from wow_forecaster.governance.preflight import (
    PreflightCheckResult,
    SourceDisabledError,
    assert_source_enabled,
    run_preflight_checks,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────


def _make_policy(
    enabled: bool = True,
    cooldown_seconds: float = 60.0,
    source_id: str = "test_source",
) -> SourcePolicy:
    return SourcePolicy(
        source_id=source_id,
        display_name="Test Source",
        source_type="auction_data",
        access_method="api",
        requires_auth=True,
        enabled=enabled,
        rate_limit=RateLimitConfig(
            requests_per_minute=10,
            cooldown_seconds=cooldown_seconds,
        ),
        backoff=BackoffConfig(),
        freshness=FreshnessConfig(
            ttl_hours=1.0,
            refresh_cadence_hours=1.0,
            stale_threshold_hours=3.0,
            critical_threshold_hours=25.0,
        ),
        provenance=ProvenanceRequirements(),
        retention=RetentionConfig(),
        policy_notes=PolicyNotes(access_type="authorized_api"),
    )


def _make_manual_policy(source_id: str = "manual_csv") -> SourcePolicy:
    """Manual source: enabled, no rate limits."""
    return SourcePolicy(
        source_id=source_id,
        display_name="Manual CSV",
        source_type="manual_event",
        access_method="manual",
        requires_auth=False,
        enabled=True,
        rate_limit=RateLimitConfig(cooldown_seconds=0.0),
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


def _utcnow() -> datetime:
    return datetime.now(tz=timezone.utc)


# ── run_preflight_checks ──────────────────────────────────────────────────────


class TestRunPreflightChecks:
    def test_enabled_no_last_call_passes(self):
        policy = _make_policy(enabled=True, cooldown_seconds=60.0)
        result = run_preflight_checks("test_source", policy, last_call_at=None)

        assert result.passed is True
        assert result.blocked_reason is None
        assert result.checks["enabled"] is True
        assert result.checks["policy_present"] is True
        assert result.checks.get("cooldown", True) is True
        assert result.errors == []

    def test_disabled_source_blocked(self):
        policy = _make_policy(enabled=False)
        result = run_preflight_checks("test_source", policy)

        assert result.passed is False
        assert result.checks["enabled"] is False
        assert result.blocked_reason is not None
        assert "disabled" in result.blocked_reason.lower()
        assert len(result.errors) == 1

    def test_cooldown_active_adds_warning_not_error(self):
        policy = _make_policy(enabled=True, cooldown_seconds=60.0)
        # Called 5 seconds ago — still in 60s cooldown
        last_call = _utcnow() - timedelta(seconds=5)
        result = run_preflight_checks("test_source", policy, last_call_at=last_call)

        # Cooldown is a WARNING (non-blocking) — passes is still True
        # because only "enabled" is a hard block; cooldown is advisory
        assert result.checks["cooldown"] is False
        assert len(result.warnings) >= 1
        assert "cooldown" in result.warnings[0].lower()
        # passed = all(checks.values()) → cooldown=False → passed=False
        assert result.passed is False

    def test_cooldown_elapsed_passes(self):
        policy = _make_policy(enabled=True, cooldown_seconds=5.0)
        # Called 10 seconds ago — cooldown of 5s has elapsed
        last_call = _utcnow() - timedelta(seconds=10)
        result = run_preflight_checks("test_source", policy, last_call_at=last_call)

        assert result.checks["cooldown"] is True
        assert result.passed is True
        assert result.errors == []

    def test_no_cooldown_set_skips_check(self):
        policy = _make_policy(enabled=True, cooldown_seconds=0.0)
        # Even with last_call_at set, cooldown=0 means no check is enforced
        last_call = _utcnow() - timedelta(seconds=1)
        result = run_preflight_checks("test_source", policy, last_call_at=last_call)

        assert result.checks["cooldown"] is True
        assert result.passed is True

    def test_manual_source_no_cooldown_passes(self):
        policy = _make_manual_policy()
        result = run_preflight_checks("manual_csv", policy)

        assert result.passed is True
        assert result.errors == []

    def test_disabled_source_blocked_reason_mentions_source_id(self):
        policy = _make_policy(enabled=False, source_id="blizzard_api")
        result = run_preflight_checks("blizzard_api", policy)

        assert "blizzard_api" in result.blocked_reason

    def test_policy_present_always_true(self):
        policy = _make_policy(enabled=True)
        result = run_preflight_checks("test_source", policy)
        assert result.checks["policy_present"] is True

    def test_result_is_frozen(self):
        policy = _make_policy(enabled=True)
        result = run_preflight_checks("test_source", policy)
        with pytest.raises(Exception):
            result.passed = False  # type: ignore[misc]

    def test_cooldown_warning_mentions_remaining_time(self):
        policy = _make_policy(enabled=True, cooldown_seconds=60.0)
        last_call = _utcnow() - timedelta(seconds=10)
        result = run_preflight_checks("test_source", policy, last_call_at=last_call)

        assert any("50" in w or "remaining" in w for w in result.warnings)

    def test_last_call_naive_datetime_handled(self):
        """Naive datetime (no tzinfo) should not raise — treated as UTC."""
        policy = _make_policy(enabled=True, cooldown_seconds=60.0)
        # Strip tzinfo from an aware datetime to produce a naive one.
        # Avoids the deprecated datetime.utcnow() (removed in future Python).
        last_call = datetime.now(tz=timezone.utc).replace(tzinfo=None) - timedelta(seconds=5)
        # Should not raise; cooldown check should still work
        result = run_preflight_checks("test_source", policy, last_call_at=last_call)
        assert result.checks["cooldown"] is False  # still in cooldown


# ── assert_source_enabled ─────────────────────────────────────────────────────


class TestAssertSourceEnabled:
    def test_enabled_source_passes_silently(self):
        policy = _make_policy(enabled=True)
        # Should not raise
        assert_source_enabled("test_source", policy)

    def test_disabled_source_raises(self):
        policy = _make_policy(enabled=False, source_id="undermine_exchange")
        with pytest.raises(SourceDisabledError) as exc_info:
            assert_source_enabled("undermine_exchange", policy)

        assert exc_info.value.source_id == "undermine_exchange"
        assert "undermine_exchange" in str(exc_info.value)

    def test_error_message_guides_user(self):
        policy = _make_policy(enabled=False)
        with pytest.raises(SourceDisabledError) as exc_info:
            assert_source_enabled("test_source", policy)

        msg = str(exc_info.value)
        assert "enabled" in msg.lower()
