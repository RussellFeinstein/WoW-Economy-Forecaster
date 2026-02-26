"""
Pre-call preflight guardrails for source integrations.

The preflight system enforces governance policies *before* a pipeline stage
makes any call to an external source.  It is the enforcement point that turns
the declarative policy config into actual blocking behaviour.

Checks performed
----------------
1. ``policy_present``  — A registered policy exists for this source_id.
                         (Always True if you reached the check via registry.)
2. ``enabled``         — source.enabled = true in sources.toml.
3. ``cooldown``        — Minimum seconds between consecutive calls has elapsed.
                         Requires the caller to supply last_call_at.

The first failing check terminates the run and populates blocked_reason.

How to use
----------
In orchestrator._run_ingest() (or any future provider call site):

    from wow_forecaster.governance.preflight import run_preflight_checks
    from wow_forecaster.governance.registry import get_source_policy

    policy = get_source_policy("blizzard_api", config.governance.sources_config_path)
    result = run_preflight_checks("blizzard_api", policy, last_call_at=last_ts)

    if not result.passed:
        logger.warning("Source blocked: %s", result.blocked_reason)
        return  # skip this call

Raising vs returning
--------------------
``run_preflight_checks()`` returns a ``PreflightCheckResult`` — it does NOT
raise by default, so the caller decides whether to skip or abort.

``assert_source_enabled()`` raises ``SourceDisabledError`` if the source is
disabled — useful when you want a hard failure rather than a soft skip.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from wow_forecaster.governance.models import SourcePolicy


# ── Custom exceptions ─────────────────────────────────────────────────────────


class SourceDisabledError(RuntimeError):
    """Raised when attempting to use a disabled source.

    Attributes:
        source_id: The ID of the blocked source.
    """

    def __init__(self, source_id: str) -> None:
        self.source_id = source_id
        super().__init__(
            f"Source '{source_id}' is disabled.  "
            "Set enabled = true in config/sources.toml and ensure credentials are configured."
        )


class SourceCooldownError(RuntimeError):
    """Raised when a source is called before its cooldown period has elapsed.

    Attributes:
        source_id:         The ID of the throttled source.
        remaining_seconds: Approximate seconds remaining in the cooldown.
    """

    def __init__(self, source_id: str, remaining_seconds: float) -> None:
        self.source_id         = source_id
        self.remaining_seconds = remaining_seconds
        super().__init__(
            f"Source '{source_id}' is in cooldown.  "
            f"Wait approximately {remaining_seconds:.0f}s before calling again."
        )


# ── Result type ───────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class PreflightCheckResult:
    """Outcome of running preflight checks for one source.

    Attributes:
        source_id:      Source that was checked.
        passed:         True only if ALL checks passed.
        checks:         Dict of check_name -> bool (True = passed).
        warnings:       Non-blocking advisory messages (e.g., approaching rate limit).
        errors:         Blocking failure messages.
        blocked_reason: First blocking error message, or None if passed.
    """

    source_id:      str
    passed:         bool
    checks:         dict[str, bool]
    warnings:       list[str]
    errors:         list[str]
    blocked_reason: Optional[str]


# ── Public functions ──────────────────────────────────────────────────────────


def run_preflight_checks(
    source_id: str,
    policy: SourcePolicy,
    last_call_at: Optional[datetime] = None,
) -> PreflightCheckResult:
    """Run all preflight checks for a source before making a call.

    Checks are evaluated in order; the first failure sets blocked_reason.

    Args:
        source_id:    Source identifier (must match policy.source_id).
        policy:       SourcePolicy from the registry.
        last_call_at: UTC datetime of the most recent call to this source,
                      used to enforce cooldown_seconds.  None = no cooldown.

    Returns:
        PreflightCheckResult.  Check result.passed to decide whether to proceed.
    """
    checks:   dict[str, bool] = {}
    warnings: list[str]       = []
    errors:   list[str]       = []

    # ── Check 1: policy_present ───────────────────────────────────────────
    # Trivially True here (we already have the policy object), but including
    # it makes the checks dict complete and consistent.
    checks["policy_present"] = True

    # ── Check 2: enabled ──────────────────────────────────────────────────
    checks["enabled"] = policy.enabled
    if not policy.enabled:
        errors.append(
            f"Source '{source_id}' is disabled (enabled=false in sources.toml). "
            "Configure credentials and set enabled=true before calling this source."
        )

    # ── Check 3: cooldown ─────────────────────────────────────────────────
    cooldown_ok = True
    if last_call_at is not None and policy.rate_limit.cooldown_seconds > 0:
        now     = datetime.now(tz=timezone.utc)
        if last_call_at.tzinfo is None:
            last_call_at = last_call_at.replace(tzinfo=timezone.utc)
        elapsed = (now - last_call_at).total_seconds()
        if elapsed < policy.rate_limit.cooldown_seconds:
            remaining   = policy.rate_limit.cooldown_seconds - elapsed
            cooldown_ok = False
            warnings.append(
                f"Source '{source_id}' is in cooldown: {remaining:.0f}s remaining "
                f"(cooldown_seconds={policy.rate_limit.cooldown_seconds:.0f})."
            )
    checks["cooldown"] = cooldown_ok

    # ── Result ────────────────────────────────────────────────────────────
    passed         = all(checks.values())
    blocked_reason = errors[0] if errors else None

    return PreflightCheckResult(
        source_id=source_id,
        passed=passed,
        checks=checks,
        warnings=warnings,
        errors=errors,
        blocked_reason=blocked_reason,
    )


def assert_source_enabled(source_id: str, policy: SourcePolicy) -> None:
    """Assert that a source is enabled, raising SourceDisabledError if not.

    Use this when a hard failure is preferred over a soft skip.

    Args:
        source_id: Source identifier.
        policy:    SourcePolicy from the registry.

    Raises:
        SourceDisabledError: If policy.enabled is False.
    """
    if not policy.enabled:
        raise SourceDisabledError(source_id)
