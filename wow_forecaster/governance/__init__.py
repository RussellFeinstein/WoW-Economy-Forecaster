"""
Source governance and compliance guardrails for the WoW Economy Forecaster.

This sub-package provides:

  governance/models.py    — Pydantic models for source policy configuration.
  governance/registry.py  — Source registry: load, look up, and list policies.
  governance/freshness.py — TTL / freshness checks per source.
  governance/preflight.py — Pre-call guardrails: enabled status, cooldown.
  governance/reporter.py  — ASCII terminal formatters for CLI commands.

Purpose
-------
When real provider integrations are added (real HTTP calls to the Blizzard
Game Data API or Undermine Exchange), the pipeline needs explicit, enforceable
contracts for each source:

  - Is this source currently enabled?
  - What are its rate-limit and backoff parameters?
  - How fresh must its data be before it is considered stale?
  - What provenance metadata must each snapshot record?

Without these guardrails, rate limits and freshness requirements are either
absent or scattered through hard-coded constants in monitoring/provenance.py
and pipeline/orchestrator.py.

This module makes them explicit, configurable, and testable.

Technical vs legal
------------------
This module provides *technical guardrails only*.  It does not interpret
platform terms of service, copyright law, or any other legal instrument.
The policy_notes fields in config/sources.toml are informational reminders
written by the researcher; they are not legal determinations.

Independent legal and compliance review is the responsibility of the
researcher using this system.
"""

from wow_forecaster.governance.models import (
    BackoffConfig,
    FreshnessConfig,
    PolicyNotes,
    ProvenanceRequirements,
    RateLimitConfig,
    RetentionConfig,
    SourcePolicy,
)
from wow_forecaster.governance.registry import (
    get_enabled_sources,
    get_source_policy,
    get_registry,
    list_sources,
)
from wow_forecaster.governance.freshness import (
    FreshnessResult,
    FreshnessStatus,
    check_all_sources_freshness,
    check_source_freshness,
)
from wow_forecaster.governance.preflight import (
    PreflightCheckResult,
    SourceDisabledError,
    run_preflight_checks,
)

__all__ = [
    # models
    "BackoffConfig",
    "FreshnessConfig",
    "PolicyNotes",
    "ProvenanceRequirements",
    "RateLimitConfig",
    "RetentionConfig",
    "SourcePolicy",
    # registry
    "get_enabled_sources",
    "get_registry",
    "get_source_policy",
    "list_sources",
    # freshness
    "FreshnessResult",
    "FreshnessStatus",
    "check_all_sources_freshness",
    "check_source_freshness",
    # preflight
    "PreflightCheckResult",
    "SourceDisabledError",
    "run_preflight_checks",
]
