"""
Pydantic v2 models for source policy configuration.

Each SourcePolicy fully describes the operational constraints and metadata for
one data source.  Instances are loaded from config/sources.toml by the
registry module and are immutable (frozen=True).

Model hierarchy
---------------
  SourcePolicy
    ├── RateLimitConfig      — requests per minute/hour, burst, cooldown
    ├── BackoffConfig        — strategy (exponential/linear/fixed), delays, retries
    ├── FreshnessConfig      — TTL, stale/critical thresholds
    ├── ProvenanceRequirements — snapshot and hash requirements
    ├── RetentionConfig      — how long raw snapshots are kept
    └── PolicyNotes          — informational notes (NOT legal determinations)

Validation
----------
All numeric fields are validated to be non-negative.  Freshness thresholds
are validated to be non-decreasing (stale >= ttl, critical >= stale).
BackoffConfig strategy must be one of: "exponential", "linear", "fixed".
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, field_validator, model_validator


# ── Rate-limit sub-model ──────────────────────────────────────────────────────


class RateLimitConfig(BaseModel):
    """Rate-limit constraints for one source.

    Fields with value 0 mean "no limit enforced" (e.g., manual sources).

    Attributes:
        requests_per_minute: Maximum calls per minute (0 = unlimited).
        requests_per_hour:   Maximum calls per hour (0 = unlimited).
        burst_limit:         Maximum concurrent / burst calls (0 = unlimited).
        cooldown_seconds:    Minimum seconds to wait between consecutive calls.
    """

    model_config = ConfigDict(frozen=True)

    requests_per_minute: int = 0
    requests_per_hour:   int = 0
    burst_limit:         int = 0
    cooldown_seconds:    float = 0.0

    @field_validator("requests_per_minute", "requests_per_hour", "burst_limit")
    @classmethod
    def non_negative_int(cls, v: int) -> int:
        if v < 0:
            raise ValueError(f"Rate limit value must be >= 0, got {v}.")
        return v

    @field_validator("cooldown_seconds")
    @classmethod
    def non_negative_float(cls, v: float) -> float:
        if v < 0.0:
            raise ValueError(f"cooldown_seconds must be >= 0.0, got {v}.")
        return v


# ── Backoff sub-model ─────────────────────────────────────────────────────────


VALID_BACKOFF_STRATEGIES = frozenset({"exponential", "linear", "fixed"})


class BackoffConfig(BaseModel):
    """Retry backoff policy for one source.

    Attributes:
        strategy:     "exponential" (2^n * base), "linear" (n * base),
                      or "fixed" (always base_seconds).
        base_seconds: Base delay for the first retry.
        max_seconds:  Upper cap on retry delay.
        jitter:       If True, add ±25% random jitter to prevent thundering herd.
        max_retries:  Stop retrying after this many attempts (0 = no retries).
    """

    model_config = ConfigDict(frozen=True)

    strategy:     str   = "exponential"
    base_seconds: float = 1.0
    max_seconds:  float = 300.0
    jitter:       bool  = True
    max_retries:  int   = 5

    @field_validator("strategy")
    @classmethod
    def valid_strategy(cls, v: str) -> str:
        if v not in VALID_BACKOFF_STRATEGIES:
            raise ValueError(
                f"BackoffConfig.strategy must be one of {sorted(VALID_BACKOFF_STRATEGIES)}, got '{v}'."
            )
        return v

    @field_validator("base_seconds", "max_seconds")
    @classmethod
    def non_negative_seconds(cls, v: float) -> float:
        if v < 0.0:
            raise ValueError(f"Delay seconds must be >= 0.0, got {v}.")
        return v

    @field_validator("max_retries")
    @classmethod
    def non_negative_retries(cls, v: int) -> int:
        if v < 0:
            raise ValueError(f"max_retries must be >= 0, got {v}.")
        return v


# ── Freshness sub-model ───────────────────────────────────────────────────────


class FreshnessConfig(BaseModel):
    """TTL and freshness threshold configuration for one source.

    Attributes:
        ttl_hours:               Data is "fresh" within this window.
        refresh_cadence_hours:   Intended call frequency (informational).
        stale_threshold_hours:   Beyond this, data is flagged [STALE].
        critical_threshold_hours: Beyond this, data is flagged [CRITICAL] and
                                  cannot be trusted for forecasting.

    All thresholds must be non-decreasing:
        ttl_hours <= stale_threshold_hours <= critical_threshold_hours
    """

    model_config = ConfigDict(frozen=True)

    ttl_hours:               float = 1.0
    refresh_cadence_hours:   float = 1.0
    stale_threshold_hours:   float = 2.0
    critical_threshold_hours: float = 25.0

    @field_validator(
        "ttl_hours", "refresh_cadence_hours",
        "stale_threshold_hours", "critical_threshold_hours",
    )
    @classmethod
    def positive_hours(cls, v: float) -> float:
        if v <= 0.0:
            raise ValueError(f"Freshness hours must be > 0, got {v}.")
        return v

    @model_validator(mode="after")
    def thresholds_non_decreasing(self) -> "FreshnessConfig":
        if self.stale_threshold_hours < self.ttl_hours:
            raise ValueError(
                f"stale_threshold_hours ({self.stale_threshold_hours}) must be >= "
                f"ttl_hours ({self.ttl_hours})."
            )
        if self.critical_threshold_hours < self.stale_threshold_hours:
            raise ValueError(
                f"critical_threshold_hours ({self.critical_threshold_hours}) must be >= "
                f"stale_threshold_hours ({self.stale_threshold_hours})."
            )
        return self


# ── Provenance sub-model ──────────────────────────────────────────────────────


class ProvenanceRequirements(BaseModel):
    """Provenance and snapshot requirements for one source.

    Attributes:
        requires_snapshot:     True if every API call must write a snapshot file.
        snapshot_format:       Expected format: "json", "csv", "csv_or_json", etc.
        content_hash_required: True if the snapshot must record a sha256 hash.
    """

    model_config = ConfigDict(frozen=True)

    requires_snapshot:     bool = True
    snapshot_format:       str  = "json"
    content_hash_required: bool = True


# ── Retention sub-model ───────────────────────────────────────────────────────


class RetentionConfig(BaseModel):
    """Raw data retention policy.

    Attributes:
        raw_snapshot_days: Days to keep raw snapshot files (0 = retain indefinitely).
        notes:             Free-text note about retention rationale.
    """

    model_config = ConfigDict(frozen=True)

    raw_snapshot_days: int = 30
    notes:             str = ""

    @field_validator("raw_snapshot_days")
    @classmethod
    def non_negative(cls, v: int) -> int:
        if v < 0:
            raise ValueError(f"raw_snapshot_days must be >= 0, got {v}.")
        return v


# ── Policy notes sub-model ────────────────────────────────────────────────────


VALID_ACCESS_TYPES = frozenset({"authorized_api", "export", "manual"})


class PolicyNotes(BaseModel):
    """Informational policy notes for one source.

    IMPORTANT: These notes are researcher-authored reminders only.
    They are NOT legal determinations or legal advice.

    Attributes:
        access_type:                "authorized_api", "export", or "manual".
        requires_registered_account: Whether a developer/API account is needed.
        personal_research_only:     Reminder that this system is for personal use.
        notes:                      Free-text notes for this source.
    """

    model_config = ConfigDict(frozen=True)

    access_type:                 str  = "manual"
    requires_registered_account: bool = False
    personal_research_only:      bool = True
    notes:                       str  = ""

    @field_validator("access_type")
    @classmethod
    def valid_access_type(cls, v: str) -> str:
        if v not in VALID_ACCESS_TYPES:
            raise ValueError(
                f"PolicyNotes.access_type must be one of {sorted(VALID_ACCESS_TYPES)}, got '{v}'."
            )
        return v


# ── Top-level SourcePolicy ────────────────────────────────────────────────────


VALID_SOURCE_TYPES  = frozenset({"auction_data", "news_event", "manual_event", "other"})
VALID_ACCESS_METHODS = frozenset({"api", "export", "manual"})


class SourcePolicy(BaseModel):
    """Complete policy definition for one data source.

    This is the primary model loaded from config/sources.toml.  One
    SourcePolicy exists per source_id.  Instances are immutable.

    Attributes:
        source_id:     Unique identifier (e.g., "blizzard_api").
        display_name:  Human-readable name.
        source_type:   Category of data provided.
        access_method: How data is obtained ("api", "export", "manual").
        requires_auth: Whether credentials are needed to use this source.
        enabled:       Whether this source is active in the pipeline.
        rate_limit:    Call frequency constraints.
        backoff:       Retry / backoff policy.
        freshness:     TTL and staleness thresholds.
        provenance:    Snapshot and hash requirements.
        retention:     How long raw data is retained.
        policy_notes:  Informational notes (NOT legal determinations).
    """

    model_config = ConfigDict(frozen=True)

    source_id:    str
    display_name: str
    source_type:  str
    access_method: str
    requires_auth: bool
    enabled:       bool

    rate_limit:  RateLimitConfig
    backoff:     BackoffConfig
    freshness:   FreshnessConfig
    provenance:  ProvenanceRequirements
    retention:   RetentionConfig
    policy_notes: PolicyNotes

    @field_validator("source_type")
    @classmethod
    def valid_source_type(cls, v: str) -> str:
        if v not in VALID_SOURCE_TYPES:
            raise ValueError(
                f"SourcePolicy.source_type must be one of {sorted(VALID_SOURCE_TYPES)}, got '{v}'."
            )
        return v

    @field_validator("access_method")
    @classmethod
    def valid_access_method(cls, v: str) -> str:
        if v not in VALID_ACCESS_METHODS:
            raise ValueError(
                f"SourcePolicy.access_method must be one of {sorted(VALID_ACCESS_METHODS)}, got '{v}'."
            )
        return v
