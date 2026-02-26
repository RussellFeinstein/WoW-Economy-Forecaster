"""
Tests for governance/registry.py — load, look up, and list source policies.

Covers:
  - Loading a valid sources.toml produces correct SourcePolicy instances
  - get_source_policy() returns the correct policy by ID
  - get_source_policy() raises KeyError for unknown IDs
  - list_sources() returns all policies sorted by source_id
  - get_enabled_sources() returns only enabled policies
  - FileNotFoundError raised for missing sources.toml
  - Registry cache is populated and respected between calls
  - clear_registry_cache() resets the cache

Tests use a temporary TOML file rather than the real config/sources.toml to
avoid coupling test outcomes to production config state changes.
"""

import tomllib
from pathlib import Path

import pytest

from wow_forecaster.governance.registry import (
    _load_registry,
    clear_registry_cache,
    get_enabled_sources,
    get_registry,
    get_source_policy,
    list_sources,
)


# ── Helpers ───────────────────────────────────────────────────────────────────


MINIMAL_SOURCE_TOML = """\
[sources.source_a]
source_id    = "source_a"
display_name = "Source A"
source_type  = "auction_data"
access_method = "api"
requires_auth = true
enabled = true

[sources.source_a.rate_limit]
requests_per_minute = 10
cooldown_seconds = 3.0

[sources.source_a.backoff]
strategy = "exponential"

[sources.source_a.freshness]
ttl_hours = 1.0
refresh_cadence_hours = 1.0
stale_threshold_hours = 3.0
critical_threshold_hours = 25.0

[sources.source_a.provenance]
requires_snapshot = true
snapshot_format = "json"
content_hash_required = true

[sources.source_a.retention]
raw_snapshot_days = 30

[sources.source_a.policy_notes]
access_type = "authorized_api"

# --- second source, disabled ---

[sources.source_b]
source_id    = "source_b"
display_name = "Source B"
source_type  = "manual_event"
access_method = "manual"
requires_auth = false
enabled = false

[sources.source_b.rate_limit]
[sources.source_b.backoff]
strategy = "fixed"
base_seconds = 0.0
max_seconds = 0.0
max_retries = 0

[sources.source_b.freshness]
ttl_hours = 720.0
refresh_cadence_hours = 720.0
stale_threshold_hours = 1440.0
critical_threshold_hours = 2160.0

[sources.source_b.provenance]
requires_snapshot = false
snapshot_format = "csv_or_json"
content_hash_required = false

[sources.source_b.retention]
raw_snapshot_days = 0

[sources.source_b.policy_notes]
access_type = "manual"
"""


@pytest.fixture()
def sources_toml(tmp_path: Path) -> Path:
    """Write a minimal sources.toml to a temp dir and return its path."""
    p = tmp_path / "sources.toml"
    p.write_text(MINIMAL_SOURCE_TOML, encoding="utf-8")
    # Ensure cache is cleared so tests are independent
    clear_registry_cache()
    return p


@pytest.fixture(autouse=True)
def _clear_cache():
    """Auto-clear registry cache before each test."""
    clear_registry_cache()
    yield
    clear_registry_cache()


# ── _load_registry ────────────────────────────────────────────────────────────


class TestLoadRegistry:
    def test_loads_two_sources(self, sources_toml):
        registry = _load_registry(sources_toml)
        assert len(registry) == 2
        assert "source_a" in registry
        assert "source_b" in registry

    def test_source_a_fields(self, sources_toml):
        registry = _load_registry(sources_toml)
        p = registry["source_a"]
        assert p.source_id == "source_a"
        assert p.display_name == "Source A"
        assert p.source_type == "auction_data"
        assert p.access_method == "api"
        assert p.requires_auth is True
        assert p.enabled is True
        assert p.rate_limit.requests_per_minute == 10
        assert p.rate_limit.cooldown_seconds == 3.0
        assert p.backoff.strategy == "exponential"
        assert p.freshness.ttl_hours == 1.0
        assert p.provenance.requires_snapshot is True

    def test_source_b_is_disabled_manual(self, sources_toml):
        registry = _load_registry(sources_toml)
        p = registry["source_b"]
        assert p.enabled is False
        assert p.access_method == "manual"
        assert p.provenance.requires_snapshot is False
        assert p.retention.raw_snapshot_days == 0

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Source registry file not found"):
            _load_registry(tmp_path / "nonexistent.toml")


# ── get_registry ──────────────────────────────────────────────────────────────


class TestGetRegistry:
    def test_returns_dict(self, sources_toml):
        registry = get_registry(str(sources_toml))
        assert isinstance(registry, dict)
        assert len(registry) == 2

    def test_cache_hit_same_result(self, sources_toml):
        r1 = get_registry(str(sources_toml))
        r2 = get_registry(str(sources_toml))
        assert r1 is r2  # exact same dict object (cached)

    def test_different_path_reloads(self, tmp_path):
        # Two different files should produce independent registries
        toml1 = tmp_path / "s1.toml"
        toml2 = tmp_path / "s2.toml"

        # Write both with the same content but different paths
        content = MINIMAL_SOURCE_TOML
        toml1.write_text(content, encoding="utf-8")
        toml2.write_text(content, encoding="utf-8")

        r1 = get_registry(str(toml1))
        clear_registry_cache()
        r2 = get_registry(str(toml2))

        # Different objects loaded from different paths
        assert r1 is not r2
        assert set(r1.keys()) == set(r2.keys())


# ── get_source_policy ─────────────────────────────────────────────────────────


class TestGetSourcePolicy:
    def test_known_source_returned(self, sources_toml):
        p = get_source_policy("source_a", str(sources_toml))
        assert p.source_id == "source_a"
        assert p.enabled is True

    def test_unknown_source_raises_key_error(self, sources_toml):
        with pytest.raises(KeyError, match="Source 'unknown_src' not found"):
            get_source_policy("unknown_src", str(sources_toml))

    def test_error_message_lists_available(self, sources_toml):
        with pytest.raises(KeyError) as exc_info:
            get_source_policy("missing", str(sources_toml))
        assert "source_a" in str(exc_info.value)
        assert "source_b" in str(exc_info.value)


# ── list_sources ──────────────────────────────────────────────────────────────


class TestListSources:
    def test_returns_all_sources(self, sources_toml):
        policies = list_sources(str(sources_toml))
        assert len(policies) == 2

    def test_sorted_by_source_id(self, sources_toml):
        policies = list_sources(str(sources_toml))
        ids = [p.source_id for p in policies]
        assert ids == sorted(ids)

    def test_returns_source_policy_instances(self, sources_toml):
        from wow_forecaster.governance.models import SourcePolicy
        for p in list_sources(str(sources_toml)):
            assert isinstance(p, SourcePolicy)


# ── get_enabled_sources ───────────────────────────────────────────────────────


class TestGetEnabledSources:
    def test_only_enabled_returned(self, sources_toml):
        enabled = get_enabled_sources(str(sources_toml))
        assert len(enabled) == 1
        assert enabled[0].source_id == "source_a"
        assert enabled[0].enabled is True

    def test_empty_when_all_disabled(self, tmp_path):
        disabled_toml = tmp_path / "disabled.toml"
        content = MINIMAL_SOURCE_TOML.replace("enabled = true", "enabled = false")
        disabled_toml.write_text(content, encoding="utf-8")

        enabled = get_enabled_sources(str(disabled_toml))
        assert enabled == []

    def test_all_enabled_when_all_true(self, tmp_path):
        enabled_toml = tmp_path / "all_enabled.toml"
        content = MINIMAL_SOURCE_TOML.replace("enabled = false", "enabled = true")
        enabled_toml.write_text(content, encoding="utf-8")

        enabled = get_enabled_sources(str(enabled_toml))
        assert len(enabled) == 2


# ── clear_registry_cache ──────────────────────────────────────────────────────


class TestClearRegistryCache:
    def test_clear_forces_reload(self, sources_toml):
        r1 = get_registry(str(sources_toml))
        clear_registry_cache()
        r2 = get_registry(str(sources_toml))
        # After clearing, a new object is loaded
        assert r1 is not r2
        # But the content should be identical
        assert set(r1.keys()) == set(r2.keys())
