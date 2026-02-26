"""
Source policy registry.

Loads SourcePolicy objects from config/sources.toml (or a caller-supplied
path) and provides lookup utilities.

Usage
-----
    from wow_forecaster.governance.registry import get_source_policy, list_sources

    policy = get_source_policy("blizzard_api")
    all_sources = list_sources()
    active = get_enabled_sources()

The registry is loaded lazily on first access and then cached for the
lifetime of the process.  To force a reload (e.g., in tests), pass an
explicit ``sources_path`` argument.

TOML structure expected in sources.toml
----------------------------------------
    [sources.<source_id>]
    source_id    = "blizzard_api"
    display_name = "Blizzard Game Data API"
    ...

    [sources.<source_id>.rate_limit]
    requests_per_minute = 20
    ...

    (and so on for backoff, freshness, provenance, retention, policy_notes)
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Optional

from wow_forecaster.governance.models import (
    BackoffConfig,
    FreshnessConfig,
    PolicyNotes,
    ProvenanceRequirements,
    RateLimitConfig,
    RetentionConfig,
    SourcePolicy,
)

# ── Module-level cache ────────────────────────────────────────────────────────

_REGISTRY_CACHE: Optional[dict[str, SourcePolicy]] = None
_CACHE_PATH: Optional[str] = None

_PROJECT_ROOT = Path(__file__).parent.parent.parent


def _default_sources_path() -> Path:
    """Return the default path to config/sources.toml."""
    return _PROJECT_ROOT / "config" / "sources.toml"


def _parse_source(source_id: str, raw: dict) -> SourcePolicy:
    """Parse one [sources.<id>] block into a SourcePolicy instance.

    Args:
        source_id: The key from the TOML table (used as fallback for source_id).
        raw:       The raw TOML dict for this source block.

    Returns:
        Validated SourcePolicy.

    Raises:
        pydantic.ValidationError: If any field fails validation.
    """
    return SourcePolicy(
        source_id=raw.get("source_id", source_id),
        display_name=raw.get("display_name", source_id),
        source_type=raw.get("source_type", "other"),
        access_method=raw.get("access_method", "manual"),
        requires_auth=raw.get("requires_auth", False),
        enabled=raw.get("enabled", False),
        rate_limit=RateLimitConfig(**raw.get("rate_limit", {})),
        backoff=BackoffConfig(**raw.get("backoff", {})),
        freshness=FreshnessConfig(**raw.get("freshness", {})),
        provenance=ProvenanceRequirements(**raw.get("provenance", {})),
        retention=RetentionConfig(**raw.get("retention", {})),
        policy_notes=PolicyNotes(**raw.get("policy_notes", {})),
    )


def _load_registry(sources_path: Path) -> dict[str, SourcePolicy]:
    """Load and parse sources.toml into a dict of source_id -> SourcePolicy.

    Args:
        sources_path: Path to the sources TOML file.

    Returns:
        Dict mapping source_id strings to SourcePolicy instances.

    Raises:
        FileNotFoundError: If sources_path does not exist.
        tomllib.TOMLDecodeError: If the TOML is malformed.
        pydantic.ValidationError: If a policy fails validation.
    """
    if not sources_path.exists():
        raise FileNotFoundError(
            f"Source registry file not found: {sources_path}\n"
            "Expected at config/sources.toml.  "
            "Set governance.sources_config_path in default.toml to override."
        )

    with open(sources_path, "rb") as f:
        raw = tomllib.load(f)

    sources_raw: dict = raw.get("sources", {})
    registry: dict[str, SourcePolicy] = {}
    for sid, block in sources_raw.items():
        policy = _parse_source(sid, block)
        registry[policy.source_id] = policy

    return registry


def get_registry(sources_path: Optional[str] = None) -> dict[str, SourcePolicy]:
    """Return the full source registry (cached after first load).

    Args:
        sources_path: Override path to sources.toml.  If None, uses the
                      default config/sources.toml relative to project root.
                      Pass an explicit path in tests to avoid cache pollution.

    Returns:
        Dict mapping source_id -> SourcePolicy.
    """
    global _REGISTRY_CACHE, _CACHE_PATH

    resolved = Path(sources_path) if sources_path else _default_sources_path()
    resolved_str = str(resolved)

    if _REGISTRY_CACHE is None or _CACHE_PATH != resolved_str:
        _REGISTRY_CACHE = _load_registry(resolved)
        _CACHE_PATH = resolved_str

    return _REGISTRY_CACHE


def get_source_policy(
    source_id: str,
    sources_path: Optional[str] = None,
) -> SourcePolicy:
    """Look up a single source policy by ID.

    Args:
        source_id:    The source identifier (e.g., "blizzard_api").
        sources_path: Optional override for the sources.toml path.

    Returns:
        SourcePolicy for the given ID.

    Raises:
        KeyError: If source_id is not found in the registry.
    """
    registry = get_registry(sources_path)
    if source_id not in registry:
        available = sorted(registry.keys())
        raise KeyError(
            f"Source '{source_id}' not found in registry.  "
            f"Available sources: {available}"
        )
    return registry[source_id]


def list_sources(sources_path: Optional[str] = None) -> list[SourcePolicy]:
    """Return all registered source policies sorted by source_id.

    Args:
        sources_path: Optional override for the sources.toml path.

    Returns:
        List of SourcePolicy instances.
    """
    registry = get_registry(sources_path)
    return sorted(registry.values(), key=lambda p: p.source_id)


def get_enabled_sources(sources_path: Optional[str] = None) -> list[SourcePolicy]:
    """Return only the enabled source policies.

    Args:
        sources_path: Optional override for the sources.toml path.

    Returns:
        List of enabled SourcePolicy instances, sorted by source_id.
    """
    return [p for p in list_sources(sources_path) if p.enabled]


def clear_registry_cache() -> None:
    """Clear the module-level registry cache.

    Useful in tests to reset state between test cases when sources_path
    is not overridden explicitly.
    """
    global _REGISTRY_CACHE, _CACHE_PATH
    _REGISTRY_CACHE = None
    _CACHE_PATH = None
