"""
ASCII terminal formatters and JSON export for governance CLI commands.

Functions here produce human-readable output for:
  - list-sources           (format_source_table)
  - validate-source-policies (format_validation_report)
  - check-source-freshness  (format_freshness_table)

No external dependencies — pure stdlib + project models.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from wow_forecaster.governance.freshness import FreshnessResult, FreshnessStatus
from wow_forecaster.governance.models import SourcePolicy


# ── Helpers ───────────────────────────────────────────────────────────────────


def _bool_icon(v: bool) -> str:
    return "Yes" if v else "No"


def _enabled_badge(enabled: bool) -> str:
    return "[ENABLED] " if enabled else "[disabled]"


def _freshness_badge(status: FreshnessStatus) -> str:
    badges = {
        FreshnessStatus.FRESH:    "[FRESH]   ",
        FreshnessStatus.AGING:    "[AGING]   ",
        FreshnessStatus.STALE:    "[STALE]   ",
        FreshnessStatus.CRITICAL: "[CRITICAL]",
        FreshnessStatus.UNKNOWN:  "[UNKNOWN] ",
    }
    return badges.get(status, "[?]       ")


# ── Source table ──────────────────────────────────────────────────────────────


def format_source_table(policies: list[SourcePolicy]) -> str:
    """Format a registry summary table for `list-sources`.

    Columns: Status | Source ID | Display Name | Type | Access | Auth | TTL
    """
    if not policies:
        return "  (no sources registered)\n"

    header = (
        f"  {'Status':<11} {'Source ID':<26} {'Display Name':<38} "
        f"{'Type':<14} {'Access':<8} {'Auth':<6} {'TTL(h)':<8}"
    )
    sep = "  " + "-" * (len(header) - 2)

    rows = [header, sep]
    for p in policies:
        rows.append(
            f"  {_enabled_badge(p.enabled):<11} {p.source_id:<26} {p.display_name:<38} "
            f"{p.source_type:<14} {p.access_method:<8} {_bool_icon(p.requires_auth):<6} "
            f"{p.freshness.ttl_hours:<8.1f}"
        )

    rows.append("")
    return "\n".join(rows)


def format_source_detail(policy: SourcePolicy) -> str:
    """Format a verbose single-source detail block."""
    lines = [
        f"  Source: {policy.source_id}",
        f"    Display name : {policy.display_name}",
        f"    Source type  : {policy.source_type}",
        f"    Access method: {policy.access_method}",
        f"    Requires auth: {_bool_icon(policy.requires_auth)}",
        f"    Enabled      : {_bool_icon(policy.enabled)}",
        "",
        f"    Rate limits",
        f"      req/min     : {policy.rate_limit.requests_per_minute or 'unlimited'}",
        f"      req/hour    : {policy.rate_limit.requests_per_hour or 'unlimited'}",
        f"      burst       : {policy.rate_limit.burst_limit or 'unlimited'}",
        f"      cooldown    : {policy.rate_limit.cooldown_seconds}s",
        "",
        f"    Backoff",
        f"      strategy    : {policy.backoff.strategy}",
        f"      base_seconds: {policy.backoff.base_seconds}s",
        f"      max_seconds : {policy.backoff.max_seconds}s",
        f"      jitter      : {_bool_icon(policy.backoff.jitter)}",
        f"      max_retries : {policy.backoff.max_retries}",
        "",
        f"    Freshness",
        f"      TTL         : {policy.freshness.ttl_hours}h",
        f"      Stale after : {policy.freshness.stale_threshold_hours}h",
        f"      Critical at : {policy.freshness.critical_threshold_hours}h",
        "",
        f"    Provenance",
        f"      Req snapshot: {_bool_icon(policy.provenance.requires_snapshot)}",
        f"      Format      : {policy.provenance.snapshot_format}",
        f"      Hash req    : {_bool_icon(policy.provenance.content_hash_required)}",
        "",
        f"    Retention",
        f"      Keep days   : {policy.retention.raw_snapshot_days or 'indefinite'}",
        f"      Notes       : {policy.retention.notes or '—'}",
        "",
        f"    Policy notes  (informational only — NOT legal advice)",
        f"      Access type : {policy.policy_notes.access_type}",
        f"      Reg account : {_bool_icon(policy.policy_notes.requires_registered_account)}",
        f"      Research use: {_bool_icon(policy.policy_notes.personal_research_only)}",
    ]
    if policy.policy_notes.notes:
        # Wrap multi-line notes with indent
        note_lines = policy.policy_notes.notes.strip().splitlines()
        lines.append(f"      Note        : {note_lines[0].strip()}")
        for nl in note_lines[1:]:
            lines.append(f"                    {nl.strip()}")
    return "\n".join(lines)


# ── Validation report ─────────────────────────────────────────────────────────


def format_validation_report(
    policies: list[SourcePolicy],
    errors: dict[str, list[str]],
) -> str:
    """Format validation outcome for `validate-source-policies`.

    Args:
        policies: All registered policies.
        errors:   Dict of source_id -> list of error strings.
                  Pass an empty dict if all policies are valid.
    """
    total   = len(policies)
    n_ok    = total - len(errors)
    n_fail  = len(errors)

    lines = [
        f"  Sources validated : {total}",
        f"  Passed            : {n_ok}",
        f"  Failed            : {n_fail}",
        "",
    ]

    if not errors:
        lines.append("  All source policies are valid.")
    else:
        lines.append("  Validation errors:")
        for sid, errs in sorted(errors.items()):
            lines.append(f"    {sid}:")
            for e in errs:
                lines.append(f"      - {e}")

    lines.append("")
    return "\n".join(lines)


# ── Freshness table ───────────────────────────────────────────────────────────


def format_freshness_table(
    results: list[FreshnessResult],
    realm_slug: Optional[str] = None,
) -> str:
    """Format a per-source freshness summary for `check-source-freshness`.

    Columns: Status | Source ID | Last Snapshot | Age (h) | TTL | Stale |
             Snapshot req
    """
    if not results:
        return "  (no sources to check)\n"

    realm_line = f"  Realm filter: {realm_slug}" if realm_slug else "  Realm filter: (all)"

    header = (
        f"  {'Status':<11} {'Source ID':<26} {'Last Snapshot':<22} "
        f"{'Age(h)':<8} {'TTL(h)':<7} {'Stale(h)':<9} {'Snap req':<9}"
    )
    sep = "  " + "-" * (len(header) - 2)

    rows = [realm_line, "", header, sep]
    for r in results:
        age_str  = f"{r.age_hours:.1f}" if r.age_hours is not None else "—"
        snap_str = r.last_snapshot_at[:19] if r.last_snapshot_at else "—"
        rows.append(
            f"  {_freshness_badge(r.status):<11} {r.source_id:<26} {snap_str:<22} "
            f"{age_str:<8} {r.ttl_hours:<7.1f} {r.stale_threshold_hours:<9.1f} "
            f"{_bool_icon(r.requires_snapshot):<9}"
        )

    rows.append("")
    return "\n".join(rows)


# ── JSON export ───────────────────────────────────────────────────────────────


def write_governance_report(
    policies: list[SourcePolicy],
    freshness_results: Optional[list[FreshnessResult]],
    output_dir: str,
) -> Path:
    """Write a combined governance JSON report.

    The report contains:
      - All source policies (serialised)
      - Freshness results (if provided)

    Args:
        policies:          All registered SourcePolicy instances.
        freshness_results: Optional list of FreshnessResult objects.
        output_dir:        Directory to write the report into.

    Returns:
        Path to the written JSON file.
    """
    from datetime import datetime, timezone
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    report: dict = {
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "sources": [p.model_dump() for p in policies],
    }
    if freshness_results is not None:
        report["freshness"] = [
            {
                "source_id":                r.source_id,
                "last_snapshot_at":         r.last_snapshot_at,
                "age_hours":                r.age_hours,
                "ttl_hours":                r.ttl_hours,
                "stale_threshold_hours":    r.stale_threshold_hours,
                "critical_threshold_hours": r.critical_threshold_hours,
                "is_within_ttl":            r.is_within_ttl,
                "is_stale":                 r.is_stale,
                "is_critical":              r.is_critical,
                "status":                   r.status.value,
                "requires_snapshot":        r.requires_snapshot,
            }
            for r in freshness_results
        ]

    ts  = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    out = out_dir / f"governance_report_{ts}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    return out
