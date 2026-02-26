"""
Reporting data reader: discovers and loads the latest output files.

All loaders return ``None`` rather than raising when no file is found,
so CLI commands can emit a friendly "no data yet" message without try/except
at the call site.

File-discovery convention:
  The pipeline writes files named ``{report_type}_{realm}_{date}.{ext}``.
  ``find_latest_file()`` picks the most-recently-modified match, so stale
  files from previous days are never silently preferred over today's run.

Freshness conventions:
  ``check_freshness()`` compares ``generated_at`` (ISO string in the report)
  against the current UTC wall clock.  Reports older than *max_hours* are
  flagged as stale so the caller can warn the user.

  - Recommendations JSON: ``generated_at`` is a date-only string (daily run).
  - Drift / health JSON:   ``checked_at`` may be a full ISO datetime.
  Both are handled by the same function.
"""

from __future__ import annotations

import csv
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


# ── File discovery ─────────────────────────────────────────────────────────────

def find_latest_file(directory: Path, glob_pattern: str) -> Path | None:
    """Return the most recently *modified* file matching ``glob_pattern``.

    Args:
        directory:    Directory to search (must exist; returns None if not).
        glob_pattern: Glob pattern relative to ``directory`` (e.g.
                      ``"recommendations_area-52_*.json"``).

    Returns:
        Path of the most recently modified matching file, or None.
    """
    if not directory.exists():
        return None
    matches = list(directory.glob(glob_pattern))
    if not matches:
        return None
    return max(matches, key=lambda p: p.stat().st_mtime)


# ── Freshness ─────────────────────────────────────────────────────────────────

def check_freshness(
    generated_at: str | None,
    max_hours: float = 4.0,
) -> tuple[bool, float | None]:
    """Check whether a report timestamp is within the freshness window.

    Handles both date-only (``"2025-01-15"``) and full ISO datetime strings.
    Date-only strings are treated as midnight UTC of that day.

    Args:
        generated_at: ISO string from a report's ``generated_at`` /
                      ``checked_at`` field.
        max_hours:    Hours beyond which the report is considered stale
                      (default 4 h).

    Returns:
        ``(is_fresh, age_hours)`` — ``age_hours`` is None when the string
        cannot be parsed.
    """
    if not generated_at:
        return False, None
    try:
        if "T" in generated_at or " " in generated_at:
            dt = datetime.fromisoformat(generated_at)
        else:
            dt = datetime.fromisoformat(generated_at + "T00:00:00")
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        now = datetime.now(tz=timezone.utc)
        age_hours = (now - dt).total_seconds() / 3600.0
        return age_hours <= max_hours, age_hours
    except (ValueError, OverflowError):
        return False, None


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_recommendations_report(
    realm: str,
    output_dir: Path,
) -> dict | None:
    """Load the latest recommendations JSON for a realm.

    Looks for files named ``recommendations_{realm}_*.json`` and returns
    the parsed dict of the most-recently-modified match.

    Returns:
        Parsed JSON dict, or None if no file is found / parse fails.
    """
    path = find_latest_file(output_dir, f"recommendations_{realm}_*.json")
    if path is None:
        logger.debug(
            "No recommendations report found for realm=%s in %s", realm, output_dir
        )
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to load recommendations report %s: %s", path, exc)
        return None


def load_forecast_records(
    realm: str,
    output_dir: Path,
) -> list[dict] | None:
    """Load the latest forecast CSV for a realm.

    Returns a list of row dicts (keyed by CSV header names), or None
    if no file is found.  Returns an empty list for an empty CSV.
    """
    path = find_latest_file(output_dir, f"forecast_{realm}_*.csv")
    if path is None:
        logger.debug(
            "No forecast CSV found for realm=%s in %s", realm, output_dir
        )
        return None
    try:
        rows: list[dict] = []
        with path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                rows.append(dict(row))
        return rows
    except (csv.Error, OSError) as exc:
        logger.warning("Failed to load forecast CSV %s: %s", path, exc)
        return None


def load_drift_report(
    realm: str,
    output_dir: Path,
) -> dict | None:
    """Load the latest drift status JSON for a realm.

    Returns:
        Parsed JSON dict, or None.
    """
    path = find_latest_file(output_dir, f"drift_status_{realm}_*.json")
    if path is None:
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to load drift report %s: %s", path, exc)
        return None


def load_health_report(
    realm: str,
    output_dir: Path,
) -> dict | None:
    """Load the latest model health JSON for a realm.

    Returns:
        Parsed JSON dict, or None.
    """
    path = find_latest_file(output_dir, f"model_health_{realm}_*.json")
    if path is None:
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to load health report %s: %s", path, exc)
        return None


def load_provenance_report(
    realm: str,
    output_dir: Path,
) -> dict | None:
    """Load the latest provenance JSON for a realm.

    Returns:
        Parsed JSON dict, or None.
    """
    path = find_latest_file(output_dir, f"provenance_{realm}_*.json")
    if path is None:
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to load provenance report %s: %s", path, exc)
        return None
