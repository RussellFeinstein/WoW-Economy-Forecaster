"""
CSV import parser for manual WoWEvent records.

Format — tab or comma delimited, with a header row.
Required columns:
  slug, display_name, event_type, scope, severity, expansion_slug, start_date

Optional columns (empty string → None):
  patch_version, end_date, announced_at, is_recurring, recurrence_rule, notes

See ``config/events/event_import_template.csv`` for a full example with all columns.

Valid enum values:
  event_type  → any EventType.value  (e.g. "expansion_launch", "rtwf", "holiday_event")
  scope       → any EventScope.value (e.g. "global", "region", "realm_cluster", "faction")
  severity    → any EventSeverity.value (e.g. "critical", "major", "moderate", "minor", "negligible")

Date formats:
  start_date / end_date   → YYYY-MM-DD
  announced_at            → ISO 8601 with timezone, e.g. 2025-11-03T18:00:00Z or +00:00

Boolean columns (is_recurring):
  true/1/yes/t/y  → True
  false/0/no/f/n  → False (default if omitted)
"""

from __future__ import annotations

import csv
import logging
from datetime import date, datetime
from pathlib import Path
from typing import Optional

from pydantic import ValidationError

from wow_forecaster.models.event import WoWEvent
from wow_forecaster.taxonomy.event_taxonomy import EventScope, EventSeverity, EventType

logger = logging.getLogger(__name__)

REQUIRED_CSV_COLUMNS = frozenset({
    "slug", "display_name", "event_type", "scope",
    "severity", "expansion_slug", "start_date",
})


def parse_event_csv(path: Path) -> list[WoWEvent]:
    """Parse a CSV file of WoW events into validated :class:`WoWEvent` objects.

    All rows are validated before any are returned. If **any** row fails,
    a single :class:`ValueError` is raised listing the first 10 failures.

    Args:
        path: Path to the CSV file (must exist).

    Returns:
        List of fully validated :class:`WoWEvent` instances.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If required columns are missing or any row fails validation.
    """
    if not path.exists():
        raise FileNotFoundError(f"Event CSV file not found: {path}")

    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)

        if reader.fieldnames is None:
            raise ValueError(f"CSV file is empty or has no header row: {path}")

        actual_cols = set(reader.fieldnames)
        missing = REQUIRED_CSV_COLUMNS - actual_cols
        if missing:
            raise ValueError(
                f"CSV missing required columns: {sorted(missing)}\n"
                f"Found columns: {sorted(actual_cols)}"
            )

        rows = list(reader)

    if not rows:
        logger.warning("Event CSV is empty (header only): %s", path)
        return []

    events: list[WoWEvent] = []
    errors: list[tuple[int, str]] = []

    for i, row in enumerate(rows):
        line_no = i + 2  # 1-based, skip header row
        try:
            event = _row_to_wow_event(row)
            events.append(event)
        except (ValueError, ValidationError) as exc:
            errors.append((line_no, str(exc)))

    if errors:
        max_shown = 10
        detail = "\n".join(f"  Row {ln}: {msg}" for ln, msg in errors[:max_shown])
        suffix = f"\n  … and {len(errors) - max_shown} more" if len(errors) > max_shown else ""
        raise ValueError(
            f"{len(errors)} row(s) failed validation in {path.name}:\n{detail}{suffix}"
        )

    logger.info("Parsed %d events from %s", len(events), path.name)
    return events


# ── Private helpers ────────────────────────────────────────────────────────────

def _row_to_wow_event(row: dict[str, str]) -> WoWEvent:
    """Convert a CSV row dict to a validated :class:`WoWEvent`.

    Args:
        row: Dict from :class:`csv.DictReader` (all values are strings).

    Returns:
        Validated :class:`WoWEvent`.

    Raises:
        ValueError: On bad enum values, date formats, or missing required fields.
        pydantic.ValidationError: On model-level validation failure.
    """
    event_type = _parse_enum(EventType, "event_type", row)
    scope = _parse_enum(EventScope, "scope", row)
    severity = _parse_enum(EventSeverity, "severity", row)

    return WoWEvent(
        slug=_req(row, "slug"),
        display_name=_req(row, "display_name"),
        event_type=event_type,
        scope=scope,
        severity=severity,
        expansion_slug=_req(row, "expansion_slug"),
        patch_version=_opt(row, "patch_version"),
        start_date=_parse_date(row, "start_date", required=True),
        end_date=_parse_date(row, "end_date"),
        announced_at=_parse_datetime(row, "announced_at"),
        is_recurring=_parse_bool(row, "is_recurring", default=False),
        recurrence_rule=_opt(row, "recurrence_rule"),
        notes=_opt(row, "notes"),
    )


def _req(row: dict[str, str], key: str) -> str:
    """Return a required string field, stripped; raise if empty."""
    v = row.get(key, "").strip()
    if not v:
        raise ValueError(f"Required field '{key}' is empty.")
    return v


def _opt(row: dict[str, str], key: str) -> Optional[str]:
    """Return an optional string field, or None if absent/empty."""
    v = row.get(key, "").strip()
    return v if v else None


def _parse_date(
    row: dict[str, str],
    key: str,
    required: bool = False,
) -> Optional[date]:
    """Parse an ISO date string (YYYY-MM-DD) from a CSV row field."""
    v = _opt(row, key)
    if v is None:
        if required:
            raise ValueError(f"Required date field '{key}' is empty.")
        return None
    try:
        return date.fromisoformat(v)
    except ValueError:
        raise ValueError(
            f"Invalid date for '{key}': '{v}'. Expected YYYY-MM-DD format."
        )


def _parse_datetime(row: dict[str, str], key: str) -> Optional[datetime]:
    """Parse an ISO 8601 datetime string (with timezone) from a CSV row field."""
    v = _opt(row, key)
    if v is None:
        return None
    try:
        # Accept both trailing 'Z' and explicit '+00:00'
        return datetime.fromisoformat(v.replace("Z", "+00:00"))
    except ValueError:
        raise ValueError(
            f"Invalid datetime for '{key}': '{v}'. "
            "Expected ISO 8601 with timezone, e.g. '2025-11-03T18:00:00Z'."
        )


def _parse_bool(row: dict[str, str], key: str, default: bool = False) -> bool:
    """Parse a boolean-ish string from a CSV row field."""
    v = _opt(row, key)
    if v is None:
        return default
    return v.lower() in ("true", "1", "yes", "t", "y")


def _parse_enum(enum_cls, key: str, row: dict[str, str]):
    """Parse an enum value from a CSV row field with a descriptive error."""
    raw = row.get(key, "").strip()
    if not raw:
        raise ValueError(f"Required enum field '{key}' is empty.")
    try:
        return enum_cls(raw)
    except ValueError:
        valid = sorted(e.value for e in enum_cls)
        raise ValueError(
            f"Invalid {key} value '{raw}'. "
            f"Valid values: {valid}"
        )
