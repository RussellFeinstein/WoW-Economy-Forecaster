"""
Time and date utilities for expansion-aware forecasting.

Key concepts:
  - Expansion epochs: each expansion has a canonical start date used to
    anchor relative time features (days since expansion launch, etc.).
  - Event windows: helpers for constructing rolling windows around event dates.
  - Forecast date generation: produce target dates for a given horizon.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from typing import Optional

# ── Expansion launch dates ─────────────────────────────────────────────────────
# Dates are the live-server launch dates (UTC) for US region.
# Add Midnight date once officially announced.

EXPANSION_LAUNCH_DATES: dict[str, date] = {
    "classic":      date(2004, 11, 23),
    "tbc":          date(2007, 1, 16),
    "wotlk":        date(2008, 11, 13),
    "cata":         date(2010, 12, 7),
    "mop":          date(2012, 9, 25),
    "wod":          date(2014, 11, 13),
    "legion":       date(2016, 8, 30),
    "bfa":          date(2018, 8, 14),
    "shadowlands":  date(2020, 11, 23),
    "dragonflight": date(2022, 11, 28),
    "tww":          date(2024, 8, 26),
    # "midnight": date(TBD)  — placeholder; fill in when announced
}


def days_since_expansion_launch(
    check_date: date,
    expansion_slug: str,
) -> Optional[int]:
    """Return the number of days since an expansion's launch, or ``None``.

    Returns ``None`` if:
      - The expansion slug is not in ``EXPANSION_LAUNCH_DATES``.
      - ``check_date`` is before the expansion's launch (would be negative).

    Args:
        check_date: The date to measure from.
        expansion_slug: Expansion identifier string, e.g. ``"tww"``.

    Returns:
        Non-negative integer days since launch, or ``None``.
    """
    launch = EXPANSION_LAUNCH_DATES.get(expansion_slug)
    if launch is None:
        return None
    delta = (check_date - launch).days
    return delta if delta >= 0 else None


def days_until_event(
    check_date: date,
    event_start: date,
) -> int:
    """Return signed number of days from ``check_date`` to ``event_start``.

    Positive: event is in the future.
    Zero: event starts today.
    Negative: event has already started.

    Args:
        check_date: Reference date.
        event_start: Event start date.

    Returns:
        ``(event_start - check_date).days``
    """
    return (event_start - check_date).days


def date_range(start: date, end: date, step_days: int = 1) -> list[date]:
    """Generate a list of dates from ``start`` to ``end`` (inclusive).

    Args:
        start: First date in the range.
        end: Last date in the range (inclusive).
        step_days: Step size in days (default 1).

    Returns:
        List of date objects.

    Raises:
        ValueError: If ``end < start`` or ``step_days < 1``.
    """
    if end < start:
        raise ValueError(f"end ({end}) must be >= start ({start}).")
    if step_days < 1:
        raise ValueError(f"step_days must be >= 1, got {step_days}.")

    result: list[date] = []
    current = start
    while current <= end:
        result.append(current)
        current += timedelta(days=step_days)
    return result


def forecast_target_dates(
    base_date: date,
    horizons: list[str],
) -> dict[str, date]:
    """Compute forecast target dates for each horizon string.

    Args:
        base_date: The date the forecast is produced (T+0).
        horizons: List of horizon strings, e.g. ``["1d", "7d", "30d"]``.

    Returns:
        Mapping of horizon string → target date.

    Raises:
        ValueError: If a horizon string cannot be parsed.
    """
    result: dict[str, date] = {}
    for h in horizons:
        days = _parse_horizon_days(h)
        result[h] = base_date + timedelta(days=days)
    return result


def _parse_horizon_days(horizon: str) -> int:
    """Parse a horizon string like ``"7d"`` into an integer day count.

    Supported formats: ``Nd`` (days), ``Nw`` (weeks), ``Nm`` (months, ≈30 days).

    Args:
        horizon: Horizon string.

    Returns:
        Number of days.

    Raises:
        ValueError: If the format is unrecognized.
    """
    horizon = horizon.strip().lower()
    if horizon.endswith("d"):
        return int(horizon[:-1])
    if horizon.endswith("w"):
        return int(horizon[:-1]) * 7
    if horizon.endswith("m"):
        return int(horizon[:-1]) * 30
    raise ValueError(
        f"Cannot parse horizon '{horizon}'. "
        "Expected format: Nd (days), Nw (weeks), or Nm (months)."
    )


def utcnow() -> datetime:
    """Return the current UTC datetime with timezone info.

    Prefer this over ``datetime.utcnow()`` (which returns naive datetimes).
    """
    return datetime.now(tz=timezone.utc)
