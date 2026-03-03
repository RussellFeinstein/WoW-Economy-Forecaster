"""
Time and date utilities for the WoW Economy Forecaster.
"""

from __future__ import annotations

from datetime import datetime, timezone


def utcnow() -> datetime:
    """Return the current UTC datetime with timezone info.

    Prefer this over ``datetime.utcnow()`` (which returns naive datetimes).
    """
    return datetime.now(tz=timezone.utc)
