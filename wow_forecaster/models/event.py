"""
WoW event model — the core of the event-aware forecasting system.

``WoWEvent`` represents any in-game or real-world event that affects the
WoW auction house economy. The ``announced_at`` field is critical for
**backtest correctness**: forecasts should only incorporate event knowledge
that existed *at the time the forecast was produced*.

Key method: ``is_known_at(as_of)``
  Returns ``True`` only if the event was publicly announced before ``as_of``.
  When ``announced_at`` is ``None``, the event is conservatively treated as
  unknown (safe default — prevents look-ahead bias).

Event attributes use three orthogonal taxonomy dimensions:
  - ``event_type``  → the *what* (EventType enum)
  - ``scope``       → the *who*  (EventScope enum)
  - ``severity``    → the *how much* (EventSeverity enum)
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict, model_validator

from wow_forecaster.taxonomy.event_taxonomy import EventScope, EventSeverity, EventType
from wow_forecaster.models.item import VALID_EXPANSIONS


class WoWEvent(BaseModel):
    """An economy-affecting WoW event with time window and announcement metadata.

    Attributes:
        event_id: Auto-assigned DB PK; ``None`` before insertion.
        slug: Unique machine-readable identifier, e.g. ``"tww-rtwf-nerubar-s1"``.
        display_name: Human-readable name, e.g. ``"RTWF: Nerub-ar Palace S1"``.
        event_type: Category of event from ``EventType`` enum.
        scope: Who is affected from ``EventScope`` enum.
        severity: Expected market impact from ``EventSeverity`` enum.
        expansion_slug: Which expansion this event belongs to.
        patch_version: Associated patch string, e.g. ``"11.0.2"``, or ``None``.
        start_date: First day this event is active (UTC date).
        end_date: Last day this event is active, or ``None`` if ongoing/unknown.
        announced_at: UTC datetime when Blizzard publicly announced this event.
            ``None`` means unknown / not yet announced.
        is_recurring: ``True`` for annual/weekly recurring events (holidays).
        recurrence_rule: Optional iCal RRULE string for recurring events.
        notes: Free-form annotation for researcher context.
    """

    model_config = ConfigDict(frozen=True)

    event_id: Optional[int] = None
    slug: str
    display_name: str
    event_type: EventType
    scope: EventScope
    severity: EventSeverity
    expansion_slug: str
    patch_version: Optional[str] = None
    start_date: date
    end_date: Optional[date] = None
    announced_at: Optional[datetime] = None
    is_recurring: bool = False
    recurrence_rule: Optional[str] = None
    notes: Optional[str] = None

    @model_validator(mode="after")
    def validate_date_ordering(self) -> "WoWEvent":
        """Ensure end_date is not before start_date when both are provided."""
        if self.end_date is not None and self.end_date < self.start_date:
            raise ValueError(
                f"end_date ({self.end_date}) must be >= start_date ({self.start_date})."
            )
        return self

    @model_validator(mode="after")
    def validate_expansion_slug(self) -> "WoWEvent":
        if self.expansion_slug not in VALID_EXPANSIONS:
            raise ValueError(
                f"Unknown expansion '{self.expansion_slug}'. "
                f"Must be one of {sorted(VALID_EXPANSIONS)}."
            )
        return self

    def is_known_at(self, as_of: datetime) -> bool:
        """Return ``True`` if this event was publicly announced before ``as_of``.

        This is the primary look-ahead bias guard for backtesting. Forecasts
        produced at time ``T`` should only factor in events where
        ``announced_at <= T``.

        If ``announced_at`` is ``None``, the event is conservatively treated
        as **unknown** (returns ``False``). This is the safe default — it is
        better to under-use event features than to accidentally include
        future information in a training window.

        Args:
            as_of: The point in time from which to evaluate knowledge.

        Returns:
            ``True`` if the event was known at ``as_of``, ``False`` otherwise.
        """
        if self.announced_at is None:
            return False
        return self.announced_at <= as_of

    def is_active_on(self, check_date: date) -> bool:
        """Return ``True`` if the event is active on ``check_date``.

        Args:
            check_date: The date to check.

        Returns:
            ``True`` if ``start_date <= check_date`` and either ``end_date``
            is ``None`` (ongoing) or ``check_date <= end_date``.
        """
        if check_date < self.start_date:
            return False
        if self.end_date is not None and check_date > self.end_date:
            return False
        return True
