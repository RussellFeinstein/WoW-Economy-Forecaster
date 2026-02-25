"""
Forecast and recommendation output models.

``ForecastOutput`` represents a single point forecast with confidence interval
for a specific (archetype OR item, realm, horizon, date) tuple.

``RecommendationOutput`` is a trading action derived from a forecast:
buy/sell/hold/avoid with priority ranking and expiry.

Both models are frozen — once a forecast is produced and persisted, it should
not be mutated. Historical forecasts form the ground truth for backtesting.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

ForecastHorizon = Literal["1d", "7d", "14d", "30d", "90d"]
VALID_HORIZONS: frozenset[str] = frozenset({"1d", "7d", "14d", "30d", "90d"})

RecommendationAction = Literal["buy", "sell", "hold", "avoid"]


class ForecastOutput(BaseModel):
    """Point forecast with confidence interval for an archetype or item.

    Either ``archetype_id`` or ``item_id`` should be set — archetype-level
    forecasts are the primary output during the transfer learning phase when
    Midnight item IDs are not yet known.

    Attributes:
        forecast_id: Auto-assigned DB PK; ``None`` before insertion.
        run_id: FK to ``run_metadata.run_id``.
        archetype_id: FK to ``economic_archetypes.archetype_id``, or ``None``.
        item_id: FK to ``items.item_id``, or ``None``.
        realm_slug: Target realm for this forecast.
        forecast_horizon: Time horizon string, e.g. ``"7d"``.
        target_date: Forecasted calendar date (UTC).
        predicted_price_gold: Central price estimate in gold.
        confidence_lower: Lower bound of confidence interval.
        confidence_upper: Upper bound of confidence interval.
        confidence_pct: Confidence level of the interval, e.g. ``0.80`` for 80%.
        model_slug: Identifier of the model that produced this forecast.
        features_hash: SHA-256 of the feature vector for reproducibility.
    """

    model_config = ConfigDict(frozen=True)

    forecast_id: Optional[int] = None
    run_id: int
    archetype_id: Optional[int] = None
    item_id: Optional[int] = None
    realm_slug: str
    forecast_horizon: ForecastHorizon
    target_date: date
    predicted_price_gold: float
    confidence_lower: float
    confidence_upper: float
    confidence_pct: float = 0.80
    model_slug: str
    features_hash: Optional[str] = None

    @model_validator(mode="after")
    def validate_forecast_consistency(self) -> "ForecastOutput":
        if self.confidence_lower > self.confidence_upper:
            raise ValueError(
                f"confidence_lower ({self.confidence_lower}) must be <= "
                f"confidence_upper ({self.confidence_upper})."
            )
        if self.predicted_price_gold < 0:
            raise ValueError("predicted_price_gold must be non-negative.")
        if self.confidence_lower < 0:
            raise ValueError("confidence_lower must be non-negative.")
        if not 0.0 < self.confidence_pct < 1.0:
            raise ValueError(
                f"confidence_pct must be in (0.0, 1.0), got {self.confidence_pct}."
            )
        if self.archetype_id is None and self.item_id is None:
            raise ValueError("At least one of archetype_id or item_id must be set.")
        return self


class RecommendationOutput(BaseModel):
    """A trading recommendation derived from a forecast.

    Recommendations have a priority (1 = most urgent) and an expiry date after
    which the recommendation should be considered stale.

    Attributes:
        rec_id: Auto-assigned DB PK; ``None`` before insertion.
        forecast_id: FK to ``forecast_outputs.forecast_id``.
        action: Trading action: ``"buy"``, ``"sell"``, ``"hold"``, or ``"avoid"``.
        reasoning: Human-readable explanation of why this action is recommended.
        priority: Urgency rank from 1 (highest) to 10 (lowest).
        expires_at: UTC datetime after which this recommendation is stale,
            or ``None`` for open-ended recommendations.
    """

    model_config = ConfigDict(frozen=True)

    rec_id: Optional[int] = None
    forecast_id: int
    action: RecommendationAction
    reasoning: str
    priority: int = 5
    expires_at: Optional[datetime] = None

    @field_validator("priority")
    @classmethod
    def validate_priority_range(cls, v: int) -> int:
        if not 1 <= v <= 10:
            raise ValueError(f"priority must be in [1, 10], got {v}.")
        return v

    @field_validator("reasoning")
    @classmethod
    def validate_reasoning_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("reasoning must not be empty.")
        return v.strip()
