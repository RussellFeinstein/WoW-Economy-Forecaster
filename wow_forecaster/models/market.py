"""
Market observation models — raw ingestion and normalized feature-ready data.

Two-stage design:
  1. ``RawMarketObservation``       — data exactly as received from source;
                                     all monetary values in copper (integer).
  2. ``NormalizedMarketObservation`` — post-processing: copper → gold,
                                      z-score outlier detection applied.

Both models are frozen (immutable) after construction. This prevents
accidental mutation as records pass through pipeline stages.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict, field_validator

# Valid source identifiers for raw observations
VALID_SOURCES = frozenset({"tsm_export", "ah_scan", "manual", "undermine_api", "blizzard_api", "auctionator_posting"})
VALID_FACTIONS = frozenset({"alliance", "horde", "neutral"})


class RawMarketObservation(BaseModel):
    """A single price observation as ingested from a data source.

    All monetary values are stored in **copper** (integer) at this layer.
    No normalization or conversion is applied here — that is the responsibility
    of the ``NormalizeStage`` pipeline step.

    Attributes:
        item_id: WoW canonical item ID.
        realm_slug: Blizzard realm slug (e.g. ``"area-52"``).
        faction: ``"alliance"``, ``"horde"``, or ``"neutral"`` (commodity AH).
        observed_at: UTC timestamp of the AH scan or export.
        source: Data provenance identifier.
        min_buyout_raw: Lowest posted buyout in copper, or ``None`` if unavailable.
        market_value_raw: TSM-style market value in copper, or ``None``.
        historical_value_raw: Long-running historical average in copper, or ``None``.
        quantity_listed: Total units listed on AH at scan time, or ``None``.
        num_auctions: Number of individual auctions, or ``None``.
        raw_json: Optional full source blob for debugging / re-processing.
    """

    model_config = ConfigDict(frozen=True)

    item_id: int
    realm_slug: str
    faction: str = "neutral"
    observed_at: datetime
    source: str
    min_buyout_raw: Optional[int] = None
    market_value_raw: Optional[int] = None
    historical_value_raw: Optional[int] = None
    quantity_listed: Optional[int] = None
    num_auctions: Optional[int] = None
    raw_json: Optional[str] = None

    @field_validator("faction")
    @classmethod
    def validate_faction(cls, v: str) -> str:
        if v not in VALID_FACTIONS:
            raise ValueError(f"Invalid faction '{v}'. Must be one of {sorted(VALID_FACTIONS)}.")
        return v

    @field_validator("source")
    @classmethod
    def validate_source(cls, v: str) -> str:
        if v not in VALID_SOURCES:
            raise ValueError(f"Unknown source '{v}'. Must be one of {sorted(VALID_SOURCES)}.")
        return v

    @field_validator("min_buyout_raw", "market_value_raw", "historical_value_raw")
    @classmethod
    def validate_copper_non_negative(cls, v: Optional[int]) -> Optional[int]:
        if v is not None and v < 0:
            raise ValueError("Copper monetary values must be non-negative.")
        return v

    @field_validator("quantity_listed", "num_auctions")
    @classmethod
    def validate_counts_non_negative(cls, v: Optional[int]) -> Optional[int]:
        if v is not None and v < 0:
            raise ValueError("Quantity and auction count values must be non-negative.")
        return v


class NormalizedMarketObservation(BaseModel):
    """Price observation after normalization pipeline processing.

    Derived from a ``RawMarketObservation`` after:
      - Copper → gold conversion (divide by 10_000).
      - Z-score calculation against recent rolling window.
      - Outlier flagging (``is_outlier=True`` if ``|z_score| > threshold``).

    Attributes:
        obs_id: Foreign key to ``market_observations_raw.obs_id``.
        item_id: WoW canonical item ID.
        archetype_id: FK to ``economic_archetypes.archetype_id`` (may be ``None``
            if the item has not yet been mapped to an archetype).
        realm_slug: Blizzard realm slug.
        faction: AH faction.
        observed_at: UTC timestamp (copied from raw).
        price_gold: Minimum buyout converted to gold.
        market_value_gold: TSM market value in gold, or ``None``.
        historical_value_gold: Historical average in gold, or ``None``.
        quantity_listed: Units listed on AH.
        num_auctions: Number of individual auctions.
        z_score: Standardized price vs. rolling window, or ``None`` if
            insufficient history.
        is_outlier: ``True`` if ``|z_score|`` exceeds the configured threshold.
    """

    model_config = ConfigDict(frozen=True)

    obs_id: int
    item_id: int
    archetype_id: Optional[int] = None
    realm_slug: str
    faction: str = "neutral"
    observed_at: datetime
    price_gold: float
    market_value_gold: Optional[float] = None
    historical_value_gold: Optional[float] = None
    quantity_listed: Optional[int] = None
    num_auctions: Optional[int] = None
    z_score: Optional[float] = None
    is_outlier: bool = False

    @field_validator("price_gold", "market_value_gold", "historical_value_gold")
    @classmethod
    def validate_gold_non_negative(cls, v: Optional[float]) -> Optional[float]:
        if v is not None and v < 0:
            raise ValueError("Gold monetary values must be non-negative.")
        return v

    @field_validator("faction")
    @classmethod
    def validate_faction(cls, v: str) -> str:
        if v not in VALID_FACTIONS:
            raise ValueError(f"Invalid faction '{v}'. Must be one of {sorted(VALID_FACTIONS)}.")
        return v
