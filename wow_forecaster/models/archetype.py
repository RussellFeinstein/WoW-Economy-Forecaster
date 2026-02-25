"""
Economic archetype models.

``EconomicArchetype`` is the central concept for cross-expansion transfer.
Rather than mapping TWW item IDs to Midnight item IDs (impossible before
Midnight ships), we map **economic behavior archetypes**.

Example: the archetype ``consumable.flask.stat`` captures:
  - Crafted by Alchemy
  - Consumed on raid nights
  - 1-hour duration, refreshed between pulls
  - Price spikes 2–3 days before RTWF, crashes at season end
  - Supply constrained by herb farming throughput

This behavior archetype will have a Midnight equivalent with different
specific items but identical economic dynamics. The ``ArchetypeMapping``
model formalizes that relationship with a confidence score and rationale.

``transfer_confidence`` is the key signal for downstream modeling:
  - 1.0 = near-identical economic role, transfer weights directly
  - 0.5 = similar but with notable differences (new crafting system)
  - 0.0 = no meaningful transfer possible
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, field_validator

from wow_forecaster.taxonomy.archetype_taxonomy import ArchetypeCategory, ArchetypeTag


class EconomicArchetype(BaseModel):
    """A behavior-based item grouping for cross-expansion economic modeling.

    Attributes:
        archetype_id: Auto-assigned DB PK; ``None`` before insertion.
        slug: Unique dot-delimited identifier, e.g. ``"consumable.flask.stat"``.
        display_name: Human-readable label.
        category_tag: Top-level ``ArchetypeCategory`` this belongs to.
        sub_tag: Specific ``ArchetypeTag`` if applicable, or ``None`` for
            category-level archetypes.
        description: Optional explanation of economic behavior patterns.
        is_transferable: Whether this archetype can be mapped to a target expansion.
            Set to ``False`` for expansion-specific mechanics with no parallel.
        transfer_confidence: Prior confidence [0.0, 1.0] that this archetype's
            patterns will transfer to the target expansion.
        transfer_notes: Human annotation explaining the confidence score.
    """

    model_config = ConfigDict(frozen=True)

    archetype_id: Optional[int] = None
    slug: str
    display_name: str
    category_tag: ArchetypeCategory
    sub_tag: Optional[ArchetypeTag] = None
    description: Optional[str] = None
    is_transferable: bool = True
    transfer_confidence: float = 0.5
    transfer_notes: Optional[str] = None

    @field_validator("transfer_confidence")
    @classmethod
    def validate_confidence_range(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError(
                f"transfer_confidence must be in [0.0, 1.0], got {v}."
            )
        return v

    @field_validator("slug")
    @classmethod
    def validate_slug_format(cls, v: str) -> str:
        if not v or " " in v or v != v.lower():
            raise ValueError(
                f"Archetype slug '{v}' must be lowercase with no spaces."
            )
        return v


class ArchetypeMapping(BaseModel):
    """Maps an archetype from one expansion to its economic equivalent in another.

    Every mapping must include a human-readable ``mapping_rationale``. This
    is intentional: it forces the researcher to reason explicitly about *why*
    a mapping is valid, creating an audit trail for when model transfer fails.

    Attributes:
        mapping_id: Auto-assigned DB PK; ``None`` before insertion.
        source_archetype_id: FK to ``economic_archetypes.archetype_id``.
        target_archetype_id: FK to ``economic_archetypes.archetype_id``.
        source_expansion: Expansion of the source archetype (usually ``"tww"``).
        target_expansion: Expansion of the target archetype (usually ``"midnight"``).
        confidence_score: How well the economic behavior transfers [0.0, 1.0].
        mapping_rationale: Required human explanation of the mapping.
        created_by: ``"manual"`` for researcher-defined, ``"model"`` for ML-derived.
    """

    model_config = ConfigDict(frozen=True)

    mapping_id: Optional[int] = None
    source_archetype_id: int
    target_archetype_id: int
    source_expansion: str = "tww"
    target_expansion: str = "midnight"
    confidence_score: float
    mapping_rationale: str
    created_by: str = "manual"

    @field_validator("confidence_score")
    @classmethod
    def validate_confidence_range(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError(
                f"confidence_score must be in [0.0, 1.0], got {v}."
            )
        return v

    @field_validator("mapping_rationale")
    @classmethod
    def validate_rationale_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("mapping_rationale must not be empty.")
        return v.strip()
