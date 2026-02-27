"""
Recommendation ranker: converts ForecastOutput + inference rows into
ScoredForecast objects, selects top-N per category, and builds
RecommendationOutput records for DB persistence.

Usage flow
----------
1. build_scored_forecasts(forecasts, inference_rows)
   -> list[ScoredForecast]  (one per matched forecast)

2. top_n_per_category(scored, n=3)
   -> dict[category_tag, list[ScoredForecast]]  (top-N per category)

3. build_recommendation_outputs(top_by_category)
   -> list[RecommendationOutput]  (ready for DB insertion)

Planned improvements to top_n_per_category
-------------------------------------------
- V2: Pareto-frontier ranking (score AND liquidity as dual objectives).
- V2: User-profile weighting (configurable per-category importance).
- V2: "Do not recommend" blocklist (user-defined archetype exclusions).
- V2: A/B test support (persist scoring parameters alongside recommendations).
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from wow_forecaster.models.forecast import ForecastOutput, RecommendationOutput
from wow_forecaster.recommendations.scorer import (
    ScoreComponents,
    build_reasoning,
    compute_score,
    determine_action,
)

_HORIZON_MAP: dict[str, int] = {
    "1d": 1, "7d": 7, "14d": 14, "28d": 28, "30d": 30, "90d": 90,
}


@dataclass
class ScoredForecast:
    """Intermediate object coupling a ForecastOutput with its recommendation data.

    Attributes:
        forecast:       The underlying ForecastOutput.
        score:          Weighted recommendation score (0–~100).
        components:     Detailed score breakdown.
        action:         Trading action ("buy", "sell", "hold", "avoid").
        reasoning:      Human-readable explanation string.
        category_tag:   ArchetypeCategory slug (e.g. "consumable").
        archetype_id:   Economic archetype PK.
        realm_slug:     Realm this forecast belongs to.
        current_price:  price_mean from the inference row (gold).
        horizon_days:   Forecast horizon in days.
    """

    forecast:      ForecastOutput
    score:         float
    components:    ScoreComponents
    action:        str
    reasoning:     str
    category_tag:       str
    archetype_sub_tag:  str | None
    archetype_id:       int
    realm_slug:         str
    current_price:      float | None
    horizon_days:       int


def build_scored_forecasts(
    forecasts:      list[ForecastOutput],
    inference_rows: list[dict[str, Any]],
) -> list[ScoredForecast]:
    """Score all forecasts against their corresponding inference rows.

    Forecasts are matched to inference rows by (archetype_id, realm_slug).
    Forecasts with no matching inference row are silently skipped.

    Args:
        forecasts:      ForecastOutput list (from run_inference or DB).
        inference_rows: Inference Parquet rows (current market state).

    Returns:
        List of ScoredForecast objects.
    """
    # Build lookup: (archetype_id, realm_slug) -> inference_row
    inf_lookup: dict[tuple[int, str], dict[str, Any]] = {}
    for row in inference_rows:
        arch_id = row.get("archetype_id")
        realm   = row.get("realm_slug")
        if arch_id is not None and realm is not None:
            inf_lookup[(int(arch_id), str(realm))] = row

    scored: list[ScoredForecast] = []

    for fc in forecasts:
        arch_id = fc.archetype_id
        if arch_id is None:
            continue

        inf_row = inf_lookup.get((arch_id, fc.realm_slug))
        if inf_row is None:
            continue

        horizon_days = _HORIZON_MAP.get(fc.forecast_horizon, 7)

        components = compute_score(
            forecast_price=fc.predicted_price_gold,
            current_price=inf_row.get("price_mean"),
            confidence_lower=fc.confidence_lower,
            confidence_upper=fc.confidence_upper,
            quantity_sum=inf_row.get("quantity_sum"),
            auctions_sum=inf_row.get("auctions_sum"),
            price_roll_std_7d=inf_row.get("price_roll_std_7d"),
            price_mean=inf_row.get("price_mean"),
            event_active=bool(inf_row.get("event_active", False)),
            event_days_to_next=inf_row.get("event_days_to_next"),
            event_severity_max=inf_row.get("event_severity_max"),
            event_archetype_impact=inf_row.get("event_archetype_impact"),
            is_cold_start=bool(inf_row.get("is_cold_start", False)),
            transfer_confidence=inf_row.get("transfer_confidence"),
        )

        action = determine_action(
            roi=components.roi,
            uncertainty_pct=components.uncertainty_pct,
            volatility_cv=components.volatility_cv,
        )

        reasoning = build_reasoning(
            components=components,
            action=action,
            is_cold_start=bool(inf_row.get("is_cold_start", False)),
            transfer_confidence=inf_row.get("transfer_confidence"),
            event_active=bool(inf_row.get("event_active", False)),
            event_days_to_next=inf_row.get("event_days_to_next"),
            event_severity_max=inf_row.get("event_severity_max"),
            horizon_days=horizon_days,
        )

        category_tag      = str(inf_row.get("archetype_category") or "unknown")
        archetype_sub_tag = inf_row.get("archetype_sub_tag")

        scored.append(
            ScoredForecast(
                forecast=fc,
                score=round(components.total, 2),
                components=components,
                action=action,
                reasoning=reasoning,
                category_tag=category_tag,
                archetype_sub_tag=archetype_sub_tag,
                archetype_id=arch_id,
                realm_slug=fc.realm_slug,
                current_price=inf_row.get("price_mean"),
                horizon_days=horizon_days,
            )
        )

    return scored


def top_n_per_category(
    scored:  list[ScoredForecast],
    n:       int = 3,
    actions: list[str] | None = None,
) -> dict[str, list[ScoredForecast]]:
    """Return top-N scored forecasts per archetype category.

    Each archetype appears at most once per category — the horizon with the
    highest score is kept.  When two horizons tie on score the shorter horizon
    wins (more immediately actionable).  After de-duplication, archetypes are
    sorted by score descending; ties broken by archetype_id ascending.

    Args:
        scored:   All ScoredForecast objects.
        n:        Max results per category (default 3).
        actions:  Optional filter — only include these action strings.
                  E.g. ``["buy"]`` for buy-only recommendations.
                  Pass ``None`` to include all actions.

    Returns:
        Dict mapping category_tag -> list of top-N ScoredForecast (desc order).
    """
    by_cat: dict[str, list[ScoredForecast]] = defaultdict(list)
    for sf in scored:
        if actions is not None and sf.action not in actions:
            continue
        by_cat[sf.category_tag].append(sf)

    result: dict[str, list[ScoredForecast]] = {}
    for cat, items in by_cat.items():
        # De-duplicate: keep only the best-scoring horizon per archetype.
        # Tie-break within same archetype: prefer shorter horizon (more actionable).
        best_by_archetype: dict[int, ScoredForecast] = {}
        for sf in items:
            existing = best_by_archetype.get(sf.archetype_id)
            if existing is None:
                best_by_archetype[sf.archetype_id] = sf
            elif sf.score > existing.score:
                best_by_archetype[sf.archetype_id] = sf
            elif sf.score == existing.score and sf.horizon_days < existing.horizon_days:
                best_by_archetype[sf.archetype_id] = sf

        deduped = list(best_by_archetype.values())
        # Primary sort: score descending. Secondary: archetype_id ascending (stable).
        sorted_items = sorted(deduped, key=lambda x: (-x.score, x.archetype_id))
        result[cat] = sorted_items[:n]

    return result


def build_recommendation_outputs(
    top_by_category:      dict[str, list[ScoredForecast]],
    default_horizon_days: int = 7,
) -> list[RecommendationOutput]:
    """Convert ScoredForecast objects into RecommendationOutput for DB persistence.

    Priority is set by rank within the category (1 = top of category).
    Forecasts without a forecast_id are silently skipped (must be persisted first).

    Args:
        top_by_category:      Output from top_n_per_category().
        default_horizon_days: Used to set expires_at when horizon unknown.

    Returns:
        List of RecommendationOutput objects ready for insert_recommendation().
    """
    outputs: list[RecommendationOutput] = []

    for cat, items in top_by_category.items():
        for rank, sf in enumerate(items, start=1):
            if sf.forecast.forecast_id is None:
                continue  # forecast_id FK required

            expires = datetime.now(tz=timezone.utc) + timedelta(
                days=sf.horizon_days or default_horizon_days
            )

            outputs.append(
                RecommendationOutput(
                    forecast_id=sf.forecast.forecast_id,
                    action=sf.action,
                    reasoning=sf.reasoning,
                    priority=rank,
                    expires_at=expires,
                    score=round(sf.score, 2),
                    score_components=json.dumps(
                        {
                            "opportunity_score":   sf.components.opportunity_score,
                            "liquidity_score":     sf.components.liquidity_score,
                            "volatility_penalty":  sf.components.volatility_penalty,
                            "event_boost":         sf.components.event_boost,
                            "uncertainty_penalty": sf.components.uncertainty_penalty,
                            "roi":                 sf.components.roi,
                            "volatility_cv":       sf.components.volatility_cv,
                            "uncertainty_pct":     sf.components.uncertainty_pct,
                        }
                    ),
                    category_tag=sf.category_tag,
                )
            )

    return outputs
