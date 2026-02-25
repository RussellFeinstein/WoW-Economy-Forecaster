"""
Recommendation report writer: CSV and JSON output for forecasts and
ranked recommendations.

All functions are pure I/O — no DB access.  They consume in-memory
ScoredForecast lists and write human-readable + machine-readable files.

Output files (written by RecommendStage)
-----------------------------------------
  data/outputs/forecasts/
    forecast_{realm}_{date}.csv          -- all forecast_outputs for this run
    forecast_{realm}_{date}_manifest.json

  data/outputs/recommendations/
    recommendations_{realm}_{date}.csv   -- top-N per category (all horizons)
    recommendations_{realm}_{date}.json  -- same data, structured JSON
"""

from __future__ import annotations

import csv
import json
import logging
from datetime import date
from pathlib import Path

from wow_forecaster.recommendations.ranker import ScoredForecast

logger = logging.getLogger(__name__)


def write_forecast_csv(
    scored: list[ScoredForecast],
    output_dir: Path,
    realm_slug: str,
    run_date: date | None = None,
) -> Path:
    """Write all scored forecasts to a CSV file.

    Columns: archetype_id, realm_slug, horizon, target_date,
             current_price, predicted_price, ci_lower, ci_upper,
             roi_pct, score, action, model_slug.

    Args:
        scored:     All ScoredForecast objects for this run.
        output_dir: Directory to write the file (created if missing).
        realm_slug: Realm slug (used in filename).
        run_date:   Date label for the filename. Defaults to today.

    Returns:
        Path to the written CSV file.
    """
    if run_date is None:
        run_date = date.today()

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"forecast_{realm_slug}_{run_date}.csv"

    fieldnames = [
        "archetype_id", "realm_slug", "horizon", "target_date",
        "current_price", "predicted_price", "ci_lower", "ci_upper",
        "roi_pct", "score", "action", "model_slug",
    ]

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for sf in sorted(scored, key=lambda x: (-x.score, x.archetype_id)):
            roi_pct = f"{sf.components.roi:+.2%}" if sf.components.roi is not None else ""
            writer.writerow(
                {
                    "archetype_id":   sf.archetype_id,
                    "realm_slug":     sf.realm_slug,
                    "horizon":        sf.forecast.forecast_horizon,
                    "target_date":    sf.forecast.target_date.isoformat(),
                    "current_price":  sf.current_price,
                    "predicted_price":sf.forecast.predicted_price_gold,
                    "ci_lower":       sf.forecast.confidence_lower,
                    "ci_upper":       sf.forecast.confidence_upper,
                    "roi_pct":        roi_pct,
                    "score":          sf.score,
                    "action":         sf.action,
                    "model_slug":     sf.forecast.model_slug,
                }
            )

    logger.info("Forecast CSV written: %s (%d rows)", csv_path, len(scored))
    return csv_path


def write_recommendation_csv(
    top_by_category: dict[str, list[ScoredForecast]],
    output_dir: Path,
    realm_slug: str,
    run_date: date | None = None,
) -> Path:
    """Write top-N recommendations per category to a CSV file.

    Columns: rank, category, archetype_id, realm_slug, horizon,
             current_price, predicted_price, roi_pct, score, action, reasoning.

    Args:
        top_by_category: Output from top_n_per_category().
        output_dir:      Target directory.
        realm_slug:      Used in filename.
        run_date:        Date label. Defaults to today.

    Returns:
        Path to the written CSV file.
    """
    if run_date is None:
        run_date = date.today()

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"recommendations_{realm_slug}_{run_date}.csv"

    fieldnames = [
        "rank", "category", "archetype_id", "realm_slug", "horizon",
        "current_price", "predicted_price", "roi_pct", "score", "action", "reasoning",
    ]

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for cat in sorted(top_by_category):
            for rank, sf in enumerate(top_by_category[cat], start=1):
                roi_pct = (
                    f"{sf.components.roi:+.2%}"
                    if sf.components.roi is not None else ""
                )
                writer.writerow(
                    {
                        "rank":           rank,
                        "category":       sf.category_tag,
                        "archetype_id":   sf.archetype_id,
                        "realm_slug":     sf.realm_slug,
                        "horizon":        sf.forecast.forecast_horizon,
                        "current_price":  sf.current_price,
                        "predicted_price":sf.forecast.predicted_price_gold,
                        "roi_pct":        roi_pct,
                        "score":          sf.score,
                        "action":         sf.action,
                        "reasoning":      sf.reasoning,
                    }
                )

    logger.info("Recommendation CSV written: %s", csv_path)
    return csv_path


def write_recommendation_json(
    top_by_category: dict[str, list[ScoredForecast]],
    output_dir: Path,
    realm_slug: str,
    run_date: date | None = None,
    run_slug: str = "",
) -> Path:
    """Write top-N recommendations to a structured JSON file.

    Args:
        top_by_category: Output from top_n_per_category().
        output_dir:      Target directory.
        realm_slug:      Used in filename + metadata.
        run_date:        Date label. Defaults to today.
        run_slug:        Pipeline run UUID for provenance.

    Returns:
        Path to the written JSON file.
    """
    if run_date is None:
        run_date = date.today()

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"recommendations_{realm_slug}_{run_date}.json"

    payload: dict = {
        "schema_version": "v0.5.0",
        "realm_slug":     realm_slug,
        "generated_at":   run_date.isoformat(),
        "run_slug":       run_slug,
        "categories":     {},
    }

    for cat in sorted(top_by_category):
        items = top_by_category[cat]
        payload["categories"][cat] = [
            {
                "rank":            rank,
                "archetype_id":    sf.archetype_id,
                "realm_slug":      sf.realm_slug,
                "horizon":         sf.forecast.forecast_horizon,
                "target_date":     sf.forecast.target_date.isoformat(),
                "current_price":   sf.current_price,
                "predicted_price": sf.forecast.predicted_price_gold,
                "ci_lower":        sf.forecast.confidence_lower,
                "ci_upper":        sf.forecast.confidence_upper,
                "roi_pct":         round(sf.components.roi, 4),
                "score":           sf.score,
                "action":          sf.action,
                "reasoning":       sf.reasoning,
                "score_components": {
                    "opportunity":   sf.components.opportunity_score,
                    "liquidity":     sf.components.liquidity_score,
                    "volatility":    sf.components.volatility_penalty,
                    "event_boost":   sf.components.event_boost,
                    "uncertainty":   sf.components.uncertainty_penalty,
                },
                "model_slug":      sf.forecast.model_slug,
            }
            for rank, sf in enumerate(items, start=1)
        ]

    json_path.write_text(json.dumps(payload, indent=2, default=str))
    logger.info("Recommendation JSON written: %s", json_path)
    return json_path
