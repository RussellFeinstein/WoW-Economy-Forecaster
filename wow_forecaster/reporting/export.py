"""
Export helpers for Power BI and manual analysis.

All functions write to disk and return the written ``Path``.
They accept generic ``list[dict]`` data to stay decoupled from
specific report shapes.

CSV exports are flat (no nested dicts) so they load directly in Power BI,
Excel, or pandas without any pre-processing step.

``flatten_recommendations_for_export()`` is the main adapter function:
it converts the nested ``categories`` structure of the recommendations
JSON into one row per item with all score components as separate columns.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path


def export_to_csv(
    records: list[dict],
    path: Path,
    fieldnames: list[str] | None = None,
) -> Path:
    """Write ``records`` to a UTF-8 CSV file.

    Args:
        records:    List of row dicts.
        path:       Destination file path (parent dirs created if missing).
        fieldnames: Column order.  If None, uses the keys of the first record.

    Returns:
        ``path`` as written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        path.write_text("", encoding="utf-8")
        return path
    cols = fieldnames or list(records[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    return path


def export_to_json(
    data: dict | list,
    path: Path,
) -> Path:
    """Write ``data`` to a pretty-printed JSON file.

    Args:
        data: Dict or list to serialise.
        path: Destination file path (parent dirs created if missing).

    Returns:
        ``path`` as written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    return path


def flatten_recommendations_for_export(recs_json: dict) -> list[dict]:
    """Flatten a recommendations JSON dict into a list of flat rows.

    Converts the nested ``categories`` structure into one row per item
    so the output is directly loadable in Power BI or Excel without any
    unpivoting.

    Each row contains:
    - ``realm_slug``, ``generated_at``, ``run_slug`` (report metadata)
    - ``category``, ``rank``, ``archetype_id``, ``horizon``, ``target_date``
    - ``current_price``, ``predicted_price``, ``ci_lower``, ``ci_upper``
    - ``roi_pct``, ``score``, ``action``, ``reasoning``
    - ``sc_opportunity``, ``sc_liquidity``, ``sc_volatility``,
      ``sc_event_boost``, ``sc_uncertainty`` (score components)
    - ``model_slug``

    Args:
        recs_json: Parsed ``recommendations_{realm}_{date}.json`` dict.

    Returns:
        List of flat row dicts.
    """
    rows: list[dict] = []
    realm        = recs_json.get("realm_slug", "")
    generated_at = recs_json.get("generated_at", "")
    run_slug     = recs_json.get("run_slug", "")

    for cat, items in recs_json.get("categories", {}).items():
        for item in items:
            comps = item.get("score_components", {})
            rows.append(
                {
                    "realm_slug":      realm,
                    "generated_at":    generated_at,
                    "run_slug":        run_slug,
                    "category":        cat,
                    "rank":            item.get("rank", ""),
                    "archetype_id":    item.get("archetype_id", ""),
                    "horizon":         item.get("horizon", ""),
                    "target_date":     item.get("target_date", ""),
                    "current_price":   item.get("current_price", ""),
                    "predicted_price": item.get("predicted_price", ""),
                    "ci_lower":        item.get("ci_lower", ""),
                    "ci_upper":        item.get("ci_upper", ""),
                    "roi_pct":         item.get("roi_pct", ""),
                    "score":           item.get("score", ""),
                    "action":          item.get("action", ""),
                    "reasoning":       item.get("reasoning", ""),
                    "sc_opportunity":  comps.get("opportunity", ""),
                    "sc_liquidity":    comps.get("liquidity", ""),
                    "sc_volatility":   comps.get("volatility", ""),
                    "sc_event_boost":  comps.get("event_boost", ""),
                    "sc_uncertainty":  comps.get("uncertainty", ""),
                    "model_slug":      item.get("model_slug", ""),
                }
            )

    return rows


def flatten_forecast_records_for_export(records: list[dict]) -> list[dict]:
    """Return forecast CSV rows enriched with the computed CI width column.

    This adds ``ci_width_gold`` and ``ci_pct_of_price`` as derived columns
    so Power BI users don't need custom measures for these common fields.

    Args:
        records: Row dicts from ``forecast_{realm}_{date}.csv``.

    Returns:
        Same rows with two extra keys appended.
    """
    result: list[dict] = []
    for r in records:
        try:
            ci_lower = float(r.get("ci_lower") or 0)
            ci_upper = float(r.get("ci_upper") or 0)
            pred     = float(r.get("predicted_price") or 0)
            ci_width = round(ci_upper - ci_lower, 4)
            ci_pct   = round(ci_width / pred, 4) if pred > 0 else None
        except (TypeError, ValueError):
            ci_width = None
            ci_pct   = None

        result.append({**r, "ci_width_gold": ci_width, "ci_pct_of_price": ci_pct})

    return result
