"""TSM export: generate TradeSkillMaster import strings from item-level forecasts.

Reads item-level forecast_outputs from the DB and filters to buy signals with
ROI >= min_roi_pct.  Outputs a TSM-compatible import string that can be pasted
into TradeSkillMaster's group import or shopping dialog.

Output format
-------------
Comma-separated TSM item string::

    i:12345,i:67890,...

Each token is a WoW item reference in the ``i:XXXXX`` format understood by
TradeSkillMaster.  The list is sorted by forecast ROI descending (best
opportunity first).

Filtering
---------
Only item-level forecasts are exported (``forecast_outputs`` rows where
``item_id IS NOT NULL`` and ``archetype_id IS NULL``).  Items are included when:

- ROI >= ``min_roi_pct`` (default 10 %)
- At least ``min_obs`` recent price observations exist (default 1)
- ``ci_quality = 'good'`` (excludes forecasts with extreme CI clamping)
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TsmExportRow:
    """A single item selected for TSM export.

    Attributes:
        item_id:       FK to items.item_id.
        name:          Item display name.
        current_price: Recent average price in gold (from normalized obs).
        forecast_price: Predicted price from the item-level forecast.
        roi_pct:       (forecast_price - current_price) / current_price.
                       Positive means the model predicts a price increase.
        horizon:       Forecast horizon label ("1d", "7d", "28d").
    """

    item_id:        int
    name:           str
    current_price:  float
    forecast_price: float
    roi_pct:        float
    horizon:        str


def fetch_tsm_export_items(
    conn:          sqlite3.Connection,
    realm_slug:    str,
    horizon:       str = "1d",
    min_roi_pct:   float = 0.10,
    lookback_days: int   = 3,
    min_obs:       int   = 1,
) -> list[TsmExportRow]:
    """Fetch item-level buy signals for TSM export.

    Queries the latest item-level forecast for each item and filters to those
    with ROI >= ``min_roi_pct``.  Only uses forecasts with ``ci_quality = 'good'``
    to exclude overly uncertain predictions.

    Args:
        conn:          Open DB connection (row_factory should be sqlite3.Row).
        realm_slug:    Realm to query (e.g. ``"us"``).
        horizon:       Forecast horizon label (``"1d"``, ``"7d"``, ``"28d"``).
        min_roi_pct:   Minimum fractional ROI threshold (0.10 = 10 %).
        lookback_days: Days of history to average for current_price (default 3).
        min_obs:       Min observation count required for current_price
                       (default 1).

    Returns:
        List of :class:`TsmExportRow` sorted by ``roi_pct`` descending.
        Empty when no item-level forecasts exist or none meet the ROI threshold.
    """
    rows = conn.execute(
        """
        SELECT
            i.item_id,
            i.name,
            cp.current_price,
            fc.predicted_price_gold AS forecast_price
        FROM items i
        JOIN (
            SELECT item_id,
                   AVG(price_gold) AS current_price,
                   COUNT(*)        AS obs_count
            FROM market_observations_normalized
            WHERE realm_slug  = :realm_slug
              AND is_outlier  = 0
              AND observed_at >= datetime('now', :since)
            GROUP BY item_id
            HAVING COUNT(*) >= :min_obs
        ) cp ON i.item_id = cp.item_id
        JOIN (
            SELECT item_id, MAX(created_at) AS max_ts
            FROM forecast_outputs
            WHERE item_id IS NOT NULL
              AND archetype_id IS NULL
              AND realm_slug       = :realm_slug
              AND forecast_horizon = :horizon
              AND ci_quality       = 'good'
            GROUP BY item_id
        ) latest ON i.item_id = latest.item_id
        JOIN forecast_outputs fc
             ON  fc.item_id          = latest.item_id
             AND fc.archetype_id     IS NULL
             AND fc.realm_slug       = :realm_slug
             AND fc.forecast_horizon = :horizon
             AND fc.created_at       = latest.max_ts
             AND fc.ci_quality       = 'good'
        WHERE cp.current_price > 0
          AND fc.predicted_price_gold > 0
        """,
        {
            "realm_slug": realm_slug,
            "horizon":    horizon,
            "since":      f"-{lookback_days} days",
            "min_obs":    min_obs,
        },
    ).fetchall()

    results: list[TsmExportRow] = []
    for row in rows:
        current  = float(row["current_price"])
        forecast = float(row["forecast_price"])
        if current <= 0:
            continue
        roi = (forecast - current) / current
        if roi < min_roi_pct:
            continue
        results.append(
            TsmExportRow(
                item_id        = int(row["item_id"]),
                name           = str(row["name"]),
                current_price  = current,
                forecast_price = forecast,
                roi_pct        = roi,
                horizon        = horizon,
            )
        )

    results.sort(key=lambda r: -r.roi_pct)
    logger.debug(
        "TSM export: %d items with ROI >= %.0f%% (realm=%s, horizon=%s)",
        len(results), min_roi_pct * 100, realm_slug, horizon,
    )
    return results


def build_tsm_import_string(items: list[TsmExportRow]) -> str:
    """Build a TSM-compatible import string.

    Formats items as ``i:XXXXX,i:YYYYY,...`` — the comma-separated item-ID list
    understood by TradeSkillMaster's group import and shopping dialogs.

    Args:
        items: Items from :func:`fetch_tsm_export_items`.

    Returns:
        Comma-separated ``i:XXXXX`` string, or ``""`` when the list is empty.
    """
    if not items:
        return ""
    return ",".join(f"i:{r.item_id}" for r in items)


def write_tsm_export(
    items:       list[TsmExportRow],
    output_path: Path,
) -> Path:
    """Write the TSM import string to a plain text file.

    Creates parent directories if they do not exist.

    Args:
        items:       Items from :func:`fetch_tsm_export_items`.
        output_path: Destination file path (e.g. ``Path("tsm_export.txt")``).

    Returns:
        Path to the written file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tsm_string = build_tsm_import_string(items)
    output_path.write_text(tsm_string + "\n", encoding="utf-8")
    logger.info("TSM export written: %s (%d items)", output_path, len(items))
    return output_path
