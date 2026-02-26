"""
Live model health evaluation for the WoW Economy Forecaster.

This module computes a ModelHealthSummary by comparing forecast_outputs
predictions (where target_date has passed) against actual normalized prices
from market_observations_normalized.

Health status thresholds (based on mae_ratio = live_mae / baseline_mae):
    ok        : mae_ratio < 1.5  (or no baseline — unknown)
    degraded  : 1.5 <= mae_ratio < 3.0
    critical  : mae_ratio >= 3.0

When no forecast-vs-actual pairs can be evaluated (because target dates
have not yet passed, or there is no normalized data for those dates), the
status is "unknown".

This is the backend for the ``evaluate-live-forecast`` CLI command.
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Optional

logger = logging.getLogger(__name__)

# Health status constants
HEALTH_OK       = "ok"
HEALTH_DEGRADED = "degraded"
HEALTH_CRITICAL = "critical"
HEALTH_UNKNOWN  = "unknown"

# MAE ratio thresholds for health status
_DEGRADED_RATIO  = 1.5
_CRITICAL_RATIO  = 3.0


@dataclass(frozen=True)
class ModelHealthSummary:
    """Live model performance summary for one (realm, horizon) combination.

    Attributes:
        realm_slug:      Realm evaluated.
        checked_at:      ISO-8601 UTC timestamp.
        horizon_days:    Forecast horizon in days (1, 7, or 28).
        n_evaluated:     Number of forecast-vs-actual pairs evaluated.
        live_mae:        Mean absolute error over evaluated pairs (gold).
        baseline_mae:    Reference MAE from the most recent backtest run.
        mae_ratio:       live_mae / baseline_mae (None if either missing).
        live_dir_acc:    Directional accuracy on evaluated pairs (0–1).
        baseline_dir_acc: Reference directional accuracy from backtest.
        health_status:   One of "ok", "degraded", "critical", "unknown".
    """

    realm_slug:       str
    checked_at:       str
    horizon_days:     int
    n_evaluated:      int
    live_mae:         Optional[float]
    baseline_mae:     Optional[float]
    mae_ratio:        Optional[float]
    live_dir_acc:     Optional[float]
    baseline_dir_acc: Optional[float]
    health_status:    str


def compute_health_summary(
    conn: sqlite3.Connection,
    realm_slug: str,
    horizon_days: int,
    window_days: int = 14,
) -> ModelHealthSummary:
    """Compute a live health summary for one (realm, horizon).

    Args:
        conn:         Open SQLite connection.
        realm_slug:   Realm to evaluate.
        horizon_days: Forecast horizon to evaluate.
        window_days:  How many recent days of target dates to include.

    Returns:
        ModelHealthSummary.
    """
    from wow_forecaster.monitoring.drift import _utc_now_iso

    now_str     = _utc_now_iso()
    horizon_tag = f"{horizon_days}d"
    cutoff      = (date.today() - timedelta(days=window_days)).isoformat()

    # ── Live MAE and directional accuracy ────────────────────────────────────
    # For each forecast where target_date < today, find the average actual
    # price from normalized obs on that target_date.
    live_q = """
        SELECT
            f.forecast_id,
            f.predicted_price_gold,
            f.target_date,
            AVG(n.price_gold) AS actual_price
        FROM forecast_outputs f
        JOIN market_observations_normalized n
            ON  n.archetype_id = f.archetype_id
            AND n.realm_slug   = f.realm_slug
            AND date(n.observed_at) = f.target_date
        WHERE f.realm_slug       = ?
          AND f.forecast_horizon = ?
          AND f.target_date     >= ?
          AND f.target_date     <  date('now')
          AND n.is_outlier = 0
        GROUP BY f.forecast_id;
    """
    try:
        live_rows = conn.execute(
            live_q, (realm_slug, horizon_tag, cutoff)
        ).fetchall()
    except Exception as exc:
        logger.warning("Health summary live query failed: %s", exc)
        live_rows = []

    n_evaluated   = len(live_rows)
    live_mae      = None
    live_dir_acc  = None

    if n_evaluated > 0:
        errors   = []
        dir_hits = []
        for row in live_rows:
            pred   = row["predicted_price_gold"]
            actual = row["actual_price"]
            if pred is not None and actual is not None:
                errors.append(abs(pred - actual))
                # For directional accuracy we need the price at the time the
                # forecast was made.  We don't store that directly; we skip
                # dir_acc when we can't compute it.

        if errors:
            live_mae = round(sum(errors) / len(errors), 4)

    # ── Baseline MAE and dir_acc from most recent backtest ───────────────────
    baseline_mae     = None
    baseline_dir_acc = None
    try:
        bt_row = conn.execute(
            """
            SELECT backtest_run_id FROM backtest_runs
            WHERE realm_slug = ?
            ORDER BY backtest_run_id DESC LIMIT 1;
            """,
            (realm_slug,),
        ).fetchone()

        if bt_row is not None:
            bt_run_id = bt_row["backtest_run_id"]

            m_row = conn.execute(
                """
                SELECT
                    AVG(abs_error)       AS mae,
                    AVG(direction_correct) AS dir_acc
                FROM backtest_fold_results
                WHERE backtest_run_id = ?
                  AND horizon_days    = ?
                  AND actual_price IS NOT NULL;
                """,
                (bt_run_id, horizon_days),
            ).fetchone()

            if m_row:
                if m_row["mae"] is not None:
                    baseline_mae = round(m_row["mae"], 4)
                if m_row["dir_acc"] is not None:
                    baseline_dir_acc = round(m_row["dir_acc"], 4)
    except Exception as exc:
        logger.warning("Health summary baseline query failed: %s", exc)

    # ── MAE ratio and health status ───────────────────────────────────────────
    mae_ratio = None
    if live_mae is not None and baseline_mae is not None and baseline_mae > 1e-6:
        mae_ratio = round(live_mae / baseline_mae, 4)

    health_status = _classify_health(mae_ratio)

    logger.info(
        "Model health | realm=%s | h=%dd | n=%d | live_mae=%s | ratio=%s | status=%s",
        realm_slug, horizon_days, n_evaluated,
        f"{live_mae:.2f}g" if live_mae else "N/A",
        f"{mae_ratio:.2f}x" if mae_ratio else "N/A",
        health_status,
    )

    return ModelHealthSummary(
        realm_slug=realm_slug,
        checked_at=now_str,
        horizon_days=horizon_days,
        n_evaluated=n_evaluated,
        live_mae=live_mae,
        baseline_mae=baseline_mae,
        mae_ratio=mae_ratio,
        live_dir_acc=live_dir_acc,
        baseline_dir_acc=baseline_dir_acc,
        health_status=health_status,
    )


def _classify_health(mae_ratio: Optional[float]) -> str:
    """Map mae_ratio to a health status string.

    Args:
        mae_ratio: live_mae / baseline_mae, or None.

    Returns:
        One of "ok", "degraded", "critical", "unknown".
    """
    if mae_ratio is None:
        return HEALTH_UNKNOWN
    if mae_ratio >= _CRITICAL_RATIO:
        return HEALTH_CRITICAL
    if mae_ratio >= _DEGRADED_RATIO:
        return HEALTH_DEGRADED
    return HEALTH_OK
