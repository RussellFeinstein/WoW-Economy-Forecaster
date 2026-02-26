"""
Drift detection for the WoW Economy Forecaster.

Three detection modes
---------------------

DATA DRIFT — Feature distribution shift
    Compares the distribution of price_gold in a "recent" window (last N hours)
    against a "baseline" window (prior M days) for each (archetype_id, realm_slug)
    series.  Test statistic: z-score of the recent mean vs the baseline
    (mean, std).

    Why z-score of means rather than PSI?
      PSI (Population Stability Index) requires binning data.  Bin-count choice
      strongly affects sensitivity and breaks down on thin series (< 50 obs).
      The z-score of means is transparent ("price moved 2.5 std from 30-day
      baseline"), interpretable, and works with sparse data.  We flag each
      series independently and then aggregate to a realm-level verdict by the
      fraction of series that are drifted.

    Drift-level thresholds (fraction of drifted series):
      NONE     < 10%   — normal variance, no action
      LOW      < 25%   — mild perturbation, CI widening advisory
      MEDIUM   < 40%   — noticeable shift, retrain recommended
      HIGH     < 60%   — strong regime change, model likely stale
      CRITICAL >= 60%  — widespread shift, model unreliable

ERROR DRIFT — Forecast residual degradation
    When target_date < today, compares actual prices (from
    market_observations_normalized) against forecast_outputs predictions.
    Computes a rolling MAE over the configured window and compares to the
    baseline MAE from the most recent backtest run.

    Ratio thresholds (live_mae / baseline_mae):
      NONE     < 1.2   — model performing within normal variance
      LOW      < 1.5   — 20–50% MAE increase, watch and wait
      MEDIUM   < 2.0   — 50–100% increase, retrain recommended
      HIGH     < 3.0   — 100–200% increase, model stale
      CRITICAL >= 3.0  — > 200% increase, model likely broken

    Why K=1.5 for LOW?  Normal market noise moves MAE ±20%; a 50% increase
    is a sustained signal, not noise.

EVENT-SHOCK DETECTION — Hook for manual event labels
    Queries wow_events for events starting within ±shock_window_days of today.
    Sets shock_active=True if any MAJOR/CRITICAL/CATASTROPHIC event is active
    or imminent.

    This is a *detection hook* — it identifies that we are in or near an event
    window so that adaptive policy can widen CIs preemptively.  Actual price
    attribution to events requires real price data (not implemented in this
    stub-mode system).
"""

from __future__ import annotations

import logging
import math
import sqlite3
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)

# Severity values that trigger shock detection
_SHOCK_SEVERITIES = {"major", "critical", "catastrophic"}

# Default thresholds (overridden by MonitoringConfig at runtime)
_DEFAULT_Z_THRESHOLD        = 2.0
_DEFAULT_DRIFT_WINDOW_HOURS = 25
_DEFAULT_BASELINE_DAYS      = 30
_DEFAULT_ERROR_WINDOW_DAYS  = 7
_DEFAULT_SHOCK_WINDOW_DAYS  = 7

# MAE ratio thresholds (low, medium, high, critical)
_DEFAULT_MAE_THRESHOLDS = (1.2, 1.5, 2.0, 3.0)


class DriftLevel(str, Enum):
    """Severity of detected drift."""

    NONE     = "none"
    LOW      = "low"
    MEDIUM   = "medium"
    HIGH     = "high"
    CRITICAL = "critical"

    def __ge__(self, other: "DriftLevel") -> bool:
        _order = list(DriftLevel)
        return _order.index(self) >= _order.index(other)

    def __gt__(self, other: "DriftLevel") -> bool:
        _order = list(DriftLevel)
        return _order.index(self) > _order.index(other)

    def __le__(self, other: "DriftLevel") -> bool:
        _order = list(DriftLevel)
        return _order.index(self) <= _order.index(other)

    def __lt__(self, other: "DriftLevel") -> bool:
        _order = list(DriftLevel)
        return _order.index(self) < _order.index(other)


# ── Per-series stats ──────────────────────────────────────────────────────────

@dataclass(frozen=True)
class SeriesDriftStats:
    """Drift statistics for a single (archetype_id, realm_slug) series.

    Attributes:
        archetype_id:   DB PK of the archetype.
        realm_slug:     Realm this series belongs to.
        recent_mean:    Mean price_gold in the recent window.
        baseline_mean:  Mean price_gold in the historical baseline window.
        baseline_std:   Std of price_gold in the historical baseline window.
        z_mean_shift:   (recent_mean - baseline_mean) / baseline_std.
                        None if baseline_std == 0 or baseline is empty.
        recent_n:       Observation count in the recent window.
        baseline_n:     Observation count in the baseline window.
        is_drifted:     True if |z_mean_shift| > threshold.
    """

    archetype_id:   int
    realm_slug:     str
    recent_mean:    Optional[float]
    baseline_mean:  Optional[float]
    baseline_std:   Optional[float]
    z_mean_shift:   Optional[float]
    recent_n:       int
    baseline_n:     int
    is_drifted:     bool


# ── Data drift ────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class DataDriftReport:
    """Result of a data drift check across all tracked series in a realm.

    Attributes:
        realm_slug:        Realm that was checked.
        checked_at:        ISO-8601 UTC timestamp of the check.
        window_hours:      Size of the "recent" observation window in hours.
        baseline_days:     Length of the historical baseline window in days.
        n_series_checked:  Total series with enough data to evaluate.
        n_series_drifted:  Series flagged as drifted.
        drift_fraction:    n_series_drifted / n_series_checked (0.0 if no data).
        drift_level:       Aggregate drift severity for the realm.
        series_stats:      Per-series statistics (may be large; serialise selectively).
    """

    realm_slug:       str
    checked_at:       str
    window_hours:     int
    baseline_days:    int
    n_series_checked: int
    n_series_drifted: int
    drift_fraction:   float
    drift_level:      DriftLevel
    series_stats:     list[SeriesDriftStats]


# ── Error drift ───────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ErrorDriftReport:
    """Result of an error drift check (forecast residual degradation).

    Attributes:
        realm_slug:     Realm that was checked.
        checked_at:     ISO-8601 UTC timestamp of the check.
        horizon_days:   Forecast horizon evaluated (1, 7, or 28).
        n_evaluated:    Number of forecast-vs-actual pairs evaluated.
        live_mae:       Mean absolute error over the recent window (gold).
        baseline_mae:   Reference MAE from most recent backtest run (gold).
        mae_ratio:      live_mae / baseline_mae; None if either is missing.
        drift_level:    Error drift severity.
    """

    realm_slug:   str
    checked_at:   str
    horizon_days: int
    n_evaluated:  int
    live_mae:     Optional[float]
    baseline_mae: Optional[float]
    mae_ratio:    Optional[float]
    drift_level:  DriftLevel


# ── Event shock ───────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class EventShockReport:
    """Event shock detection result.

    This is a detection hook, not attribution.  It signals that we are in or
    near a major event window so that CIs can be widened preemptively.

    Attributes:
        checked_at:      ISO-8601 UTC timestamp of the check.
        active_events:   Events currently active (start_date <= today <= end_date).
        upcoming_events: Events starting within shock_window_days.
        shock_active:    True if any MAJOR/CRITICAL/CATASTROPHIC event is active
                         or imminent.
    """

    checked_at:      str
    active_events:   list[dict]
    upcoming_events: list[dict]
    shock_active:    bool


# ── Composite result ──────────────────────────────────────────────────────────

@dataclass(frozen=True)
class DriftCheckResult:
    """Composite drift check result for one realm.

    Attributes:
        realm_slug:            Realm that was checked.
        checked_at:            ISO-8601 UTC timestamp.
        data_drift:            Data distribution drift report.
        error_drift:           Forecast residual drift report.
        event_shock:           Event shock detection report.
        overall_drift_level:   Highest severity across all three checks.
        uncertainty_multiplier: Applied by adaptive policy (1.0 = no change).
        retrain_recommended:   True if drift warrants retraining.
    """

    realm_slug:            str
    checked_at:            str
    data_drift:            DataDriftReport
    error_drift:           ErrorDriftReport
    event_shock:           EventShockReport
    overall_drift_level:   DriftLevel
    uncertainty_multiplier: float
    retrain_recommended:   bool


# ── Classification helpers (module-level for testability) ─────────────────────

def _classify_data_drift(
    drift_fraction: float,
    thresholds: tuple[float, float, float, float] = (0.10, 0.25, 0.40, 0.60),
) -> DriftLevel:
    """Map drift_fraction to a DriftLevel.

    Args:
        drift_fraction: Fraction of series flagged as drifted (0.0–1.0).
        thresholds:     (low, medium, high, critical) boundary fractions.

    Returns:
        DriftLevel corresponding to drift_fraction.
    """
    low, medium, high, critical = thresholds
    if drift_fraction >= critical:
        return DriftLevel.CRITICAL
    if drift_fraction >= high:
        return DriftLevel.HIGH
    if drift_fraction >= medium:
        return DriftLevel.MEDIUM
    if drift_fraction >= low:
        return DriftLevel.LOW
    return DriftLevel.NONE


def _classify_error_drift(
    mae_ratio: Optional[float],
    thresholds: tuple[float, float, float, float] = _DEFAULT_MAE_THRESHOLDS,
) -> DriftLevel:
    """Map mae_ratio to a DriftLevel.

    Args:
        mae_ratio:  live_mae / baseline_mae.  None → NONE (can't detect).
        thresholds: (low, medium, high, critical) boundary ratios.

    Returns:
        DriftLevel corresponding to mae_ratio.
    """
    if mae_ratio is None:
        return DriftLevel.NONE
    low, medium, high, critical = thresholds
    if mae_ratio >= critical:
        return DriftLevel.CRITICAL
    if mae_ratio >= high:
        return DriftLevel.HIGH
    if mae_ratio >= medium:
        return DriftLevel.MEDIUM
    if mae_ratio >= low:
        return DriftLevel.LOW
    return DriftLevel.NONE


def _overall_drift_level(
    data_level: DriftLevel,
    error_level: DriftLevel,
    shock_active: bool,
) -> DriftLevel:
    """Take the maximum across data drift, error drift, and event shock.

    Event shock on its own bumps level by at most one tier (to ensure CIs
    widen proactively during events even when error drift hasn't materialized).

    Args:
        data_level:   Data distribution drift level.
        error_level:  Forecast residual drift level.
        shock_active: Whether a major event shock is detected.

    Returns:
        Combined drift level.
    """
    _order = list(DriftLevel)
    base_idx = max(_order.index(data_level), _order.index(error_level))
    if shock_active:
        base_idx = min(base_idx + 1, len(_order) - 1)
    return _order[base_idx]


# ── Main checker ──────────────────────────────────────────────────────────────

class DriftChecker:
    """Run all three drift checks for a given realm.

    Args:
        conn:                An open sqlite3.Connection.
        drift_window_hours:  Recent observation window for data drift (default 25h).
        baseline_days:       Historical baseline length (default 30d).
        z_threshold:         Z-score threshold for flagging a series (default 2.0).
        error_window_days:   Look-back for forecast residual drift (default 7d).
        mae_thresholds:      (low, medium, high, critical) MAE ratio thresholds.
        shock_window_days:   Days ahead to check for upcoming events (default 7d).
    """

    def __init__(
        self,
        conn: sqlite3.Connection,
        drift_window_hours: int   = _DEFAULT_DRIFT_WINDOW_HOURS,
        baseline_days: int        = _DEFAULT_BASELINE_DAYS,
        z_threshold: float        = _DEFAULT_Z_THRESHOLD,
        error_window_days: int    = _DEFAULT_ERROR_WINDOW_DAYS,
        mae_thresholds: tuple[float, float, float, float] = _DEFAULT_MAE_THRESHOLDS,
        shock_window_days: int    = _DEFAULT_SHOCK_WINDOW_DAYS,
    ) -> None:
        self._conn             = conn
        self._drift_window_h   = drift_window_hours
        self._baseline_days    = baseline_days
        self._z_threshold      = z_threshold
        self._error_window_days = error_window_days
        self._mae_thresholds   = mae_thresholds
        self._shock_window_days = shock_window_days

    def run_all(self, realm_slug: str) -> DriftCheckResult:
        """Run all three drift checks and return a composite result.

        Args:
            realm_slug: Realm to check.

        Returns:
            DriftCheckResult with data drift, error drift, event shock,
            overall level, uncertainty multiplier, and retrain flag.
        """
        now_str = _utc_now_iso()

        data_report  = self.check_data_drift(realm_slug)
        error_report = self.check_error_drift(realm_slug, horizon_days=1)
        shock_report = self.check_event_shocks()

        overall = _overall_drift_level(
            data_report.drift_level,
            error_report.drift_level,
            shock_report.shock_active,
        )

        from wow_forecaster.monitoring.adaptive import evaluate_policy
        policy = evaluate_policy(overall)

        return DriftCheckResult(
            realm_slug=realm_slug,
            checked_at=now_str,
            data_drift=data_report,
            error_drift=error_report,
            event_shock=shock_report,
            overall_drift_level=overall,
            uncertainty_multiplier=policy.uncertainty_multiplier,
            retrain_recommended=policy.retrain_recommended,
        )

    # ── Data drift ────────────────────────────────────────────────────────────

    def check_data_drift(self, realm_slug: str) -> DataDriftReport:
        """Compare recent vs baseline price distributions per archetype series.

        Queries market_observations_normalized.  Uses price_gold as the
        primary metric.

        Args:
            realm_slug: Realm to check.

        Returns:
            DataDriftReport.
        """
        now_str = _utc_now_iso()

        # Cutoffs
        cutoff_recent   = _days_ago(self._drift_window_h / 24.0)
        cutoff_baseline = _days_ago(self._baseline_days)

        # ── Baseline stats per series ─────────────────────────────────────────
        baseline_q = """
            SELECT archetype_id,
                   AVG(price_gold)                     AS mean_price,
                   SUM((price_gold - sub.avg_p) * (price_gold - sub.avg_p)) AS ss,
                   COUNT(*)                            AS n
            FROM market_observations_normalized
            JOIN (
                SELECT archetype_id AS a_id, AVG(price_gold) AS avg_p
                FROM market_observations_normalized
                WHERE realm_slug = ?
                  AND date(observed_at) >= ?
                  AND date(observed_at) < ?
                  AND is_outlier = 0
                  AND archetype_id IS NOT NULL
                GROUP BY archetype_id
            ) sub ON archetype_id = sub.a_id
            WHERE realm_slug = ?
              AND date(observed_at) >= ?
              AND date(observed_at) < ?
              AND is_outlier = 0
              AND archetype_id IS NOT NULL
            GROUP BY archetype_id;
        """
        baseline_cutoff_str = cutoff_baseline.isoformat()
        recent_cutoff_str   = cutoff_recent.isoformat()

        try:
            baseline_rows = self._conn.execute(
                baseline_q,
                (realm_slug, baseline_cutoff_str, recent_cutoff_str,
                 realm_slug, baseline_cutoff_str, recent_cutoff_str),
            ).fetchall()
        except Exception as exc:
            logger.warning("Data drift baseline query failed: %s", exc)
            baseline_rows = []

        baseline_by_arch: dict[int, dict] = {}
        for row in baseline_rows:
            arch_id = row["archetype_id"]
            n       = row["n"] or 0
            mean_p  = row["mean_price"]
            # std = sqrt(SS / n) (population std for consistency)
            std_p   = math.sqrt(row["ss"] / n) if (n > 1 and row["ss"] is not None) else 0.0
            baseline_by_arch[arch_id] = {
                "mean": mean_p,
                "std":  std_p,
                "n":    n,
            }

        # ── Recent stats per series ───────────────────────────────────────────
        recent_q = """
            SELECT archetype_id,
                   AVG(price_gold) AS mean_price,
                   COUNT(*)        AS n
            FROM market_observations_normalized
            WHERE realm_slug = ?
              AND date(observed_at) >= ?
              AND is_outlier = 0
              AND archetype_id IS NOT NULL
            GROUP BY archetype_id;
        """
        try:
            recent_rows = self._conn.execute(
                recent_q,
                (realm_slug, recent_cutoff_str),
            ).fetchall()
        except Exception as exc:
            logger.warning("Data drift recent query failed: %s", exc)
            recent_rows = []

        recent_by_arch: dict[int, dict] = {}
        for row in recent_rows:
            recent_by_arch[row["archetype_id"]] = {
                "mean": row["mean_price"],
                "n":    row["n"] or 0,
            }

        # ── Compute per-series stats ──────────────────────────────────────────
        all_series: set[int] = set(baseline_by_arch) | set(recent_by_arch)
        series_stats: list[SeriesDriftStats] = []

        for arch_id in sorted(all_series):
            b = baseline_by_arch.get(arch_id, {})
            r = recent_by_arch.get(arch_id, {})

            b_mean = b.get("mean")
            b_std  = b.get("std", 0.0)
            b_n    = b.get("n", 0)
            r_mean = r.get("mean")
            r_n    = r.get("n", 0)

            # Compute z-score of mean shift
            z: Optional[float] = None
            if b_mean is not None and r_mean is not None and b_std and b_std > 1e-6:
                z = (r_mean - b_mean) / b_std

            is_drifted = z is not None and abs(z) > self._z_threshold

            series_stats.append(SeriesDriftStats(
                archetype_id=arch_id,
                realm_slug=realm_slug,
                recent_mean=r_mean,
                baseline_mean=b_mean,
                baseline_std=b_std if b_n > 0 else None,
                z_mean_shift=round(z, 4) if z is not None else None,
                recent_n=r_n,
                baseline_n=b_n,
                is_drifted=is_drifted,
            ))

        n_checked = len([s for s in series_stats if s.baseline_n > 0])
        n_drifted = sum(1 for s in series_stats if s.is_drifted)
        drift_frac = n_drifted / n_checked if n_checked > 0 else 0.0
        drift_level = _classify_data_drift(drift_frac)

        logger.info(
            "Data drift | realm=%s | series=%d checked, %d drifted (%.0f%%) | level=%s",
            realm_slug, n_checked, n_drifted, drift_frac * 100, drift_level.value,
        )

        return DataDriftReport(
            realm_slug=realm_slug,
            checked_at=now_str,
            window_hours=self._drift_window_h,
            baseline_days=self._baseline_days,
            n_series_checked=n_checked,
            n_series_drifted=n_drifted,
            drift_fraction=round(drift_frac, 4),
            drift_level=drift_level,
            series_stats=series_stats,
        )

    # ── Error drift ───────────────────────────────────────────────────────────

    def check_error_drift(
        self, realm_slug: str, horizon_days: int = 1
    ) -> ErrorDriftReport:
        """Compare recent live MAE vs baseline MAE from backtests.

        Args:
            realm_slug:   Realm to check.
            horizon_days: Forecast horizon to evaluate.

        Returns:
            ErrorDriftReport.
        """
        now_str = _utc_now_iso()
        horizon_tag = f"{horizon_days}d"
        window_cutoff = _days_ago(self._error_window_days).isoformat()

        # ── Live MAE: forecast vs actual ──────────────────────────────────────
        # Join forecast_outputs with normalized obs where target_date has passed
        live_q = """
            SELECT
                f.forecast_id,
                ABS(f.predicted_price_gold - AVG(n.price_gold)) AS abs_error
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
            live_rows = self._conn.execute(
                live_q, (realm_slug, horizon_tag, window_cutoff)
            ).fetchall()
        except Exception as exc:
            logger.warning("Error drift live query failed: %s", exc)
            live_rows = []

        n_evaluated = len(live_rows)
        live_mae: Optional[float] = None
        if n_evaluated > 0:
            errors = [r["abs_error"] for r in live_rows if r["abs_error"] is not None]
            if errors:
                live_mae = sum(errors) / len(errors)

        # ── Baseline MAE: from most recent backtest run ────────────────────────
        baseline_mae: Optional[float] = None
        try:
            bt_row = self._conn.execute(
                """
                SELECT backtest_run_id FROM backtest_runs
                WHERE realm_slug = ?
                ORDER BY backtest_run_id DESC LIMIT 1;
                """,
                (realm_slug,),
            ).fetchone()

            if bt_row is not None:
                bt_run_id = bt_row["backtest_run_id"]
                metrics_row = self._conn.execute(
                    """
                    SELECT AVG(abs_error) AS baseline_mae
                    FROM backtest_fold_results
                    WHERE backtest_run_id = ?
                      AND horizon_days = ?
                      AND actual_price IS NOT NULL;
                    """,
                    (bt_run_id, horizon_days),
                ).fetchone()
                if metrics_row and metrics_row["baseline_mae"] is not None:
                    baseline_mae = metrics_row["baseline_mae"]
        except Exception as exc:
            logger.warning("Error drift baseline query failed: %s", exc)

        mae_ratio: Optional[float] = None
        if live_mae is not None and baseline_mae is not None and baseline_mae > 1e-6:
            mae_ratio = round(live_mae / baseline_mae, 4)

        drift_level = _classify_error_drift(mae_ratio, self._mae_thresholds)

        logger.info(
            "Error drift | realm=%s | h=%dd | n=%d | live_mae=%s | baseline_mae=%s | ratio=%s | level=%s",
            realm_slug, horizon_days, n_evaluated,
            f"{live_mae:.2f}g" if live_mae else "N/A",
            f"{baseline_mae:.2f}g" if baseline_mae else "N/A",
            f"{mae_ratio:.2f}x" if mae_ratio else "N/A",
            drift_level.value,
        )

        return ErrorDriftReport(
            realm_slug=realm_slug,
            checked_at=now_str,
            horizon_days=horizon_days,
            n_evaluated=n_evaluated,
            live_mae=round(live_mae, 4) if live_mae is not None else None,
            baseline_mae=round(baseline_mae, 4) if baseline_mae is not None else None,
            mae_ratio=mae_ratio,
            drift_level=drift_level,
        )

    # ── Event shock ───────────────────────────────────────────────────────────

    def check_event_shocks(self) -> EventShockReport:
        """Detect active or imminent major events from the wow_events table.

        This is a *detection hook* — it identifies event proximity so that
        adaptive policy can widen CIs proactively.  Actual price attribution
        to events requires real price data.

        Returns:
            EventShockReport.
        """
        now_str  = _utc_now_iso()
        today    = date.today()
        horizon  = today + timedelta(days=self._shock_window_days)

        try:
            # Active events: start_date <= today and (end_date IS NULL or end_date >= today)
            active_rows = self._conn.execute(
                """
                SELECT slug, display_name, event_type, severity, start_date, end_date
                FROM wow_events
                WHERE date(start_date) <= ?
                  AND (end_date IS NULL OR date(end_date) >= ?)
                ORDER BY start_date;
                """,
                (today.isoformat(), today.isoformat()),
            ).fetchall()

            # Upcoming events within shock window
            upcoming_rows = self._conn.execute(
                """
                SELECT slug, display_name, event_type, severity, start_date, end_date
                FROM wow_events
                WHERE date(start_date) > ?
                  AND date(start_date) <= ?
                ORDER BY start_date;
                """,
                (today.isoformat(), horizon.isoformat()),
            ).fetchall()
        except Exception as exc:
            logger.warning("Event shock query failed: %s", exc)
            active_rows, upcoming_rows = [], []

        def _to_dict(row: sqlite3.Row) -> dict:
            return {
                "slug": row["slug"],
                "display_name": row["display_name"],
                "event_type": row["event_type"],
                "severity": row["severity"],
                "start_date": row["start_date"],
                "end_date": row["end_date"],
            }

        active   = [_to_dict(r) for r in active_rows]
        upcoming = [_to_dict(r) for r in upcoming_rows]

        shock_active = any(
            e["severity"] in _SHOCK_SEVERITIES
            for e in active + upcoming
        )

        logger.info(
            "Event shock | active=%d, upcoming=%d | shock_active=%s",
            len(active), len(upcoming), shock_active,
        )

        return EventShockReport(
            checked_at=now_str,
            active_events=active,
            upcoming_events=upcoming,
            shock_active=shock_active,
        )


# ── Internal helpers ──────────────────────────────────────────────────────────

def _utc_now_iso() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _days_ago(n: float) -> date:
    """Return the date n days ago (fractional days truncated)."""
    return date.today() - timedelta(days=int(n))
