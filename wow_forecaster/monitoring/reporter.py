"""
Monitoring report writer for the WoW Economy Forecaster.

Writes drift status, model health, and provenance summaries to
``data/outputs/monitoring/`` as JSON files.

File layout:
    data/outputs/monitoring/
        drift_status_{realm}_{date}.json
        model_health_{realm}_{date}.json
        provenance_{realm}_{date}.json

Each file is a self-contained JSON document.  Old files from previous
runs are NOT automatically deleted — they serve as an audit trail.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from wow_forecaster.monitoring.drift import DriftCheckResult
    from wow_forecaster.monitoring.health import ModelHealthSummary
    from wow_forecaster.monitoring.provenance import ProvenanceSummary

logger = logging.getLogger(__name__)


def write_drift_report(
    result: "DriftCheckResult",
    output_dir: Path,
) -> Path:
    """Serialise a DriftCheckResult to a dated JSON file.

    Args:
        result:     DriftCheckResult from DriftChecker.run_all().
        output_dir: Directory to write the file into.

    Returns:
        Path of the written file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    today = date.today().isoformat()
    filename = f"drift_status_{result.realm_slug}_{today}.json"
    path = output_dir / filename

    payload = {
        "realm_slug":             result.realm_slug,
        "checked_at":             result.checked_at,
        "overall_drift_level":    result.overall_drift_level.value,
        "uncertainty_multiplier": result.uncertainty_multiplier,
        "retrain_recommended":    result.retrain_recommended,
        "data_drift": {
            "drift_level":       result.data_drift.drift_level.value,
            "drift_fraction":    result.data_drift.drift_fraction,
            "n_series_checked":  result.data_drift.n_series_checked,
            "n_series_drifted":  result.data_drift.n_series_drifted,
            "window_hours":      result.data_drift.window_hours,
            "baseline_days":     result.data_drift.baseline_days,
        },
        "error_drift": {
            "drift_level":  result.error_drift.drift_level.value,
            "horizon_days": result.error_drift.horizon_days,
            "n_evaluated":  result.error_drift.n_evaluated,
            "live_mae":     result.error_drift.live_mae,
            "baseline_mae": result.error_drift.baseline_mae,
            "mae_ratio":    result.error_drift.mae_ratio,
        },
        "event_shock": {
            "shock_active":    result.event_shock.shock_active,
            "active_count":    len(result.event_shock.active_events),
            "upcoming_count":  len(result.event_shock.upcoming_events),
            "active_events":   result.event_shock.active_events,
            "upcoming_events": result.event_shock.upcoming_events,
        },
    }

    _write_json(path, payload)
    return path


def write_health_report(
    summaries: list["ModelHealthSummary"],
    output_dir: Path,
    realm_slug: str,
) -> Path:
    """Serialise model health summaries for all horizons to JSON.

    Args:
        summaries:  List of ModelHealthSummary (one per horizon).
        output_dir: Directory to write the file into.
        realm_slug: Realm label used in the filename.

    Returns:
        Path of the written file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    today    = date.today().isoformat()
    filename = f"model_health_{realm_slug}_{today}.json"
    path     = output_dir / filename

    payload = {
        "realm_slug": realm_slug,
        "checked_at": summaries[0].checked_at if summaries else "",
        "horizons": [
            {
                "horizon_days":     s.horizon_days,
                "health_status":    s.health_status,
                "n_evaluated":      s.n_evaluated,
                "live_mae":         s.live_mae,
                "baseline_mae":     s.baseline_mae,
                "mae_ratio":        s.mae_ratio,
                "live_dir_acc":     s.live_dir_acc,
                "baseline_dir_acc": s.baseline_dir_acc,
            }
            for s in summaries
        ],
    }

    _write_json(path, payload)
    return path


def write_provenance_report(
    summary: "ProvenanceSummary",
    output_dir: Path,
) -> Path:
    """Serialise a ProvenanceSummary to a dated JSON file.

    Args:
        summary:    ProvenanceSummary from build_provenance_summary().
        output_dir: Directory to write the file into.

    Returns:
        Path of the written file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    today    = date.today().isoformat()
    filename = f"provenance_{summary.realm_slug}_{today}.json"
    path     = output_dir / filename

    payload = {
        "realm_slug":      summary.realm_slug,
        "checked_at":      summary.checked_at,
        "freshness_hours": summary.freshness_hours,
        "is_fresh":        summary.is_fresh,
        "sources": [
            {
                "source":              s.source,
                "last_snapshot_at":    s.last_snapshot_at,
                "snapshot_count_24h":  s.snapshot_count_24h,
                "total_records_24h":   s.total_records_24h,
                "success_rate_24h":    s.success_rate_24h,
                "is_stale":            s.is_stale,
            }
            for s in summary.sources
        ],
    }

    _write_json(path, payload)
    return path


def persist_drift_to_db(
    conn,
    run_id: int,
    result: "DriftCheckResult",
) -> None:
    """Persist a DriftCheckResult to the drift_check_results table.

    Args:
        conn:    Open SQLite connection.
        run_id:  run_metadata.run_id for this orchestration run.
        result:  DriftCheckResult to persist.
    """
    import json as _json

    drift_details = _json.dumps({
        "data_drift_fraction":  result.data_drift.drift_fraction,
        "n_series_checked":     result.data_drift.n_series_checked,
        "n_series_drifted":     result.data_drift.n_series_drifted,
        "error_mae_ratio":      result.error_drift.mae_ratio,
        "event_shock_count":    len(result.event_shock.active_events),
    })

    try:
        conn.execute(
            """
            INSERT INTO drift_check_results
                (run_id, realm_slug, checked_at,
                 data_drift_level, error_drift_level, event_shock_active,
                 drift_details, uncertainty_mult, retrain_recommended)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            (
                run_id,
                result.realm_slug,
                result.checked_at,
                result.data_drift.drift_level.value,
                result.error_drift.drift_level.value,
                1 if result.event_shock.shock_active else 0,
                drift_details,
                result.uncertainty_multiplier,
                1 if result.retrain_recommended else 0,
            ),
        )
        conn.commit()
    except Exception as exc:
        logger.error("Failed to persist drift result for realm=%s: %s", result.realm_slug, exc)


def persist_health_to_db(
    conn,
    run_id: int,
    summary: "ModelHealthSummary",
) -> None:
    """Persist a ModelHealthSummary to the model_health_snapshots table.

    Args:
        conn:    Open SQLite connection.
        run_id:  run_metadata.run_id for this orchestration run.
        summary: ModelHealthSummary to persist.
    """
    try:
        conn.execute(
            """
            INSERT INTO model_health_snapshots
                (run_id, realm_slug, horizon_days, n_evaluated,
                 live_mae, baseline_mae, mae_ratio,
                 live_dir_acc, baseline_dir_acc, health_status, checked_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            (
                run_id,
                summary.realm_slug,
                summary.horizon_days,
                summary.n_evaluated,
                summary.live_mae,
                summary.baseline_mae,
                summary.mae_ratio,
                summary.live_dir_acc,
                summary.baseline_dir_acc,
                summary.health_status,
                summary.checked_at,
            ),
        )
        conn.commit()
    except Exception as exc:
        logger.error(
            "Failed to persist health summary for realm=%s h=%dd: %s",
            summary.realm_slug, summary.horizon_days, exc,
        )


def get_latest_uncertainty_multiplier(
    conn,
    realm_slug: str,
) -> float:
    """Read the most recent uncertainty_multiplier from drift_check_results.

    Returns 1.0 (no adjustment) if no drift check results exist for the realm.

    Args:
        conn:       Open SQLite connection.
        realm_slug: Realm to look up.

    Returns:
        uncertainty_multiplier as a float.
    """
    try:
        row = conn.execute(
            """
            SELECT uncertainty_mult
            FROM drift_check_results
            WHERE realm_slug = ?
            ORDER BY drift_id DESC LIMIT 1;
            """,
            (realm_slug,),
        ).fetchone()
        if row is not None and row["uncertainty_mult"] is not None:
            return float(row["uncertainty_mult"])
    except Exception as exc:
        logger.warning(
            "Could not read uncertainty_mult for realm=%s: %s", realm_slug, exc
        )
    return 1.0


# ── Internal helpers ──────────────────────────────────────────────────────────

def _write_json(path: Path, payload: dict) -> None:
    """Write a JSON payload to path, logging success or failure."""
    try:
        path.write_text(
            json.dumps(payload, indent=2, default=str), encoding="utf-8"
        )
        logger.info("Wrote monitoring report: %s", path)
    except OSError as exc:
        logger.error("Failed to write monitoring report %s: %s", path, exc)
