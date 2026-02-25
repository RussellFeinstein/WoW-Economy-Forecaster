"""
Backtest result reporting: CSV files, JSON manifest, and SQLite persistence.

Output layout (one backtest run, one horizon):
  outputs/backtest/{realm_slug}_{backtest_start}_{backtest_end}/
    horizon_{N}d/
      summary.csv           — aggregate metrics by (model, horizon)
      by_category.csv       — aggregate metrics by category
      per_prediction.csv    — one row per PredictionRecord (full raw data)
      manifest.json         — run config, fold count, dataset version

SQLite tables (written by this module):
  backtest_runs           — one row per backtest invocation
  backtest_fold_results   — one row per PredictionRecord

These outputs enable:
  - Quick review in a spreadsheet from the CSV files.
  - Long-term storage and cross-run comparison in SQLite.
  - The `report-backtest` CLI command to query and display results.
"""

from __future__ import annotations

import csv
import json
import logging
import sqlite3
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from wow_forecaster.backtest.metrics import BacktestMetrics, PredictionRecord, compute_metrics
from wow_forecaster.backtest.slices import slice_by_category, slice_by_model_and_horizon
from wow_forecaster.backtest.splits import BacktestFold

log = logging.getLogger(__name__)


# ── SQLite persistence ─────────────────────────────────────────────────────────

def persist_backtest_run(
    conn: sqlite3.Connection,
    run_id: int | None,
    realm_slug: str,
    backtest_start: date,
    backtest_end: date,
    window_days: int,
    step_days: int,
    fold_count: int,
    model_names: list[str],
    config_snapshot: dict[str, Any],
) -> int:
    """Insert a backtest_runs row and return its backtest_run_id."""
    cursor = conn.execute(
        """
        INSERT INTO backtest_runs
            (run_id, realm_slug, backtest_start, backtest_end,
             window_days, step_days, fold_count, models, config_snapshot)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            run_id,
            realm_slug,
            backtest_start.isoformat(),
            backtest_end.isoformat(),
            window_days,
            step_days,
            fold_count,
            json.dumps(model_names),
            json.dumps(config_snapshot, default=str),
        ),
    )
    conn.commit()
    return cursor.lastrowid  # type: ignore[return-value]


def persist_prediction_records(
    conn: sqlite3.Connection,
    backtest_run_id: int,
    records: list[PredictionRecord],
) -> int:
    """Bulk-insert PredictionRecords into backtest_fold_results.

    Computes per-record error columns inline (abs_error, pct_error,
    direction_actual/predicted/correct) rather than storing them in
    PredictionRecord — keeps the model pure.

    Returns the number of rows inserted.
    """
    rows_to_insert: list[tuple] = []
    for r in records:
        if r.actual_price is not None and r.predicted_price is not None:
            abs_err = abs(r.actual_price - r.predicted_price)
            pct_err = abs_err / max(r.actual_price, 0.01)
        else:
            abs_err = None
            pct_err = None

        if (
            r.last_known_price is not None
            and r.actual_price is not None
            and r.predicted_price is not None
            and r.actual_price != r.last_known_price
        ):
            dir_actual    = 1 if r.actual_price    > r.last_known_price else -1
            dir_predicted = 1 if r.predicted_price > r.last_known_price else -1
            dir_correct   = 1 if dir_actual == dir_predicted else 0
        else:
            dir_actual = dir_predicted = dir_correct = None

        rows_to_insert.append((
            backtest_run_id,
            r.fold_index,
            r.train_end.isoformat(),
            r.test_date.isoformat(),
            r.horizon_days,
            r.archetype_id,
            r.realm_slug,
            r.category_tag,
            r.model_name,
            r.actual_price,
            r.predicted_price,
            abs_err,
            pct_err,
            dir_actual,
            dir_predicted,
            dir_correct,
            1 if r.is_event_window else 0,
        ))

    conn.executemany(
        """
        INSERT INTO backtest_fold_results
            (backtest_run_id, fold_index, train_end, test_date, horizon_days,
             archetype_id, realm_slug, category_tag, model_name,
             actual_price, predicted_price, abs_error, pct_error,
             direction_actual, direction_predicted, direction_correct,
             is_event_window)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows_to_insert,
    )
    conn.commit()
    return len(rows_to_insert)


# ── CSV output ─────────────────────────────────────────────────────────────────

def write_summary_csv(
    metrics_by_model_horizon: dict[tuple[str, int], BacktestMetrics],
    path: Path,
) -> None:
    """Write aggregate (model × horizon) metrics as CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model_name", "horizon_days",
        "n_predictions", "n_evaluated",
        "mae", "rmse", "mape",
        "directional_accuracy", "n_directional",
        "mean_actual", "mean_predicted",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for (model_name, horizon_days), m in sorted(metrics_by_model_horizon.items()):
            writer.writerow({
                "model_name": model_name,
                "horizon_days": horizon_days,
                "n_predictions": m.n_predictions,
                "n_evaluated": m.n_evaluated,
                "mae":  _fmt(m.mae),
                "rmse": _fmt(m.rmse),
                "mape": _fmt(m.mape),
                "directional_accuracy": _fmt(m.directional_accuracy),
                "n_directional": m.n_directional,
                "mean_actual":    _fmt(m.mean_actual),
                "mean_predicted": _fmt(m.mean_predicted),
            })
    log.info("Summary CSV written: %s", path)


def write_by_category_csv(
    category_metrics: dict[str, BacktestMetrics],
    path: Path,
) -> None:
    """Write category-sliced aggregate metrics as CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "category", "n_evaluated",
        "mae", "rmse", "mape", "directional_accuracy",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for cat, m in sorted(category_metrics.items()):
            writer.writerow({
                "category":            cat,
                "n_evaluated":         m.n_evaluated,
                "mae":                 _fmt(m.mae),
                "rmse":                _fmt(m.rmse),
                "mape":                _fmt(m.mape),
                "directional_accuracy": _fmt(m.directional_accuracy),
            })
    log.info("By-category CSV written: %s", path)


def write_per_prediction_csv(records: list[PredictionRecord], path: Path) -> None:
    """Write all raw PredictionRecords as CSV (for detailed inspection)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "fold_index", "archetype_id", "realm_slug", "category_tag",
        "model_name", "train_end", "test_date", "horizon_days",
        "actual_price", "predicted_price", "last_known_price", "is_event_window",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            writer.writerow({
                "fold_index":       r.fold_index,
                "archetype_id":     r.archetype_id,
                "realm_slug":       r.realm_slug,
                "category_tag":     r.category_tag,
                "model_name":       r.model_name,
                "train_end":        r.train_end.isoformat(),
                "test_date":        r.test_date.isoformat(),
                "horizon_days":     r.horizon_days,
                "actual_price":     r.actual_price,
                "predicted_price":  r.predicted_price,
                "last_known_price": r.last_known_price,
                "is_event_window":  int(r.is_event_window),
            })
    log.info("Per-prediction CSV written: %s (%d rows)", path, len(records))


# ── JSON manifest ──────────────────────────────────────────────────────────────

def build_backtest_manifest(
    realm_slug: str,
    backtest_start: date,
    backtest_end: date,
    run_slug: str,
    folds: list[BacktestFold],
    model_names: list[str],
    records: list[PredictionRecord],
    output_dir: Path,
    config_snapshot: dict[str, Any],
) -> dict[str, Any]:
    """Build a JSON manifest summarising this backtest run."""
    overall = compute_metrics(records)
    return {
        "schema_version": "1.0",
        "built_at":   datetime.now(tz=timezone.utc).isoformat(),
        "run_slug":   run_slug,
        "realm_slug": realm_slug,
        "date_range": {
            "backtest_start": backtest_start.isoformat(),
            "backtest_end":   backtest_end.isoformat(),
        },
        "fold_count":   len(folds),
        "model_names":  model_names,
        "evaluation_summary": {
            "n_predictions":       overall.n_predictions,
            "n_evaluated":         overall.n_evaluated,
            "mae":                 overall.mae,
            "rmse":                overall.rmse,
            "mape":                overall.mape,
            "directional_accuracy": overall.directional_accuracy,
        },
        "output_files": {
            "summary_csv":        str(output_dir / "summary.csv"),
            "by_category_csv":    str(output_dir / "by_category.csv"),
            "per_prediction_csv": str(output_dir / "per_prediction.csv"),
        },
        "config_snapshot": config_snapshot,
    }


def write_manifest(manifest: dict[str, Any], path: Path) -> None:
    """Write the manifest dict as pretty-printed JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, default=str)
    log.info("Backtest manifest written: %s", path)


def make_output_dir(base_dir: str, realm_slug: str, start: date, end: date) -> Path:
    """Build the deterministic output directory path for one backtest run."""
    slug = realm_slug.replace("-", "_")
    return Path(base_dir) / "backtest" / f"{slug}_{start}_{end}"


def _fmt(v: float | None) -> str:
    """Format float to 4 decimal places, or empty string for None."""
    if v is None:
        return ""
    return f"{v:.4f}"
