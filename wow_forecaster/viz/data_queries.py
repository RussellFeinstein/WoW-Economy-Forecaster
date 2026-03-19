"""
Data query layer for the visualization module.

Pure functions that fetch data from SQLite / CSV / JSON / model artifacts
and return pandas DataFrames. This is the viz layer's only interface to
persistent storage — chart modules never open DB connections directly.

All functions return empty DataFrames (not None) when no data exists,
so callers can safely check ``df.empty`` without null guards.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


# ── Forecast data ─────────────────────────────────────────────────────────────


def fetch_forecast_data(
    db_path: str | Path,
    realm: str,
    horizon: str | None = None,
) -> pd.DataFrame:
    """Fetch forecast outputs from the DB.

    Returns columns: archetype_id, item_id, realm_slug, forecast_horizon,
    target_date, predicted_price_gold, confidence_lower, confidence_upper,
    ci_quality, model_slug, created_at.
    """
    sql = """
        SELECT fo.archetype_id, fo.item_id, fo.realm_slug,
               fo.forecast_horizon, fo.target_date,
               fo.predicted_price_gold, fo.confidence_lower,
               fo.confidence_upper, fo.ci_quality, fo.model_slug,
               fo.created_at
        FROM   forecast_outputs fo
        JOIN   run_metadata rm ON fo.run_id = rm.run_id
        WHERE  fo.realm_slug = ?
    """
    params: list = [realm]
    if horizon:
        sql += " AND fo.forecast_horizon = ?"
        params.append(horizon)
    sql += " ORDER BY fo.created_at DESC, fo.archetype_id"
    return _query_db(db_path, sql, params)


# ── Historical prices ─────────────────────────────────────────────────────────


def fetch_historical_prices(
    db_path: str | Path,
    realm: str,
    archetype_id: int,
    days: int = 90,
) -> pd.DataFrame:
    """Fetch daily average prices for one archetype.

    Returns columns: obs_date, avg_price_gold, min_price_gold,
    max_price_gold, obs_count.
    """
    sql = """
        SELECT date(n.observed_at) AS obs_date,
               AVG(n.price_gold)   AS avg_price_gold,
               MIN(n.price_gold)   AS min_price_gold,
               MAX(n.price_gold)   AS max_price_gold,
               COUNT(*)            AS obs_count
        FROM   market_observations_normalized n
        JOIN   items i ON n.item_id = i.item_id
        WHERE  i.archetype_id  = ?
          AND  n.realm_slug    = ?
          AND  n.is_outlier    = 0
          AND  date(n.observed_at) >= date('now', ? || ' days')
        GROUP  BY obs_date
        ORDER  BY obs_date
    """
    return _query_db(db_path, sql, [archetype_id, realm, f"-{days}"])


# ── Backtest predictions ──────────────────────────────────────────────────────


def fetch_backtest_predictions(
    backtest_dir: str | Path,
    realm: str,
    run_id: str | None = None,
) -> pd.DataFrame:
    """Load per-prediction backtest results from CSV.

    Searches for the latest backtest run directory matching ``realm``
    and reads the ``per_prediction.csv`` within each horizon subdirectory.
    """
    base = Path(backtest_dir)
    if not base.exists():
        return pd.DataFrame()

    # Find run directories matching realm
    if run_id:
        run_dirs = [base / run_id]
    else:
        run_dirs = sorted(
            [d for d in base.iterdir() if d.is_dir() and realm in d.name],
            key=lambda p: p.name,
            reverse=True,
        )

    frames: list[pd.DataFrame] = []
    for run_dir in run_dirs[:1]:  # latest only
        for horizon_dir in sorted(run_dir.iterdir()):
            pred_file = horizon_dir / "per_prediction.csv"
            if pred_file.exists():
                try:
                    df = pd.read_csv(pred_file)
                    df["run_dir"] = run_dir.name
                    frames.append(df)
                except Exception:
                    logger.warning("Failed to read %s", pred_file)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def fetch_backtest_summary(
    backtest_dir: str | Path,
    realm: str,
    run_id: str | None = None,
) -> pd.DataFrame:
    """Load aggregated backtest summary from CSV."""
    base = Path(backtest_dir)
    if not base.exists():
        return pd.DataFrame()

    if run_id:
        run_dirs = [base / run_id]
    else:
        run_dirs = sorted(
            [d for d in base.iterdir() if d.is_dir() and realm in d.name],
            key=lambda p: p.name,
            reverse=True,
        )

    frames: list[pd.DataFrame] = []
    for run_dir in run_dirs[:1]:
        for horizon_dir in sorted(run_dir.iterdir()):
            summary_file = horizon_dir / "summary.csv"
            if summary_file.exists():
                try:
                    df = pd.read_csv(summary_file)
                    df["horizon"] = horizon_dir.name
                    df["run_dir"] = run_dir.name
                    frames.append(df)
                except Exception:
                    logger.warning("Failed to read %s", summary_file)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ── Feature importance ────────────────────────────────────────────────────────


def fetch_feature_importance(
    artifact_dir: str | Path,
    realm: str,
    horizon: str | None = None,
) -> pd.DataFrame:
    """Extract feature importance from LightGBM model artifacts.

    Returns columns: feature, gain, gain_pct, split, split_pct, horizon.
    """
    base = Path(artifact_dir)
    if not base.exists():
        return pd.DataFrame()

    horizons = [horizon] if horizon else ["1d", "7d", "28d"]
    frames: list[pd.DataFrame] = []

    for h in horizons:
        # Find the latest artifact for this horizon + realm
        pattern = f"lgbm_{h}_{realm}_*.pkl"
        artifacts = sorted(base.glob(pattern), key=lambda p: p.name, reverse=True)
        if not artifacts:
            continue

        try:
            from wow_forecaster.ml.lgbm_model import LightGBMForecaster

            model = LightGBMForecaster.load(artifacts[0])
            if not model.is_fitted:
                continue

            gain_vals = model._booster.feature_importance(importance_type="gain")
            split_vals = model._booster.feature_importance(importance_type="split")
            feat_cols = model._feature_cols

            total_gain = max(float(sum(gain_vals)), 1.0)
            total_split = max(float(sum(split_vals)), 1.0)

            df = pd.DataFrame({
                "feature": feat_cols,
                "gain": gain_vals.astype(float),
                "gain_pct": gain_vals.astype(float) / total_gain * 100,
                "split": split_vals.astype(float),
                "split_pct": split_vals.astype(float) / total_split * 100,
                "horizon": h,
            })
            frames.append(df)
        except Exception:
            logger.warning("Failed to load feature importance for %s_%s", h, realm)

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ── Drift history ─────────────────────────────────────────────────────────────


def fetch_drift_history(
    monitoring_dir: str | Path,
    realm: str,
) -> pd.DataFrame:
    """Load all drift status JSON files for a realm into a timeline DataFrame.

    Returns columns: checked_at, overall_drift_level, uncertainty_multiplier,
    retrain_recommended, data_drift_level, data_drift_fraction,
    error_drift_level, mae_ratio, shock_active.
    """
    base = Path(monitoring_dir)
    if not base.exists():
        return pd.DataFrame()

    files = sorted(base.glob(f"drift_status_{realm}_*.json"))
    records: list[dict] = []

    for f in files:
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            record = {
                "checked_at": data.get("checked_at"),
                "overall_drift_level": data.get("overall_drift_level"),
                "uncertainty_multiplier": data.get("uncertainty_multiplier"),
                "retrain_recommended": data.get("retrain_recommended"),
            }
            dd = data.get("data_drift", {})
            record["data_drift_level"] = dd.get("drift_level")
            record["data_drift_fraction"] = dd.get("drift_fraction")
            ed = data.get("error_drift", {})
            record["error_drift_level"] = ed.get("drift_level")
            record["mae_ratio"] = ed.get("mae_ratio")
            es = data.get("event_shock", {})
            record["shock_active"] = es.get("shock_active")
            records.append(record)
        except Exception:
            logger.warning("Failed to parse drift file: %s", f)

    return pd.DataFrame(records) if records else pd.DataFrame()


# ── Recommendation scores ────────────────────────────────────────────────────


def fetch_recommendation_scores(
    recs_dir: str | Path,
    realm: str,
) -> pd.DataFrame:
    """Flatten the latest recommendations JSON into a DataFrame.

    Returns columns: category, rank, archetype_id, horizon, action,
    score, sc_opportunity, sc_liquidity, sc_volatility, sc_event_boost,
    sc_uncertainty, roi_pct, current_price, predicted_price, risk_level.
    """
    base = Path(recs_dir)
    if not base.exists():
        return pd.DataFrame()

    files = sorted(base.glob(f"recommendations_{realm}_*.json"), reverse=True)
    if not files:
        return pd.DataFrame()

    try:
        data = json.loads(files[0].read_text(encoding="utf-8"))
    except Exception:
        return pd.DataFrame()

    records: list[dict] = []
    categories = data.get("categories", {})
    for cat_name, cat_recs in categories.items():
        for rec in cat_recs:
            sc = rec.get("score_components", {})
            records.append({
                "category": cat_name,
                "rank": rec.get("rank"),
                "archetype_id": rec.get("archetype_id"),
                "horizon": rec.get("horizon"),
                "action": rec.get("action"),
                "score": rec.get("score"),
                "sc_opportunity": sc.get("opportunity"),
                "sc_liquidity": sc.get("liquidity"),
                "sc_volatility": sc.get("volatility"),
                "sc_event_boost": sc.get("event_boost"),
                "sc_uncertainty": sc.get("uncertainty"),
                "roi_pct": rec.get("roi_pct"),
                "current_price": rec.get("current_price"),
                "predicted_price": rec.get("predicted_price"),
                "risk_level": rec.get("risk_level"),
                "ci_lower": rec.get("ci_lower"),
                "ci_upper": rec.get("ci_upper"),
                "model_slug": rec.get("model_slug"),
            })

    return pd.DataFrame(records) if records else pd.DataFrame()


# ── Crafting margins ──────────────────────────────────────────────────────────


def fetch_crafting_margins(
    db_path: str | Path,
    realm: str,
    days: int = 30,
) -> pd.DataFrame:
    """Fetch crafting margin snapshots from the DB.

    Returns columns: recipe_id, realm_slug, obs_date, craft_cost_gold,
    output_price_gold, margin_gold, margin_pct.
    """
    sql = """
        SELECT cms.recipe_id, cms.realm_slug, cms.obs_date,
               cms.craft_cost_gold, cms.output_price_gold,
               cms.margin_gold, cms.margin_pct,
               r.output_item_id, r.profession, r.recipe_name
        FROM   crafting_margin_snapshots cms
        JOIN   recipes r ON cms.recipe_id = r.recipe_id
        WHERE  cms.realm_slug = ?
          AND  date(cms.obs_date) >= date('now', ? || ' days')
        ORDER  BY cms.obs_date DESC, cms.margin_pct DESC
    """
    return _query_db(db_path, sql, [realm, f"-{days}"])


# ── Archetype metadata ────────────────────────────────────────────────────────


def fetch_archetypes(db_path: str | Path) -> pd.DataFrame:
    """Fetch all economic archetypes.

    Returns columns: archetype_id, slug, display_name, category_tag,
    sub_tag, is_transferable, transfer_confidence.
    """
    sql = """
        SELECT archetype_id, slug, display_name, category_tag,
               sub_tag, is_transferable, transfer_confidence
        FROM   economic_archetypes
        ORDER  BY category_tag, slug
    """
    return _query_db(db_path, sql, [])


# ── Events ────────────────────────────────────────────────────────────────────


def fetch_events(
    db_path: str | Path,
    days_back: int = 90,
    days_ahead: int = 30,
) -> pd.DataFrame:
    """Fetch WoW events within a time window for chart annotation.

    Returns columns: event_id, slug, display_name, event_type,
    severity, start_date, end_date.
    """
    sql = """
        SELECT event_id, slug, display_name, event_type,
               severity, start_date, end_date
        FROM   wow_events
        WHERE  date(start_date) >= date('now', ? || ' days')
          AND  date(start_date) <= date('now', ? || ' days')
        ORDER  BY start_date
    """
    return _query_db(db_path, sql, [f"-{days_back}", str(days_ahead)])


# ── Internal helpers ──────────────────────────────────────────────────────────


def _query_db(
    db_path: str | Path,
    sql: str,
    params: list,
) -> pd.DataFrame:
    """Execute a SQL query and return a pandas DataFrame."""
    try:
        conn = sqlite3.connect(str(db_path))
        df = pd.read_sql_query(sql, conn, params=params)
        conn.close()
        return df
    except Exception:
        logger.warning("DB query failed on %s", db_path)
        return pd.DataFrame()
