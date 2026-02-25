"""
BacktestStage — walk-forward evaluation pipeline stage.

This stage:
1. Loads normalised market observations from the database and computes
   daily-aggregated feature rows (reusing the existing feature pipeline).
2. Generates walk-forward folds from the requested date range.
3. Runs all baseline models over the folds via the evaluator.
4. Persists PredictionRecords to backtest_runs + backtest_fold_results.
5. Writes CSV summaries and a JSON manifest to disk.

One BacktestStage run covers ONE realm and ONE set of horizons.
If multiple realms are needed, the CLI calls this stage once per realm.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Any

from wow_forecaster.config import AppConfig
from wow_forecaster.models.meta import RunMetadata
from wow_forecaster.pipeline.base import PipelineStage

log = logging.getLogger(__name__)


class BacktestStage(PipelineStage):
    """Walk-forward backtest pipeline stage."""

    stage_name = "backtest"

    def _execute(
        self,
        run: RunMetadata,
        realm_slug: str,
        start_date: date,
        end_date: date,
        horizons_days: list[int] | None = None,
        window_days: int | None = None,
        step_days: int | None = None,
    ) -> int:
        """Run the backtest for one realm.

        Args:
            run:           RunMetadata being tracked.
            realm_slug:    Realm to evaluate.
            start_date:    Start of the backtest window (first fold origin).
            end_date:      End of the backtest window (last possible test_date).
            horizons_days: List of forecast horizons in days.  Defaults to
                           config.backtest.horizons_days.
            window_days:   Training window size.  Defaults to
                           config.backtest.window_days.
            step_days:     Steps between folds.  Defaults to
                           config.backtest.step_days.

        Returns:
            Total number of PredictionRecords written across all horizons.
        """
        from wow_forecaster.backtest.evaluator import run_backtest
        from wow_forecaster.backtest.models import all_baseline_models
        from wow_forecaster.backtest.reporter import (
            build_backtest_manifest,
            make_output_dir,
            persist_backtest_run,
            persist_prediction_records,
            write_by_category_csv,
            write_manifest,
            write_per_prediction_csv,
            write_summary_csv,
        )
        from wow_forecaster.backtest.slices import (
            slice_by_category,
            slice_by_model_and_horizon,
        )
        from wow_forecaster.backtest.splits import generate_walk_forward_splits
        from wow_forecaster.db.connection import get_connection
        from wow_forecaster.features.daily_agg import fetch_daily_agg
        from wow_forecaster.features.lag_rolling import compute_lag_rolling_features

        cfg    = self.config
        cfg_bt = cfg.backtest
        cfg_ft = cfg.features

        _window   = window_days   or cfg_bt.window_days
        _step     = step_days     or cfg_bt.step_days
        _horizons = horizons_days or cfg_bt.horizons_days
        _min_rows = cfg_bt.min_train_rows

        total_records = 0

        with get_connection(self.db_path) as conn:
            # ── Load active event dates for is_event_window classification ──
            event_rows = conn.execute(
                "SELECT start_date, end_date FROM wow_events WHERE start_date IS NOT NULL;"
            ).fetchall()
            active_event_dates: set[date] = set()
            for ev in event_rows:
                try:
                    ev_start = date.fromisoformat(ev["start_date"])
                    ev_end = (
                        date.fromisoformat(ev["end_date"])
                        if ev["end_date"]
                        else ev_start
                    )
                    d = ev_start
                    while d <= ev_end:
                        active_event_dates.add(d)
                        d += timedelta(days=1)
                except (ValueError, TypeError):
                    pass

            # ── Load archetype categories for per-category slicing ──────────
            arch_rows = conn.execute(
                "SELECT archetype_id, category_tag FROM economic_archetypes;"
            ).fetchall()
            archetype_categories: dict[int, str] = {
                int(r["archetype_id"]): r["category_tag"] for r in arch_rows
            }

            # ── Load feature rows for the full date range ───────────────────
            # Fetch an extended window before start_date so lag features
            # (up to 28 days lookback) are valid from the first fold origin.
            lookback_buffer = _window + max(cfg_ft.lag_days)
            data_start = start_date - timedelta(days=lookback_buffer)
            agg_rows = fetch_daily_agg(conn, realm_slug, data_start, end_date)

            if not agg_rows:
                log.warning(
                    "No normalised data found for realm=%s %s → %s",
                    realm_slug, start_date, end_date,
                )
                return 0

            feature_rows = compute_lag_rolling_features(agg_rows, cfg_ft)
            log.info(
                "Loaded %d feature rows for realm=%s  buffer=[%s..%s]",
                len(feature_rows), realm_slug, data_start, end_date,
            )

            # ── Run backtest for each horizon ───────────────────────────────
            for h in _horizons:
                folds = generate_walk_forward_splits(
                    start_date=start_date,
                    end_date=end_date,
                    window_days=_window,
                    step_days=_step,
                    horizon_days=h,
                )
                if not folds:
                    log.warning(
                        "No folds for horizon=%dd window=%d step=%d range=%s→%s",
                        h, _window, _step, start_date, end_date,
                    )
                    continue

                models = all_baseline_models()
                model_names = [m.name for m in models]

                records = run_backtest(
                    feature_rows=feature_rows,
                    folds=folds,
                    models=models,
                    archetype_categories=archetype_categories,
                    active_event_dates=active_event_dates,
                    min_train_rows=_min_rows,
                )
                total_records += len(records)

                # ── Persist to SQLite ───────────────────────────────────────
                bt_run_id = persist_backtest_run(
                    conn=conn,
                    run_id=run.run_id,
                    realm_slug=realm_slug,
                    backtest_start=start_date,
                    backtest_end=end_date,
                    window_days=_window,
                    step_days=_step,
                    fold_count=len(folds),
                    model_names=model_names,
                    config_snapshot=run.config_snapshot,
                )
                persist_prediction_records(conn, bt_run_id, records)

                # ── Write CSV + manifest output files ───────────────────────
                out_dir = make_output_dir(
                    cfg.data.processed_dir, realm_slug, start_date, end_date
                )
                h_dir = out_dir / f"horizon_{h}d"
                h_dir.mkdir(parents=True, exist_ok=True)

                summary_metrics = slice_by_model_and_horizon(records)
                write_summary_csv(summary_metrics, h_dir / "summary.csv")

                cat_metrics = slice_by_category(records)
                write_by_category_csv(cat_metrics, h_dir / "by_category.csv")

                write_per_prediction_csv(records, h_dir / "per_prediction.csv")

                manifest = build_backtest_manifest(
                    realm_slug=realm_slug,
                    backtest_start=start_date,
                    backtest_end=end_date,
                    run_slug=run.run_slug,
                    folds=folds,
                    model_names=model_names,
                    records=records,
                    output_dir=h_dir,
                    config_snapshot=run.config_snapshot,
                )
                write_manifest(manifest, h_dir / "manifest.json")

                log.info(
                    "Backtest horizon=%dd done | folds=%d | records=%d | realm=%s",
                    h, len(folds), len(records), realm_slug,
                )

        return total_records
