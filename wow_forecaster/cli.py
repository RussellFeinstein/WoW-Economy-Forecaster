"""
WoW Economy Forecaster — CLI entry point.

All commands follow this pattern:
  1. Load ``AppConfig`` via ``load_config()``.
  2. Configure logging.
  3. Validate inputs.
  4. Execute action (DB init, event import, pipeline stub, etc.).
  5. Report result to stdout.

Install and run::

    pip install -e .
    wow-forecaster --help
    wow-forecaster init-db
    wow-forecaster validate-config
    wow-forecaster import-events
    wow-forecaster run-hourly-refresh
    wow-forecaster run-daily-forecast
    wow-forecaster backtest --start-date 2024-09-10 --end-date 2024-12-01
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import typer

app = typer.Typer(
    name="wow-forecaster",
    help="WoW Auction House Economy Forecaster — local-first research CLI.",
    add_completion=False,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_config_or_exit(config_path: Optional[str] = None):
    """Load AppConfig, printing a friendly error and exiting on failure."""
    from wow_forecaster.config import load_config

    try:
        cfg_path = Path(config_path) if config_path else None
        return load_config(cfg_path)
    except FileNotFoundError as exc:
        typer.echo(f"[ERROR] {exc}", err=True)
        raise typer.Exit(code=1)
    except Exception as exc:
        typer.echo(f"[ERROR] Config validation failed: {exc}", err=True)
        raise typer.Exit(code=1)


def _configure_logging(config):
    """Set up logging from config."""
    from wow_forecaster.utils.logging import configure_logging
    configure_logging(config.logging)


# ── Commands ──────────────────────────────────────────────────────────────────

@app.command("init-db")
def init_db(
    db_path: Optional[str] = typer.Option(
        None,
        "--db-path",
        help="Override DB path from config (e.g. data/db/test.db).",
    ),
    config_path: Optional[str] = typer.Option(
        None,
        "--config",
        help="Path to TOML config file.",
    ),
) -> None:
    """Initialize the SQLite database and apply the full schema.

    Safe to run multiple times — all DDL uses IF NOT EXISTS.
    Also runs pending schema migrations.
    """
    from wow_forecaster.db.connection import get_connection
    from wow_forecaster.db.migrations import run_migrations
    from wow_forecaster.db.schema import apply_schema, ALL_TABLE_NAMES

    config = _load_config_or_exit(config_path)
    _configure_logging(config)

    target_path = db_path or config.database.db_path
    typer.echo(f"Initializing database at: {target_path}")

    with get_connection(
        target_path,
        wal_mode=config.database.wal_mode,
        busy_timeout_ms=config.database.busy_timeout_ms,
    ) as conn:
        apply_schema(conn)
        migrations_applied = run_migrations(conn)

    typer.echo(f"  Tables: {len(ALL_TABLE_NAMES)} created/verified.")
    typer.echo(f"  Migrations applied: {migrations_applied}")
    typer.echo("[OK] Database ready.")


@app.command("validate-config")
def validate_config(
    config_path: Optional[str] = typer.Option(
        None,
        "--config",
        help="Path to TOML config file (default: config/default.toml).",
    ),
    show_full: bool = typer.Option(
        False,
        "--full",
        help="Print full config including all fields.",
    ),
) -> None:
    """Validate the configuration file and print parsed values.

    Exits with code 1 if the config fails validation.
    """
    config = _load_config_or_exit(config_path)

    typer.echo("Configuration validated successfully.")
    typer.echo("")
    typer.echo(f"  Database path:    {config.database.db_path}")
    typer.echo(f"  Active expansion: {config.expansions.active}")
    typer.echo(f"  Transfer target:  {config.expansions.transfer_target}")
    typer.echo(f"  Default realms:   {', '.join(config.realms.defaults)}")
    typer.echo(f"  Forecast horizons:{', '.join(config.forecast.horizons)}")
    typer.echo(f"  Log level:        {config.logging.level}")
    typer.echo(f"  Debug mode:       {config.debug}")

    if show_full:
        typer.echo("")
        typer.echo("Full config (JSON):")
        typer.echo(json.dumps(config.model_dump(), indent=2, default=str))

    typer.echo("")
    typer.echo("[OK] Config valid.")


@app.command("import-events")
def import_events(
    events_file: Optional[str] = typer.Option(
        None,
        "--file",
        "-f",
        help=(
            "Path to events file (.json or .csv). "
            "Defaults to config.data.events_seed_file."
        ),
    ),
    config_path: Optional[str] = typer.Option(
        None,
        "--config",
        help="Path to TOML config file.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Validate events but do not write to the database.",
    ),
) -> None:
    """Import WoW events from a JSON or CSV file into the database.

    Accepts two file formats (detected by extension):

    \b
      .json — Array of event objects matching the WoWEvent schema.
              See config/events/tww_events.json for an example.

      .csv  — Comma-separated with header row.
              See config/events/event_import_template.csv for the schema.

    Uses UPSERT semantics — existing events with the same slug are updated.
    """
    from pydantic import ValidationError

    from wow_forecaster.db.connection import get_connection
    from wow_forecaster.db.repositories.event_repo import WoWEventRepository
    from wow_forecaster.ingestion.event_csv import parse_event_csv
    from wow_forecaster.models.event import WoWEvent

    config = _load_config_or_exit(config_path)
    _configure_logging(config)

    events_path = Path(events_file) if events_file else Path(config.data.events_seed_file)

    if not events_path.exists():
        typer.echo(f"[ERROR] Events file not found: {events_path}", err=True)
        raise typer.Exit(code=1)

    typer.echo(f"Loading events from: {events_path}")
    fmt = events_path.suffix.lower()
    validated: list[WoWEvent] = []

    if fmt == ".csv":
        try:
            validated = parse_event_csv(events_path)
        except (FileNotFoundError, ValueError) as exc:
            typer.echo(f"[ERROR] CSV parse failed:\n{exc}", err=True)
            raise typer.Exit(code=1)

    elif fmt == ".json":
        try:
            with open(events_path, encoding="utf-8") as f:
                raw_events = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            typer.echo(f"[ERROR] JSON parse error: {exc}", err=True)
            raise typer.Exit(code=1)

        if not isinstance(raw_events, list):
            typer.echo("[ERROR] JSON events file must contain an array.", err=True)
            raise typer.Exit(code=1)

        errors: list[tuple[int, str]] = []
        for i, raw in enumerate(raw_events):
            try:
                validated.append(WoWEvent(**raw))
            except (ValidationError, Exception) as exc:
                errors.append((i, str(exc)))

        if errors:
            typer.echo(f"[ERROR] {len(errors)} event(s) failed validation:", err=True)
            for idx, msg in errors[:5]:
                typer.echo(f"  Event #{idx}: {msg}", err=True)
            if len(errors) > 5:
                typer.echo(f"  ... and {len(errors) - 5} more.", err=True)
            raise typer.Exit(code=1)

    else:
        typer.echo(
            f"[ERROR] Unsupported file format '{fmt}'. Use .json or .csv.", err=True
        )
        raise typer.Exit(code=1)

    typer.echo(f"  Validated {len(validated)} event(s) from {fmt} file.")

    if dry_run:
        typer.echo("[DRY RUN] No events written to database.")
        for ev in validated:
            typer.echo(f"  {ev.slug} | {ev.event_type.value} | {ev.start_date}")
        return

    with get_connection(
        config.database.db_path,
        wal_mode=config.database.wal_mode,
        busy_timeout_ms=config.database.busy_timeout_ms,
    ) as conn:
        repo = WoWEventRepository(conn)
        for ev in validated:
            repo.upsert(ev)

    typer.echo(f"  Upserted {len(validated)} event(s) into database.")
    typer.echo("[OK] Events imported.")


@app.command("run-hourly-refresh")
def run_hourly_refresh(
    realm: Optional[str] = typer.Option(
        None,
        "--realm",
        help="Realm slug to refresh (e.g. area-52). Uses config defaults if omitted.",
    ),
    db_path: Optional[str] = typer.Option(
        None,
        "--db-path",
        help="Override DB path from config.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Print what would run without executing.",
    ),
    check_drift: bool = typer.Option(
        True,
        "--check-drift/--no-check-drift",
        help="Run drift detection after normalize (default: on).",
    ),
    config_path: Optional[str] = typer.Option(
        None,
        "--config",
        help="Path to TOML config file.",
    ),
) -> None:
    """Run the full hourly refresh: ingest, normalize, drift check.

    \b
    Steps:
      1. IngestStage     — fetch AH snapshots (fixture mode until API keys set).
                           Per-realm failures are isolated; other realms continue.
      2. NormalizeStage  — convert copper->gold, z-score, flag outliers.
      3. Drift Check     — data drift + error drift + event shock detection.
      4. Adaptive policy — compute uncertainty multiplier from drift severity.
      5. Provenance      — record source freshness and attribution.

    \b
    Outputs:
      data/outputs/monitoring/drift_status_{realm}_{date}.json
      data/outputs/monitoring/provenance_{realm}_{date}.json

    \b
    Drift levels and effects:
      none     -> uncertainty x1.0 (no change)
      low      -> uncertainty x1.25
      medium   -> uncertainty x1.5  + retrain recommended
      high     -> uncertainty x2.0  + retrain recommended
      critical -> uncertainty x3.0  + retrain recommended

    \b
    Credential setup (.env, gitignored):
      UNDERMINE_API_KEY=...          -> enables real Undermine data
      BLIZZARD_CLIENT_ID=...         -> enables real Blizzard AH data
      BLIZZARD_CLIENT_SECRET=...

    Without credentials the pipeline runs in fixture mode (synthetic sample data).
    """
    from wow_forecaster.pipeline.orchestrator import HourlyOrchestrator

    config = _load_config_or_exit(config_path)
    _configure_logging(config)

    target_db     = db_path or config.database.db_path
    target_realms = [realm] if realm else list(config.realms.defaults)

    typer.echo(
        f"run-hourly-refresh | realms={', '.join(target_realms)} | "
        f"check_drift={check_drift} | db={target_db}"
    )

    if dry_run:
        typer.echo("[DRY RUN] Would run:")
        typer.echo(f"  [1] IngestStage    -> realms={target_realms}")
        typer.echo("  [2] NormalizeStage -> all unprocessed raw observations")
        if check_drift:
            typer.echo("  [3] DriftChecker   -> data drift + error drift + event shock")
            typer.echo("  [4] AdaptivePolicy -> uncertainty multiplier")
            typer.echo("  [5] Provenance     -> source freshness summary")
        return

    orchestrator = HourlyOrchestrator(config=config, db_path=target_db)
    result = orchestrator.run(
        realm_slugs=target_realms,
        check_drift=check_drift,
        apply_adaptive=check_drift,
    )

    # ── Print summary ─────────────────────────────────────────────────────────
    typer.echo("")
    for rr in result.realm_results:
        status_str = "ok" if rr.success else "FAIL"
        msg = f"  Ingest [{rr.realm_slug}] {status_str} | rows={rr.rows_written}"
        if rr.error:
            msg += f" | err={rr.error}"
        typer.echo(msg)

    norm_status = "ok" if result.normalize_success else "FAIL"
    typer.echo(f"  Normalize       {norm_status} | rows={result.normalize_rows}")

    if result.drift_results:
        typer.echo("")
        typer.echo("  Drift check results:")
        for realm_slug, dr in result.drift_results.items():
            retrain_flag = " [RETRAIN RECOMMENDED]" if dr.retrain_recommended else ""
            shock_flag   = " [EVENT SHOCK]" if dr.event_shock.shock_active else ""
            typer.echo(
                f"    {realm_slug:<20} drift={dr.overall_drift_level.value:<8} "
                f"mult=x{dr.uncertainty_multiplier:.2f}{retrain_flag}{shock_flag}"
            )
            typer.echo(
                f"      data: {dr.data_drift.drift_level.value} "
                f"({dr.data_drift.n_series_drifted}/{dr.data_drift.n_series_checked} series) | "
                f"error: {dr.error_drift.drift_level.value} "
                f"(mae_ratio={dr.error_drift.mae_ratio or 'N/A'})"
            )

    if result.monitoring_files:
        typer.echo("")
        typer.echo("  Monitoring outputs:")
        for f in result.monitoring_files:
            typer.echo(f"    {f}")

    if result.errors:
        typer.echo("")
        for e in result.errors:
            typer.echo(f"  [WARN] {e}", err=True)

    typer.echo("")
    status_tag = (
        "[OK]"      if result.status == "success" else
        "[PARTIAL]" if result.status == "partial"  else
        "[FAIL]"
    )
    fixture_note = ""
    if result.status in ("success", "partial") and not any(
        rr.success for rr in result.realm_results
    ):
        fixture_note = " (fixture mode — set API keys in .env for live data)"
    typer.echo(f"{status_tag} Hourly refresh {result.status}.{fixture_note}")


@app.command("train-model")
def train_model(
    realm: Optional[list[str]] = typer.Option(
        None,
        "--realm",
        help="Realm slug(s) to train for. Repeatable; uses config defaults if omitted.",
    ),
    db_path: Optional[str] = typer.Option(
        None,
        "--db-path",
        help="Override DB path from config.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Show what would train without executing.",
    ),
    config_path: Optional[str] = typer.Option(
        None,
        "--config",
        help="Path to TOML config file.",
    ),
) -> None:
    """Train LightGBM forecasting models from the latest feature Parquet.

    \b
    Steps:
      1. Locate the latest training Parquet for each realm.
      2. Apply a time-based validation split (last N days held out).
      3. Train one LightGBM model per configured horizon (1d, 7d, 28d).
      4. Write .pkl + .json artifacts to config.model.artifact_dir.
      5. Register each model in model_metadata (marking old models inactive).

    \b
    Run 'build-datasets' first to generate the training Parquet.
    """
    from wow_forecaster.db.connection import get_connection
    from wow_forecaster.db.schema import apply_schema
    from wow_forecaster.pipeline.train import TrainStage

    config = _load_config_or_exit(config_path)
    _configure_logging(config)

    target_db     = db_path or config.database.db_path
    target_realms = list(realm) if realm else list(config.realms.defaults)
    artifact_dir  = config.model.artifact_dir

    typer.echo(
        f"train-model | realms={', '.join(target_realms)} | "
        f"horizons={list(config.features.target_horizons_days)} | db={target_db}"
    )

    if dry_run:
        typer.echo("[DRY RUN] Would train:")
        for r in target_realms:
            typer.echo(
                f"  realm={r} | horizons={list(config.features.target_horizons_days)} | "
                f"artifacts -> {artifact_dir}"
            )
        return

    with get_connection(
        target_db,
        wal_mode=config.database.wal_mode,
        busy_timeout_ms=config.database.busy_timeout_ms,
    ) as conn:
        apply_schema(conn)

    try:
        stage      = TrainStage(config=config, db_path=target_db)
        result_run = stage.run(realm_slugs=target_realms)
    except Exception as exc:
        typer.echo(f"[ERROR] Training failed: {exc}", err=True)
        raise typer.Exit(code=1)

    typer.echo(
        f"  status={result_run.status} | models_trained={result_run.rows_processed}"
    )
    typer.echo("")
    typer.echo(f"[OK] Training complete. Artifacts in: {artifact_dir}")


@app.command("run-daily-forecast")
def run_daily_forecast(
    realm: Optional[str] = typer.Option(
        None,
        "--realm",
        help="Realm slug to forecast (e.g. area-52). Uses config defaults if omitted.",
    ),
    skip_train: bool = typer.Option(
        False,
        "--skip-train",
        help="Skip TrainStage and reuse the most recent model artifacts.",
    ),
    skip_recommend: bool = typer.Option(
        False,
        "--skip-recommend",
        help="Skip RecommendStage (forecast only).",
    ),
    db_path: Optional[str] = typer.Option(
        None,
        "--db-path",
        help="Override DB path from config.",
    ),
    config_path: Optional[str] = typer.Option(
        None,
        "--config",
        help="Path to TOML config file.",
    ),
) -> None:
    """Run the full daily forecast pipeline: train -> forecast -> recommend.

    \b
    Steps:
      1. TrainStage:     Fit/update LightGBM models from the latest training Parquet.
      2. ForecastStage:  Predict prices for all archetypes (1d, 7d, 28d horizons).
      3. RecommendStage: Rank forecast outputs into buy/sell/hold/avoid recommendations.

    \b
    Outputs:
      data/outputs/model_artifacts/  -- updated .pkl model files
      data/outputs/forecasts/        -- forecast_{realm}_{date}.csv
      data/outputs/recommendations/  -- recommendations_{realm}_{date}.csv/.json

    \b
    Run 'build-datasets' first to generate the required feature Parquet files.
    Use --skip-train to reuse existing artifacts (useful for intra-day re-runs).
    """
    from wow_forecaster.db.connection import get_connection
    from wow_forecaster.db.schema import apply_schema
    from wow_forecaster.pipeline.forecast import ForecastStage
    from wow_forecaster.pipeline.recommend import RecommendStage
    from wow_forecaster.pipeline.train import TrainStage

    config = _load_config_or_exit(config_path)
    _configure_logging(config)

    target_db    = db_path or config.database.db_path
    target_realm = realm or config.realms.defaults[0]

    typer.echo(
        f"run-daily-forecast | realm={target_realm} | date={date.today()} | "
        f"skip_train={skip_train} | skip_recommend={skip_recommend}"
    )

    # Ensure schema exists (idempotent).
    with get_connection(
        target_db,
        wal_mode=config.database.wal_mode,
        busy_timeout_ms=config.database.busy_timeout_ms,
    ) as conn:
        apply_schema(conn)

    # ── Step 1: TrainStage ─────────────────────────────────────────────────────
    if not skip_train:
        typer.echo("  [1/3] TrainStage ...")
        try:
            train_stage  = TrainStage(config=config, db_path=target_db)
            train_result = train_stage.run(realm_slugs=[target_realm])
            typer.echo(
                f"        status={train_result.status} | "
                f"models={train_result.rows_processed}"
            )
        except Exception as exc:
            typer.echo(f"        [WARN] TrainStage failed: {exc}", err=True)
            typer.echo("        Proceeding with existing model artifacts.")
    else:
        typer.echo("  [1/3] TrainStage skipped (--skip-train).")

    # ── Step 2: ForecastStage ──────────────────────────────────────────────────
    typer.echo("  [2/3] ForecastStage ...")
    try:
        fc_stage  = ForecastStage(config=config, db_path=target_db)
        fc_result = fc_stage.run(realm_slug=target_realm)
        typer.echo(
            f"        status={fc_result.status} | "
            f"forecast_rows={fc_result.rows_processed}"
        )
    except Exception as exc:
        typer.echo(f"[ERROR] ForecastStage failed: {exc}", err=True)
        raise typer.Exit(code=1)

    # ── Step 3: RecommendStage ─────────────────────────────────────────────────
    if not skip_recommend:
        typer.echo("  [3/3] RecommendStage ...")
        try:
            rec_stage  = RecommendStage(config=config, db_path=target_db)
            rec_result = rec_stage.run(
                realm_slug=target_realm,
                forecast_run_id=fc_result.run_id,
            )
            typer.echo(
                f"        status={rec_result.status} | "
                f"recommendations={rec_result.rows_processed}"
            )
        except Exception as exc:
            typer.echo(f"        [WARN] RecommendStage failed: {exc}", err=True)
    else:
        typer.echo("  [3/3] RecommendStage skipped (--skip-recommend).")

    typer.echo("")
    typer.echo(
        f"[OK] Daily forecast complete. "
        f"Reports in: {config.model.recommendation_output_dir}"
    )


@app.command("recommend-top-items")
def recommend_top_items(
    realm: Optional[str] = typer.Option(
        None,
        "--realm",
        help="Realm slug (e.g. area-52). Uses first config default if omitted.",
    ),
    top_n: Optional[int] = typer.Option(
        None,
        "--top-n",
        help="Max recommendations per category. Defaults to config.model.top_n_per_category.",
    ),
    forecast_run_id: Optional[int] = typer.Option(
        None,
        "--forecast-run-id",
        help="Score forecasts from a specific run_id. Defaults to the most recent.",
    ),
    db_path: Optional[str] = typer.Option(
        None,
        "--db-path",
        help="Override DB path from config.",
    ),
    config_path: Optional[str] = typer.Option(
        None,
        "--config",
        help="Path to TOML config file.",
    ),
) -> None:
    """Score forecast outputs and write ranked buy/sell/hold/avoid recommendations.

    \b
    Loads the most recent ForecastOutput rows from the database, scores them
    with the 5-component formula, and writes CSV + JSON report files.

    \b
    Score formula (0-100):
      total = 0.35 x opportunity  +  0.20 x liquidity
            - 0.20 x volatility   +  0.15 x event_boost
            - 0.10 x uncertainty

    \b
    Actions:
      buy   -- predicted ROI >= 10%
      sell  -- predicted ROI <= -10%
      avoid -- CI width > 80% or CV > 80% (too risky)
      hold  -- otherwise

    \b
    Run 'run-daily-forecast' first to generate forecast outputs.
    """
    from wow_forecaster.pipeline.recommend import RecommendStage

    config = _load_config_or_exit(config_path)
    _configure_logging(config)

    target_db    = db_path or config.database.db_path
    target_realm = realm or config.realms.defaults[0]
    n            = top_n or config.model.top_n_per_category

    typer.echo(
        f"recommend-top-items | realm={target_realm} | "
        f"top_n={n} | forecast_run_id={forecast_run_id or 'latest'}"
    )

    try:
        stage  = RecommendStage(config=config, db_path=target_db)
        result = stage.run(
            realm_slug=target_realm,
            top_n_per_category=n,
            forecast_run_id=forecast_run_id,
        )
    except Exception as exc:
        typer.echo(f"[ERROR] Recommend failed: {exc}", err=True)
        raise typer.Exit(code=1)

    typer.echo(
        f"  status={result.status} | recommendations={result.rows_processed}"
    )
    typer.echo("")
    typer.echo(
        f"[OK] Recommendations written to: {config.model.recommendation_output_dir}"
    )


@app.command("backtest")
def backtest(
    start_date: str = typer.Option(
        ...,
        "--start-date",
        help="Start of backtest window (ISO date, e.g. 2024-09-10).",
    ),
    end_date: str = typer.Option(
        ...,
        "--end-date",
        help="End of backtest window (ISO date, e.g. 2024-12-01).",
    ),
    realm: Optional[str] = typer.Option(
        None,
        "--realm",
        help="Realm to backtest. Uses config defaults if omitted.",
    ),
    window_days: Optional[int] = typer.Option(
        None,
        "--window-days",
        help="Walk-forward training window size in days. Uses config default if omitted.",
    ),
    step_days: Optional[int] = typer.Option(
        None,
        "--step-days",
        help="Days to advance between folds. Uses config default if omitted.",
    ),
    horizons: Optional[str] = typer.Option(
        None,
        "--horizons",
        help=(
            "Comma-separated forecast horizons in days (e.g. '1,3'). "
            "Uses config default if omitted."
        ),
    ),
    db_path: Optional[str] = typer.Option(
        None,
        "--db-path",
        help="Override DB path from config.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Show what would run (fold count, model names) without executing.",
    ),
    config_path: Optional[str] = typer.Option(
        None,
        "--config",
        help="Path to TOML config file.",
    ),
) -> None:
    """Run walk-forward backtest over historical TWW data.

    \b
    For each realm:
      1. Load normalised market observations and build daily feature rows.
      2. Generate walk-forward folds from start_date to end_date.
      3. Fit all baseline models (last_value, rolling_mean, day_of_week,
         simple_volatility) on each fold's training window.
      4. Evaluate predictions vs actual prices; compute MAE, RMSE, MAPE,
         directional accuracy.
      5. Persist results to backtest_runs + backtest_fold_results (SQLite).
      6. Write CSV summaries and JSON manifest to:
           data/processed/backtest/{realm}_{start}_{end}/horizon_{N}d/

    \b
    Leakage prevention:
      - Models only see data up to (and including) each fold's train_end date.
      - Event window classification is post-hoc (never a model input).
      - target_price_* columns are never passed to any model.
    """
    from wow_forecaster.db.connection import get_connection
    from wow_forecaster.db.schema import apply_schema
    from wow_forecaster.pipeline.backtest import BacktestStage

    config = _load_config_or_exit(config_path)
    _configure_logging(config)

    # Validate dates.
    try:
        start = date.fromisoformat(start_date)
        end   = date.fromisoformat(end_date)
    except ValueError as exc:
        typer.echo(f"[ERROR] Invalid date format: {exc}", err=True)
        raise typer.Exit(code=1)

    if end <= start:
        typer.echo("[ERROR] --end-date must be after --start-date.", err=True)
        raise typer.Exit(code=1)

    # Parse optional horizon override.
    parsed_horizons: Optional[list[int]] = None
    if horizons:
        try:
            parsed_horizons = [int(h.strip()) for h in horizons.split(",")]
            if any(h < 1 for h in parsed_horizons):
                raise ValueError("All horizons must be >= 1.")
        except ValueError as exc:
            typer.echo(f"[ERROR] Invalid --horizons value: {exc}", err=True)
            raise typer.Exit(code=1)

    target_db    = db_path or config.database.db_path
    target_realm = realm or config.realms.defaults[0]
    win  = window_days or config.backtest.window_days
    step = step_days   or config.backtest.step_days
    hors = parsed_horizons or config.backtest.horizons_days

    typer.echo(
        f"backtest | realm={target_realm} | {start} -> {end} | "
        f"window={win}d | step={step}d | horizons={hors}"
    )

    if dry_run:
        from wow_forecaster.backtest.splits import generate_walk_forward_splits
        from wow_forecaster.backtest.models import all_baseline_models
        models = all_baseline_models()
        typer.echo("[DRY RUN] Would execute:")
        for h in hors:
            folds = generate_walk_forward_splits(start, end, win, step, h)
            typer.echo(
                f"  horizon={h}d | folds={len(folds)} | "
                f"models={[m.name for m in models]}"
            )
        return

    # Ensure schema exists (idempotent).
    with get_connection(
        target_db,
        wal_mode=config.database.wal_mode,
        busy_timeout_ms=config.database.busy_timeout_ms,
    ) as conn:
        apply_schema(conn)

    typer.echo(f"  Running BacktestStage ...")
    try:
        stage = BacktestStage(config=config, db_path=target_db)
        result_run = stage.run(
            realm_slug=target_realm,
            start_date=start,
            end_date=end,
            horizons_days=hors,
            window_days=win,
            step_days=step,
        )
    except Exception as exc:
        typer.echo(f"[ERROR] Backtest failed: {exc}", err=True)
        raise typer.Exit(code=1)

    typer.echo(
        f"  status={result_run.status} | records={result_run.rows_processed}"
    )
    typer.echo("")
    typer.echo(
        f"[OK] Backtest complete. "
        f"Results in: data/processed/backtest/{target_realm.replace('-', '_')}_{start}_{end}/"
    )


@app.command("report-backtest")
def report_backtest(
    realm: Optional[str] = typer.Option(
        None,
        "--realm",
        help="Filter results to this realm slug.",
    ),
    backtest_run_id: Optional[int] = typer.Option(
        None,
        "--run-id",
        help="Show results for a specific backtest_run_id.",
    ),
    horizon: Optional[int] = typer.Option(
        None,
        "--horizon",
        help="Filter to a specific horizon in days (e.g. 1 or 3).",
    ),
    db_path: Optional[str] = typer.Option(
        None,
        "--db-path",
        help="Override DB path from config.",
    ),
    config_path: Optional[str] = typer.Option(
        None,
        "--config",
        help="Path to TOML config file.",
    ),
) -> None:
    """Print a summary of the most recent backtest results.

    \b
    Shows:
      - Run metadata (realm, date range, window, fold count).
      - Per-model aggregate metrics (MAE, RMSE, MAPE, directional accuracy).
      - Event vs non-event accuracy split.

    \b
    Use --run-id to target a specific run; otherwise the most recent run
    matching --realm is shown.
    """
    from wow_forecaster.backtest.metrics import BacktestMetrics, PredictionRecord
    from wow_forecaster.backtest.slices import (
        slice_by_event_window,
        slice_by_model_and_horizon,
    )
    from wow_forecaster.db.connection import get_connection

    config = _load_config_or_exit(config_path)
    _configure_logging(config)

    target_db = db_path or config.database.db_path

    with get_connection(
        target_db,
        wal_mode=config.database.wal_mode,
        busy_timeout_ms=config.database.busy_timeout_ms,
    ) as conn:
        # ── Find the target backtest run ─────────────────────────────────────
        if backtest_run_id is not None:
            run_row = conn.execute(
                "SELECT * FROM backtest_runs WHERE backtest_run_id = ?;",
                (backtest_run_id,),
            ).fetchone()
        else:
            q = "SELECT * FROM backtest_runs"
            params: list = []
            if realm:
                q += " WHERE realm_slug = ?"
                params.append(realm)
            q += " ORDER BY backtest_run_id DESC LIMIT 1;"
            run_row = conn.execute(q, params).fetchone()

        if run_row is None:
            typer.echo("[ERROR] No backtest runs found in the database.", err=True)
            typer.echo(
                "  Run 'wow-forecaster backtest --start-date ... --end-date ...' first.",
                err=True,
            )
            raise typer.Exit(code=1)

        bt_run_id   = run_row["backtest_run_id"]
        realm_slug  = run_row["realm_slug"]
        bt_start    = run_row["backtest_start"]
        bt_end      = run_row["backtest_end"]
        fold_count  = run_row["fold_count"]
        window_days = run_row["window_days"]

        typer.echo("")
        typer.echo("=== Backtest Report ===")
        typer.echo(f"  Run ID:      {bt_run_id}")
        typer.echo(f"  Realm:       {realm_slug}")
        typer.echo(f"  Date range:  {bt_start} -> {bt_end}")
        typer.echo(f"  Window:      {window_days}d | Folds: {fold_count}")

        # ── Load prediction records ──────────────────────────────────────────
        q_results = """
            SELECT fold_index, archetype_id, realm_slug, category_tag,
                   model_name, train_end, test_date, horizon_days,
                   actual_price, predicted_price,
                   direction_actual, direction_predicted, direction_correct,
                   is_event_window
            FROM backtest_fold_results
            WHERE backtest_run_id = ?
        """
        params_r: list = [bt_run_id]
        if horizon is not None:
            q_results += " AND horizon_days = ?"
            params_r.append(horizon)
        result_rows = conn.execute(q_results, params_r).fetchall()

    if not result_rows:
        typer.echo("  No prediction records found for this run.")
        raise typer.Exit(code=0)

    # ── Reconstruct PredictionRecords ────────────────────────────────────────
    import datetime as _dt
    records: list[PredictionRecord] = []
    for r in result_rows:
        try:
            train_end  = _dt.date.fromisoformat(r["train_end"])
            test_date  = _dt.date.fromisoformat(r["test_date"])
        except (ValueError, TypeError):
            continue
        records.append(PredictionRecord(
            fold_index=r["fold_index"],
            archetype_id=r["archetype_id"],
            realm_slug=r["realm_slug"],
            category_tag=r["category_tag"],
            model_name=r["model_name"],
            train_end=train_end,
            test_date=test_date,
            horizon_days=r["horizon_days"],
            actual_price=r["actual_price"],
            predicted_price=r["predicted_price"],
            last_known_price=None,  # not stored; not needed for reporting
            is_event_window=bool(r["is_event_window"]),
        ))

    typer.echo(f"  Predictions: {len(records)}")

    # ── Per-model × horizon metrics ──────────────────────────────────────────
    model_metrics = slice_by_model_and_horizon(records)
    typer.echo("")
    typer.echo("Per-model metrics (MAE in gold | RMSE | MAPE | Dir.Acc):")
    header = f"  {'Model':<22} {'H':>3}  {'N':>6}  {'MAE':>8}  {'RMSE':>8}  {'MAPE':>7}  {'DirAcc':>7}"
    typer.echo(header)
    typer.echo("  " + "-" * (len(header) - 2))
    for (model_name, h), m in sorted(model_metrics.items()):
        mae_str  = f"{m.mae:.2f}"  if m.mae  is not None else "  N/A"
        rmse_str = f"{m.rmse:.2f}" if m.rmse is not None else "  N/A"
        mape_str = f"{m.mape:.1%}" if m.mape is not None else "  N/A"
        dir_str  = (
            f"{m.directional_accuracy:.1%}"
            if m.directional_accuracy is not None else "  N/A"
        )
        typer.echo(
            f"  {model_name:<22} {h:>3}  {m.n_evaluated:>6}  "
            f"{mae_str:>8}  {rmse_str:>8}  {mape_str:>7}  {dir_str:>7}"
        )

    # ── Event vs non-event split ─────────────────────────────────────────────
    event_metrics = slice_by_event_window(records)
    if len(event_metrics) > 1:
        typer.echo("")
        typer.echo("Event vs non-event accuracy (all models combined):")
        for window_key, m in sorted(event_metrics.items()):
            mae_str = f"{m.mae:.2f}" if m.mae is not None else "N/A"
            dir_str = (
                f"{m.directional_accuracy:.1%}"
                if m.directional_accuracy is not None else "N/A"
            )
            typer.echo(
                f"  {window_key:<20} N={m.n_evaluated:>5}  MAE={mae_str}  DirAcc={dir_str}"
            )

    typer.echo("")
    typer.echo("[OK] Report complete.")


@app.command("build-datasets")
def build_datasets_cmd(
    realm: Optional[list[str]] = typer.Option(
        None,
        "--realm",
        help="Realm slug to process (e.g. area-52). Repeatable; uses config defaults if omitted.",
    ),
    start_date: Optional[str] = typer.Option(
        None,
        "--start-date",
        help="ISO date for the start of the training window (e.g. 2024-09-01). "
             "Defaults to today minus config.features.training_lookback_days.",
    ),
    end_date: Optional[str] = typer.Option(
        None,
        "--end-date",
        help="ISO date for the end of the training window (e.g. 2025-01-31). "
             "Defaults to today.",
    ),
    no_inference: bool = typer.Option(
        False,
        "--no-inference",
        help="Skip writing the inference Parquet file (training only).",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Validate inputs and print what would be built, then exit without writing files.",
    ),
    db_path: Optional[str] = typer.Option(
        None,
        "--db-path",
        help="Override DB path from config.",
    ),
    config_path: Optional[str] = typer.Option(
        None,
        "--config",
        help="Path to TOML config file.",
    ),
) -> None:
    """Build training + inference Parquet feature datasets.

    \b
    Steps:
      1. Aggregates market_observations_normalized to daily (archetype, realm, date) grain.
      2. Adds lag (1/3/7/14/28d), rolling (7/14/28d), momentum, and target label features.
      3. Adds event proximity features with strict leakage guard (announced_at IS NOT NULL).
      4. Adds archetype category, cold-start, and transfer mapping features.
      5. Writes training Parquet (45 cols) and inference Parquet (42 cols, no targets).
      6. Writes a JSON manifest with quality report and file checksums.

    \b
    Output paths:
      data/processed/features/training/train_{realm}_{start}_{end}.parquet
      data/processed/features/inference/inference_{realm}_{today}.parquet
      data/processed/features/manifests/manifest_{realm}_{today}.json
    """
    from wow_forecaster.db.connection import get_connection
    from wow_forecaster.db.schema import apply_schema
    from wow_forecaster.features.dataset_builder import build_datasets
    from wow_forecaster.models.meta import RunMetadata
    from wow_forecaster.utils.time_utils import utcnow

    config = _load_config_or_exit(config_path)
    _configure_logging(config)

    target_db = db_path or config.database.db_path
    target_realms: list[str] = list(realm) if realm else list(config.realms.defaults)

    # Parse / default dates.
    try:
        end = date.fromisoformat(end_date) if end_date else date.today()
        start = (
            date.fromisoformat(start_date)
            if start_date
            else end - timedelta(days=config.features.training_lookback_days)
        )
    except ValueError as exc:
        typer.echo(f"[ERROR] Invalid date format: {exc}", err=True)
        raise typer.Exit(code=1)

    if end <= start:
        typer.echo("[ERROR] --end-date must be after --start-date.", err=True)
        raise typer.Exit(code=1)

    typer.echo(
        f"build-datasets | realms={', '.join(target_realms)} | "
        f"{start} -> {end} | db={target_db}"
    )

    if dry_run:
        typer.echo("[DRY RUN] Would build:")
        for r in target_realms:
            typer.echo(f"  training:  data/processed/features/training/train_{r}_{start}_{end}.parquet")
            if not no_inference:
                typer.echo(f"  inference: data/processed/features/inference/inference_{r}_{date.today()}.parquet")
            typer.echo(f"  manifest:  data/processed/features/manifests/manifest_{r}_{date.today()}.json")
        return

    # Ensure schema exists.
    with get_connection(
        target_db,
        wal_mode=config.database.wal_mode,
        busy_timeout_ms=config.database.busy_timeout_ms,
    ) as conn:
        apply_schema(conn)

    # Create a RunMetadata for this build.
    run = RunMetadata(
        run_slug=f"feature-build-{date.today().isoformat()}",
        pipeline_stage="feature_build",
        config_snapshot={"features": config.features.model_dump()},
        started_at=utcnow(),
    )

    try:
        with get_connection(
            target_db,
            wal_mode=config.database.wal_mode,
            busy_timeout_ms=config.database.busy_timeout_ms,
        ) as conn:
            total = build_datasets(
                conn=conn,
                config=config,
                run=run,
                realm_slugs=target_realms,
                start_date=start,
                end_date=end,
                build_training=True,
                build_inference=not no_inference,
            )
    except Exception as exc:
        typer.echo(f"[ERROR] Dataset build failed: {exc}", err=True)
        raise typer.Exit(code=1)

    typer.echo(f"[OK] Built {total} training rows across {len(target_realms)} realm(s).")


@app.command("validate-datasets")
def validate_datasets_cmd(
    manifest: str = typer.Option(
        ...,
        "--manifest",
        help="Path to a manifest JSON file (written by build-datasets).",
    ),
    strict: bool = typer.Option(
        False,
        "--strict",
        help="Exit with code 1 if any quality warnings are present (not just hard errors).",
    ),
    config_path: Optional[str] = typer.Option(
        None,
        "--config",
        help="Path to TOML config file.",
    ),
) -> None:
    """Validate a feature dataset using its manifest JSON.

    Loads the training Parquet file referenced in the manifest and runs
    a full data quality report.  Prints a human-readable summary.

    \b
    Exit codes:
      0 — report is clean (or warnings-only when --strict is not set).
      1 — hard quality errors found (duplicates or leakage warnings).
          Also 1 if --strict is set and any warnings are present.
    """
    import pyarrow.parquet as pq

    from wow_forecaster.features.quality import build_quality_report

    config = _load_config_or_exit(config_path)
    _configure_logging(config)

    manifest_path = Path(manifest)
    if not manifest_path.exists():
        typer.echo(f"[ERROR] Manifest file not found: {manifest_path}", err=True)
        raise typer.Exit(code=1)

    try:
        with open(manifest_path, encoding="utf-8") as f:
            manifest_data = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        typer.echo(f"[ERROR] Failed to read manifest: {exc}", err=True)
        raise typer.Exit(code=1)

    training_info = manifest_data.get("files", {}).get("training", {})
    training_path = Path(training_info.get("path", ""))

    if not training_path.exists():
        typer.echo(f"[ERROR] Training Parquet not found: {training_path}", err=True)
        raise typer.Exit(code=1)

    typer.echo(f"Loading Parquet: {training_path}")
    try:
        table = pq.read_table(str(training_path))
        rows = table.to_pylist()
    except Exception as exc:
        typer.echo(f"[ERROR] Failed to read Parquet: {exc}", err=True)
        raise typer.Exit(code=1)

    typer.echo(f"Running quality report on {len(rows)} rows...")
    items_excluded = manifest_data.get("quality", {}).get("items_excluded_no_archetype", 0)
    report = build_quality_report(rows, items_excluded=items_excluded)

    # ── Print report ──────────────────────────────────────────────────────────
    typer.echo("")
    typer.echo("=== Data Quality Report ===")
    typer.echo(f"  Realm:            {manifest_data.get('realm_slug', '?')}")
    typer.echo(f"  Date range:       {manifest_data.get('date_range', {}).get('start')} -> {manifest_data.get('date_range', {}).get('end')}")
    typer.echo(f"  Total rows:       {report.total_rows}")
    typer.echo(f"  Total archetypes: {report.total_archetypes}")
    typer.echo(f"  Total realms:     {report.total_realms}")
    typer.echo(f"  Duplicates:       {report.duplicate_key_count}")
    typer.echo(f"  Series w/ gaps:   {report.date_gap_series_count}")
    typer.echo(f"  Volume proxy:     {report.volume_proxy_pct:.1%}")
    typer.echo(f"  Cold-start rows:  {report.cold_start_pct:.1%}")
    typer.echo(f"  Items excl. (no archetype): {report.items_excluded_no_archetype}")
    typer.echo(f"  Is clean:         {report.is_clean}")

    if report.high_missingness_cols:
        typer.echo(f"\n  High-missingness columns (>{report.total_rows * 0:.0f} threshold):")
        for col in report.high_missingness_cols:
            frac = report.missingness.get(col, 0.0)
            typer.echo(f"    {col}: {frac:.1%} null")

    if report.leakage_warnings:
        typer.echo(f"\n  [WARN] Leakage warnings ({len(report.leakage_warnings)}):")
        for w in report.leakage_warnings[:5]:
            typer.echo(f"    {w}")
        if len(report.leakage_warnings) > 5:
            typer.echo(f"    ... and {len(report.leakage_warnings) - 5} more.")

    typer.echo("")

    has_errors   = not report.is_clean
    has_warnings = bool(report.high_missingness_cols or report.date_gap_series_count > 0)

    if has_errors:
        typer.echo("[FAIL] Dataset has hard quality errors.")
        raise typer.Exit(code=1)
    elif strict and has_warnings:
        typer.echo("[FAIL] Dataset has quality warnings (--strict mode).")
        raise typer.Exit(code=1)
    else:
        typer.echo("[OK] Dataset quality check passed.")


@app.command("evaluate-live-forecast")
def evaluate_live_forecast(
    realm: Optional[str] = typer.Option(
        None,
        "--realm",
        help="Realm slug (e.g. area-52). Uses first config default if omitted.",
    ),
    window_days: int = typer.Option(
        14,
        "--window-days",
        help="How many recent days of forecast target dates to evaluate.",
    ),
    db_path: Optional[str] = typer.Option(
        None,
        "--db-path",
        help="Override DB path from config.",
    ),
    output_json: bool = typer.Option(
        True,
        "--output-json/--no-output-json",
        help="Write model_health_{realm}_{date}.json (default: on).",
    ),
    config_path: Optional[str] = typer.Option(
        None,
        "--config",
        help="Path to TOML config file.",
    ),
) -> None:
    """Evaluate live forecast accuracy against actual prices.

    \b
    For each configured horizon (1d, 7d, 28d), finds forecasts whose
    target_date has already passed, looks up actual prices in
    market_observations_normalized, and computes live MAE.

    \b
    Compares live MAE to baseline MAE from the most recent backtest run:
      ok        : mae_ratio < 1.5
      degraded  : 1.5 <= mae_ratio < 3.0
      critical  : mae_ratio >= 3.0
      unknown   : no baseline or no actuals available yet

    \b
    Output:
      Prints a per-horizon health table.
      Optionally writes data/outputs/monitoring/model_health_{realm}_{date}.json.
    """
    from wow_forecaster.db.connection import get_connection
    from wow_forecaster.monitoring.health import compute_health_summary
    from wow_forecaster.monitoring.reporter import (
        persist_health_to_db,
        write_health_report,
    )

    config = _load_config_or_exit(config_path)
    _configure_logging(config)

    target_db    = db_path or config.database.db_path
    target_realm = realm or config.realms.defaults[0]
    horizons     = config.features.target_horizons_days

    typer.echo(
        f"evaluate-live-forecast | realm={target_realm} | "
        f"horizons={horizons} | window={window_days}d"
    )

    summaries = []
    with get_connection(
        target_db,
        wal_mode=config.database.wal_mode,
        busy_timeout_ms=config.database.busy_timeout_ms,
    ) as conn:
        for h in horizons:
            s = compute_health_summary(
                conn=conn,
                realm_slug=target_realm,
                horizon_days=h,
                window_days=window_days,
            )
            summaries.append(s)
            persist_health_to_db(conn, run_id=0, summary=s)

    # ── Print table ───────────────────────────────────────────────────────────
    typer.echo("")
    typer.echo("=== Live Forecast Health ===")
    typer.echo(f"  Realm: {target_realm}")
    typer.echo("")
    header = f"  {'Horizon':>8}  {'Status':>10}  {'N':>5}  {'LiveMAE':>9}  {'BaseMAE':>9}  {'Ratio':>7}  {'DirAcc':>7}"
    typer.echo(header)
    typer.echo("  " + "-" * (len(header) - 2))
    for s in summaries:
        live_mae_s  = f"{s.live_mae:.2f}g"  if s.live_mae  else "N/A"
        base_mae_s  = f"{s.baseline_mae:.2f}g" if s.baseline_mae else "N/A"
        ratio_s     = f"{s.mae_ratio:.2f}x"  if s.mae_ratio else "N/A"
        dir_acc_s   = f"{s.live_dir_acc:.1%}" if s.live_dir_acc else "N/A"
        typer.echo(
            f"  {s.horizon_days:>6}d  {s.health_status:>10}  {s.n_evaluated:>5}  "
            f"{live_mae_s:>9}  {base_mae_s:>9}  {ratio_s:>7}  {dir_acc_s:>7}"
        )

    if output_json and summaries:
        from pathlib import Path
        out_dir = Path(config.monitoring.monitoring_output_dir)
        p = write_health_report(summaries, out_dir, target_realm)
        typer.echo("")
        typer.echo(f"  Health report written: {p}")

    typer.echo("")
    typer.echo("[OK] Live forecast evaluation complete.")


@app.command("check-drift")
def check_drift(
    realm: Optional[str] = typer.Option(
        None,
        "--realm",
        help="Realm slug (e.g. area-52). Uses first config default if omitted.",
    ),
    db_path: Optional[str] = typer.Option(
        None,
        "--db-path",
        help="Override DB path from config.",
    ),
    output_json: bool = typer.Option(
        True,
        "--output-json/--no-output-json",
        help="Write drift_status_{realm}_{date}.json (default: on).",
    ),
    config_path: Optional[str] = typer.Option(
        None,
        "--config",
        help="Path to TOML config file.",
    ),
) -> None:
    """Check for market drift and model degradation.

    \b
    Runs three drift checks and prints a summary table:

    \b
      DATA DRIFT — Compares price_gold distribution (last 25h) vs 30-day
                   baseline per archetype series.  Flags series where mean
                   shifted by more than 2 standard deviations.

    \b
      ERROR DRIFT — Compares live forecast MAE (last 7 days) vs baseline
                    MAE from the most recent backtest run.

    \b
      EVENT SHOCK — Detects active or upcoming major WoW events from the
                    database (detection hook, not price attribution).

    \b
    Drift levels and uncertainty multipliers:
      none     -> x1.0  (no adjustment)
      low      -> x1.25
      medium   -> x1.5  + retrain recommended
      high     -> x2.0  + retrain recommended
      critical -> x3.0  + retrain recommended

    \b
    The latest uncertainty_mult is read by RecommendStage to widen CIs.
    """
    from pathlib import Path

    from wow_forecaster.db.connection import get_connection
    from wow_forecaster.monitoring.drift import DriftChecker
    from wow_forecaster.monitoring.reporter import (
        persist_drift_to_db,
        write_drift_report,
    )

    config = _load_config_or_exit(config_path)
    _configure_logging(config)

    target_db    = db_path or config.database.db_path
    target_realm = realm or config.realms.defaults[0]
    mc           = config.monitoring

    typer.echo(f"check-drift | realm={target_realm} | db={target_db}")

    with get_connection(
        target_db,
        wal_mode=config.database.wal_mode,
        busy_timeout_ms=config.database.busy_timeout_ms,
    ) as conn:
        checker = DriftChecker(
            conn=conn,
            drift_window_hours=mc.drift_window_hours,
            baseline_days=mc.drift_baseline_days,
            z_threshold=mc.drift_z_threshold,
            error_window_days=mc.error_drift_window_days,
            mae_thresholds=(
                mc.error_drift_mae_ratio_low,
                mc.error_drift_mae_ratio_medium,
                mc.error_drift_mae_ratio_high,
                mc.error_drift_mae_ratio_critical,
            ),
            shock_window_days=mc.event_shock_window_days,
        )
        result = checker.run_all(target_realm)
        persist_drift_to_db(conn, run_id=0, result=result)

    # ── Print table ───────────────────────────────────────────────────────────
    typer.echo("")
    typer.echo("=== Drift Check Report ===")
    typer.echo(f"  Realm:         {target_realm}")
    typer.echo(f"  Checked at:    {result.checked_at}")
    typer.echo("")
    typer.echo("  Data drift:")
    typer.echo(f"    Level:       {result.data_drift.drift_level.value}")
    typer.echo(
        f"    Series:      {result.data_drift.n_series_drifted} / "
        f"{result.data_drift.n_series_checked} drifted "
        f"({result.data_drift.drift_fraction:.0%})"
    )
    typer.echo(
        f"    Window:      recent {result.data_drift.window_hours}h vs "
        f"baseline {result.data_drift.baseline_days}d"
    )
    typer.echo("")
    typer.echo("  Error drift (horizon=1d):")
    typer.echo(f"    Level:       {result.error_drift.drift_level.value}")
    typer.echo(f"    Evaluated:   {result.error_drift.n_evaluated} forecast-vs-actual pairs")
    if result.error_drift.live_mae is not None:
        typer.echo(f"    Live MAE:    {result.error_drift.live_mae:.2f}g")
    if result.error_drift.baseline_mae is not None:
        typer.echo(f"    Baseline MAE:{result.error_drift.baseline_mae:.2f}g")
    if result.error_drift.mae_ratio is not None:
        typer.echo(f"    MAE ratio:   {result.error_drift.mae_ratio:.2f}x baseline")
    typer.echo("")
    typer.echo("  Event shock:")
    typer.echo(
        f"    Active:      {len(result.event_shock.active_events)} event(s)"
    )
    typer.echo(
        f"    Upcoming:    {len(result.event_shock.upcoming_events)} event(s) "
        f"within {mc.event_shock_window_days}d"
    )
    typer.echo(f"    Shock flag:  {'YES' if result.event_shock.shock_active else 'no'}")
    typer.echo("")
    typer.echo("  Overall:")
    typer.echo(f"    Drift level:         {result.overall_drift_level.value}")
    typer.echo(f"    Uncertainty mult:    x{result.uncertainty_multiplier:.2f}")
    typer.echo(
        f"    Retrain recommended: {'YES' if result.retrain_recommended else 'no'}"
    )

    if output_json:
        out_dir = Path(mc.monitoring_output_dir)
        p = write_drift_report(result, out_dir)
        typer.echo("")
        typer.echo(f"  Drift report written: {p}")

    typer.echo("")
    typer.echo("[OK] Drift check complete.")


# ── Report commands ───────────────────────────────────────────────────────────
#
# These commands READ already-persisted output files and display them in
# human-readable form.  They do NOT re-run the pipeline.
#
# File discovery: each command looks for the most-recently-modified file
# matching the expected pattern in the configured output directory.  If no
# file is found it prints a "no data yet" message and exits cleanly.
#
# Freshness banners: every report shows [FRESH] / [STALE] with the hours
# since the report was written so readers can judge whether to trust the
# output without needing to check file timestamps manually.


@app.command("report-top-items")
def report_top_items_cmd(
    realm: Optional[str] = typer.Option(
        None,
        "--realm",
        help="Realm slug (e.g. area-52). Uses first config default if omitted.",
    ),
    horizon: Optional[str] = typer.Option(
        None,
        "--horizon",
        help="Filter to a single horizon (e.g. 1d, 7d, 28d). Shows all if omitted.",
    ),
    export: Optional[str] = typer.Option(
        None,
        "--export",
        help=(
            "Write a flat CSV export to this path (for Power BI / manual analysis). "
            "E.g. --export data/exports/top_items.csv"
        ),
    ),
    freshness_hours: float = typer.Option(
        4.0,
        "--freshness-hours",
        help="Reports older than this many hours are flagged as [STALE].",
    ),
    config_path: Optional[str] = typer.Option(
        None,
        "--config",
        help="Path to TOML config file.",
    ),
) -> None:
    """Show the current top recommendations per category.

    \b
    Reads the most recent recommendations_{realm}_{date}.json from
    data/outputs/recommendations/ and prints a ranked table grouped
    by archetype category.

    \b
    Columns: rank, archetype, horizon, current price, predicted price,
             ROI, composite score, recommended action.

    \b
    Freshness: reports older than --freshness-hours are flagged [STALE].
    Run 'run-daily-forecast' to refresh the underlying data.

    \b
    Use --export PATH to write a flat CSV for Power BI or Excel.
    Each row includes all score components as separate columns.
    """
    from pathlib import Path as _Path

    from wow_forecaster.reporting.export import (
        export_to_csv,
        flatten_recommendations_for_export,
    )
    from wow_forecaster.reporting.formatters import format_top_items_table
    from wow_forecaster.reporting.reader import (
        check_freshness,
        load_recommendations_report,
    )

    config = _load_config_or_exit(config_path)
    _configure_logging(config)

    target_realm = realm or config.realms.defaults[0]
    out_dir      = _Path(config.model.recommendation_output_dir)

    recs = load_recommendations_report(target_realm, out_dir)
    if recs is None:
        typer.echo(
            f"[INFO] No recommendations found for realm={target_realm} in {out_dir}"
        )
        typer.echo("       Run 'wow-forecaster run-daily-forecast' first.")
        raise typer.Exit(code=0)

    generated_at = recs.get("generated_at", "")
    is_fresh, age_hours = check_freshness(generated_at, freshness_hours)

    categories = recs.get("categories", {})

    # Optional horizon filter: keep only items whose horizon matches.
    if horizon:
        categories = {
            cat: [i for i in items if i.get("horizon") == horizon]
            for cat, items in categories.items()
        }
        categories = {cat: items for cat, items in categories.items() if items}

    typer.echo(
        format_top_items_table(
            categories=categories,
            realm=target_realm,
            generated_at=generated_at,
            is_fresh=is_fresh,
            age_hours=age_hours,
        )
    )

    if export:
        rows = flatten_recommendations_for_export(recs)
        if horizon:
            rows = [r for r in rows if r.get("horizon") == horizon]
        p = export_to_csv(rows, _Path(export))
        typer.echo(f"\n  Exported {len(rows)} rows to: {p}")

    typer.echo("")
    typer.echo("[OK] report-top-items complete.")


@app.command("report-forecasts")
def report_forecasts_cmd(
    realm: Optional[str] = typer.Option(
        None,
        "--realm",
        help="Realm slug. Uses first config default if omitted.",
    ),
    horizon: Optional[str] = typer.Option(
        None,
        "--horizon",
        help="Filter to a single horizon (e.g. 1d, 7d, 28d).",
    ),
    top_n: int = typer.Option(
        15,
        "--top-n",
        help="How many rows to print (sorted by score descending).",
    ),
    export: Optional[str] = typer.Option(
        None,
        "--export",
        help="Write full forecast CSV (with ci_width_gold column) to this path.",
    ),
    freshness_hours: float = typer.Option(
        4.0,
        "--freshness-hours",
        help="Reports older than this many hours are flagged [STALE].",
    ),
    config_path: Optional[str] = typer.Option(
        None,
        "--config",
        help="Path to TOML config file.",
    ),
) -> None:
    """Print a forecast summary sorted by composite score.

    \b
    Reads the most recent forecast_{realm}_{date}.csv from
    data/outputs/forecasts/ and prints the top-N rows (all horizons unless
    --horizon is specified).

    \b
    Columns: archetype, horizon, current price, predicted price,
             CI width (forecast uncertainty), ROI, score, action.

    \b
    The CI width column shows absolute uncertainty in gold — a wide CI
    means the model is less certain about the prediction.

    \b
    Use --export PATH to write the full forecast set (with computed
    ci_width_gold and ci_pct_of_price columns) to a CSV file.
    """
    from pathlib import Path as _Path

    from wow_forecaster.reporting.export import (
        export_to_csv,
        flatten_forecast_records_for_export,
    )
    from wow_forecaster.reporting.formatters import format_forecast_summary
    from wow_forecaster.reporting.reader import check_freshness, load_forecast_records

    config = _load_config_or_exit(config_path)
    _configure_logging(config)

    target_realm = realm or config.realms.defaults[0]
    out_dir      = _Path(config.model.forecast_output_dir)

    records = load_forecast_records(target_realm, out_dir)
    if records is None:
        typer.echo(
            f"[INFO] No forecast CSV found for realm={target_realm} in {out_dir}"
        )
        typer.echo("       Run 'wow-forecaster run-daily-forecast' first.")
        raise typer.Exit(code=0)

    # Freshness: use file mtime as proxy (CSV has no embedded timestamp).
    from wow_forecaster.reporting.reader import find_latest_file
    csv_path = find_latest_file(out_dir, f"forecast_{target_realm}_*.csv")
    age_hours: Optional[float] = None
    is_fresh  = True
    if csv_path is not None:
        import os as _os
        mtime     = _os.path.getmtime(csv_path)
        import time as _time
        age_hours = (_time.time() - mtime) / 3600.0
        is_fresh  = age_hours <= freshness_hours

    typer.echo(
        format_forecast_summary(
            records=records,
            realm=target_realm,
            top_n=top_n,
            horizon_filter=horizon,
            is_fresh=is_fresh,
            age_hours=age_hours,
        )
    )

    if export:
        enriched = flatten_forecast_records_for_export(records)
        if horizon:
            enriched = [r for r in enriched if r.get("horizon") == horizon]
        p = export_to_csv(enriched, _Path(export))
        typer.echo(f"\n  Exported {len(enriched)} rows to: {p}")

    typer.echo("")
    typer.echo("[OK] report-forecasts complete.")


@app.command("report-volatility")
def report_volatility_cmd(
    realm: Optional[str] = typer.Option(
        None,
        "--realm",
        help="Realm slug. Uses first config default if omitted.",
    ),
    top_n: int = typer.Option(
        20,
        "--top-n",
        help="How many items to show (default 20).",
    ),
    export: Optional[str] = typer.Option(
        None,
        "--export",
        help="Write watchlist CSV to this path.",
    ),
    freshness_hours: float = typer.Option(
        4.0,
        "--freshness-hours",
        help="Reports older than this many hours are flagged [STALE].",
    ),
    config_path: Optional[str] = typer.Option(
        None,
        "--config",
        help="Path to TOML config file.",
    ),
) -> None:
    """Show the volatility watchlist: items with the widest forecast CI bands.

    \b
    A wide confidence interval means the model is uncertain — the actual
    price could deviate significantly from the prediction.  These items
    carry the most risk regardless of their predicted ROI.

    \b
    Items are ranked by absolute CI width (ci_upper - ci_lower) in gold.
    A relative CI % column normalises by predicted price so high-value and
    low-value items can be compared fairly.

    \b
    Reads forecast_{realm}_{date}.csv from data/outputs/forecasts/.
    Use --export PATH to write the sorted watchlist to a CSV file.
    """
    from pathlib import Path as _Path
    import os as _os
    import time as _time

    from wow_forecaster.reporting.export import export_to_csv
    from wow_forecaster.reporting.formatters import format_volatility_watchlist
    from wow_forecaster.reporting.reader import find_latest_file, load_forecast_records

    config = _load_config_or_exit(config_path)
    _configure_logging(config)

    target_realm = realm or config.realms.defaults[0]
    out_dir      = _Path(config.model.forecast_output_dir)

    records = load_forecast_records(target_realm, out_dir)
    if records is None:
        typer.echo(
            f"[INFO] No forecast CSV found for realm={target_realm} in {out_dir}"
        )
        typer.echo("       Run 'wow-forecaster run-daily-forecast' first.")
        raise typer.Exit(code=0)

    csv_path = find_latest_file(out_dir, f"forecast_{target_realm}_*.csv")
    age_hours_: Optional[float] = None
    is_fresh_  = True
    if csv_path is not None:
        age_hours_ = (_time.time() - _os.path.getmtime(csv_path)) / 3600.0
        is_fresh_  = age_hours_ <= freshness_hours

    typer.echo(
        format_volatility_watchlist(
            records=records,
            realm=target_realm,
            top_n=top_n,
            is_fresh=is_fresh_,
            age_hours=age_hours_,
        )
    )

    if export:
        # Enrich with ci_width / ci_pct and sort before export.
        from wow_forecaster.reporting.export import flatten_forecast_records_for_export
        enriched = flatten_forecast_records_for_export(records)
        try:
            enriched.sort(
                key=lambda r: -(float(r.get("ci_width_gold") or 0))
            )
        except (TypeError, ValueError):
            pass
        p = export_to_csv(enriched, _Path(export))
        typer.echo(f"\n  Exported {len(enriched)} rows to: {p}")

    typer.echo("")
    typer.echo("[OK] report-volatility complete.")


@app.command("report-drift")
def report_drift_cmd(
    realm: Optional[str] = typer.Option(
        None,
        "--realm",
        help="Realm slug. Uses first config default if omitted.",
    ),
    export: Optional[str] = typer.Option(
        None,
        "--export",
        help="Write a combined drift+health JSON export to this path.",
    ),
    freshness_hours: float = typer.Option(
        4.0,
        "--freshness-hours",
        help="Reports older than this many hours are flagged [STALE].",
    ),
    config_path: Optional[str] = typer.Option(
        None,
        "--config",
        help="Path to TOML config file.",
    ),
) -> None:
    """Show drift level, model health, and retrain recommendation.

    \b
    Reads the two most recent monitoring reports:
      drift_status_{realm}_{date}.json   -- data/error drift, event shock
      model_health_{realm}_{date}.json   -- live MAE vs backtest baseline

    \b
    Key columns in model health table:
      Ratio  = live_MAE / baseline_MAE
        < 1.5x  -> ok
        1.5-3x  -> degraded
        >= 3x   -> critical (retrain recommended)

    \b
    The uncertainty multiplier from the drift report is what actually widens
    confidence intervals in live forecasts.  A x2.0 multiplier means CI bands
    are doubled; a x3.0 means the model is highly uncertain.

    \b
    Run 'check-drift' to refresh the drift report.
    Run 'evaluate-live-forecast' to refresh the model health report.
    """
    from pathlib import Path as _Path

    from wow_forecaster.reporting.export import export_to_json
    from wow_forecaster.reporting.formatters import format_drift_health_summary
    from wow_forecaster.reporting.reader import (
        check_freshness,
        load_drift_report,
        load_health_report,
    )

    config = _load_config_or_exit(config_path)
    _configure_logging(config)

    target_realm = realm or config.realms.defaults[0]
    mon_dir      = _Path(config.monitoring.monitoring_output_dir)

    drift  = load_drift_report(target_realm, mon_dir)
    health = load_health_report(target_realm, mon_dir)

    is_fresh_d, age_d = check_freshness(
        drift.get("checked_at")  if drift  else None, freshness_hours
    )
    is_fresh_h, age_h = check_freshness(
        health.get("checked_at") if health else None, freshness_hours
    )

    typer.echo(
        format_drift_health_summary(
            drift=drift,
            health=health,
            realm=target_realm,
            is_fresh_drift=is_fresh_d,
            age_hours_drift=age_d,
            is_fresh_health=is_fresh_h,
            age_hours_health=age_h,
        )
    )

    if export:
        payload = {
            "drift":  drift  or {},
            "health": health or {},
        }
        p = export_to_json(payload, _Path(export))
        typer.echo(f"\n  Exported drift+health to: {p}")

    typer.echo("")
    typer.echo("[OK] report-drift complete.")


@app.command("report-status")
def report_status_cmd(
    realm: Optional[str] = typer.Option(
        None,
        "--realm",
        help="Realm slug. Uses first config default if omitted.",
    ),
    export: Optional[str] = typer.Option(
        None,
        "--export",
        help="Write provenance JSON export to this path.",
    ),
    freshness_hours: float = typer.Option(
        4.0,
        "--freshness-hours",
        help="Reports older than this many hours are flagged [STALE].",
    ),
    config_path: Optional[str] = typer.Option(
        None,
        "--config",
        help="Path to TOML config file.",
    ),
) -> None:
    """Show last-refresh timestamps and data source health.

    \b
    Reads the most recent provenance_{realm}_{date}.json from
    data/outputs/monitoring/ and prints a per-source summary.

    \b
    IMPORTANT: Report age != data freshness.
      A provenance file written 5 minutes ago can still report stale data
      if the Undermine or Blizzard API did not respond during that run.
      This command surfaces both values explicitly.

    \b
    Per-source columns:
      Source        -- undermine / blizzard_api / blizzard_news
      Last Snapshot -- timestamp of the most recent ingested snapshot
      Snaps/24h     -- number of snapshots in the last 24 hours
      Records       -- total market records ingested in the last 24 hours
      SuccRate      -- fraction of ingestion attempts that succeeded
      Status        -- [OK] or [STALE]

    \b
    Run 'run-hourly-refresh' to produce a fresh provenance report.
    """
    from pathlib import Path as _Path

    from wow_forecaster.reporting.export import export_to_json
    from wow_forecaster.reporting.formatters import format_status_summary
    from wow_forecaster.reporting.reader import (
        check_freshness,
        load_provenance_report,
    )

    config = _load_config_or_exit(config_path)
    _configure_logging(config)

    target_realm = realm or config.realms.defaults[0]
    mon_dir      = _Path(config.monitoring.monitoring_output_dir)

    prov = load_provenance_report(target_realm, mon_dir)

    is_fresh_p, age_p = check_freshness(
        prov.get("checked_at") if prov else None, freshness_hours
    )

    typer.echo(
        format_status_summary(
            provenance=prov,
            realm=target_realm,
            is_fresh_prov=is_fresh_p,
            age_hours_prov=age_p,
        )
    )

    if export and prov:
        p = export_to_json(prov, _Path(export))
        typer.echo(f"\n  Exported provenance to: {p}")

    typer.echo("")
    typer.echo("[OK] report-status complete.")


# ── Governance commands ───────────────────────────────────────────────────────


@app.command("list-sources")
def list_sources_cmd(
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Print full detail for each source (rate limits, backoff, retention, policy notes).",
    ),
    config_path: Optional[str] = typer.Option(
        None,
        "--config",
        help="Path to TOML config file.",
    ),
) -> None:
    """List all registered data sources and their enabled/disabled status.

    \b
    Reads config/sources.toml (or the path set in governance.sources_config_path)
    and prints a summary table of every declared source.

    \b
    Columns:
      Status       -- [ENABLED] or [disabled]
      Source ID    -- unique identifier used throughout the pipeline
      Display Name -- human-readable name
      Type         -- auction_data | news_event | manual_event
      Access       -- api | export | manual
      Auth         -- whether credentials are required
      TTL(h)       -- hours before data is considered aging

    \b
    Use --verbose to see full policy detail per source (rate limits, backoff,
    freshness thresholds, retention, provenance requirements, and policy notes).

    \b
    IMPORTANT: The "policy_notes" section contains researcher-authored
    informational reminders only. It does NOT constitute legal advice.
    """
    from wow_forecaster.governance.registry import list_sources
    from wow_forecaster.governance.reporter import format_source_detail, format_source_table

    config = _load_config_or_exit(config_path)
    _configure_logging(config)

    sources_path = config.governance.sources_config_path

    try:
        policies = list_sources(sources_path)
    except FileNotFoundError as exc:
        typer.echo(f"[ERROR] {exc}", err=True)
        raise typer.Exit(code=1)

    n_enabled  = sum(1 for p in policies if p.enabled)
    n_disabled = len(policies) - n_enabled

    typer.echo(f"\n  Source Registry  ({len(policies)} sources, {n_enabled} enabled, {n_disabled} disabled)")
    typer.echo(f"  Config: {sources_path}")
    typer.echo("")

    if verbose:
        for p in policies:
            typer.echo(format_source_detail(p))
            typer.echo("")
    else:
        typer.echo(format_source_table(policies))
        typer.echo("  Use --verbose for full rate-limit, backoff, and policy details.")

    typer.echo("[OK] list-sources complete.")


@app.command("validate-source-policies")
def validate_source_policies_cmd(
    config_path: Optional[str] = typer.Option(
        None,
        "--config",
        help="Path to TOML config file.",
    ),
) -> None:
    """Validate all source policies in sources.toml.

    \b
    Parses every [sources.*] block in config/sources.toml and runs Pydantic
    validation on each one.  Reports which sources pass and which fail.

    \b
    Validation checks include:
      - All required fields are present
      - Rate limit values are non-negative integers
      - Backoff strategy is "exponential", "linear", or "fixed"
      - Freshness thresholds are non-decreasing (ttl <= stale <= critical)
      - source_type and access_method are recognised values
      - PolicyNotes access_type is recognised

    \b
    Exit codes:
      0 -- all policies valid
      1 -- one or more policies failed validation (details printed to stdout)

    \b
    Run this after editing config/sources.toml to catch mistakes early.
    """
    import tomllib as _tomllib
    from pathlib import Path as _Path

    from pydantic import ValidationError

    from wow_forecaster.governance.models import (
        BackoffConfig,
        FreshnessConfig,
        PolicyNotes,
        ProvenanceRequirements,
        RateLimitConfig,
        RetentionConfig,
        SourcePolicy,
    )
    from wow_forecaster.governance.reporter import format_validation_report

    config = _load_config_or_exit(config_path)
    _configure_logging(config)

    sources_path = _Path(config.governance.sources_config_path)
    typer.echo(f"\n  Validating source policies in: {sources_path}\n")

    if not sources_path.exists():
        typer.echo(
            f"[ERROR] Sources config not found: {sources_path}\n"
            "Expected at config/sources.toml.",
            err=True,
        )
        raise typer.Exit(code=1)

    with open(sources_path, "rb") as f:
        raw = _tomllib.load(f)

    sources_raw = raw.get("sources", {})
    if not sources_raw:
        typer.echo("  [WARNING] No [sources.*] blocks found in sources.toml.\n")
        raise typer.Exit(code=0)

    errors: dict[str, list[str]] = {}
    policies = []

    for sid, block in sources_raw.items():
        try:
            policy = SourcePolicy(
                source_id=block.get("source_id", sid),
                display_name=block.get("display_name", sid),
                source_type=block.get("source_type", "other"),
                access_method=block.get("access_method", "manual"),
                requires_auth=block.get("requires_auth", False),
                enabled=block.get("enabled", False),
                rate_limit=RateLimitConfig(**block.get("rate_limit", {})),
                backoff=BackoffConfig(**block.get("backoff", {})),
                freshness=FreshnessConfig(**block.get("freshness", {})),
                provenance=ProvenanceRequirements(**block.get("provenance", {})),
                retention=RetentionConfig(**block.get("retention", {})),
                policy_notes=PolicyNotes(**block.get("policy_notes", {})),
            )
            policies.append(policy)
        except (ValidationError, TypeError) as exc:
            errors[sid] = [str(e) for e in (exc.errors() if isinstance(exc, ValidationError) else [exc])]

    typer.echo(format_validation_report(policies, errors))

    if errors:
        typer.echo("[FAIL] validate-source-policies: one or more policies are invalid.")
        raise typer.Exit(code=1)

    typer.echo("[OK] validate-source-policies complete.")


@app.command("check-source-freshness")
def check_source_freshness_cmd(
    realm: Optional[str] = typer.Option(
        None,
        "--realm",
        help="Filter freshness check to one realm slug.  Omit to check across all realms.",
    ),
    export: Optional[str] = typer.Option(
        None,
        "--export",
        help="Write a governance JSON report to this directory path.",
    ),
    db_path: Optional[str] = typer.Option(
        None,
        "--db-path",
        help="Override DB path from config.",
    ),
    config_path: Optional[str] = typer.Option(
        None,
        "--config",
        help="Path to TOML config file.",
    ),
) -> None:
    """Check freshness of each registered data source against its TTL policy.

    \b
    Queries ingestion_snapshots to find the most recent successful snapshot
    per source, then classifies each one against the FreshnessConfig thresholds
    declared in config/sources.toml.

    \b
    Status codes:
      [FRESH]    -- age < ttl_hours
      [AGING]    -- ttl_hours <= age < stale_threshold_hours
      [STALE]    -- stale_threshold_hours <= age < critical_threshold_hours
      [CRITICAL] -- age >= critical_threshold_hours
      [UNKNOWN]  -- no snapshot found, or source does not require snapshots

    \b
    Manual sources (access_method=manual, requires_snapshot=false) always
    report [UNKNOWN] — they have no ingestion_snapshots records.

    \b
    Use --export <dir> to write a JSON governance report file.
    Use --realm <slug> to restrict the snapshot query to one realm.

    \b
    Run 'run-hourly-refresh' to generate fresh ingestion snapshots.
    """
    from pathlib import Path as _Path

    from wow_forecaster.db.connection import get_connection
    from wow_forecaster.governance.freshness import check_all_sources_freshness
    from wow_forecaster.governance.registry import list_sources
    from wow_forecaster.governance.reporter import format_freshness_table, write_governance_report

    config = _load_config_or_exit(config_path)
    _configure_logging(config)

    sources_path = config.governance.sources_config_path
    target_db    = db_path or config.database.db_path

    try:
        policies = list_sources(sources_path)
    except FileNotFoundError as exc:
        typer.echo(f"[ERROR] {exc}", err=True)
        raise typer.Exit(code=1)

    typer.echo(f"\n  Source Freshness Check")
    typer.echo(f"  Sources config : {sources_path}")
    typer.echo(f"  DB             : {target_db}")
    if realm:
        typer.echo(f"  Realm filter   : {realm}")
    typer.echo("")

    with get_connection(
        target_db,
        wal_mode=config.database.wal_mode,
        busy_timeout_ms=config.database.busy_timeout_ms,
    ) as conn:
        results = check_all_sources_freshness(conn, policies, realm_slug=realm)

    typer.echo(format_freshness_table(results, realm_slug=realm))

    # Summary counts
    from wow_forecaster.governance.freshness import FreshnessStatus
    n_fresh    = sum(1 for r in results if r.status == FreshnessStatus.FRESH)
    n_aging    = sum(1 for r in results if r.status == FreshnessStatus.AGING)
    n_stale    = sum(1 for r in results if r.status == FreshnessStatus.STALE)
    n_critical = sum(1 for r in results if r.status == FreshnessStatus.CRITICAL)
    n_unknown  = sum(1 for r in results if r.status == FreshnessStatus.UNKNOWN)

    typer.echo(
        f"  Summary: {n_fresh} fresh, {n_aging} aging, {n_stale} stale, "
        f"{n_critical} critical, {n_unknown} unknown"
    )

    if export:
        out = write_governance_report(policies, results, export)
        typer.echo(f"\n  Exported governance report to: {out}")

    typer.echo("")
    typer.echo("[OK] check-source-freshness complete.")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app()
