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
        help="Realm slug to refresh (e.g. area-52). Repeatable; uses config defaults if omitted.",
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
    config_path: Optional[str] = typer.Option(
        None,
        "--config",
        help="Path to TOML config file.",
    ),
) -> None:
    """Run ingest + normalize pipeline for one or more realms.

    \b
    Steps:
      1. IngestStage  — fetch AH snapshots (fixture mode until API keys are set).
                        Writes JSON to data/raw/snapshots/ and records metadata
                        in the ingestion_snapshots SQLite table.
      2. NormalizeStage — convert copper→gold, flag outliers, write normalized obs.

    \b
    Credential setup (.env, gitignored):
      UNDERMINE_API_KEY=...          → enables real Undermine data
      BLIZZARD_CLIENT_ID=...         → enables real Blizzard AH data
      BLIZZARD_CLIENT_SECRET=...

    Without credentials the pipeline runs in fixture mode (synthetic sample data).
    """
    from wow_forecaster.db.connection import get_connection
    from wow_forecaster.db.schema import apply_schema
    from wow_forecaster.pipeline.ingest import IngestStage
    from wow_forecaster.pipeline.normalize import NormalizeStage

    config = _load_config_or_exit(config_path)
    _configure_logging(config)

    target_db = db_path or config.database.db_path
    target_realms = [realm] if realm else list(config.realms.defaults)

    typer.echo(
        f"run-hourly-refresh | realms={', '.join(target_realms)} | db={target_db}"
    )

    if dry_run:
        typer.echo("[DRY RUN] Would run:")
        typer.echo(f"  IngestStage   → realms={target_realms}")
        typer.echo("  NormalizeStage → all unprocessed raw observations")
        return

    # Ensure schema exists (idempotent)
    with get_connection(
        target_db,
        wal_mode=config.database.wal_mode,
        busy_timeout_ms=config.database.busy_timeout_ms,
    ) as conn:
        apply_schema(conn)

    # ── IngestStage ───────────────────────────────────────────────────────────
    typer.echo("  [1/2] IngestStage ...")
    ingest_ok = False
    try:
        ingest = IngestStage(config=config, db_path=target_db)
        ingest_run = ingest.run(realm_slugs=target_realms)
        typer.echo(
            f"        status={ingest_run.status} | "
            f"snapshots written (market_obs_raw: {ingest_run.rows_processed} rows)"
        )
        ingest_ok = ingest_run.status == "success"
    except Exception as exc:
        typer.echo(f"        FAILED: {exc}", err=True)

    # ── NormalizeStage ────────────────────────────────────────────────────────
    typer.echo("  [2/2] NormalizeStage ...")
    try:
        norm = NormalizeStage(config=config, db_path=target_db)
        norm_run = norm.run()
        typer.echo(
            f"        status={norm_run.status} | "
            f"normalized={norm_run.rows_processed} rows"
        )
    except Exception as exc:
        typer.echo(f"        FAILED: {exc}", err=True)

    typer.echo("")
    if ingest_ok:
        typer.echo("[OK] Hourly refresh complete.")
    else:
        typer.echo(
            "[OK] Hourly refresh complete (fixture mode — set API keys in .env for live data)."
        )


@app.command("run-daily-forecast")
def run_daily_forecast(
    realm: Optional[str] = typer.Option(
        None,
        "--realm",
        help="Realm slug to forecast (e.g. area-52). Uses config defaults if omitted.",
    ),
    horizon: str = typer.Option(
        "7d",
        "--horizon",
        help="Forecast horizon (1d, 7d, 14d, 30d, 90d).",
    ),
    config_path: Optional[str] = typer.Option(
        None,
        "--config",
        help="Path to TOML config file.",
    ),
) -> None:
    """[STUB] Run feature_build + train + forecast + recommend pipeline.

    This command is a placeholder. When implemented it will:
      1. FeatureBuildStage: engineer features from normalized observations.
      2. TrainStage: update models with latest data.
      3. ForecastStage: generate price forecasts for configured horizons.
      4. RecommendStage: produce ranked buy/sell/hold/avoid recommendations.
    """
    config = _load_config_or_exit(config_path)
    _configure_logging(config)

    target_realm = realm or config.realms.defaults[0]

    typer.echo(f"[STUB] run-daily-forecast | realm={target_realm} | horizon={horizon}")
    typer.echo("  FeatureBuildStage → not yet implemented")
    typer.echo("  TrainStage        → not yet implemented")
    typer.echo("  ForecastStage     → not yet implemented")
    typer.echo("  RecommendStage    → not yet implemented")
    typer.echo("")
    typer.echo("This command is a scaffold stub. Implement pipeline stages first.")


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
        help="Walk-forward window size in days. Uses config default if omitted.",
    ),
    config_path: Optional[str] = typer.Option(
        None,
        "--config",
        help="Path to TOML config file.",
    ),
) -> None:
    """[STUB] Run walk-forward backtest over historical TWW data.

    This command is a placeholder. When implemented it will:
      1. Iterate over (start_date, end_date) in walk-forward windows.
      2. For each window: build features, train, forecast, evaluate.
      3. Respect WoWEvent.is_known_at() to prevent look-ahead bias.
      4. Compute MAE, RMSE, directional accuracy per archetype.
      5. Write evaluation results to outputs/backtest/.
    """
    config = _load_config_or_exit(config_path)
    _configure_logging(config)

    # Validate date format
    try:
        start = date.fromisoformat(start_date)
        end = date.fromisoformat(end_date)
    except ValueError as exc:
        typer.echo(f"[ERROR] Invalid date format: {exc}", err=True)
        raise typer.Exit(code=1)

    if end <= start:
        typer.echo("[ERROR] --end-date must be after --start-date.", err=True)
        raise typer.Exit(code=1)

    target_realm = realm or config.realms.defaults[0]
    win = window_days or config.backtest.window_days

    typer.echo(
        f"[STUB] backtest | realm={target_realm} | "
        f"{start} → {end} | window={win}d"
    )
    typer.echo("")
    typer.echo("This command is a scaffold stub. Implement pipeline stages first.")


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
        f"{start} → {end} | db={target_db}"
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
    typer.echo(f"  Date range:       {manifest_data.get('date_range', {}).get('start')} → {manifest_data.get('date_range', {}).get('end')}")
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


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app()
