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
import sys
from datetime import date
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
        help="Path to events JSON file. Defaults to config.data.events_seed_file.",
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
    """Import WoW events from a JSON seed file into the database.

    Uses UPSERT semantics — existing events with the same slug are updated.
    The seed file format is a JSON array of event objects. See
    config/events/tww_events.json for the schema.
    """
    from wow_forecaster.db.connection import get_connection
    from wow_forecaster.db.repositories.event_repo import WoWEventRepository
    from wow_forecaster.models.event import WoWEvent
    from pydantic import ValidationError

    config = _load_config_or_exit(config_path)
    _configure_logging(config)

    # Resolve events file path
    if events_file:
        events_path = Path(events_file)
    else:
        events_path = Path(config.data.events_seed_file)

    if not events_path.exists():
        typer.echo(f"[ERROR] Events file not found: {events_path}", err=True)
        raise typer.Exit(code=1)

    typer.echo(f"Loading events from: {events_path}")

    with open(events_path, encoding="utf-8") as f:
        raw_events = json.load(f)

    if not isinstance(raw_events, list):
        typer.echo("[ERROR] Events file must contain a JSON array.", err=True)
        raise typer.Exit(code=1)

    # Validate all events first
    validated: list[WoWEvent] = []
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

    typer.echo(f"  Validated {len(validated)} event(s).")

    if dry_run:
        typer.echo("[DRY RUN] No events written to database.")
        for ev in validated:
            typer.echo(f"  {ev.slug} | {ev.event_type} | {ev.start_date}")
        return

    # Write to DB
    with get_connection(
        config.database.db_path,
        wal_mode=config.database.wal_mode,
        busy_timeout_ms=config.database.busy_timeout_ms,
    ) as conn:
        repo = WoWEventRepository(conn)
        upserted = 0
        for ev in validated:
            repo.upsert(ev)
            upserted += 1

    typer.echo(f"  Upserted {upserted} event(s) into database.")
    typer.echo("[OK] Events imported.")


@app.command("run-hourly-refresh")
def run_hourly_refresh(
    realm: Optional[str] = typer.Option(
        None,
        "--realm",
        help="Realm slug to refresh (e.g. area-52). Uses config defaults if omitted.",
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
    """[STUB] Run ingest + normalize pipeline for a realm.

    This command is a placeholder. When implemented it will:
      1. Call IngestStage to fetch new AH snapshots.
      2. Call NormalizeStage to convert and flag outliers.
      3. Write a RunMetadata record for the combined hourly refresh.
    """
    config = _load_config_or_exit(config_path)
    _configure_logging(config)

    target_realm = realm or config.realms.defaults[0]

    typer.echo(f"[STUB] run-hourly-refresh | realm={target_realm}")
    typer.echo("  IngestStage   → not yet implemented")
    typer.echo("  NormalizeStage → not yet implemented")
    typer.echo("")
    typer.echo("This command is a scaffold stub. Implement IngestStage and NormalizeStage first.")


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


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app()
