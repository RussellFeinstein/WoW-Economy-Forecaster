"""
CLI smoke tests for wow_forecaster/cli.py.

Strategy
--------
Smoke tests verify that:
  1. Every command exposes a parseable --help without crashing.
  2. Simple commands (validate-config, init-db) succeed end-to-end.
  3. Commands with --dry-run exit 0 and print the expected dry-run banner.
  4. Error paths (missing files, bad dates, missing credentials) exit with
     code 1 and do NOT raise unhandled exceptions.

We deliberately avoid:
  - Running live pipeline stages (IngestStage, ForecastStage, etc.).
  - Commands that require Blizzard API credentials unless we test the
    ``exit 1`` path when credentials are absent.

All tests use typer.testing.CliRunner, which invokes the Typer app
in-process without spawning a subprocess.
"""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from wow_forecaster.cli import app

runner = CliRunner()


# ── --help smoke ──────────────────────────────────────────────────────────────

class TestCLIHelp:
    """Every registered command must handle --help and exit 0."""

    @pytest.mark.parametrize(
        "command",
        [
            "init-db",
            "validate-config",
            "import-events",
            "import-auctionator",
            "bootstrap-items",
            "run-hourly-refresh",
            "train-model",
            "run-daily-forecast",
            "recommend-top-items",
            "backtest",
            "report-backtest",
            "build-events",
            "list-events",
            "build-datasets",
            "validate-datasets",
            "evaluate-live-forecast",
            "check-drift",
            "report-top-items",
            "report-forecasts",
            "report-volatility",
            "report-drift",
            "report-status",
            "list-sources",
            "validate-source-policies",
            "check-source-freshness",
            "start-scheduler",
        ],
    )
    def test_help_exits_zero(self, command):
        result = runner.invoke(app, [command, "--help"])
        assert result.exit_code == 0, (
            f"`{command} --help` exited {result.exit_code}:\n{result.output}"
        )
        assert "Usage:" in result.output


# ── validate-config ───────────────────────────────────────────────────────────

class TestValidateConfig:
    def test_exits_zero_with_default_config(self):
        result = runner.invoke(app, ["validate-config"])
        assert result.exit_code == 0
        assert "[OK]" in result.output

    def test_prints_database_path(self):
        result = runner.invoke(app, ["validate-config"])
        assert "Database path" in result.output

    def test_full_flag_prints_json(self):
        result = runner.invoke(app, ["validate-config", "--full"])
        assert result.exit_code == 0
        assert "database" in result.output


# ── init-db ───────────────────────────────────────────────────────────────────

class TestInitDb:
    def test_creates_db_file(self, tmp_path):
        db = str(tmp_path / "smoke.db")
        result = runner.invoke(app, ["init-db", "--db-path", db])
        assert result.exit_code == 0, result.output
        assert "[OK]" in result.output
        assert (tmp_path / "smoke.db").exists()

    def test_idempotent_second_run(self, tmp_path):
        """Running init-db twice on the same file must not raise."""
        db = str(tmp_path / "smoke.db")
        runner.invoke(app, ["init-db", "--db-path", db])
        result = runner.invoke(app, ["init-db", "--db-path", db])
        assert result.exit_code == 0


# ── dry-run commands ──────────────────────────────────────────────────────────

class TestDryRunCommands:
    def test_run_hourly_refresh_dry_run(self):
        result = runner.invoke(app, ["run-hourly-refresh", "--dry-run"])
        assert result.exit_code == 0
        assert "DRY RUN" in result.output

    def test_build_datasets_dry_run(self):
        result = runner.invoke(
            app,
            [
                "build-datasets",
                "--dry-run",
                "--start-date", "2025-01-01",
                "--end-date", "2025-01-31",
            ],
        )
        assert result.exit_code == 0
        assert "DRY RUN" in result.output

    def test_train_model_dry_run(self):
        result = runner.invoke(app, ["train-model", "--dry-run"])
        assert result.exit_code == 0
        assert "DRY RUN" in result.output

    def test_backtest_dry_run(self):
        result = runner.invoke(
            app,
            [
                "backtest",
                "--dry-run",
                "--start-date", "2024-09-10",
                "--end-date", "2024-12-01",
            ],
        )
        assert result.exit_code == 0
        assert "DRY RUN" in result.output

    def test_build_datasets_dry_run_mentions_parquet(self):
        result = runner.invoke(
            app,
            [
                "build-datasets",
                "--dry-run",
                "--start-date", "2025-01-01",
                "--end-date", "2025-01-31",
            ],
        )
        assert "parquet" in result.output.lower()


# ── error paths ───────────────────────────────────────────────────────────────

class TestErrorPaths:
    def test_import_events_missing_file_exits_1(self):
        result = runner.invoke(
            app, ["import-events", "--file", "/nonexistent/path/events.json"]
        )
        assert result.exit_code == 1

    def test_import_auctionator_missing_file_exits_1(self):
        result = runner.invoke(
            app, ["import-auctionator", "--path", "/nonexistent/Auctionator.lua"]
        )
        assert result.exit_code == 1

    def test_backtest_bad_date_format_exits_1(self):
        result = runner.invoke(
            app,
            [
                "backtest",
                "--start-date", "not-a-date",
                "--end-date", "2024-12-01",
            ],
        )
        assert result.exit_code == 1

    def test_backtest_end_before_start_exits_1(self):
        result = runner.invoke(
            app,
            [
                "backtest",
                "--start-date", "2024-12-01",
                "--end-date", "2024-09-01",
            ],
        )
        assert result.exit_code == 1

    def test_build_datasets_end_before_start_exits_1(self):
        result = runner.invoke(
            app,
            [
                "build-datasets",
                "--start-date", "2025-01-31",
                "--end-date", "2025-01-01",
            ],
        )
        assert result.exit_code == 1

    def test_bootstrap_items_without_creds_exits_1(self):
        # Pass empty strings via CliRunner's env override so that even if
        # load_dotenv() has already populated os.environ, these values win.
        result = runner.invoke(
            app,
            ["bootstrap-items"],
            env={"BLIZZARD_CLIENT_ID": "", "BLIZZARD_CLIENT_SECRET": ""},
        )
        assert result.exit_code == 1
        assert "BLIZZARD_CLIENT_ID" in result.output

    def test_build_events_missing_events_file_exits_1(self):
        result = runner.invoke(
            app, ["build-events", "--events-file", "/nonexistent/events.json"]
        )
        assert result.exit_code == 1


# ── import-events with real seed file ────────────────────────────────────────

class TestImportEventsWithRealFile:
    def test_dry_run_with_tww_seed_json_exits_zero(self):
        result = runner.invoke(
            app,
            [
                "import-events",
                "--dry-run",
                "--file", "config/events/tww_events.json",
            ],
        )
        assert result.exit_code == 0, result.output
        assert "DRY RUN" in result.output

    def test_dry_run_validates_event_count(self):
        """The dry-run output should report at least 1 validated event."""
        result = runner.invoke(
            app,
            [
                "import-events",
                "--dry-run",
                "--file", "config/events/tww_events.json",
            ],
        )
        assert "Validated" in result.output

    def test_unsupported_file_format_exits_1(self, tmp_path):
        bad_file = tmp_path / "events.xml"
        bad_file.write_text("<events/>")
        result = runner.invoke(
            app, ["import-events", "--file", str(bad_file)]
        )
        assert result.exit_code == 1
        assert "Unsupported" in result.output
