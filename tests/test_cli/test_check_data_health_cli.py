"""CLI-level tests for check-data-health options (issue #105)."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from typer.testing import CliRunner

from wow_forecaster.cli import app
from wow_forecaster.db.schema import apply_schema

runner = CliRunner()


def _schema_db(tmp_path: Path) -> Path:
    db_path = tmp_path / "health_cli.db"
    conn = sqlite3.connect(str(db_path))
    apply_schema(conn)
    conn.commit()
    conn.close()
    return db_path


def test_unknown_integrity_scope_is_a_usage_error(tmp_path: Path) -> None:
    db_path = _schema_db(tmp_path)
    result = runner.invoke(
        app,
        ["check-data-health", "--integrity-scope", "bogus"],
        env={"WOW_FORECASTER_DB_PATH": str(db_path)},
    )
    assert result.exit_code == 2


def test_durable_scope_runs_and_reports(tmp_path: Path) -> None:
    """An empty schema DB is stale (exit 1) but the integrity line must show
    the scope ran clean; plain typer.echo output, so the substring is safe to
    assert (the rich-wrap caveat applies to --help tables only)."""
    db_path = _schema_db(tmp_path)
    result = runner.invoke(
        app,
        ["check-data-health", "--integrity-scope", "durable"],
        env={"WOW_FORECASTER_DB_PATH": str(db_path)},
    )
    assert result.exit_code == 1  # empty DB: stale, but not corrupt
    assert "Integrity        : ok" in result.output
