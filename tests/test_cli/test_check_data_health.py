"""
CLI exit-code tests for check-data-health (issue #5 acceptance).

The report-level logic lives in tests/test_reporting/test_health.py; these
tests prove the wiring: a stale hourly lock or a retention violation must
flip the command's exit code to 1 even when the market data itself is fresh,
because run_daily.bat and run_healthcheck.bat consume only the exit code.

Each test builds a real schema DB in tmp_path via apply_schema() and points
the CLI at it with a minimal --config TOML.  The TOML disables file logging
(log_file = "") because the default data/logs/forecaster.log is CWD-relative
and would write into the repo.  WOW_FORECASTER_* env overrides are cleared so
a developer's environment cannot redirect the tmp DB.
"""

from __future__ import annotations

import os
import sqlite3
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from typer.testing import CliRunner

from wow_forecaster.cli import app
from wow_forecaster.db.schema import apply_schema

runner = CliRunner()


@pytest.fixture
def cli_tree(tmp_path: Path, monkeypatch) -> Path:
    """Tmp project tree: schema DB + minimal config TOML pointing at it."""
    monkeypatch.delenv("WOW_FORECASTER_DB_PATH", raising=False)
    monkeypatch.delenv("WOW_FORECASTER_LOG_LEVEL", raising=False)

    db_path = tmp_path / "db" / "health.db"
    db_path.parent.mkdir()
    conn = sqlite3.connect(str(db_path))
    apply_schema(conn)
    conn.commit()
    conn.close()

    config = tmp_path / "config.toml"
    config.write_text(
        "\n".join(
            [
                "[database]",
                f'db_path = "{db_path.as_posix()}"',
                "[realms]",
                'defaults = ["us"]',
                "[logging]",
                'log_file = ""',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return tmp_path


def _insert_fresh_raw(tree: Path, observed_days_ago: float = 0.02) -> None:
    db_path = tree / "db" / "health.db"
    ts = (datetime.now(tz=UTC) - timedelta(days=observed_days_ago)).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "PRAGMA foreign_keys = OFF;"
    )
    conn.execute(
        "INSERT INTO market_observations_raw "
        "(item_id, realm_slug, observed_at, source, ingested_at) "
        "VALUES (1, 'us', ?, 'test', ?)",
        (ts, ts),
    )
    conn.commit()
    conn.close()


def _run(tree: Path) -> tuple[int, str]:
    result = runner.invoke(
        app,
        ["check-data-health", "--config", str(tree / "config.toml")],
    )
    return result.exit_code, result.output


def test_healthy_db_exits_zero(cli_tree: Path) -> None:
    _insert_fresh_raw(cli_tree)
    exit_code, output = _run(cli_tree)
    assert "[HEALTHY]" in output
    assert exit_code == 0


def test_stale_lock_exits_one_despite_fresh_data(cli_tree: Path) -> None:
    _insert_fresh_raw(cli_tree)
    lock = cli_tree / "db" / ".hourly.lock"
    lock.write_text("leaked", encoding="ascii")
    past = time.time() - 300 * 60.0  # 300 minutes, well past the 180 threshold
    os.utime(lock, (past, past))
    exit_code, output = _run(cli_tree)
    assert "[STALE LOCK]" in output
    assert exit_code == 1


def test_retention_violation_exits_one_despite_fresh_data(cli_tree: Path) -> None:
    _insert_fresh_raw(cli_tree)
    _insert_fresh_raw(cli_tree, observed_days_ago=40.0)
    exit_code, output = _run(cli_tree)
    assert "[RETENTION]" in output
    assert exit_code == 1
