"""Windows-only integration tests for scripts/run_daily.bat.

Tests cover the freshness gate (issue #12):
  - Health check fails: HEALTH ALERT ACTIVE is logged, the forecast steps
    are skipped, and the script exits non-zero.
  - Health check passes: all three steps run and the exit code is 0.
  - Build failure: run-daily-forecast is skipped, exit code mirrors the build.
  - No venv at all: the gate itself fails to launch, which still blocks the
    forecast steps and exits non-zero.

The script cd's to its parent's parent, so a copy in tmp_path/scripts makes
tmp_path the project root.  The WOWFC environment variable (a test seam in
run_daily.bat) points the CLI calls at a stub .bat that logs the subcommand
it received and exits with a scripted code, so both gate outcomes can be
exercised without a real venv.

These tests are skipped on non-Windows platforms (CI runs ubuntu-latest).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    sys.platform != "win32",
    reason="run_daily.bat requires cmd.exe",
)

REPO_ROOT = Path(__file__).resolve().parents[2]
BAT_SOURCE = REPO_ROOT / "scripts" / "run_daily.bat"
CMD_EXE = os.environ.get("ComSpec", r"C:\Windows\System32\cmd.exe")


# ── Harness ───────────────────────────────────────────────────────────────────


@pytest.fixture
def bat_tree(tmp_path: Path) -> Path:
    """Isolated project tree containing only what run_daily.bat touches."""
    (tmp_path / "scripts").mkdir()
    shutil.copyfile(BAT_SOURCE, tmp_path / "scripts" / "run_daily.bat")
    return tmp_path


def _make_stub(tree: Path, fail_on: str | None = None, exit_code: int = 1) -> Path:
    """Write a CLI stub that echoes its subcommand and exits with a scripted code.

    The stub's stdout lands in logs/daily.log via the parent's redirection, so
    assertions can check which subcommands actually ran.

    Args:
        tree:      Project tree root.
        fail_on:   Subcommand that should fail (None = everything succeeds).
        exit_code: Exit code returned when fail_on matches.
    """
    stub = tree / "wowfc_stub.bat"
    lines = ["@echo off", "echo STUB %1"]
    if fail_on is not None:
        lines.append(f'if "%~1"=="{fail_on}" exit /b {exit_code}')
    lines.append("exit /b 0")
    stub.write_text("\r\n".join(lines) + "\r\n", encoding="ascii")
    return stub


def _run_bat(tree: Path, stub: Path | None = None) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env.pop("WOWFC", None)
    if stub is not None:
        env["WOWFC"] = str(stub)
    return subprocess.run(
        [CMD_EXE, "/c", str(tree / "scripts" / "run_daily.bat")],
        capture_output=True,
        text=True,
        timeout=60,
        cwd=str(tree),
        env=env,
    )


def _read_log(tree: Path) -> str:
    return (tree / "logs" / "daily.log").read_text(encoding="utf-8", errors="replace")


# ── Freshness gate ────────────────────────────────────────────────────────────


def test_stale_health_blocks_forecast_steps(bat_tree: Path) -> None:
    """A failing health check logs HEALTH ALERT ACTIVE and skips everything else."""
    stub = _make_stub(bat_tree, fail_on="check-data-health", exit_code=1)
    result = _run_bat(bat_tree, stub)
    log = _read_log(bat_tree)
    assert "STUB check-data-health" in log
    assert "HEALTH ALERT ACTIVE" in log
    assert "STUB build-datasets" not in log
    assert "STUB run-daily-forecast" not in log
    assert result.returncode == 1


def test_fresh_health_runs_all_steps(bat_tree: Path) -> None:
    """A passing health check lets build-datasets and run-daily-forecast run."""
    stub = _make_stub(bat_tree)
    result = _run_bat(bat_tree, stub)
    log = _read_log(bat_tree)
    assert "Freshness gate OK" in log
    assert "STUB build-datasets" in log
    assert "STUB run-daily-forecast" in log
    assert "Daily pipeline complete" in log
    assert "HEALTH ALERT ACTIVE" not in log
    assert result.returncode == 0


def test_build_failure_skips_forecast(bat_tree: Path) -> None:
    """A build-datasets failure still skips the forecast step (pre-gate behavior)."""
    stub = _make_stub(bat_tree, fail_on="build-datasets", exit_code=7)
    result = _run_bat(bat_tree, stub)
    log = _read_log(bat_tree)
    assert "Freshness gate OK" in log
    assert "build-datasets FAILED" in log
    assert "STUB run-daily-forecast" not in log
    assert result.returncode == 7


def test_missing_venv_blocks_at_gate(bat_tree: Path) -> None:
    """Without a venv the gate cannot even launch; forecast steps never run.

    `call` on an unresolvable path sets errorlevel 1 (unlike direct .exe
    invocation, which gives ERROR_PATH_NOT_FOUND 3 in test_run_hourly.py).
    The contract is only that the gate exits non-zero.
    """
    result = _run_bat(bat_tree)  # no stub: WOWFC defaults to .venv\Scripts\wowfc.exe
    log = _read_log(bat_tree)
    assert "HEALTH ALERT ACTIVE" in log
    assert "Step 2/3" not in log
    assert result.returncode != 0
