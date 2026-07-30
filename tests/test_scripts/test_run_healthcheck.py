"""Windows-only integration tests for scripts/run_healthcheck.bat.

Tests cover the scheduled health check alert surfaces (issue #4):
  - Healthy run: alert file and suppression flag are deleted, exit 0.
  - Failing run: health_alert.json is written (timestamp, exit code, log
    snippet), the alert window decision is logged, and the exit code mirrors
    check-data-health.
  - 24h window suppression: a fresh flag suppresses the window; a stale or
    unverifiable flag raises again (bias toward raising).
  - No venv at all: "cannot even run the check" takes the alert path.

The script cd's to its parent's parent, so a copy in tmp_path/scripts makes
tmp_path the project root.  The WOWFC environment variable (test seam) points
the CLI call at a stub .bat with a scripted exit code.  Every run also sets
WOWFC_NO_ALERT_WINDOW so no test can pop a console window; tests assert the
logged decision (ALERT WINDOW RAISED / ALERT SUPPRESSED) instead.

These tests are skipped on non-Windows platforms (CI runs ubuntu-latest).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    sys.platform != "win32",
    reason="run_healthcheck.bat requires cmd.exe and Windows PowerShell",
)

REPO_ROOT = Path(__file__).resolve().parents[2]
BAT_SOURCE = REPO_ROOT / "scripts" / "run_healthcheck.bat"
PS1_SOURCE = REPO_ROOT / "scripts" / "sleep_back.ps1"
CMD_EXE = os.environ.get("ComSpec", r"C:\Windows\System32\cmd.exe")

FLAG_SENTINEL = "flag-sentinel-issue-4"


# ── Harness ───────────────────────────────────────────────────────────────────


@pytest.fixture
def bat_tree(tmp_path: Path) -> Path:
    """Isolated project tree containing only what run_healthcheck.bat touches.

    data/outputs/monitoring is deliberately NOT pre-created here: the script's
    own mkdir guard is part of the surface under test.

    sleep_back.ps1 comes along because the .bat calls it on both exit paths
    (issue #78).
    """
    (tmp_path / "scripts").mkdir()
    shutil.copyfile(BAT_SOURCE, tmp_path / "scripts" / "run_healthcheck.bat")
    shutil.copyfile(PS1_SOURCE, tmp_path / "scripts" / "sleep_back.ps1")
    return tmp_path


def _make_stub(tree: Path, exit_code: int = 0) -> Path:
    """Write a CLI stub that echoes its subcommand and exits with a scripted code.

    The stub's stdout lands in logs/health.log via the parent's redirection,
    so assertions (and the alert JSON's log_snippet) can see it ran.
    """
    stub = tree / "wowfc_stub.bat"
    lines = ["@echo off", "echo STUB %1", f"exit /b {exit_code}"]
    stub.write_text("\r\n".join(lines) + "\r\n", encoding="ascii")
    return stub


def _monitoring_dir(tree: Path) -> Path:
    return tree / "data" / "outputs" / "monitoring"


def _alert_path(tree: Path) -> Path:
    return _monitoring_dir(tree) / "health_alert.json"


def _flag_path(tree: Path) -> Path:
    return _monitoring_dir(tree) / "health_window_raised.json"


def _seed_flag(tree: Path, age_hours: float = 0.0) -> Path:
    """Create the suppression flag with sentinel content and an artificial age."""
    _monitoring_dir(tree).mkdir(parents=True, exist_ok=True)
    flag = _flag_path(tree)
    flag.write_text(FLAG_SENTINEL, encoding="ascii")
    if age_hours:
        past = time.time() - age_hours * 3600.0
        os.utime(flag, (past, past))
    return flag


def _run_bat(
    tree: Path,
    stub: Path | None = None,
    path_override: str | None = None,
) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env.pop("WOWFC", None)
    env["WOWFC_NO_ALERT_WINDOW"] = "1"  # no test may ever pop a console window
    env["WOWFC_NO_SLEEP"] = "1"  # nor may one ever suspend the machine (#78)
    if stub is not None:
        env["WOWFC"] = str(stub)
    if path_override is not None:
        env["PATH"] = path_override
    return subprocess.run(
        [CMD_EXE, "/c", str(tree / "scripts" / "run_healthcheck.bat")],
        capture_output=True,
        text=True,
        timeout=60,
        cwd=str(tree),
        env=env,
    )


def _read_log(tree: Path) -> str:
    return (tree / "logs" / "health.log").read_text(encoding="utf-8", errors="replace")


def test_integrity_scope_flag_reaches_the_cli(bat_tree: Path) -> None:
    """The scheduled health check opts in to the durable integrity scope
    (issue #105). A stub that echoes its full argument list proves the flag
    is passed; the daily gate (run_daily.bat) deliberately does not pass it."""
    stub = bat_tree / "wowfc_stub_args.bat"
    stub.write_text(
        "@echo off\r\necho STUB %*\r\nexit /b 0\r\n", encoding="ascii"
    )
    result = _run_bat(bat_tree, stub=stub)
    assert result.returncode == 0
    assert "--integrity-scope durable" in _read_log(bat_tree)


# ── Healthy path ──────────────────────────────────────────────────────────────


def test_healthy_run_clears_alert_and_flag(bat_tree: Path) -> None:
    """A healthy run deletes both the alert file and the suppression flag."""
    _seed_flag(bat_tree)
    _alert_path(bat_tree).write_text("{}", encoding="ascii")
    stub = _make_stub(bat_tree, exit_code=0)
    result = _run_bat(bat_tree, stub)
    log = _read_log(bat_tree)
    assert "STUB check-data-health" in log
    assert "Health check OK" in log
    assert "cleared health_alert.json" in log
    assert not _alert_path(bat_tree).exists()
    assert not _flag_path(bat_tree).exists()
    assert result.returncode == 0


def test_healthy_run_with_no_prior_alert(bat_tree: Path) -> None:
    """A healthy run with nothing to clear succeeds quietly (if-exist guards)."""
    stub = _make_stub(bat_tree, exit_code=0)
    result = _run_bat(bat_tree, stub)
    log = _read_log(bat_tree)
    assert "Health check OK" in log
    assert "ALERT" not in log
    assert "cleared health_alert.json" not in log
    assert result.returncode == 0


# ── Failure path ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize("code", [1, 5])
def test_stale_run_writes_alert_and_raises(bat_tree: Path, code: int) -> None:
    """A failing check writes the alert JSON, raises, and mirrors the exit code."""
    stub = _make_stub(bat_tree, exit_code=code)
    result = _run_bat(bat_tree, stub)
    log = _read_log(bat_tree)
    assert "HEALTH CHECK FAILED" in log
    assert "ALERT WINDOW RAISED" in log
    assert "WOWFC_NO_ALERT_WINDOW test seam" in log
    payload = json.loads(_alert_path(bat_tree).read_text(encoding="utf-8"))
    assert payload["raised_at"]
    assert payload["exit_code"] == code
    assert isinstance(payload["log_snippet"], list)
    assert any("STUB check-data-health" in line for line in payload["log_snippet"])
    assert _flag_path(bat_tree).exists()
    assert result.returncode == code


def test_second_failure_within_window_is_suppressed(bat_tree: Path) -> None:
    """A provably fresh flag suppresses the window; the alert JSON still refreshes."""
    _seed_flag(bat_tree)
    stub = _make_stub(bat_tree, exit_code=1)
    result = _run_bat(bat_tree, stub)
    log = _read_log(bat_tree)
    assert "ALERT SUPPRESSED" in log
    assert "ALERT WINDOW RAISED" not in log
    assert _alert_path(bat_tree).exists()
    assert _flag_path(bat_tree).read_text(encoding="ascii") == FLAG_SENTINEL
    assert result.returncode == 1


def test_flag_older_than_24h_reraises(bat_tree: Path) -> None:
    """A flag past the 24h window no longer suppresses; the flag is rewritten."""
    _seed_flag(bat_tree, age_hours=25.0)
    stub = _make_stub(bat_tree, exit_code=1)
    result = _run_bat(bat_tree, stub)
    log = _read_log(bat_tree)
    assert "ALERT WINDOW RAISED" in log
    assert "ALERT SUPPRESSED" not in log
    assert FLAG_SENTINEL not in _flag_path(bat_tree).read_text(encoding="ascii")
    assert result.returncode == 1


def test_missing_venv_takes_alert_path(bat_tree: Path) -> None:
    """Without a venv the check cannot even launch; that is itself alert-worthy.

    `call` on an unresolvable path sets errorlevel 1 (unlike direct .exe
    invocation, which gives ERROR_PATH_NOT_FOUND 3 in test_run_hourly.py).
    The contract is only that the script exits non-zero.
    """
    result = _run_bat(bat_tree)  # no stub: WOWFC defaults to .venv\Scripts\wowfc.exe
    log = _read_log(bat_tree)
    assert "HEALTH CHECK FAILED" in log
    assert "ALERT WINDOW RAISED" in log
    assert _alert_path(bat_tree).exists()
    assert result.returncode != 0


def test_powershell_failure_biases_toward_raise(bat_tree: Path) -> None:
    """With PowerShell unreachable, a fresh flag is unverifiable: raise anyway.

    PATH points at an empty directory so `powershell` cannot resolve; the stub
    and cmd.exe are invoked by absolute path and still run.  The JSON write
    fails (logged WARNING), the suppression check fails (raise), and the saved
    exit code still comes through untouched.
    """
    _seed_flag(bat_tree)  # fresh flag would suppress if it were verifiable
    stub = _make_stub(bat_tree, exit_code=1)
    empty = bat_tree / "emptypath"
    empty.mkdir()
    result = _run_bat(bat_tree, stub, path_override=str(empty))
    log = _read_log(bat_tree)
    assert "WARNING: failed to write health_alert.json" in log
    assert "ALERT WINDOW RAISED" in log
    assert "ALERT SUPPRESSED" not in log
    assert result.returncode == 1


# ── Sleep-back wiring (issue #78) ─────────────────────────────────────────────
#
# The helper is invoked detached (`start "" /b`), so its log lands after the
# .bat has already exited and these two tests poll for it.  Detaching is not
# incidental: SetSuspendState does not return until the machine resumes, and
# run_silent.vbs waits on the .bat, so an inline call would hold the task in
# the Running state for the whole sleep and the next trigger could be skipped.


def _wait_for_sleep_log(tree: Path, timeout: float = 20.0) -> str:
    """Poll for the detached helper's log, which outlives the .bat."""
    path = tree / "logs" / "sleep_back.log"
    deadline = time.time() + timeout
    while time.time() < deadline:
        if path.exists():
            text = path.read_text(encoding="utf-8", errors="replace")
            if "STAYING AWAKE" in text or "SLEEP BACK" in text:
                return text
        time.sleep(0.25)
    return path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""


def test_failing_check_cannot_sleep_away_its_own_alert(bat_tree: Path) -> None:
    """Ordering is the point: the alert JSON is written before the handoff.

    So the helper's condition 4 sees it and refuses.  Without that ordering a
    failing health check would raise a console window and then suspend the
    machine underneath it.
    """
    stub = _make_stub(bat_tree, exit_code=1)
    result = _run_bat(bat_tree, stub)
    assert result.returncode == 1
    assert _alert_path(bat_tree).exists()
    assert "health alert present" in _wait_for_sleep_log(bat_tree)


def test_healthy_check_gets_past_the_alert_condition(bat_tree: Path) -> None:
    """On the healthy path the alert file was just deleted, so it may proceed.

    What this pins is the ordering, so the assertion is that the alert check
    passed, not what stopped the run afterwards.  Two things can: no
    WoWForecaster task woke this machine for a test run (live-acceptance
    territory), or someone is at the keyboard while the suite runs.  Either
    reason proves the alert condition was cleared first.
    """
    _seed_flag(bat_tree)
    _alert_path(bat_tree).write_text('{"raised_at": "old"}', encoding="ascii")
    stub = _make_stub(bat_tree, exit_code=0)
    result = _run_bat(bat_tree, stub)
    assert result.returncode == 0
    assert not _alert_path(bat_tree).exists()
    log = _wait_for_sleep_log(bat_tree)
    assert "health alert present" not in log
    assert ("wake check" in log) or ("user input during run" in log)
