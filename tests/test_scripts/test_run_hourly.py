"""
Windows-only integration tests for scripts/run_hourly.bat.

Tests cover the overlap guard with age-based stale-lock takeover (issue #3):
  - No lock: the run proceeds.
  - Fresh lock: SKIPPED is logged, exit 0, lock preserved.
  - Lock older than STALE_MINUTES: STALE LOCK TAKEOVER is logged, the old
    lock is removed, the run proceeds.
  - Lock just under the threshold: still skips (guards the minutes unit).
  - PowerShell unavailable: failure biases toward takeover.

The script cd's to its parent's parent, so a copy in tmp_path/scripts makes
tmp_path the project root. No .venv exists in the tree, so the CLI call
fails with cmd exit code 3 (ERROR_PATH_NOT_FOUND for an explicit path whose
directory is missing; bare-name PATH lookup failures like powershell get
9009 instead) and the real pipeline can never start. A mid-execution
PowerShell crash is not simulated here; it shares the same nonzero-exit
contract as the unresolvable-powershell case, which is.

These tests are skipped on non-Windows platforms (CI runs ubuntu-latest).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    sys.platform != "win32",
    reason="run_hourly.bat requires cmd.exe and Windows PowerShell",
)

REPO_ROOT = Path(__file__).resolve().parents[2]
BAT_SOURCE = REPO_ROOT / "scripts" / "run_hourly.bat"
CMD_EXE = os.environ.get("ComSpec", r"C:\Windows\System32\cmd.exe")

STALE_MINUTES = 180  # mirrors STALE_MINUTES in run_hourly.bat
MISSING_CLI_EXIT = 3  # ERROR_PATH_NOT_FOUND: the tree has no .venv directory
LOCK_SENTINEL = "lock-sentinel-issue-3"


# ── Harness ───────────────────────────────────────────────────────────────────


@pytest.fixture
def bat_tree(tmp_path: Path) -> Path:
    """Isolated project tree containing only what run_hourly.bat touches."""
    (tmp_path / "scripts").mkdir()
    shutil.copyfile(BAT_SOURCE, tmp_path / "scripts" / "run_hourly.bat")
    (tmp_path / "data" / "db").mkdir(parents=True)
    return tmp_path


def _lock_path(tree: Path) -> Path:
    return tree / "data" / "db" / ".hourly.lock"


def _make_lock(tree: Path, age_minutes: float = 0.0) -> Path:
    lock = _lock_path(tree)
    lock.write_text(LOCK_SENTINEL, encoding="ascii")
    if age_minutes:
        old = time.time() - age_minutes * 60
        os.utime(lock, (old, old))
    return lock


def _run_bat(tree: Path, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [CMD_EXE, "/c", str(tree / "scripts" / "run_hourly.bat")],
        capture_output=True,
        text=True,
        timeout=60,
        cwd=str(tree),
        env=env,
    )


def _read_log(tree: Path) -> str:
    # cmd writes the log in the console codepage; assertions are ASCII-only.
    return (tree / "logs" / "hourly.log").read_text(encoding="utf-8", errors="replace")


# ── Overlap guard ─────────────────────────────────────────────────────────────


def test_no_lock_run_proceeds(bat_tree: Path) -> None:
    """Without a lock, the run starts and the exit code mirrors the CLI's."""
    result = _run_bat(bat_tree)
    log = _read_log(bat_tree)
    assert "Hourly refresh starting" in log
    assert f"Hourly refresh complete (exit {MISSING_CLI_EXIT})" in log
    assert "SKIPPED" not in log
    assert "TAKEOVER" not in log
    assert result.returncode == MISSING_CLI_EXIT
    assert not _lock_path(bat_tree).exists()  # cleaned up at end of run


def test_fresh_lock_skips_and_preserves_lock(bat_tree: Path) -> None:
    """A fresh lock logs SKIPPED, exits 0, and is left untouched."""
    lock = _make_lock(bat_tree)
    result = _run_bat(bat_tree)
    log = _read_log(bat_tree)
    assert "SKIPPED: previous run still active" in log
    assert "Hourly refresh starting" not in log
    assert "TAKEOVER" not in log
    assert result.returncode == 0
    assert lock.read_text(encoding="ascii") == LOCK_SENTINEL  # not recreated


def test_stale_lock_taken_over(bat_tree: Path) -> None:
    """A lock aged past the threshold is removed and the run proceeds."""
    _make_lock(bat_tree, age_minutes=STALE_MINUTES + 60)  # 4 hours old
    result = _run_bat(bat_tree)
    log = _read_log(bat_tree)
    assert "STALE LOCK TAKEOVER" in log
    assert "Hourly refresh starting" in log
    assert "SKIPPED" not in log
    assert result.returncode == MISSING_CLI_EXIT
    assert not _lock_path(bat_tree).exists()


def test_lock_just_under_threshold_still_skips(bat_tree: Path) -> None:
    """A 170-minute lock skips: the threshold compares minutes, not seconds."""
    _make_lock(bat_tree, age_minutes=STALE_MINUTES - 10)
    result = _run_bat(bat_tree)
    log = _read_log(bat_tree)
    assert "SKIPPED" in log
    assert "TAKEOVER" not in log
    assert result.returncode == 0
    assert _lock_path(bat_tree).exists()


def test_powershell_failure_biases_toward_takeover(bat_tree: Path) -> None:
    """When the age check cannot run, a fresh lock is still taken over."""
    _make_lock(bat_tree)  # fresh lock
    empty = bat_tree / "emptybin"
    empty.mkdir()
    env = dict(os.environ)
    env["PATH"] = str(empty)  # powershell unresolvable -> errorlevel 9009
    result = _run_bat(bat_tree, env=env)
    log = _read_log(bat_tree)
    assert "STALE LOCK TAKEOVER" in log
    assert "Hourly refresh starting" in log
    assert result.returncode == MISSING_CLI_EXIT
    assert not _lock_path(bat_tree).exists()
