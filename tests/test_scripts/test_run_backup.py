"""Windows-only integration tests for scripts/run_backup.bat.

Tests cover the scheduled durable-backup wrapper (issue #80):
  - The wrapper invokes ``backup-durable-db --upload`` via the venv CLI.
  - The exit code mirrors the CLI (Task Scheduler's Last Run Result is a
    truthful, independent backup-health signal).
  - Output is appended to logs/backup.log.

The script cd's to its parent's parent, so a copy in tmp_path/scripts makes
tmp_path the project root. The WOWFC environment variable (test seam) points
the CLI call at a stub .bat with a scripted exit code.

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
    reason="run_backup.bat requires cmd.exe",
)

REPO_ROOT = Path(__file__).resolve().parents[2]
BAT_SOURCE = REPO_ROOT / "scripts" / "run_backup.bat"
CMD_EXE = os.environ.get("ComSpec", r"C:\Windows\System32\cmd.exe")


@pytest.fixture
def bat_tree(tmp_path: Path) -> Path:
    """Isolated project tree containing only what run_backup.bat touches."""
    (tmp_path / "scripts").mkdir()
    shutil.copyfile(BAT_SOURCE, tmp_path / "scripts" / "run_backup.bat")
    return tmp_path


def _make_stub(tree: Path, exit_code: int = 0) -> Path:
    """A CLI stub that echoes its subcommand and args, then exits with a code."""
    stub = tree / "wowfc_stub.bat"
    lines = ["@echo off", "echo STUB %*", f"exit /b {exit_code}"]
    stub.write_text("\r\n".join(lines) + "\r\n", encoding="ascii")
    return stub


def _run_bat(tree: Path, stub: Path | None = None) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env.pop("WOWFC", None)
    if stub is not None:
        env["WOWFC"] = str(stub)
    return subprocess.run(
        [CMD_EXE, "/c", str(tree / "scripts" / "run_backup.bat")],
        capture_output=True,
        text=True,
        timeout=60,
        cwd=str(tree),
        env=env,
    )


def _read_log(tree: Path) -> str:
    return (tree / "logs" / "backup.log").read_text(encoding="utf-8", errors="replace")


def test_invokes_backup_command_with_upload(bat_tree: Path) -> None:
    stub = _make_stub(bat_tree, exit_code=0)
    result = _run_bat(bat_tree, stub)
    log = _read_log(bat_tree)
    assert "STUB backup-durable-db --upload" in log
    assert "Durable backup starting" in log
    assert "Durable backup complete" in log
    assert result.returncode == 0


@pytest.mark.parametrize("code", [1, 3])
def test_exit_code_mirrors_cli(bat_tree: Path, code: int) -> None:
    stub = _make_stub(bat_tree, exit_code=code)
    result = _run_bat(bat_tree, stub)
    log = _read_log(bat_tree)
    assert f"exited with code {code}" in log
    assert result.returncode == code


def test_missing_venv_exits_nonzero(bat_tree: Path) -> None:
    """Without a venv the CLI cannot launch; the wrapper still exits non-zero.

    `call` on an unresolvable path sets errorlevel 1, so Task Scheduler records
    the failure rather than a false success.
    """
    result = _run_bat(bat_tree)  # no stub: WOWFC defaults to .venv\Scripts\wowfc.exe
    assert result.returncode != 0
    assert (bat_tree / "logs" / "backup.log").exists()
