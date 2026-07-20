"""Windows-only integration tests for scripts/setup_tasks.bat.

Tests cover the Task Scheduler registration script (issue #6):
  - Fresh registration: three /Create calls with the pinned /ST anchors
    (hourly :16, daily 07:00, health-check every 6h at :45), all silent via
    wscript.exe + run_silent.vbs.
  - State preservation: a task that was Disabled before re-registration is
    re-disabled immediately after its /Create (/Create /F recreates tasks
    ENABLED; WoWForecaster-Hourly must stay disabled until the issue #1
    runbook).  A PowerShell state-query failure also takes the disabled
    path: disable-on-uncertainty is the safe bias for tasks whose
    accidental enablement can destroy data.
  - Guard order: missing wrapper scripts abort before any /Create.
  - Failure paths: a failed /Create stops the script; a failed re-disable
    exits 1 with a disable-manually error (the hazard-critical path).

No test ever touches the real Task Scheduler: every schtasks call goes
through the WOWFC_SCHTASKS seam to a logging stub, and PowerShell is stubbed
via a PATH override (precedent: test_run_healthcheck.py).  The stubs invoke
findstr by absolute %SystemRoot% path because PATH points only at the stub
directory.

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
    reason="setup_tasks.bat requires cmd.exe and Windows PowerShell",
)

REPO_ROOT = Path(__file__).resolve().parents[2]
BAT_SOURCE = REPO_ROOT / "scripts" / "setup_tasks.bat"
CMD_EXE = os.environ.get("ComSpec", r"C:\Windows\System32\cmd.exe")


# ── Harness ───────────────────────────────────────────────────────────────────


@pytest.fixture
def bat_tree(tmp_path: Path) -> Path:
    """Isolated tree: the bat under test plus dummy wrapper scripts.

    setup_tasks.bat resolves its wrappers relative to its own directory
    (%~dp0), so dummies next to the copied bat satisfy the existence guards.
    """
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    shutil.copyfile(BAT_SOURCE, scripts / "setup_tasks.bat")
    for name in ("run_hourly.bat", "run_daily.bat", "run_healthcheck.bat"):
        (scripts / name).write_text("@echo off\r\nexit /b 0\r\n", encoding="ascii")
    (scripts / "run_silent.vbs").write_text("' dummy\r\n", encoding="ascii")

    # schtasks stub: logs its full arg line, optionally fails when the args
    # contain the STUB_FAIL_ON token.
    (tmp_path / "schtasks_stub.bat").write_text(
        "\r\n".join(
            [
                "@echo off",
                'echo %* >> "%STUB_LOG%"',
                "if not defined STUB_FAIL_ON exit /b 0",
                'echo %* | %SystemRoot%\\System32\\findstr.exe /C:"%STUB_FAIL_ON%" >nul',
                "if not errorlevel 1 exit /b 1",
                "exit /b 0",
            ]
        )
        + "\r\n",
        encoding="ascii",
    )

    # powershell stub (found via PATH override): reports Disabled (exit 2)
    # only when its args contain STUB_PS_DISABLED_MATCH, else Enabled/absent.
    psstub = tmp_path / "psstub"
    psstub.mkdir()
    (psstub / "powershell.bat").write_text(
        "\r\n".join(
            [
                "@echo off",
                "if not defined STUB_PS_DISABLED_MATCH exit /b 0",
                "echo %* | %SystemRoot%\\System32\\findstr.exe "
                '/C:"%STUB_PS_DISABLED_MATCH%" >nul',
                "if not errorlevel 1 exit /b 2",
                "exit /b 0",
            ]
        )
        + "\r\n",
        encoding="ascii",
    )

    (tmp_path / "emptypath").mkdir()
    return tmp_path


def _run_bat(
    tree: Path,
    ps_disabled_match: str | None = None,
    fail_on: str | None = None,
    ps_unreachable: bool = False,
) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["WOWFC_SCHTASKS"] = str(tree / "schtasks_stub.bat")
    env["STUB_LOG"] = str(tree / "schtasks_calls.log")
    env["PATH"] = str(tree / ("emptypath" if ps_unreachable else "psstub"))
    env.pop("STUB_PS_DISABLED_MATCH", None)
    env.pop("STUB_FAIL_ON", None)
    if ps_disabled_match is not None:
        env["STUB_PS_DISABLED_MATCH"] = ps_disabled_match
    if fail_on is not None:
        env["STUB_FAIL_ON"] = fail_on
    return subprocess.run(
        [CMD_EXE, "/c", str(tree / "scripts" / "setup_tasks.bat")],
        capture_output=True,
        text=True,
        timeout=60,
        cwd=str(tree),
        env=env,
    )


def _calls(tree: Path) -> list[str]:
    log = tree / "schtasks_calls.log"
    if not log.exists():
        return []
    return [ln for ln in log.read_text(encoding="ascii").splitlines() if ln.strip()]


# ── Fresh registration ────────────────────────────────────────────────────────


def test_fresh_registration_creates_three_silent_tasks(bat_tree: Path) -> None:
    """No existing tasks: three /Create calls, pinned anchors, no disables."""
    result = _run_bat(bat_tree)
    calls = _calls(bat_tree)
    creates = [c for c in calls if "/Create" in c]
    assert len(creates) == 3
    assert len(calls) == 3  # nothing but the creates
    hourly, daily, health = creates
    assert "/SC HOURLY /ST 07:16" in hourly
    assert '"WoWForecaster-Hourly"' in hourly
    assert "/SC DAILY /ST 07:00" in daily
    assert '"WoWForecaster-Daily"' in daily
    assert "/SC HOURLY /MO 6 /ST 00:45" in health
    assert '"WoWForecaster-HealthCheck"' in health
    for c in creates:
        assert "wscript.exe" in c
        assert "run_silent.vbs" in c
    assert "/DISABLE" not in "\n".join(calls)
    assert result.returncode == 0


# ── State preservation ────────────────────────────────────────────────────────


def test_disabled_task_stays_disabled_after_reregistration(bat_tree: Path) -> None:
    """The hazard case: a Disabled hourly task is re-disabled after /Create."""
    result = _run_bat(bat_tree, ps_disabled_match="WoWForecaster-Hourly")
    calls = _calls(bat_tree)
    disables = [c for c in calls if "/DISABLE" in c]
    assert len(disables) == 1
    assert '"WoWForecaster-Hourly"' in disables[0]
    assert "/Change" in disables[0]
    # The disable follows the hourly create, before the daily create
    assert calls.index(disables[0]) == 1
    assert "preserved DISABLED state" in result.stdout
    assert result.returncode == 0


def test_ps_failure_biases_toward_disable(bat_tree: Path) -> None:
    """With PowerShell unreachable, every task is created then re-disabled.

    Disable-on-uncertainty: errorlevel 9009 from the missing powershell
    lands in the >= 2 branch.  Wrong-but-safe on a fresh machine; correct
    on this machine, where an enabled hourly task destroys data.
    """
    result = _run_bat(bat_tree, ps_unreachable=True)
    calls = _calls(bat_tree)
    assert len([c for c in calls if "/Create" in c]) == 3
    assert len([c for c in calls if "/DISABLE" in c]) == 3
    assert result.returncode == 0


# ── Guards and failure paths ──────────────────────────────────────────────────


def test_missing_healthcheck_bat_aborts_before_any_create(bat_tree: Path) -> None:
    (bat_tree / "scripts" / "run_healthcheck.bat").unlink()
    result = _run_bat(bat_tree)
    assert result.returncode == 1
    assert "[ERROR]" in result.stdout
    assert "run_healthcheck.bat" in result.stdout
    assert _calls(bat_tree) == []


def test_missing_vbs_aborts_before_any_create(bat_tree: Path) -> None:
    (bat_tree / "scripts" / "run_silent.vbs").unlink()
    result = _run_bat(bat_tree)
    assert result.returncode == 1
    assert "[ERROR]" in result.stdout
    assert "run_silent.vbs" in result.stdout
    assert _calls(bat_tree) == []


def test_create_failure_stops_the_script(bat_tree: Path) -> None:
    result = _run_bat(bat_tree, fail_on="/Create")
    calls = _calls(bat_tree)
    assert len(calls) == 1  # the failed hourly create; nothing after
    assert "Failed to create hourly task" in result.stdout
    assert result.returncode == 1


def test_disable_failure_exits_loudly(bat_tree: Path) -> None:
    """A failed re-disable leaves a data-destroying task enabled: exit 1
    with an instruction to disable manually."""
    result = _run_bat(
        bat_tree,
        ps_disabled_match="WoWForecaster-Hourly",
        fail_on="/Change",
    )
    calls = _calls(bat_tree)
    assert len([c for c in calls if "/Create" in c]) == 1  # daily never reached
    assert len([c for c in calls if "/Change" in c]) == 1
    assert "Disable it manually NOW" in result.stdout
    assert result.returncode == 1
