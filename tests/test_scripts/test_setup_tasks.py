"""Windows-only integration tests for scripts/setup_tasks.bat.

Tests cover the Task Scheduler registration script (issues #6, #40, and #80):
  - Fresh registration: four /Create calls with the pinned /ST anchors
    (hourly :16, daily 07:00, health-check every 6h at :45, backup 07:30), all
    silent via wscript.exe + run_silent.vbs, each followed by a wake-to-run set.
  - State preservation: a task that was Disabled before re-registration is
    re-disabled immediately after its /Create (/Create /F recreates tasks
    ENABLED; WoWForecaster-Hourly must stay disabled until the issue #1
    runbook).  A PowerShell state-query failure also takes the disabled
    path: disable-on-uncertainty is the safe bias for tasks whose
    accidental enablement can destroy data.
  - Wake-to-run (issue #40): each registration flips WakeToRun via a
    PowerShell fetch-modify-write; for a was-disabled task the script
    re-asserts /DISABLE after the wake-set (belt and braces around the
    cmdlet round-trip).  A wake-set failure is fatal, and it happens after
    the re-disable, so the failure exit leaves the task DISABLED.
  - Power plan check: a warn-only RTCWAKE query after registration; any AC
    index other than Enable (0x1) prints remediation, exit stays 0.
  - Guard order: missing wrapper scripts abort before any /Create.
  - Failure paths: a failed /Create stops the script; a failed re-disable
    exits 1 with a disable-manually error (the hazard-critical path).

No test ever touches the real Task Scheduler or power plan: every schtasks
call goes through the WOWFC_SCHTASKS seam to a logging stub, powercfg goes
through the WOWFC_POWERCFG seam, and PowerShell is stubbed via a PATH
override (precedent: test_run_healthcheck.py).  The stubs invoke findstr by
absolute %SystemRoot% path because PATH points only at the stub directory.

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
    for name in ("run_hourly.bat", "run_daily.bat", "run_healthcheck.bat", "run_backup.bat"):
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

    # powershell stub (found via PATH override).  Two personalities keyed on
    # the args: wake-set calls (contain WakeToRun) are logged to STUB_LOG as
    # "PS-WAKE ..." lines and fail when STUB_PS_WAKE_FAIL is set; state
    # queries report Disabled (exit 2) only when the args contain
    # STUB_PS_DISABLED_MATCH, else Enabled/absent.
    psstub = tmp_path / "psstub"
    psstub.mkdir()
    (psstub / "powershell.bat").write_text(
        "\r\n".join(
            [
                "@echo off",
                'echo %* | %SystemRoot%\\System32\\findstr.exe /C:"WakeToRun" >nul',
                "if not errorlevel 1 goto :wake",
                "if not defined STUB_PS_DISABLED_MATCH exit /b 0",
                "echo %* | %SystemRoot%\\System32\\findstr.exe "
                '/C:"%STUB_PS_DISABLED_MATCH%" >nul',
                "if not errorlevel 1 exit /b 2",
                "exit /b 0",
                ":wake",
                'echo PS-WAKE %* >> "%STUB_LOG%"',
                "if defined STUB_PS_WAKE_FAIL exit /b 1",
                "exit /b 0",
            ]
        )
        + "\r\n",
        encoding="ascii",
    )

    # powercfg stub: emits the one line the RTCWAKE check greps for, with
    # the index driven by STUB_RTCWAKE_INDEX so the real machine's power
    # plan never leaks into assertions.
    (tmp_path / "powercfg_stub.bat").write_text(
        "\r\n".join(
            [
                "@echo off",
                "echo     Current AC Power Setting Index: %STUB_RTCWAKE_INDEX%",
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
    wake_fail: bool = False,
    rtcwake_index: str = "0x00000001",
) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["WOWFC_SCHTASKS"] = str(tree / "schtasks_stub.bat")
    env["WOWFC_POWERCFG"] = str(tree / "powercfg_stub.bat")
    env["STUB_LOG"] = str(tree / "schtasks_calls.log")
    env["STUB_RTCWAKE_INDEX"] = rtcwake_index
    env["PATH"] = str(tree / ("emptypath" if ps_unreachable else "psstub"))
    env.pop("STUB_PS_DISABLED_MATCH", None)
    env.pop("STUB_FAIL_ON", None)
    env.pop("STUB_PS_WAKE_FAIL", None)
    if ps_disabled_match is not None:
        env["STUB_PS_DISABLED_MATCH"] = ps_disabled_match
    if fail_on is not None:
        env["STUB_FAIL_ON"] = fail_on
    if wake_fail:
        env["STUB_PS_WAKE_FAIL"] = "1"
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


def _wakes(calls: list[str]) -> list[str]:
    return [c for c in calls if c.startswith("PS-WAKE")]


# ── Fresh registration ────────────────────────────────────────────────────────


def test_fresh_registration_creates_four_silent_tasks(bat_tree: Path) -> None:
    """No existing tasks: create + wake-set per task, pinned anchors, no
    disables."""
    result = _run_bat(bat_tree)
    calls = _calls(bat_tree)
    creates = [c for c in calls if "/Create" in c]
    wakes = _wakes(calls)
    assert len(creates) == 4
    assert len(wakes) == 4
    assert len(calls) == 8  # each create immediately followed by its wake-set
    hourly, daily, health, backup = creates
    assert "/SC HOURLY /ST 07:16" in hourly
    assert '"WoWForecaster-Hourly"' in hourly
    assert "/SC DAILY /ST 07:00" in daily
    assert '"WoWForecaster-Daily"' in daily
    assert "/SC HOURLY /MO 6 /ST 00:45" in health
    assert '"WoWForecaster-HealthCheck"' in health
    assert "/SC DAILY /ST 07:30" in backup
    assert '"WoWForecaster-Backup"' in backup
    for c in creates:
        assert "wscript.exe" in c
        assert "run_silent.vbs" in c
    for wake, name in zip(
        wakes,
        (
            "WoWForecaster-Hourly",
            "WoWForecaster-Daily",
            "WoWForecaster-HealthCheck",
            "WoWForecaster-Backup",
        ),
        strict=True,
    ):
        assert name in wake
        assert "WakeToRun" in wake
    # Interleaving: create, wake, create, wake, create, wake, create, wake
    assert [i for i, c in enumerate(calls) if c.startswith("PS-WAKE")] == [1, 3, 5, 7]
    assert "/DISABLE" not in "\n".join(calls)
    assert result.returncode == 0


# ── State preservation ────────────────────────────────────────────────────────


def test_disabled_task_stays_disabled_after_reregistration(bat_tree: Path) -> None:
    """The hazard case: a Disabled hourly task is re-disabled after /Create,
    then re-disabled AGAIN after the wake-set (belt and braces around the
    Set-ScheduledTask round-trip)."""
    result = _run_bat(bat_tree, ps_disabled_match="WoWForecaster-Hourly")
    calls = _calls(bat_tree)
    disables = [c for c in calls if "/DISABLE" in c]
    assert len(disables) == 2
    for d in disables:
        assert '"WoWForecaster-Hourly"' in d
        assert "/Change" in d
    # Order: create H, disable H, wake H, disable H again, then the rest
    assert "/Create" in calls[0]
    assert "/DISABLE" in calls[1]
    assert calls[2].startswith("PS-WAKE")
    assert "WoWForecaster-Hourly" in calls[2]
    assert "/DISABLE" in calls[3]
    assert len(calls) == 10  # 4 creates + 4 wakes + 2 hourly disables
    assert "preserved DISABLED state" in result.stdout
    assert result.returncode == 0


def test_ps_failure_biases_toward_disable(bat_tree: Path) -> None:
    """With PowerShell unreachable, the hourly task is created and disabled,
    then the script dies at its wake-set.

    Disable-on-uncertainty: errorlevel 1 from the missing powershell takes
    the disabled branch, so the /Create is followed by a /DISABLE.  The
    wake-set (issue #40) also needs PowerShell and its failure is fatal,
    but it runs after the re-disable, so the failure exit leaves the task
    DISABLED.  Wrong-but-safe on a fresh machine; correct on this machine,
    where an enabled hourly task destroys data.
    """
    result = _run_bat(bat_tree, ps_unreachable=True)
    calls = _calls(bat_tree)
    assert len([c for c in calls if "/Create" in c]) == 1
    assert len([c for c in calls if "/DISABLE" in c]) == 1
    assert _wakes(calls) == []  # the wake logger IS powershell; it never ran
    assert "[ERROR]" in result.stdout
    assert "wake-to-run" in result.stdout
    assert result.returncode == 1


# ── Guards and failure paths ──────────────────────────────────────────────────


def test_missing_healthcheck_bat_aborts_before_any_create(bat_tree: Path) -> None:
    (bat_tree / "scripts" / "run_healthcheck.bat").unlink()
    result = _run_bat(bat_tree)
    assert result.returncode == 1
    assert "[ERROR]" in result.stdout
    assert "run_healthcheck.bat" in result.stdout
    assert _calls(bat_tree) == []


def test_missing_backup_bat_aborts_before_any_create(bat_tree: Path) -> None:
    (bat_tree / "scripts" / "run_backup.bat").unlink()
    result = _run_bat(bat_tree)
    assert result.returncode == 1
    assert "[ERROR]" in result.stdout
    assert "run_backup.bat" in result.stdout
    assert _calls(bat_tree) == []


def test_missing_vbs_aborts_before_any_create(bat_tree: Path) -> None:
    (bat_tree / "scripts" / "run_silent.vbs").unlink()
    result = _run_bat(bat_tree)
    assert result.returncode == 1
    assert "[ERROR]" in result.stdout
    assert "run_silent.vbs" in result.stdout
    assert _calls(bat_tree) == []


def test_disabled_backup_task_stays_disabled(bat_tree: Path) -> None:
    """A Disabled backup task is re-disabled after /Create and again after the
    wake-set, exactly like the hourly hazard case (the last-registered task must
    honour the same state-preservation contract)."""
    result = _run_bat(bat_tree, ps_disabled_match="WoWForecaster-Backup")
    calls = _calls(bat_tree)
    disables = [c for c in calls if "/DISABLE" in c]
    assert len(disables) == 2
    for d in disables:
        assert '"WoWForecaster-Backup"' in d
    assert len([c for c in calls if "/Create" in c]) == 4
    assert "preserved DISABLED state" in result.stdout
    assert result.returncode == 0


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
    assert _wakes(calls) == []  # exit happened before the wake-set
    assert "Disable it manually NOW" in result.stdout
    assert result.returncode == 1


# ── Wake-to-run and power plan (issue #40) ────────────────────────────────────


def test_wake_set_failure_stops_the_script(bat_tree: Path) -> None:
    """A failed wake-set is fatal: a registration that cannot wake the
    machine defeats the point of #40."""
    result = _run_bat(bat_tree, wake_fail=True)
    calls = _calls(bat_tree)
    assert len([c for c in calls if "/Create" in c]) == 1  # daily never reached
    assert len(_wakes(calls)) == 1  # the hourly attempt was made, then failed
    assert "WoWForecaster-Hourly" in _wakes(calls)[0]
    assert "[ERROR]" in result.stdout
    assert "wake-to-run" in result.stdout
    assert result.returncode == 1


@pytest.mark.parametrize("index", ["0x00000000", "0x00000002"])
def test_wake_timers_not_enabled_warns(bat_tree: Path, index: str) -> None:
    """Disable (0x0) and Important Wake Timers Only (0x2) both block Task
    Scheduler wake timers: warn with remediation, but registration stands
    (exit 0 -- fixing the power plan needs an elevated shell)."""
    result = _run_bat(bat_tree, rtcwake_index=index)
    assert "[WARNING]" in result.stdout
    assert "SETACVALUEINDEX" in result.stdout
    assert result.returncode == 0


def test_wake_timers_enabled_no_warning(bat_tree: Path) -> None:
    result = _run_bat(bat_tree)  # stub defaults to Enable (0x1)
    assert "[WARNING]" not in result.stdout
    assert result.returncode == 0
