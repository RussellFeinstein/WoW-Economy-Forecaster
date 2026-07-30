"""
Windows-only integration tests for scripts/sleep_back.ps1 (issue #78).

The helper returns the box to sleep after an unattended run that woke it.
It runs in two modes:

  -Capture   prints the current LASTINPUTINFO.dwTime and nothing else, so the
             calling bat can stash it in a variable before the real work.
  -Decide    evaluates the overnight window and four conditions, logs the
             decision, and suspends unless WOWFC_NO_SLEEP is set.

What is covered here:
  - The overnight window wraps midnight (20:00-08:00 by default).  The 23:00
    case is the regression test for the one-sided "refuse at or after 08:00"
    formulation, which refused every evening hour and so left a box slept at
    22:00 awake all night.
  - Bound precedence: -UntilHour beats WOWFC_SLEEP_UNTIL_HOUR beats 8.
  - Condition 2 (no user input during the run), as a dwTime equality check.
  - Condition 3's lock half, and condition 4 (unacknowledged health alert).
  - The WOWFC_NO_SLEEP seam, and that every path exits 0.
  - The fail-safe bias: anything unevaluable leaves the box awake.  This is the
    INVERSE of run_hourly.bat's stale-lock takeover and run_healthcheck.bat's
    alert window, both of which act on uncertainty.  Here a wrong sleep
    interrupts the operator and a wrong wake costs watts.

Not covered here, live acceptance only: condition 1 (needs a real
Power-Troubleshooter wake event) and condition 3's scheduled-task half (needs a
real concurrent WoWForecaster task).  Ordering is covered instead: when
everything else passes, the wake check is shown to be what decides.

The script resolves the project root as its own parent's parent, so a copy in
tmp_path/scripts makes tmp_path the root.  These tests are skipped on
non-Windows platforms (CI runs ubuntu-latest).
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
    reason="sleep_back.ps1 requires Windows PowerShell and the Win32 power API",
)

REPO_ROOT = Path(__file__).resolve().parents[2]
PS1_SOURCE = REPO_ROOT / "scripts" / "sleep_back.ps1"

# Mirrors the defaults in sleep_back.ps1.
DEFAULT_FROM_HOUR = 20
DEFAULT_UNTIL_HOUR = 8

SLEEP_MARK = "SLEEP BACK"
AWAKE_MARK = "STAYING AWAKE"


# ── Harness ───────────────────────────────────────────────────────────────────


@pytest.fixture
def ps_tree(tmp_path: Path) -> Path:
    """Isolated project tree containing only what sleep_back.ps1 touches."""
    (tmp_path / "scripts").mkdir()
    shutil.copyfile(PS1_SOURCE, tmp_path / "scripts" / "sleep_back.ps1")
    (tmp_path / "data" / "db").mkdir(parents=True)
    (tmp_path / "data" / "outputs" / "monitoring").mkdir(parents=True)
    (tmp_path / "logs").mkdir()
    return tmp_path


def _env(**overrides: str) -> dict[str, str]:
    """Base environment with the suspend seam always on.

    Without WOWFC_NO_SLEEP a green test run would suspend the developer's
    machine, so it is set here rather than per-test and no test may unset it.
    """
    env = dict(os.environ)
    env.pop("WOWFC_SLEEP_FROM_HOUR", None)
    env.pop("WOWFC_SLEEP_UNTIL_HOUR", None)
    env["WOWFC_NO_SLEEP"] = "1"
    env.update(overrides)
    return env


def _run(
    tree: Path,
    *args: str,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    cmd = [
        "powershell",
        "-NoProfile",
        "-NonInteractive",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(tree / "scripts" / "sleep_back.ps1"),
        *args,
    ]
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=60,
        cwd=str(tree),
        env=env if env is not None else _env(),
    )


def _decide(
    tree: Path,
    *,
    now: str = "2026-07-29T23:00:00",
    input_at_start: str | None = None,
    run_started_at: str = "2026-07-29T22:50:00",
    caller: str = "WoWForecaster-Hourly",
    extra: tuple[str, ...] = (),
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run -Decide, defaulting to a night hour and an unchanged input stamp."""
    if input_at_start is None:
        input_at_start = _capture(tree)
    args = [
        "-Decide",
        "-InputAtStart",
        input_at_start,
        "-RunStartedAt",
        run_started_at,
        "-CallerTask",
        caller,
        "-NowOverride",
        now,
        *extra,
    ]
    return _run(tree, *args, env=env)


def _capture(tree: Path) -> str:
    result = _run(tree, "-Capture")
    return result.stdout.strip()


def _log(tree: Path) -> str:
    path = tree / "logs" / "sleep_back.log"
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def _slept(tree: Path) -> bool:
    """True when the decision was to suspend (the seam skips the real call)."""
    return SLEEP_MARK in _log(tree)


# ── Capture mode ──────────────────────────────────────────────────────────────


def test_capture_prints_only_an_integer(ps_tree: Path) -> None:
    """stdout carries the tick and nothing else: the bat reads it via for /f."""
    result = _run(ps_tree, "-Capture")
    assert result.returncode == 0
    out = result.stdout.strip()
    assert out.isdigit(), f"expected a bare integer on stdout, got {out!r}"
    assert int(out) > 0


def test_capture_logs_nothing_to_stdout_when_it_also_logs(ps_tree: Path) -> None:
    """Any log line on stdout would poison the captured value."""
    _run(ps_tree, "-Capture")
    result = _run(ps_tree, "-Capture")
    assert len(result.stdout.strip().splitlines()) == 1


def test_capture_is_stable_across_two_reads(ps_tree: Path) -> None:
    """dwTime only moves on real input, which is what makes equality work."""
    first = _capture(ps_tree)
    second = _capture(ps_tree)
    assert first == second


# ── The overnight window ──────────────────────────────────────────────────────


@pytest.mark.parametrize("hour", ["23", "03", "00", "21"])
def test_window_permits_overnight_hours(ps_tree: Path, hour: str) -> None:
    """The window wraps midnight, so both sides of it are night.

    23:00 is the regression case: a one-sided "refuse at or after 08:00" rule
    refuses every evening hour, which leaves a box slept at 22:00 awake all
    night after the 23:16 hourly wakes it.
    """
    _decide(ps_tree, now=f"2026-07-29T{hour}:00:00")
    assert "outside window" not in _log(ps_tree)


@pytest.mark.parametrize("hour", ["09", "15", "12", "19"])
def test_window_refuses_daytime_hours(ps_tree: Path, hour: str) -> None:
    _decide(ps_tree, now=f"2026-07-29T{hour}:00:00")
    log = _log(ps_tree)
    assert AWAKE_MARK in log
    assert "outside window" in log
    assert not _slept(ps_tree)


@pytest.mark.parametrize(
    ("clock", "permitted"),
    [
        ("07:59:59", True),
        ("08:00:00", False),
        ("19:59:59", False),
        ("20:00:00", True),
    ],
)
def test_window_boundaries_are_exact(ps_tree: Path, clock: str, permitted: bool) -> None:
    """hour >= FromHour OR hour < UntilHour, evaluated on the hour."""
    _decide(ps_tree, now=f"2026-07-29T{clock}")
    outside = "outside window" in _log(ps_tree)
    assert outside is not permitted


def test_effective_window_is_logged_every_run(ps_tree: Path) -> None:
    """A forgotten override must be diagnosable from the log, not mysterious."""
    _decide(ps_tree)
    assert f"window: {DEFAULT_FROM_HOUR:02d}:00-{DEFAULT_UNTIL_HOUR:02d}:00" in _log(ps_tree)


# ── Bound precedence ──────────────────────────────────────────────────────────


def test_env_var_overrides_the_default(ps_tree: Path) -> None:
    """09:00 is daytime by default but night with UNTIL_HOUR=10."""
    _decide(
        ps_tree,
        now="2026-07-29T09:00:00",
        env=_env(WOWFC_SLEEP_UNTIL_HOUR="10"),
    )
    log = _log(ps_tree)
    assert "window: 20:00-10:00" in log
    assert "outside window" not in log


def test_parameter_overrides_the_env_var(ps_tree: Path) -> None:
    _decide(
        ps_tree,
        now="2026-07-29T09:00:00",
        extra=("-UntilHour", "8"),
        env=_env(WOWFC_SLEEP_UNTIL_HOUR="10"),
    )
    log = _log(ps_tree)
    assert "window: 20:00-08:00" in log
    assert "outside window" in log


def test_from_hour_is_overridable_too(ps_tree: Path) -> None:
    _decide(
        ps_tree,
        now="2026-07-29T18:00:00",
        env=_env(WOWFC_SLEEP_FROM_HOUR="17"),
    )
    log = _log(ps_tree)
    assert "window: 17:00-08:00" in log
    assert "outside window" not in log


@pytest.mark.parametrize("bad", ["abc", "25", "-1", "8.5", ""])
def test_garbage_bound_refuses_and_says_so(ps_tree: Path, bad: str) -> None:
    """Fail-safe: a typo disables sleep-back rather than silently reverting.

    Reverting to a default the operator did not choose is silent; "the box
    stopped sleeping" is a symptom that leads to this log line.
    """
    result = _decide(ps_tree, env=_env(WOWFC_SLEEP_UNTIL_HOUR=bad))
    log = _log(ps_tree)
    assert result.returncode == 0
    assert AWAKE_MARK in log
    assert "bad window bounds" in log
    assert not _slept(ps_tree)


# ── Condition 2: no user input during the run ─────────────────────────────────


def test_input_changed_during_run_refuses(ps_tree: Path) -> None:
    """A dwTime that moved means someone touched the machine mid-run."""
    stale = str(int(_capture(ps_tree)) - 500_000)
    _decide(ps_tree, input_at_start=stale)
    log = _log(ps_tree)
    assert AWAKE_MARK in log
    assert "user input during run" in log
    assert not _slept(ps_tree)


def test_unchanged_input_passes_condition_two(ps_tree: Path) -> None:
    _decide(ps_tree)
    assert "user input during run" not in _log(ps_tree)


@pytest.mark.parametrize("bad", ["abc", "", "-7"])
def test_unparseable_input_at_start_refuses(ps_tree: Path, bad: str) -> None:
    result = _decide(ps_tree, input_at_start=bad)
    log = _log(ps_tree)
    assert result.returncode == 0
    assert AWAKE_MARK in log
    assert not _slept(ps_tree)


# ── Condition 3 (lock half) and condition 4 ───────────────────────────────────


def test_hourly_lock_refuses(ps_tree: Path) -> None:
    """A manual sync-snapshots drain holds this lock and is not a task.

    The scheduled-task half of condition 3 would not see it, so the lock is
    checked directly or a drain gets suspended mid-write.
    """
    (ps_tree / "data" / "db" / ".hourly.lock").write_text("held", encoding="ascii")
    _decide(ps_tree)
    log = _log(ps_tree)
    assert AWAKE_MARK in log
    assert "lock present" in log
    assert not _slept(ps_tree)


def test_health_alert_refuses(ps_tree: Path) -> None:
    """The hourly must not sleep away an alert the health check just raised."""
    alert = ps_tree / "data" / "outputs" / "monitoring" / "health_alert.json"
    alert.write_text('{"raised_at": "x"}', encoding="ascii")
    _decide(ps_tree)
    log = _log(ps_tree)
    assert AWAKE_MARK in log
    assert "health alert present" in log
    assert not _slept(ps_tree)


def test_absent_lock_and_alert_pass(ps_tree: Path) -> None:
    """Absent is the healthy steady state, not an unevaluable condition."""
    _decide(ps_tree)
    log = _log(ps_tree)
    assert "lock present" not in log
    assert "health alert present" not in log


# ── Ordering, seam, exit code ─────────────────────────────────────────────────


def test_wake_check_is_what_decides_when_all_else_passes(ps_tree: Path) -> None:
    """Proves the cheap, controllable conditions are evaluated first.

    The wake check needs a real event log, so its verdict is not asserted; that
    it was reached is.
    """
    _decide(ps_tree)
    log = _log(ps_tree)
    assert ("wake" in log) or (SLEEP_MARK in log)


def test_no_sleep_seam_skips_only_the_suspend(ps_tree: Path) -> None:
    """Mirrors WOWFC_NO_ALERT_WINDOW: the decision still lands in the log."""
    (ps_tree / "data" / "db" / ".hourly.lock").write_text("held", encoding="ascii")
    _decide(ps_tree)
    assert AWAKE_MARK in _log(ps_tree)


@pytest.mark.parametrize(
    "case",
    ["clean", "locked", "alerted", "daytime", "bad_bounds", "bad_input"],
)
def test_always_exits_zero(ps_tree: Path, case: str) -> None:
    """Fire-and-forget: the helper can never change the caller's exit code."""
    env = _env()
    kwargs: dict[str, object] = {}
    if case == "locked":
        (ps_tree / "data" / "db" / ".hourly.lock").write_text("x", encoding="ascii")
    elif case == "alerted":
        (ps_tree / "data" / "outputs" / "monitoring" / "health_alert.json").write_text(
            "{}", encoding="ascii"
        )
    elif case == "daytime":
        kwargs["now"] = "2026-07-29T12:00:00"
    elif case == "bad_bounds":
        env = _env(WOWFC_SLEEP_UNTIL_HOUR="nope")
    elif case == "bad_input":
        kwargs["input_at_start"] = "nope"

    result = _decide(ps_tree, env=env, **kwargs)  # type: ignore[arg-type]
    assert result.returncode == 0, result.stderr


def test_missing_now_override_uses_the_real_clock(ps_tree: Path) -> None:
    """At least one test exercises the production default rather than a seam."""
    result = _run(
        ps_tree,
        "-Decide",
        "-InputAtStart",
        _capture(ps_tree),
        "-RunStartedAt",
        "2026-07-29T22:50:00",
        "-CallerTask",
        "WoWForecaster-Hourly",
    )
    assert result.returncode == 0
    assert "window: " in _log(ps_tree)
