"""Scheduler daemon for automated hourly/daily pipeline execution.

No external scheduler library is required — uses stdlib ``time``,
``signal``, and ``subprocess`` only.

Typical usage via the CLI::

    wowfc start-scheduler --daily-time 07:00

Or import directly::

    from wow_forecaster.scheduler import SchedulerDaemon
    daemon = SchedulerDaemon(realm="us", db_path="data/wow_forecaster.db")
    daemon.start()  # blocks until Ctrl-C

Pipelines executed:
  - **Hourly** — ``run-hourly-refresh`` (ingest -> normalize -> drift -> provenance)
  - **Daily**  — ``build-datasets`` then ``run-daily-forecast``
                  (train -> forecast -> recommend)
                  Runs at *daily_time* (local HH:MM clock).

Each pipeline step is invoked as a subprocess (the installed CLI),
so each run has its own process, logging, and exit code.
A failure in one run is logged but does not stop the daemon.
"""

from __future__ import annotations

import logging
import platform
import signal
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _find_cli_exe() -> str:
    """Locate the wow-forecaster CLI executable inside the active virtual env.

    Tries ``wowfc`` (alias) before ``wow-forecaster``, and adds ``.exe``
    suffix on Windows.  Raises ``RuntimeError`` if neither is found.
    """
    scripts_dir = Path(sys.executable).parent
    candidates = (
        ["wowfc.exe", "wow-forecaster.exe", "wowfc", "wow-forecaster"]
        if platform.system() == "Windows"
        else ["wowfc", "wow-forecaster"]
    )
    for name in candidates:
        candidate = scripts_dir / name
        if candidate.exists():
            return str(candidate)
    raise RuntimeError(
        f"Could not find wow-forecaster executable in {scripts_dir}. "
        "Run: pip install -e ."
    )


def _next_daily_run(daily_time: str) -> datetime:
    """Return the next local datetime matching *daily_time* (``HH:MM``)."""
    hour, minute = (int(p) for p in daily_time.split(":"))
    now = datetime.now()
    candidate = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if candidate <= now:
        candidate += timedelta(days=1)
    return candidate


# ── Daemon ────────────────────────────────────────────────────────────────────


class SchedulerDaemon:
    """Runs ``run-hourly-refresh`` every hour and the daily pipeline once per day.

    Parameters
    ----------
    realm:
        Realm slug forwarded to every CLI sub-command (e.g. ``"us"``).
    db_path:
        Path to the SQLite database file.
    daily_time:
        Local 24-hour ``HH:MM`` time to fire the daily pipeline.
        Defaults to ``"07:00"``.
    skip_initial_hourly:
        When *True*, skip the immediate hourly run on daemon start and wait
        for the first scheduled slot (1 hour from now).
    log_dir:
        Directory for scheduler-level log files.
        Defaults to ``logs/`` relative to the current working directory.
    cli_exe:
        Full path to the CLI executable.  Auto-detected from the active
        virtual environment when *None*.
    """

    def __init__(
        self,
        realm: str,
        db_path: str,
        daily_time: str = "07:00",
        skip_initial_hourly: bool = False,
        log_dir: Optional[Path] = None,
        cli_exe: Optional[str] = None,
    ) -> None:
        self.realm = realm
        self.db_path = db_path
        self.daily_time = daily_time
        self.skip_initial_hourly = skip_initial_hourly
        self.log_dir = log_dir or Path("logs")
        self.cli_exe = cli_exe or _find_cli_exe()
        self._running = False

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _run_cmd(self, args: list[str], label: str) -> bool:
        """Run a CLI sub-command.  Returns ``True`` on success (exit code 0).

        Output from the sub-process is printed directly to stdout/stderr so
        it appears in whatever logging backend the parent process has set up.
        Timeout is 3600 s per step.
        """
        cmd = [self.cli_exe] + args
        log.info("[%s] Running: %s", label, " ".join(cmd))
        try:
            result = subprocess.run(cmd, timeout=3600)
            if result.returncode == 0:
                log.info("[%s] Completed successfully (exit 0).", label)
                return True
            log.error("[%s] Exited with code %d.", label, result.returncode)
            return False
        except subprocess.TimeoutExpired:
            log.error("[%s] Timed out after 3600 s.", label)
            return False
        except Exception as exc:
            log.error("[%s] Unexpected error: %s", label, exc, exc_info=True)
            return False

    # ── Pipeline jobs ─────────────────────────────────────────────────────────

    def run_hourly(self) -> None:
        """Execute the hourly pipeline: ingest -> normalize -> drift -> provenance."""
        log.info(
            "=== Hourly refresh starting at %s ===",
            datetime.now().isoformat(timespec="seconds"),
        )
        self._run_cmd(
            ["run-hourly-refresh", "--realm", self.realm, "--db-path", self.db_path],
            "hourly-refresh",
        )

    def run_daily(self) -> None:
        """Execute the daily pipeline: build-datasets -> run-daily-forecast."""
        log.info(
            "=== Daily pipeline starting at %s ===",
            datetime.now().isoformat(timespec="seconds"),
        )
        ok = self._run_cmd(
            ["build-datasets", "--realm", self.realm, "--db-path", self.db_path],
            "build-datasets",
        )
        if not ok:
            log.warning("build-datasets failed — skipping run-daily-forecast.")
            return
        self._run_cmd(
            ["run-daily-forecast", "--realm", self.realm, "--db-path", self.db_path],
            "run-daily-forecast",
        )

    # ── Main loop ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the daemon.  Blocks until Ctrl-C (or SIGTERM on Linux/macOS)."""
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Schedule the first runs
        next_hourly: datetime = (
            datetime.now() + timedelta(hours=1)
            if self.skip_initial_hourly
            else datetime.now()
        )
        next_daily: datetime = _next_daily_run(self.daily_time)

        log.info(
            "Scheduler started.  realm=%s  daily_time=%s  db=%s",
            self.realm,
            self.daily_time,
            self.db_path,
        )
        log.info(
            "Next hourly: %s  |  Next daily: %s",
            next_hourly.isoformat(timespec="seconds"),
            next_daily.isoformat(timespec="seconds"),
        )

        # Signal handlers for clean shutdown
        self._running = True

        def _shutdown(signum, frame):  # noqa: ANN001
            log.info("Signal %d received — stopping scheduler.", signum)
            self._running = False

        signal.signal(signal.SIGINT, _shutdown)
        if platform.system() != "Windows":
            signal.signal(signal.SIGTERM, _shutdown)

        # Main loop — tick every 30 s
        while self._running:
            now = datetime.now()

            if now >= next_hourly:
                self.run_hourly()
                next_hourly = datetime.now() + timedelta(hours=1)
                log.info(
                    "Next hourly scheduled: %s",
                    next_hourly.isoformat(timespec="seconds"),
                )

            if now >= next_daily:
                self.run_daily()
                next_daily = _next_daily_run(self.daily_time)
                log.info(
                    "Next daily scheduled: %s",
                    next_daily.isoformat(timespec="seconds"),
                )

            time.sleep(30)

        log.info("Scheduler stopped.")
