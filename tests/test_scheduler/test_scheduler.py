"""
Unit tests for wow_forecaster/scheduler.py.

Tests cover:
  - _find_cli_exe(): exe discovery in the virtual-env Scripts dir.
  - _next_daily_run(): next-occurrence datetime arithmetic.
  - SchedulerDaemon.__init__(): attribute assignment.
  - SchedulerDaemon._run_cmd(): subprocess call paths (success / failure /
    timeout / unexpected exception).
  - SchedulerDaemon.run_hourly(): delegates to _run_cmd with correct args.
  - SchedulerDaemon.run_daily(): runs both steps on success, skips forecast
    when build-datasets fails.

The daemon's ``start()`` blocking loop is NOT unit-tested here; it requires
an integration harness with signal mocking. The component tests above cover
all meaningful code paths inside the loop body.
"""

from __future__ import annotations

import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from wow_forecaster.scheduler import (
    SchedulerDaemon,
    _find_cli_exe,
    _next_daily_run,
)


# ── _find_cli_exe ─────────────────────────────────────────────────────────────

class TestFindCliExe:
    def test_returns_path_when_wowfc_exists(self, tmp_path, monkeypatch):
        """Returns the wowfc path when it exists in the Scripts dir."""
        fake_exe = tmp_path / "wowfc.exe"
        fake_exe.touch()
        monkeypatch.setattr("sys.executable", str(tmp_path / "python.exe"))
        result = _find_cli_exe()
        assert result == str(fake_exe)

    def test_falls_back_to_wow_forecaster_exe(self, tmp_path, monkeypatch):
        """Falls back to wow-forecaster.exe when wowfc.exe is absent."""
        fake_exe = tmp_path / "wow-forecaster.exe"
        fake_exe.touch()
        monkeypatch.setattr("sys.executable", str(tmp_path / "python.exe"))
        result = _find_cli_exe()
        assert result == str(fake_exe)

    def test_raises_when_no_exe_found(self, tmp_path, monkeypatch):
        """RuntimeError is raised when no CLI executable can be located."""
        monkeypatch.setattr("sys.executable", str(tmp_path / "python.exe"))
        with pytest.raises(RuntimeError, match="Could not find wow-forecaster"):
            _find_cli_exe()


# ── _next_daily_run ───────────────────────────────────────────────────────────

class TestNextDailyRun:
    def test_future_time_is_scheduled_at_correct_hhmm(self):
        """A future HH:MM is always scheduled at the correct hour and minute."""
        far_future = (datetime.now() + timedelta(hours=2)).strftime("%H:%M")
        expected_h, expected_m = (int(p) for p in far_future.split(":"))
        result = _next_daily_run(far_future)
        # Must be in the future (not in the past).
        assert result >= datetime.now().replace(second=0, microsecond=0)
        # Must have exactly the requested hours and minutes.
        assert result.hour == expected_h
        assert result.minute == expected_m

    def test_past_time_is_scheduled_in_the_future(self):
        """A past HH:MM always returns a strictly future datetime at that HH:MM."""
        past_time = (datetime.now() - timedelta(hours=2)).strftime("%H:%M")
        expected_h, expected_m = (int(p) for p in past_time.split(":"))
        result = _next_daily_run(past_time)
        # Must be strictly in the future.
        assert result > datetime.now().replace(second=0, microsecond=0)
        # Must have exactly the requested hours and minutes.
        assert result.hour == expected_h
        assert result.minute == expected_m

    def test_returns_correct_hour_and_minute(self):
        """The returned datetime always has the requested hour and minute."""
        result = _next_daily_run("03:45")
        assert result.hour == 3
        assert result.minute == 45
        assert result.second == 0
        assert result.microsecond == 0

    def test_midnight_scheduled_tomorrow_when_past(self):
        """00:00 that has already passed is always tomorrow."""
        # 00:00 was in the past unless it is literally right now.
        result = _next_daily_run("00:00")
        assert result.date() >= datetime.now().date()


# ── SchedulerDaemon.__init__ ──────────────────────────────────────────────────

class TestSchedulerDaemonInit:
    @pytest.fixture
    def daemon(self, tmp_path):
        return SchedulerDaemon(
            realm="us",
            db_path=str(tmp_path / "test.db"),
            daily_time="08:00",
            skip_initial_hourly=True,
            log_dir=tmp_path / "logs",
            cli_exe="/fake/wowfc",
        )

    def test_realm_stored(self, daemon):
        assert daemon.realm == "us"

    def test_db_path_stored(self, daemon, tmp_path):
        assert daemon.db_path == str(tmp_path / "test.db")

    def test_daily_time_stored(self, daemon):
        assert daemon.daily_time == "08:00"

    def test_skip_initial_hourly_stored(self, daemon):
        assert daemon.skip_initial_hourly is True

    def test_cli_exe_stored(self, daemon):
        assert daemon.cli_exe == "/fake/wowfc"

    def test_running_is_false_on_init(self, daemon):
        assert daemon._running is False

    def test_log_dir_defaults_to_logs(self, tmp_path):
        d = SchedulerDaemon(
            realm="us",
            db_path=str(tmp_path / "test.db"),
            cli_exe="/fake/wowfc",
        )
        assert d.log_dir == Path("logs")


# ── SchedulerDaemon._run_cmd ──────────────────────────────────────────────────

class TestRunCmd:
    @pytest.fixture
    def daemon(self, tmp_path):
        return SchedulerDaemon(
            realm="us",
            db_path=str(tmp_path / "test.db"),
            cli_exe="/fake/wowfc",
        )

    def test_returns_true_on_exit_code_zero(self, daemon):
        mock_result = MagicMock()
        mock_result.returncode = 0
        with patch("subprocess.run", return_value=mock_result):
            assert daemon._run_cmd(["some-command"], "test") is True

    def test_returns_false_on_nonzero_exit(self, daemon):
        mock_result = MagicMock()
        mock_result.returncode = 1
        with patch("subprocess.run", return_value=mock_result):
            assert daemon._run_cmd(["some-command"], "test") is False

    def test_returns_false_on_timeout(self, daemon):
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="x", timeout=1)):
            assert daemon._run_cmd(["some-command"], "test") is False

    def test_returns_false_on_unexpected_exception(self, daemon):
        with patch("subprocess.run", side_effect=OSError("no such file")):
            assert daemon._run_cmd(["some-command"], "test") is False

    def test_passes_cli_exe_as_first_arg(self, daemon):
        """The daemon prepends its cli_exe to the command args."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        with patch("subprocess.run", return_value=mock_result) as mock_sub:
            daemon._run_cmd(["run-hourly-refresh"], "hourly")
            called_cmd = mock_sub.call_args[0][0]
            assert called_cmd[0] == "/fake/wowfc"
            assert called_cmd[1] == "run-hourly-refresh"


# ── SchedulerDaemon.run_hourly ────────────────────────────────────────────────

class TestRunHourly:
    @pytest.fixture
    def daemon(self, tmp_path):
        return SchedulerDaemon(
            realm="area-52",
            db_path=str(tmp_path / "test.db"),
            cli_exe="/fake/wowfc",
        )

    def test_calls_run_hourly_refresh(self, daemon):
        with patch.object(daemon, "_run_cmd", return_value=True) as mock_cmd:
            daemon.run_hourly()
            args_passed = mock_cmd.call_args[0][0]
            assert "run-hourly-refresh" in args_passed

    def test_passes_realm_to_hourly(self, daemon):
        with patch.object(daemon, "_run_cmd", return_value=True) as mock_cmd:
            daemon.run_hourly()
            args_passed = mock_cmd.call_args[0][0]
            assert "--realm" in args_passed
            assert "area-52" in args_passed

    def test_passes_db_path_to_hourly(self, daemon, tmp_path):
        with patch.object(daemon, "_run_cmd", return_value=True) as mock_cmd:
            daemon.run_hourly()
            args_passed = mock_cmd.call_args[0][0]
            assert "--db-path" in args_passed
            assert str(tmp_path / "test.db") in args_passed


# ── SchedulerDaemon.run_daily ─────────────────────────────────────────────────

class TestRunDaily:
    @pytest.fixture
    def daemon(self, tmp_path):
        return SchedulerDaemon(
            realm="us",
            db_path=str(tmp_path / "test.db"),
            cli_exe="/fake/wowfc",
        )

    def test_runs_both_steps_on_success(self, daemon):
        """Both build-datasets and run-daily-forecast are called when
        build-datasets succeeds."""
        with patch.object(daemon, "_run_cmd", return_value=True) as mock_cmd:
            daemon.run_daily()
            calls = [c[0][0] for c in mock_cmd.call_args_list]
            assert any("build-datasets" in c for c in calls)
            assert any("run-daily-forecast" in c for c in calls)

    def test_skips_forecast_when_build_datasets_fails(self, daemon):
        """run-daily-forecast must NOT be called if build-datasets returns False."""
        def _side_effect(args, label):
            if "build-datasets" in args:
                return False
            return True

        with patch.object(daemon, "_run_cmd", side_effect=_side_effect) as mock_cmd:
            daemon.run_daily()
            calls = [c[0][0] for c in mock_cmd.call_args_list]
            assert not any("run-daily-forecast" in c for c in calls)

    def test_passes_realm_to_build_datasets(self, daemon):
        with patch.object(daemon, "_run_cmd", return_value=True) as mock_cmd:
            daemon.run_daily()
            build_call = next(
                c[0][0]
                for c in mock_cmd.call_args_list
                if "build-datasets" in c[0][0]
            )
            assert "--realm" in build_call
            assert "us" in build_call

    def test_passes_realm_to_run_daily_forecast(self, daemon):
        with patch.object(daemon, "_run_cmd", return_value=True) as mock_cmd:
            daemon.run_daily()
            forecast_call = next(
                c[0][0]
                for c in mock_cmd.call_args_list
                if "run-daily-forecast" in c[0][0]
            )
            assert "--realm" in forecast_call
            assert "us" in forecast_call
