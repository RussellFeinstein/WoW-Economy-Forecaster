@echo off
setlocal

rem ---------------------------------------------------------------------------
rem  run_hourly.bat -- WoW Economy Forecaster hourly pipeline wrapper
rem
rem  Intended for use with Windows Task Scheduler (see setup_tasks.bat).
rem  Can also be run manually from any command prompt.
rem
rem  What it does:
rem    1. Navigates to the project root (one level above this scripts/ folder)
rem    2. Checks for a lock file to prevent overlapping runs
rem    3. Calls run-hourly-refresh via the venv CLI executable
rem       (ingest -> normalize -> drift -> provenance)
rem    4. Appends timestamped output to logs\hourly.log
rem
rem  Exit code mirrors the CLI exit code (0 = success, skipped = 0).
rem ---------------------------------------------------------------------------

cd /d "%~dp0.."

if not exist logs mkdir logs

rem ── Overlap guard ─────────────────────────────────────────────────────────
rem  If a previous hourly run is still active, skip this run.
set LOCKFILE=data\db\.hourly.lock
if exist "%LOCKFILE%" (
    echo [%DATE% %TIME%] SKIPPED: previous run still active ^(lockfile exists^) >> logs\hourly.log
    exit /b 0
)

rem Create lockfile
echo %DATE% %TIME% > "%LOCKFILE%"

echo [%DATE% %TIME%] ============================================================ >> logs\hourly.log
echo [%DATE% %TIME%] Hourly refresh starting >> logs\hourly.log

.venv\Scripts\wowfc.exe run-hourly-refresh >> logs\hourly.log 2>&1
set EXIT_CODE=%ERRORLEVEL%

echo [%DATE% %TIME%] Hourly refresh complete (exit %EXIT_CODE%) >> logs\hourly.log

rem Remove lockfile
del "%LOCKFILE%" 2>nul

exit /b %EXIT_CODE%
