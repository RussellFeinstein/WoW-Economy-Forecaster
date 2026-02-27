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
rem    2. Calls run-hourly-refresh via the venv CLI executable
rem       (ingest -> normalize -> drift -> provenance)
rem    3. Appends timestamped output to logs\hourly.log
rem
rem  Exit code mirrors the CLI exit code (0 = success).
rem ---------------------------------------------------------------------------

cd /d "%~dp0.."

if not exist logs mkdir logs

echo [%DATE% %TIME%] ============================================================ >> logs\hourly.log
echo [%DATE% %TIME%] Hourly refresh starting >> logs\hourly.log

.venv\Scripts\wowfc.exe run-hourly-refresh >> logs\hourly.log 2>&1
set EXIT_CODE=%ERRORLEVEL%

echo [%DATE% %TIME%] Hourly refresh complete (exit %EXIT_CODE%) >> logs\hourly.log

exit /b %EXIT_CODE%
