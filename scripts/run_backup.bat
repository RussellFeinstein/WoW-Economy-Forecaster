@echo off
setlocal EnableDelayedExpansion

rem ---------------------------------------------------------------------------
rem  run_backup.bat -- WoW Economy Forecaster durable-table backup
rem
rem  Intended for use with Windows Task Scheduler (registered by setup_tasks.bat
rem  as WoWForecaster-Backup, daily at 07:30 -- after the 07:00 daily forecast,
rem  so each backup includes that morning's fresh forecasts and recommendations).
rem  Can also be run manually.
rem
rem  What it does:
rem    1. Navigates to the project root (one level above this scripts/ folder)
rem    2. Runs backup-durable-db --upload via the venv CLI: builds a restorable
rem       .db.gz of the durable tables (everything except the two large
rem       per-observation tables) and uploads it to the separate, no-expiry R2
rem       backup bucket configured via BACKUP_S3_* in .env (see docs/db-backup.md)
rem    3. Appends timestamped output to logs\backup.log
rem
rem  Exit code mirrors backup-durable-db (0 = success), so Task Scheduler's Last
rem  Run Result is a truthful, independent backup-health signal. Staleness
rem  alerting is handled separately by the scheduled health check.
rem
rem  WOWFC may be preset in the environment to point at an alternate CLI
rem  executable (test seam); it defaults to the project venv executable.
rem ---------------------------------------------------------------------------

cd /d "%~dp0.."

if not defined WOWFC set "WOWFC=.venv\Scripts\wowfc.exe"

if not exist logs mkdir logs

echo [%DATE% %TIME%] ============================================================ >> logs\backup.log
echo [%DATE% %TIME%] Durable backup starting >> logs\backup.log

call "%WOWFC%" backup-durable-db --upload >> logs\backup.log 2>&1
set "BACKUP_CODE=!ERRORLEVEL!"

echo [%DATE% %TIME%] backup-durable-db exited with code !BACKUP_CODE! >> logs\backup.log
echo [%DATE% %TIME%] Durable backup complete >> logs\backup.log

exit /b !BACKUP_CODE!
