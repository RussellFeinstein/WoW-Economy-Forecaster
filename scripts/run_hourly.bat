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
rem    2. Checks the lock file guarding against overlapping runs:
rem         - lock older than STALE_MINUTES: treated as leaked by a crashed
rem           run, logged as STALE LOCK TAKEOVER, deleted, run continues
rem         - lock younger than STALE_MINUTES: run skipped (exit 0)
rem         - the age check failing biases toward takeover: a wedged lock
rem           already caused a 96-day silent outage; a rare double run is
rem           the cheaper failure (SQLite busy_timeout 30s covers it)
rem    3. Calls run-hourly-refresh via the venv CLI executable
rem       (ingest -> normalize -> drift -> provenance)
rem    4. Appends timestamped output to logs\hourly.log
rem
rem  Exit code mirrors the CLI exit code (0 = success; fresh-lock skip = 0).
rem ---------------------------------------------------------------------------

cd /d "%~dp0.."

if not exist logs mkdir logs

set LOCKFILE=data\db\.hourly.lock
rem  Locks older than this many minutes are considered leaked. A healthy
rem  hourly run finishes in well under an hour.
set STALE_MINUTES=180

rem -- Overlap guard with age-based stale-lock takeover ------------------------
rem  PowerShell exits 0 only when the lock is provably fresh. A stale lock,
rem  a vanished file (race), or any PowerShell failure exits nonzero and the
rem  run proceeds. "if errorlevel 1" means ">= 1" and is evaluated at
rem  execution time, so no delayed expansion is needed inside this block.
if exist "%LOCKFILE%" (
    powershell -NoProfile -NonInteractive -Command "try { $age = ((Get-Date) - (Get-Item -LiteralPath '%LOCKFILE%' -ErrorAction Stop).LastWriteTime).TotalMinutes; if ($age -gt %STALE_MINUTES%) { exit 1 } else { exit 0 } } catch { exit 1 }"
    if errorlevel 1 (
        echo [%DATE% %TIME%] STALE LOCK TAKEOVER: lock older than %STALE_MINUTES% minutes or age check failed -- deleting lock and continuing >> logs\hourly.log
        del "%LOCKFILE%" 2>nul
    ) else (
        echo [%DATE% %TIME%] SKIPPED: previous run still active ^(lockfile exists^) >> logs\hourly.log
        exit /b 0
    )
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
