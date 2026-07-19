@echo off
setlocal EnableDelayedExpansion

rem ---------------------------------------------------------------------------
rem  run_daily.bat -- WoW Economy Forecaster daily pipeline wrapper
rem
rem  Intended for use with Windows Task Scheduler (see setup_tasks.bat).
rem  Can also be run manually from any command prompt.
rem
rem  What it does:
rem    1. Navigates to the project root (one level above this scripts/ folder)
rem    2. Runs check-data-health as a freshness gate: if the last successful
rem       ingest is older than 26 hours, the forecast steps are skipped and
rem       the script exits non-zero so Task Scheduler records the failure
rem       (issue #12 -- forecasts must never be generated from stale data)
rem    3. Runs build-datasets (builds training + inference Parquets from the
rem       normalized observations written by run-hourly-refresh)
rem    4. If step 3 succeeds, runs run-daily-forecast
rem       (train LightGBM -> forecast 1d/7d/28d -> score recommendations)
rem    5. Appends timestamped output to logs\daily.log
rem
rem  If any step fails, the remaining steps are skipped and the script exits
rem  with the failing exit code so Task Scheduler can detect the error.
rem
rem  WOWFC may be preset in the environment to point at an alternate CLI
rem  executable (test seam); it defaults to the project venv executable.
rem ---------------------------------------------------------------------------

cd /d "%~dp0.."

if not defined WOWFC set "WOWFC=.venv\Scripts\wowfc.exe"

if not exist logs mkdir logs

echo [%DATE% %TIME%] ============================================================ >> logs\daily.log
echo [%DATE% %TIME%] Daily pipeline starting >> logs\daily.log

rem ---- Step 1: freshness gate ------------------------------------------------
echo [%DATE% %TIME%] Step 1/3: check-data-health (freshness gate) >> logs\daily.log

call "%WOWFC%" check-data-health --stale-hours 26 >> logs\daily.log 2>&1
set "HEALTH_CODE=!ERRORLEVEL!"

echo [%DATE% %TIME%] check-data-health exited with code !HEALTH_CODE! >> logs\daily.log

if !HEALTH_CODE! neq 0 (
    echo [%DATE% %TIME%] HEALTH ALERT ACTIVE -- data stale, skipping forecast steps >> logs\daily.log
    exit /b !HEALTH_CODE!
)

echo [%DATE% %TIME%] Freshness gate OK >> logs\daily.log

rem ---- Step 2: build feature datasets ----------------------------------------
echo [%DATE% %TIME%] Step 2/3: build-datasets >> logs\daily.log

call "%WOWFC%" build-datasets >> logs\daily.log 2>&1
set "BUILD_CODE=!ERRORLEVEL!"

echo [%DATE% %TIME%] build-datasets exited with code !BUILD_CODE! >> logs\daily.log

if !BUILD_CODE! neq 0 (
    echo [%DATE% %TIME%] build-datasets FAILED -- skipping forecast >> logs\daily.log
    exit /b !BUILD_CODE!
)

echo [%DATE% %TIME%] build-datasets OK >> logs\daily.log

rem ---- Step 3: train + forecast + recommend -----------------------------------
echo [%DATE% %TIME%] Step 3/3: run-daily-forecast >> logs\daily.log

call "%WOWFC%" run-daily-forecast >> logs\daily.log 2>&1
set "FORECAST_CODE=!ERRORLEVEL!"

echo [%DATE% %TIME%] run-daily-forecast exited with code !FORECAST_CODE! >> logs\daily.log
echo [%DATE% %TIME%] Daily pipeline complete >> logs\daily.log

exit /b !FORECAST_CODE!
