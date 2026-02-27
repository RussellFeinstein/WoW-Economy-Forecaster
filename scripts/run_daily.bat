@echo off
setlocal

rem ---------------------------------------------------------------------------
rem  run_daily.bat -- WoW Economy Forecaster daily pipeline wrapper
rem
rem  Intended for use with Windows Task Scheduler (see setup_tasks.bat).
rem  Can also be run manually from any command prompt.
rem
rem  What it does:
rem    1. Navigates to the project root (one level above this scripts/ folder)
rem    2. Runs build-datasets (builds training + inference Parquets from the
rem       normalized observations written by run-hourly-refresh)
rem    3. If step 2 succeeds, runs run-daily-forecast
rem       (train LightGBM -> forecast 1d/7d/28d -> score recommendations)
rem    4. Appends timestamped output to logs\daily.log
rem
rem  If build-datasets fails, run-daily-forecast is skipped and the script
rem  exits with the failing exit code so Task Scheduler can detect the error.
rem
rem  Exit code mirrors the last CLI step that ran (0 = success).
rem ---------------------------------------------------------------------------

cd /d "%~dp0.."

if not exist logs mkdir logs

echo [%DATE% %TIME%] ============================================================ >> logs\daily.log
echo [%DATE% %TIME%] Daily pipeline starting >> logs\daily.log

rem ---- Step 1: build feature datasets ----------------------------------------
echo [%DATE% %TIME%] Step 1/2: build-datasets >> logs\daily.log

.venv\Scripts\wowfc.exe build-datasets >> logs\daily.log 2>&1
set BUILD_CODE=%ERRORLEVEL%

if %BUILD_CODE% neq 0 (
    echo [%DATE% %TIME%] build-datasets FAILED (exit %BUILD_CODE%) -- skipping forecast >> logs\daily.log
    exit /b %BUILD_CODE%
)

echo [%DATE% %TIME%] build-datasets OK >> logs\daily.log

rem ---- Step 2: train + forecast + recommend -----------------------------------
echo [%DATE% %TIME%] Step 2/2: run-daily-forecast >> logs\daily.log

.venv\Scripts\wowfc.exe run-daily-forecast >> logs\daily.log 2>&1
set FORECAST_CODE=%ERRORLEVEL%

echo [%DATE% %TIME%] Daily pipeline complete (exit %FORECAST_CODE%) >> logs\daily.log

exit /b %FORECAST_CODE%
