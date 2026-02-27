@echo off
setlocal

rem ---------------------------------------------------------------------------
rem  setup_tasks.bat -- Register WoW Economy Forecaster with Windows Task Scheduler
rem
rem  Run once (no Administrator rights needed for per-user tasks).
rem  Idempotent: /F overwrites any existing task with the same name.
rem
rem  Creates two tasks:
rem    WoWForecaster-Hourly  -- runs run_hourly.bat every hour
rem    WoWForecaster-Daily   -- runs run_daily.bat once per day at DAILY_TIME
rem
rem  To remove tasks later:
rem    schtasks /Delete /TN "WoWForecaster-Hourly" /F
rem    schtasks /Delete /TN "WoWForecaster-Daily" /F
rem ---------------------------------------------------------------------------

rem ---- Configuration (edit as needed) ----------------------------------------

set HOURLY_TASK=WoWForecaster-Hourly
set DAILY_TASK=WoWForecaster-Daily

rem Time to run the daily pipeline (HH:MM, 24-hour local clock)
set DAILY_TIME=07:00

rem ---------------------------------------------------------------------------

set SCRIPTS_DIR=%~dp0
set HOURLY_BAT=%SCRIPTS_DIR%run_hourly.bat
set DAILY_BAT=%SCRIPTS_DIR%run_daily.bat

echo.
echo  WoW Economy Forecaster -- Task Scheduler Setup
echo  ===============================================
echo  Scripts dir  : %SCRIPTS_DIR%
echo  Hourly task  : %HOURLY_TASK%
echo  Daily task   : %DAILY_TASK% at %DAILY_TIME%
echo.

rem ---- Verify wrapper scripts exist ------------------------------------------
if not exist "%HOURLY_BAT%" (
    echo [ERROR] Cannot find run_hourly.bat at: %HOURLY_BAT%
    exit /b 1
)
if not exist "%DAILY_BAT%" (
    echo [ERROR] Cannot find run_daily.bat at: %DAILY_BAT%
    exit /b 1
)

rem ---- Register hourly task --------------------------------------------------
echo Registering hourly task...
schtasks /Create /SC HOURLY ^
    /TN "%HOURLY_TASK%" ^
    /TR "cmd /c \"%HOURLY_BAT%\"" ^
    /F

if %ERRORLEVEL% neq 0 (
    echo [ERROR] Failed to create hourly task.
    exit /b 1
)
echo [OK] Hourly task registered.

rem ---- Register daily task ---------------------------------------------------
echo Registering daily task...
schtasks /Create /SC DAILY /ST %DAILY_TIME% ^
    /TN "%DAILY_TASK%" ^
    /TR "cmd /c \"%DAILY_BAT%\"" ^
    /F

if %ERRORLEVEL% neq 0 (
    echo [ERROR] Failed to create daily task.
    exit /b 1
)
echo [OK] Daily task registered.

echo.
echo  Done.  To verify:
echo    schtasks /Query /TN "%HOURLY_TASK%"
echo    schtasks /Query /TN "%DAILY_TASK%"
echo.
echo  To run immediately (for testing):
echo    schtasks /Run /TN "%HOURLY_TASK%"
echo    schtasks /Run /TN "%DAILY_TASK%"
echo.
echo  To remove:
echo    schtasks /Delete /TN "%HOURLY_TASK%" /F
echo    schtasks /Delete /TN "%DAILY_TASK%" /F
echo.
