@echo off
setlocal

rem ---------------------------------------------------------------------------
rem  setup_tasks.bat -- Register WoW Economy Forecaster with Windows Task Scheduler
rem
rem  Run once (no Administrator rights needed for per-user tasks).
rem  Idempotent: /F overwrites any existing task with the same name, and a
rem  task an operator has Disabled stays Disabled after re-registration
rem  (see the state-preservation blocks below).
rem
rem  Creates three tasks (all run hidden via run_silent.vbs; the vbs waits
rem  for the batch and propagates its exit code, so Task Scheduler's Last
rem  Run Result stays truthful):
rem    WoWForecaster-Hourly      -- run_hourly.bat every hour at :16
rem    WoWForecaster-Daily       -- run_daily.bat once per day at DAILY_TIME
rem    WoWForecaster-HealthCheck -- run_healthcheck.bat every 6 hours at :45
rem
rem  WOWFC_SCHTASKS may be preset in the environment to point at an
rem  alternate schtasks executable (test seam); defaults to schtasks.
rem
rem  To remove tasks later:
rem    schtasks /Delete /TN "WoWForecaster-Hourly" /F
rem    schtasks /Delete /TN "WoWForecaster-Daily" /F
rem    schtasks /Delete /TN "WoWForecaster-HealthCheck" /F
rem ---------------------------------------------------------------------------

rem ---- Configuration (edit as needed) ----------------------------------------

set HOURLY_TASK=WoWForecaster-Hourly
set DAILY_TASK=WoWForecaster-Daily
set HEALTH_TASK=WoWForecaster-HealthCheck

rem Time to run the daily pipeline (HH:MM, 24-hour local clock)
set DAILY_TIME=07:00

rem Hourly anchor. The :16 minute is deliberate (issue #6): it avoids a
rem head-on collision with the 07:00 daily task, samples away from
rem Blizzard's top-of-hour snapshot refresh, and matches the cloud capture
rem cron (:16 in cloud-snapshot.yml) so the two capture streams stay
rem phase-aligned for catch-up dedup. Pinning /ST means re-running this
rem script cannot silently move the phase.
set HOURLY_TIME=07:16

rem Health-check anchor. 6-hourly firings land 00:45/06:45/12:45/18:45:
rem 29 minutes clear of the :16 ingest (concurrent readers on the large DB
rem have produced transient "database disk image is malformed" errors), and
rem the 06:45 run finishes before the 07:00 daily task starts.
set HEALTH_TIME=00:45

rem ---------------------------------------------------------------------------

if not defined WOWFC_SCHTASKS set "WOWFC_SCHTASKS=schtasks"

set SCRIPTS_DIR=%~dp0
set HOURLY_BAT=%SCRIPTS_DIR%run_hourly.bat
set DAILY_BAT=%SCRIPTS_DIR%run_daily.bat
set HEALTH_BAT=%SCRIPTS_DIR%run_healthcheck.bat
set SILENT_VBS=%SCRIPTS_DIR%run_silent.vbs

echo.
echo  WoW Economy Forecaster -- Task Scheduler Setup
echo  ===============================================
echo  Scripts dir  : %SCRIPTS_DIR%
echo  Hourly task  : %HOURLY_TASK% at :16 past each hour
echo  Daily task   : %DAILY_TASK% at %DAILY_TIME%
echo  Health task  : %HEALTH_TASK% every 6h at %HEALTH_TIME% anchor
echo  Silent mode  : Yes (via run_silent.vbs)
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
if not exist "%HEALTH_BAT%" (
    echo [ERROR] Cannot find run_healthcheck.bat at: %HEALTH_BAT%
    exit /b 1
)
if not exist "%SILENT_VBS%" (
    echo [ERROR] Cannot find run_silent.vbs at: %SILENT_VBS%
    exit /b 1
)

rem ---------------------------------------------------------------------------
rem  State preservation: /Create /F recreates a task ENABLED, so before each
rem  registration the current state is queried and a Disabled task is
rem  re-disabled immediately after its /Create. An operator's decision to
rem  disable a task survives re-running this script (WoWForecaster-Hourly
rem  must stay disabled until the issue #1 runbook re-enables it; one firing
rem  would take over the leaked lock and prune every row past retention).
rem  Exit 2 from the query = task exists and is Disabled. Exit 0 = enabled
rem  or not registered (fresh install). The query itself never exits 1, so
rem  "if errorlevel 1" catches both an affirmative Disabled and any
rem  PowerShell failure (call on an unresolvable name yields errorlevel 1):
rem  for tasks whose accidental enablement can destroy data,
rem  disable-on-uncertainty is the safe bias (the inverse of
rem  run_hourly.bat's raise-on-uncertainty).
rem ---------------------------------------------------------------------------

rem ---- Register hourly task --------------------------------------------------
echo Registering hourly task...

set "HOURLY_WAS_DISABLED=0"
call powershell -NoProfile -NonInteractive -Command "try { if ((Get-ScheduledTask -TaskName '%HOURLY_TASK%' -ErrorAction Stop).State -eq 'Disabled') { exit 2 } else { exit 0 } } catch { exit 0 }"
if errorlevel 1 set "HOURLY_WAS_DISABLED=1"

call "%WOWFC_SCHTASKS%" /Create /SC HOURLY /ST %HOURLY_TIME% ^
    /TN "%HOURLY_TASK%" ^
    /TR "wscript.exe \"%SILENT_VBS%\" \"%HOURLY_BAT%\"" ^
    /F

if %ERRORLEVEL% neq 0 (
    echo [ERROR] Failed to create hourly task.
    exit /b 1
)

if "%HOURLY_WAS_DISABLED%"=="0" goto :hourly_done
call "%WOWFC_SCHTASKS%" /Change /TN "%HOURLY_TASK%" /DISABLE
if errorlevel 1 goto :hourly_disable_failed
echo [OK] %HOURLY_TASK% re-registered; preserved DISABLED state.
goto :hourly_registered
:hourly_disable_failed
echo [ERROR] %HOURLY_TASK% was Disabled before re-registration but re-disabling
echo [ERROR] FAILED. The task is now ENABLED. Disable it manually NOW:
echo [ERROR]     schtasks /Change /TN "%HOURLY_TASK%" /DISABLE
exit /b 1
:hourly_done
echo [OK] Hourly task registered.
:hourly_registered

rem ---- Register daily task ---------------------------------------------------
echo Registering daily task...

set "DAILY_WAS_DISABLED=0"
call powershell -NoProfile -NonInteractive -Command "try { if ((Get-ScheduledTask -TaskName '%DAILY_TASK%' -ErrorAction Stop).State -eq 'Disabled') { exit 2 } else { exit 0 } } catch { exit 0 }"
if errorlevel 1 set "DAILY_WAS_DISABLED=1"

call "%WOWFC_SCHTASKS%" /Create /SC DAILY /ST %DAILY_TIME% ^
    /TN "%DAILY_TASK%" ^
    /TR "wscript.exe \"%SILENT_VBS%\" \"%DAILY_BAT%\"" ^
    /F

if %ERRORLEVEL% neq 0 (
    echo [ERROR] Failed to create daily task.
    exit /b 1
)

if "%DAILY_WAS_DISABLED%"=="0" goto :daily_done
call "%WOWFC_SCHTASKS%" /Change /TN "%DAILY_TASK%" /DISABLE
if errorlevel 1 goto :daily_disable_failed
echo [OK] %DAILY_TASK% re-registered; preserved DISABLED state.
goto :daily_registered
:daily_disable_failed
echo [ERROR] %DAILY_TASK% was Disabled before re-registration but re-disabling
echo [ERROR] FAILED. The task is now ENABLED. Disable it manually NOW:
echo [ERROR]     schtasks /Change /TN "%DAILY_TASK%" /DISABLE
exit /b 1
:daily_done
echo [OK] Daily task registered.
:daily_registered

rem ---- Register health-check task --------------------------------------------
echo Registering health-check task...

set "HEALTH_WAS_DISABLED=0"
call powershell -NoProfile -NonInteractive -Command "try { if ((Get-ScheduledTask -TaskName '%HEALTH_TASK%' -ErrorAction Stop).State -eq 'Disabled') { exit 2 } else { exit 0 } } catch { exit 0 }"
if errorlevel 1 set "HEALTH_WAS_DISABLED=1"

call "%WOWFC_SCHTASKS%" /Create /SC HOURLY /MO 6 /ST %HEALTH_TIME% ^
    /TN "%HEALTH_TASK%" ^
    /TR "wscript.exe \"%SILENT_VBS%\" \"%HEALTH_BAT%\"" ^
    /F

if %ERRORLEVEL% neq 0 (
    echo [ERROR] Failed to create health-check task.
    exit /b 1
)

if "%HEALTH_WAS_DISABLED%"=="0" goto :health_done
call "%WOWFC_SCHTASKS%" /Change /TN "%HEALTH_TASK%" /DISABLE
if errorlevel 1 goto :health_disable_failed
echo [OK] %HEALTH_TASK% re-registered; preserved DISABLED state.
goto :health_registered
:health_disable_failed
echo [ERROR] %HEALTH_TASK% was Disabled before re-registration but re-disabling
echo [ERROR] FAILED. The task is now ENABLED. Disable it manually NOW:
echo [ERROR]     schtasks /Change /TN "%HEALTH_TASK%" /DISABLE
exit /b 1
:health_done
echo [OK] Health-check task registered.
:health_registered

echo.
echo  Done.  To verify:
echo    schtasks /Query /TN "%HOURLY_TASK%"
echo    schtasks /Query /TN "%DAILY_TASK%"
echo    schtasks /Query /TN "%HEALTH_TASK%"
echo.
echo  To run immediately (for testing):
echo    schtasks /Run /TN "%HOURLY_TASK%"
echo    schtasks /Run /TN "%DAILY_TASK%"
echo    schtasks /Run /TN "%HEALTH_TASK%"
echo.
echo  To remove:
echo    schtasks /Delete /TN "%HOURLY_TASK%" /F
echo    schtasks /Delete /TN "%DAILY_TASK%" /F
echo    schtasks /Delete /TN "%HEALTH_TASK%" /F
echo.
