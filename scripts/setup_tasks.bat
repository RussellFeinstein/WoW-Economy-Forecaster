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
rem  Creates four tasks (all run hidden via run_silent.vbs; the vbs waits
rem  for the batch and propagates its exit code, so Task Scheduler's Last
rem  Run Result stays truthful):
rem    WoWForecaster-Hourly      -- run_hourly.bat every hour at :16
rem    WoWForecaster-Daily       -- run_daily.bat once per day at DAILY_TIME
rem    WoWForecaster-HealthCheck -- run_healthcheck.bat every 6 hours at :45
rem    WoWForecaster-Backup      -- run_backup.bat once per day at BACKUP_TIME
rem
rem  All four tasks are set to wake the machine from sleep (WakeToRun,
rem  issue #40), so the machine may sleep between runs without losing
rem  capture hours. The script warns at the end if the active power plan
rem  blocks wake timers. Wake covers sleep (and hibernate on supporting
rem  hardware); a powered-off machine does not wake.
rem
rem  WOWFC_SCHTASKS and WOWFC_POWERCFG may be preset in the environment to
rem  point at alternate executables (test seams); they default to schtasks
rem  and powercfg.
rem
rem  To remove tasks later:
rem    schtasks /Delete /TN "WoWForecaster-Hourly" /F
rem    schtasks /Delete /TN "WoWForecaster-Daily" /F
rem    schtasks /Delete /TN "WoWForecaster-HealthCheck" /F
rem    schtasks /Delete /TN "WoWForecaster-Backup" /F
rem ---------------------------------------------------------------------------

rem ---- Configuration (edit as needed) ----------------------------------------

set HOURLY_TASK=WoWForecaster-Hourly
set DAILY_TASK=WoWForecaster-Daily
set HEALTH_TASK=WoWForecaster-HealthCheck
set BACKUP_TASK=WoWForecaster-Backup

rem Time to run the daily pipeline (HH:MM, 24-hour local clock)
set DAILY_TIME=07:00

rem Time to run the durable-table backup. 07:30 sits after the 07:00 daily
rem forecast so each backup includes that morning's forecasts and
rem recommendations. Pinning /ST keeps re-runs from moving the phase.
set BACKUP_TIME=07:30

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
if not defined WOWFC_POWERCFG set "WOWFC_POWERCFG=powercfg"

set SCRIPTS_DIR=%~dp0
set HOURLY_BAT=%SCRIPTS_DIR%run_hourly.bat
set DAILY_BAT=%SCRIPTS_DIR%run_daily.bat
set HEALTH_BAT=%SCRIPTS_DIR%run_healthcheck.bat
set BACKUP_BAT=%SCRIPTS_DIR%run_backup.bat
set SILENT_VBS=%SCRIPTS_DIR%run_silent.vbs

echo.
echo  WoW Economy Forecaster -- Task Scheduler Setup
echo  ===============================================
echo  Scripts dir  : %SCRIPTS_DIR%
echo  Hourly task  : %HOURLY_TASK% at :16 past each hour
echo  Daily task   : %DAILY_TASK% at %DAILY_TIME%
echo  Health task  : %HEALTH_TASK% every 6h at %HEALTH_TIME% anchor
echo  Backup task  : %BACKUP_TASK% at %BACKUP_TIME%
echo  Silent mode  : Yes (via run_silent.vbs)
echo  Wake to run  : Yes (tasks wake the machine from sleep)
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
if not exist "%BACKUP_BAT%" (
    echo [ERROR] Cannot find run_backup.bat at: %BACKUP_BAT%
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
echo [ERROR] FAILED. The task may now be ENABLED. Disable it manually NOW:
echo [ERROR]     schtasks /Change /TN "%HOURLY_TASK%" /DISABLE
exit /b 1
:hourly_done
echo [OK] Hourly task registered.
:hourly_registered

rem Wake-to-run (issue #40): schtasks /Create cannot set WakeToRun, so flip
rem it on the live task by fetch-modify-write, which preserves every other
rem setting (including Enabled=false on a disabled task). Failure is fatal:
rem a registration that cannot wake the machine defeats the point of the
rem wake-to-run change. Running after the re-disable means a failure exit
rem leaves a was-disabled task safely DISABLED.
call powershell -NoProfile -NonInteractive -Command "try { $t = Get-ScheduledTask -TaskName '%HOURLY_TASK%' -ErrorAction Stop; $t.Settings.WakeToRun = $true; Set-ScheduledTask -InputObject $t -ErrorAction Stop | Out-Null; exit 0 } catch { exit 1 }"
if errorlevel 1 goto :hourly_wake_failed
if "%HOURLY_WAS_DISABLED%"=="0" goto :hourly_wake_done
rem Belt and braces: Set-ScheduledTask writes the whole definition back, so
rem re-assert Disabled with the proven schtasks primitive (a no-op when the
rem cmdlet preserved it) rather than trusting the round-trip with a task
rem whose accidental enablement destroys data.
call "%WOWFC_SCHTASKS%" /Change /TN "%HOURLY_TASK%" /DISABLE
if errorlevel 1 goto :hourly_disable_failed
:hourly_wake_done
echo [OK] Wake-to-run set on %HOURLY_TASK%.
goto :hourly_finished
:hourly_wake_failed
echo [ERROR] Could not set wake-to-run on %HOURLY_TASK%. Check that Windows
echo [ERROR] PowerShell is available, then re-run this script.
exit /b 1
:hourly_finished

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
echo [ERROR] FAILED. The task may now be ENABLED. Disable it manually NOW:
echo [ERROR]     schtasks /Change /TN "%DAILY_TASK%" /DISABLE
exit /b 1
:daily_done
echo [OK] Daily task registered.
:daily_registered

rem Wake-to-run (same pattern as the hourly task above).
call powershell -NoProfile -NonInteractive -Command "try { $t = Get-ScheduledTask -TaskName '%DAILY_TASK%' -ErrorAction Stop; $t.Settings.WakeToRun = $true; Set-ScheduledTask -InputObject $t -ErrorAction Stop | Out-Null; exit 0 } catch { exit 1 }"
if errorlevel 1 goto :daily_wake_failed
if "%DAILY_WAS_DISABLED%"=="0" goto :daily_wake_done
call "%WOWFC_SCHTASKS%" /Change /TN "%DAILY_TASK%" /DISABLE
if errorlevel 1 goto :daily_disable_failed
:daily_wake_done
echo [OK] Wake-to-run set on %DAILY_TASK%.
goto :daily_finished
:daily_wake_failed
echo [ERROR] Could not set wake-to-run on %DAILY_TASK%. Check that Windows
echo [ERROR] PowerShell is available, then re-run this script.
exit /b 1
:daily_finished

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
echo [ERROR] FAILED. The task may now be ENABLED. Disable it manually NOW:
echo [ERROR]     schtasks /Change /TN "%HEALTH_TASK%" /DISABLE
exit /b 1
:health_done
echo [OK] Health-check task registered.
:health_registered

rem Wake-to-run (same pattern as the hourly task above).
call powershell -NoProfile -NonInteractive -Command "try { $t = Get-ScheduledTask -TaskName '%HEALTH_TASK%' -ErrorAction Stop; $t.Settings.WakeToRun = $true; Set-ScheduledTask -InputObject $t -ErrorAction Stop | Out-Null; exit 0 } catch { exit 1 }"
if errorlevel 1 goto :health_wake_failed
if "%HEALTH_WAS_DISABLED%"=="0" goto :health_wake_done
call "%WOWFC_SCHTASKS%" /Change /TN "%HEALTH_TASK%" /DISABLE
if errorlevel 1 goto :health_disable_failed
:health_wake_done
echo [OK] Wake-to-run set on %HEALTH_TASK%.
goto :health_finished
:health_wake_failed
echo [ERROR] Could not set wake-to-run on %HEALTH_TASK%. Check that Windows
echo [ERROR] PowerShell is available, then re-run this script.
exit /b 1
:health_finished

rem ---- Register backup task --------------------------------------------------
echo Registering backup task...

set "BACKUP_WAS_DISABLED=0"
call powershell -NoProfile -NonInteractive -Command "try { if ((Get-ScheduledTask -TaskName '%BACKUP_TASK%' -ErrorAction Stop).State -eq 'Disabled') { exit 2 } else { exit 0 } } catch { exit 0 }"
if errorlevel 1 set "BACKUP_WAS_DISABLED=1"

call "%WOWFC_SCHTASKS%" /Create /SC DAILY /ST %BACKUP_TIME% ^
    /TN "%BACKUP_TASK%" ^
    /TR "wscript.exe \"%SILENT_VBS%\" \"%BACKUP_BAT%\"" ^
    /F

if %ERRORLEVEL% neq 0 (
    echo [ERROR] Failed to create backup task.
    exit /b 1
)

if "%BACKUP_WAS_DISABLED%"=="0" goto :backup_done
call "%WOWFC_SCHTASKS%" /Change /TN "%BACKUP_TASK%" /DISABLE
if errorlevel 1 goto :backup_disable_failed
echo [OK] %BACKUP_TASK% re-registered; preserved DISABLED state.
goto :backup_registered
:backup_disable_failed
echo [ERROR] %BACKUP_TASK% was Disabled before re-registration but re-disabling
echo [ERROR] FAILED. The task may now be ENABLED. Disable it manually NOW:
echo [ERROR]     schtasks /Change /TN "%BACKUP_TASK%" /DISABLE
exit /b 1
:backup_done
echo [OK] Backup task registered.
:backup_registered

rem Wake-to-run (same pattern as the hourly task above).
call powershell -NoProfile -NonInteractive -Command "try { $t = Get-ScheduledTask -TaskName '%BACKUP_TASK%' -ErrorAction Stop; $t.Settings.WakeToRun = $true; Set-ScheduledTask -InputObject $t -ErrorAction Stop | Out-Null; exit 0 } catch { exit 1 }"
if errorlevel 1 goto :backup_wake_failed
if "%BACKUP_WAS_DISABLED%"=="0" goto :backup_wake_done
call "%WOWFC_SCHTASKS%" /Change /TN "%BACKUP_TASK%" /DISABLE
if errorlevel 1 goto :backup_disable_failed
:backup_wake_done
echo [OK] Wake-to-run set on %BACKUP_TASK%.
goto :backup_finished
:backup_wake_failed
echo [ERROR] Could not set wake-to-run on %BACKUP_TASK%. Check that Windows
echo [ERROR] PowerShell is available, then re-run this script.
exit /b 1
:backup_finished

rem ---- Verify the power plan allows wake timers (issue #40) ------------------
rem Wake-to-run only works when the active power plan's "Allow wake timers"
rem is Enable (0x1). Disable (0x0) and Important Wake Timers Only (0x2) both
rem block Task Scheduler wake timers. AC value only: this targets a desktop
rem capture rig (on a laptop, check the DC value too). Warn-only: fixing it
rem needs an elevated shell, which this script deliberately does not require.
rem findstr by absolute path: the test harness runs with a stripped PATH.
call "%WOWFC_POWERCFG%" /Q SCHEME_CURRENT SUB_SLEEP RTCWAKE > "%TEMP%\wowfc_rtcwake.txt"
%SystemRoot%\System32\findstr.exe /C:"Current AC Power Setting Index: 0x00000001" "%TEMP%\wowfc_rtcwake.txt" >nul
if errorlevel 1 (
    echo [WARNING] The active power plan does not allow wake timers, so the
    echo [WARNING] tasks cannot wake the machine from sleep. Fix it from an
    echo [WARNING] elevated shell:
    echo [WARNING]     powercfg /SETACVALUEINDEX SCHEME_CURRENT SUB_SLEEP RTCWAKE 1
    echo [WARNING]     powercfg /SetActive SCHEME_CURRENT
)
del "%TEMP%\wowfc_rtcwake.txt" >nul 2>&1

echo.
echo  Done.  To verify:
echo    schtasks /Query /TN "%HOURLY_TASK%"
echo    schtasks /Query /TN "%DAILY_TASK%"
echo    schtasks /Query /TN "%HEALTH_TASK%"
echo    schtasks /Query /TN "%BACKUP_TASK%"
echo.
echo  To run immediately (for testing):
echo    schtasks /Run /TN "%HOURLY_TASK%"
echo    schtasks /Run /TN "%DAILY_TASK%"
echo    schtasks /Run /TN "%HEALTH_TASK%"
echo    schtasks /Run /TN "%BACKUP_TASK%"
echo.
echo  To remove:
echo    schtasks /Delete /TN "%HOURLY_TASK%" /F
echo    schtasks /Delete /TN "%DAILY_TASK%" /F
echo    schtasks /Delete /TN "%HEALTH_TASK%" /F
echo    schtasks /Delete /TN "%BACKUP_TASK%" /F
echo.
