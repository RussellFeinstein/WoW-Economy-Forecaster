@echo off
setlocal EnableDelayedExpansion

rem ---------------------------------------------------------------------------
rem  run_healthcheck.bat -- WoW Economy Forecaster scheduled health check
rem
rem  Intended for use with Windows Task Scheduler (registration ships with
rem  setup_tasks.bat in issue #6).  Can also be run manually.
rem
rem  What it does:
rem    1. Navigates to the project root (one level above this scripts/ folder)
rem    2. Runs check-data-health --stale-hours %STALE_HOURS%
rem       --backup-stale-hours %BACKUP_STALE_HOURS% --integrity-scope durable
rem       via the venv CLI (the backup and integrity checks are opt-in here so
rem       a stale backup or a corrupt durable table surfaces through this
rem       alert window without ever blocking the daily forecast gate) and
rem       appends its output to logs\health.log
rem    3. On failure:
rem         - writes data\outputs\monitoring\health_alert.json (timestamp,
rem           exit code, last 20 log lines) as a durable machine-readable
rem           alert surface
rem         - raises a persistent red console window (start + cmd /k) unless
rem           one was already raised within the last SUPPRESS_HOURS, tracked
rem           by the mtime of the flag file health_window_raised.json
rem         - an unverifiable flag (vanished file, PowerShell failure) raises
rem           anyway: skip-on-uncertainty is how the 96-day silent outage
rem           happened, and the worst case here is one extra window
rem    4. On success: deletes the alert file and the flag file (a healthy run
rem       ends the outage episode, so the next failure alerts immediately)
rem    5. Hands off to sleep_back.ps1 on both paths (issue #78).  The ordering
rem       is self-documenting: on the failure path the alert JSON has just
rem       been written, so the helper's condition 4 refuses and this check
rem       cannot sleep away its own alert; on the healthy path it has just
rem       been deleted, so the machine may go back down.  Only the 00:45 and
rem       06:45 runs are ever eligible anyway, since the helper's default
rem       window ends at 08:00.
rem
rem  Exit code always mirrors check-data-health (0 = healthy).  Alert-surface
rem  failures are logged but never change the exit code, so Task Scheduler's
rem  Last Run Result stays a truthful health signal.
rem
rem  WOWFC may be preset in the environment to point at an alternate CLI
rem  executable (test seam); it defaults to the project venv executable.
rem  WOWFC_NO_ALERT_WINDOW, when defined, skips only the start window (test
rem  seam so the suite never pops consoles); all logging still happens.
rem  WOWFC_NO_SLEEP, when defined, likewise skips only the suspend in
rem  sleep_back.ps1; its decision is still logged.
rem ---------------------------------------------------------------------------

cd /d "%~dp0.."

if not defined WOWFC set "WOWFC=.venv\Scripts\wowfc.exe"

set "STALE_HOURS=4"
set "BACKUP_STALE_HOURS=30"
set "SUPPRESS_HOURS=24"
set "ALERTFILE=data\outputs\monitoring\health_alert.json"
set "FLAGFILE=data\outputs\monitoring\health_window_raised.json"

if not exist logs mkdir logs
if not exist data\outputs\monitoring mkdir data\outputs\monitoring

rem -- Sleep-back capture (issue #78) ------------------------------------------
rem  Read before the check runs, so the end-of-run comparison covers the whole
rem  run.  See run_hourly.bat for why this is an equality check and not a
rem  threshold.
set "INPUT_AT_START="
set "RUN_STARTED_AT="
for /f "usebackq delims=" %%i in (`powershell -NoProfile -NonInteractive -ExecutionPolicy Bypass -File "%~dp0sleep_back.ps1" -Capture`) do set "INPUT_AT_START=%%i"
for /f "usebackq delims=" %%i in (`powershell -NoProfile -NonInteractive -Command "(Get-Date).ToString('s')"`) do set "RUN_STARTED_AT=%%i"

rem  Entry separator, untimestamped so it does not read as one more log line.
rem  Redirection leads: cmd keeps the space before a trailing `>>` in the text.
>> logs\health.log echo =========================================================================================
echo [%DATE% %TIME%] Health check starting ^(stale threshold %STALE_HOURS%h^) >> logs\health.log

call "%WOWFC%" check-data-health --stale-hours %STALE_HOURS% --backup-stale-hours %BACKUP_STALE_HOURS% --integrity-scope durable >> logs\health.log 2>&1
set "HC_CODE=!ERRORLEVEL!"

echo [%DATE% %TIME%] check-data-health exited with code !HC_CODE! >> logs\health.log

if !HC_CODE! equ 0 goto :healthy

rem ---- Failure path ----------------------------------------------------------
rem  Every step below is fire-and-forget: the final exit uses the saved
rem  HC_CODE, so a failed PowerShell or start can never mask the health code.
echo [%DATE% %TIME%] HEALTH CHECK FAILED ^(exit !HC_CODE!^) -- raising alert surfaces >> logs\health.log

rem  Alert JSON: timestamp + exit code + log tail.  HC_CODE crosses into
rem  PowerShell via the environment (no string-interpolation quoting hazards).
rem  [IO.File]::ReadAllLines instead of Get-Content: provider cmdlets decorate
rem  each line with PSPath/ReadCount ETS properties and ConvertTo-Json then
rem  serializes objects, not strings.  [IO.File]::WriteAllText writes UTF-8
rem  without BOM; Set-Content -Encoding UTF8 in PowerShell 5.1 writes a BOM,
rem  which breaks json.loads.
powershell -NoProfile -NonInteractive -Command "try { $root = (Get-Location).Path; $snip = @([IO.File]::ReadAllLines((Join-Path $root 'logs\health.log')) | Select-Object -Last 20); $obj = [pscustomobject]@{ raised_at = [DateTime]::UtcNow.ToString('yyyy-MM-ddTHH:mm:ssZ'); exit_code = [int]$env:HC_CODE; log_snippet = $snip }; [IO.File]::WriteAllText((Join-Path $root '%ALERTFILE%'), (ConvertTo-Json $obj)); exit 0 } catch { exit 1 }"
if errorlevel 1 echo [%DATE% %TIME%] WARNING: failed to write health_alert.json >> logs\health.log

rem  Window suppression: suppress ONLY when the flag file is provably younger
rem  than SUPPRESS_HOURS.  A stale flag, a vanished file (race), or any
rem  PowerShell failure falls through to raise.
if not exist "%FLAGFILE%" goto :raise
powershell -NoProfile -NonInteractive -Command "try { $age = ((Get-Date) - (Get-Item -LiteralPath '%FLAGFILE%' -ErrorAction Stop).LastWriteTime).TotalHours; if ($age -lt %SUPPRESS_HOURS%) { exit 0 } else { exit 1 } } catch { exit 1 }"
if errorlevel 1 goto :raise
echo [%DATE% %TIME%] ALERT SUPPRESSED: window already raised within last %SUPPRESS_HOURS% hours >> logs\health.log
goto :finish_fail

:raise
echo [%DATE% %TIME%] ALERT WINDOW RAISED >> logs\health.log
echo {"raised_at": "%DATE% %TIME%"} > "%FLAGFILE%"
if not defined WOWFC_NO_ALERT_WINDOW goto :window
echo [%DATE% %TIME%] alert window skipped ^(WOWFC_NO_ALERT_WINDOW test seam^) >> logs\health.log
goto :finish_fail

:window
rem  2>nul on type: the alert JSON may not exist when the PowerShell write
rem  failed; the title, color, and log pointer still carry the message.
start "WOW FORECASTER: DATA STALE" cmd /k "color CF & echo WOW FORECASTER: DATA STALE & echo. & type %ALERTFILE% 2>nul & echo. & echo See logs\health.log for details"

:finish_fail
call :sleep_back
exit /b !HC_CODE!

:healthy
echo [%DATE% %TIME%] Health check OK >> logs\health.log
if exist "%ALERTFILE%" (
    del "%ALERTFILE%" 2>nul
    echo [%DATE% %TIME%] cleared health_alert.json >> logs\health.log
)
if exist "%FLAGFILE%" del "%FLAGFILE%" 2>nul
call :sleep_back
exit /b 0

rem ---- Sleep back (issue #78) ------------------------------------------------
rem  DETACHED on purpose: SetSuspendState does not return until the machine
rem  resumes and run_silent.vbs waits on this .bat, so an inline call would
rem  hold the task Running for the whole sleep.  `start "" /b` lets the .bat
rem  exit and the helper decide afterwards.  The helper always exits 0 and its
rem  result is ignored here, so HC_CODE reaches Task Scheduler untouched.
:sleep_back
start "" /b powershell -NoProfile -NonInteractive -ExecutionPolicy Bypass -File "%~dp0sleep_back.ps1" -Decide -InputAtStart "!INPUT_AT_START!" -RunStartedAt "!RUN_STARTED_AT!" -CallerTask "WoWForecaster-HealthCheck"
goto :eof
