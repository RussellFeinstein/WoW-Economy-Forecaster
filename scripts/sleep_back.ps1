<#
    sleep_back.ps1 -- return rex-desktop to sleep after an unattended run
                      that woke it (issue #78).

    Background. Issue #40 gave the scheduled tasks WakeToRun and delegated the
    return to sleep to Windows' unattended idle timeout. That timeout never
    applies, because the tasks run "Interactive only". A 30-minute STANDBYIDLE
    covered for it during the #40 acceptance and was set back to Never, so in
    steady state nothing returns the machine to sleep and a deliberate
    overnight sleep ends at the first wake timer.

    Two modes:

      -Capture   Print the current LASTINPUTINFO.dwTime and NOTHING else. The
                 caller stashes it in a variable before doing its real work.
                 Any other output on stdout would poison the caller's `for /f`.

      -Decide    Evaluate the overnight window and the four conditions, log the
                 decision, and suspend if they all hold.

    The four conditions, all required, all read at END of run:

      1. This run was itself a wake, attributed by name to a WoWForecaster task
         within -WakeWindowMinutes of the run starting. Of the wake events on
         this box roughly a quarter are Unknown and a tenth are the NIC, so an
         unqualified "did a wake happen" test misfires. This is what separates
         "we woke it" from "you woke it".
      2. No user input during the run, as a dwTime EQUALITY check against the
         value -Capture returned at run start. Not an idle threshold: the
         conditions authorise an action taken up to 43 minutes after the run
         began, and a threshold reads as idle when input arrived early in a
         long run, which suspends the box with the operator sitting at it.
      3. No other WoWForecaster task running, and no hourly lock held. The lock
         half is not redundant: a manual `sync-snapshots` drain takes that lock
         and is not a scheduled task, so the task check alone would suspend the
         machine mid-write.
      4. No unacknowledged health alert. run_healthcheck.bat raises a `cmd /k`
         window on the interactive desktop; if the HOURLY finishes afterwards
         and suspends, that alert is buried even though the hourly did nothing
         wrong. So the guard reads shared state, not the caller's own outcome.

    They are numbered as the issue numbers them. Evaluation order differs and
    is set below: all four are ANDed, so the order changes only which reason
    reaches the log when several hold.

    Plus an overnight window, 20:00-08:00 by default, adjustable per machine
    through WOWFC_SLEEP_FROM_HOUR / WOWFC_SLEEP_UNTIL_HOUR (or the -FromHour /
    -UntilHour parameters, which win). The window WRAPS midnight on purpose: a
    one-sided "refuse at or after 08:00" rule refuses every evening hour too,
    so a machine slept at 22:00 would be woken by the 23:16 hourly and stay up
    all night, which is the failure this script exists to prevent.

    FAIL-SAFE BIAS. Anything that cannot be evaluated leaves the machine AWAKE.
    This is deliberately the INVERSE of the two other biases in scripts/:
    run_hourly.bat takes over a lock whose age it cannot read, and
    run_healthcheck.bat raises its window on a flag it cannot verify. Both act
    on uncertainty because inaction caused a 96-day silent outage. Here the
    costs run the other way round: a wrong sleep interrupts whoever is at the
    keyboard, a wrong wake costs some watts. Do not "fix" this into
    consistency with its neighbours.

    Exit code is ALWAYS 0. The callers invoke this fire-and-forget, detached,
    so it can never change the exit code Task Scheduler records.

    WOWFC_NO_SLEEP, when defined, skips only the suspend call; the decision is
    still logged. Test seam, matching WOWFC_NO_ALERT_WINDOW in
    run_healthcheck.bat. The test suite sets it for every test, because
    without it a green run would suspend the developer's machine.
#>

[CmdletBinding()]
param(
    [switch]$Capture,
    [switch]$Decide,

    # Strings, not typed values, so a malformed argument reaches this script's
    # own validation and the fail-safe path instead of failing PowerShell's
    # parameter binder, which would exit non-zero and break the always-0 contract.
    [string]$InputAtStart,
    [string]$RunStartedAt,
    [string]$CallerTask,
    [string]$FromHour,
    [string]$UntilHour,
    [string]$NowOverride,
    [string]$LogFile,

    [int]$WakeWindowMinutes = 5
)

$DEFAULT_FROM_HOUR  = 20
$DEFAULT_UNTIL_HOUR = 8

# The script lives in scripts/, so its parent's parent is the project root.
# Same convention as the .bat wrappers, which is what lets the test harness
# stand up a throwaway tree by copying this file into tmp_path/scripts.
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$LockPath    = Join-Path $ProjectRoot 'data\db\.hourly.lock'
$AlertPath   = Join-Path $ProjectRoot 'data\outputs\monitoring\health_alert.json'
if (-not $LogFile) { $LogFile = Join-Path $ProjectRoot 'logs\sleep_back.log' }


# ── Win32 ─────────────────────────────────────────────────────────────────────

if (-not ('WowfcPower' -as [type])) {
    Add-Type @'
using System;
using System.Runtime.InteropServices;

public class WowfcPower {
    [StructLayout(LayoutKind.Sequential)]
    public struct LASTINPUTINFO {
        public uint cbSize;
        public uint dwTime;
    }

    [DllImport("user32.dll")]
    private static extern bool GetLastInputInfo(ref LASTINPUTINFO plii);

    // bDisableWakeEvent MUST be passed false. Passing true disables wake
    // timers and silently undoes issue #40.
    [DllImport("powrprof.dll", SetLastError = true)]
    private static extern bool SetSuspendState(bool bHibernate, bool bForce, bool bDisableWakeEvent);

    public static uint LastInputTick() {
        LASTINPUTINFO lii = new LASTINPUTINFO();
        lii.cbSize = (uint)Marshal.SizeOf(lii);
        if (!GetLastInputInfo(ref lii)) {
            throw new Exception("GetLastInputInfo failed");
        }
        return lii.dwTime;
    }

    // Suspend to S3. Never hibernate: this box already suspends to S3 on
    // explicit requests, and writing a full RAM image to disk every hour is
    // the last thing a machine with its memory history needs. The widespread
    // "SetSuspendState hibernates instead" reports come from the invocation
    // form `rundll32 powrprof.dll,SetSuspendState 0,1,0`, which hands "0,1,0"
    // to a three-BOOL function as a single string and drops the arguments.
    public static bool Suspend() {
        return SetSuspendState(false, false, false);
    }
}
'@
}


# ── Capture mode ──────────────────────────────────────────────────────────────
# Deliberately silent apart from the single integer, and never logs.

if ($Capture) {
    try {
        [Console]::Out.WriteLine([WowfcPower]::LastInputTick())
    } catch {
        # Emit nothing. The caller then passes an empty -InputAtStart, which
        # fails to parse and refuses the sleep. Fail-safe by construction.
    }
    exit 0
}


# ── Logging ───────────────────────────────────────────────────────────────────

function Write-Line([string]$Message) {
    try {
        $dir = Split-Path -Parent $LogFile
        if ($dir -and -not (Test-Path -LiteralPath $dir)) {
            New-Item -ItemType Directory -Path $dir -Force | Out-Null
        }
        $stamp = (Get-Date).ToString('yyyy-MM-dd HH:mm:ss')
        [IO.File]::AppendAllText($LogFile, "[$stamp] $Message`r`n")
    } catch {
        # A log write that fails must not take the script down; the exit code
        # contract outranks the log.
    }
}

function Deny([string]$Reason) {
    Write-Line "STAYING AWAKE: $Reason"
    exit 0
}


# ── Bounds ────────────────────────────────────────────────────────────────────

function Resolve-Bound([string]$Param, [string]$EnvName, [int]$Default) {
    # Precedence: explicit parameter, then environment, then default. The
    # operator override and the test seam are the same mechanism, matching the
    # WOWFC override in run_healthcheck.bat.
    $raw = $Param
    if ([string]::IsNullOrWhiteSpace($raw)) {
        $raw = [Environment]::GetEnvironmentVariable($EnvName)
    }
    if ([string]::IsNullOrWhiteSpace($raw)) { return $Default }

    # Whole numbers only. "8.5" and "-1" are rejected rather than coerced,
    # because a silent coercion is a setting the operator did not choose.
    if ($raw -notmatch '^[0-9]+$') { return $null }
    $value = [int]$raw
    if ($value -lt 0 -or $value -gt 23) { return $null }
    return $value
}


# ── Decide mode ───────────────────────────────────────────────────────────────

if (-not $Decide) {
    Write-Line 'STAYING AWAKE: no mode given (expected -Capture or -Decide)'
    exit 0
}

try {
    $from  = Resolve-Bound $FromHour  'WOWFC_SLEEP_FROM_HOUR'  $DEFAULT_FROM_HOUR
    $until = Resolve-Bound $UntilHour 'WOWFC_SLEEP_UNTIL_HOUR' $DEFAULT_UNTIL_HOUR
    if ($null -eq $from -or $null -eq $until) {
        Deny 'bad window bounds (WOWFC_SLEEP_FROM_HOUR / WOWFC_SLEEP_UNTIL_HOUR must be whole hours 0-23)'
    }

    Write-Line ("window: {0:00}:00-{1:00}:00" -f $from, $until)

    # ---- Clock -------------------------------------------------------------
    if ($NowOverride) {
        $now = [datetime]::Parse($NowOverride, [Globalization.CultureInfo]::InvariantCulture)
    } else {
        $now = Get-Date
    }

    # Wrapping window: hour >= from OR hour < until.
    $hour = $now.Hour
    $inWindow = if ($from -le $until) { ($hour -ge $from) -and ($hour -lt $until) }
                else                  { ($hour -ge $from) -or  ($hour -lt $until) }
    if (-not $inWindow) {
        Deny ("outside window (hour {0:00}, window {1:00}:00-{2:00}:00)" -f $hour, $from, $until)
    }

    # The remaining checks are ANDed, so their order changes only which reason
    # gets logged. Shared machine state comes first, ahead of the operator
    # check: when a drain is running AND someone is at the keyboard, "a drain
    # is running" is the more actionable line, and these two are the cheapest
    # checks here (one Test-Path each).

    # ---- Condition 3a: hourly lock ------------------------------------------
    if (Test-Path -LiteralPath $LockPath) {
        Deny 'hourly lock present (a run or a sync-snapshots drain holds it)'
    }

    # ---- Condition 4: unacknowledged health alert ---------------------------
    if (Test-Path -LiteralPath $AlertPath) {
        Deny 'health alert present (an alert window is waiting to be seen)'
    }

    # ---- Condition 2: no user input during the run --------------------------
    if ($InputAtStart -notmatch '^[0-9]+$') {
        Deny "unreadable input stamp from run start ('$InputAtStart')"
    }
    $startTick = [uint64]$InputAtStart
    try {
        $nowTick = [uint64][WowfcPower]::LastInputTick()
    } catch {
        Deny 'could not read current input stamp'
    }
    if ($nowTick -ne $startTick) {
        Deny "user input during run (stamp moved $startTick -> $nowTick)"
    }

    # ---- Condition 1: this run was a wake -----------------------------------
    # Logged unconditionally once reached, so "why did it not sleep" is always
    # answerable from this file.
    $wakeOk = $false
    try {
        $evt = Get-WinEvent -MaxEvents 1 -FilterHashtable @{
            LogName      = 'System'
            ProviderName = 'Microsoft-Windows-Power-Troubleshooter'
            Id           = 1
        } -ErrorAction Stop

        $runStart = if ($RunStartedAt) {
            [datetime]::Parse($RunStartedAt, [Globalization.CultureInfo]::InvariantCulture)
        } else { $now }

        $named = $evt.Message -match 'WoWForecaster'
        # A wake is logged only on resume, so between the wake and the end of
        # the run there is exactly one. The tolerance stops a stale wake from
        # an earlier hour reading as "this run was a wake"; Windows pre-wakes
        # by a variable amount, measured here at 26s and 71s.
        $ageMin = ($runStart - $evt.TimeCreated).TotalMinutes
        $timely = ($ageMin -ge -1) -and ($ageMin -le $WakeWindowMinutes)
        $wakeOk = $named -and $timely

        Write-Line ("wake check: newest event {0:yyyy-MM-dd HH:mm:ss}, named={1}, timely={2} ({3:0.0} min before run start)" -f `
            $evt.TimeCreated, $named, $timely, $ageMin)
    } catch {
        Write-Line 'wake check: no Power-Troubleshooter event could be read'
    }
    if (-not $wakeOk) {
        Deny 'not a wake attributable to a WoWForecaster task'
    }

    # ---- Condition 3b: another WoWForecaster task running -------------------
    # Last because it is the only check that queries Task Scheduler. The
    # caller is excluded by name: the detached child normally evaluates after
    # its own task has already gone Ready, so this guards a race, not the
    # common case.
    try {
        $busy = @(Get-ScheduledTask -TaskName 'WoWForecaster-*' -ErrorAction Stop |
                  Where-Object { $_.State -eq 'Running' -and $_.TaskName -ne $CallerTask })
        if ($busy.Count -gt 0) {
            Deny ("another task running ({0})" -f ($busy.TaskName -join ', '))
        }
    } catch {
        Deny 'could not read scheduled task state'
    }

    # ---- All four hold ------------------------------------------------------
    Write-Line 'SLEEP BACK: all conditions met'
    if ($env:WOWFC_NO_SLEEP) {
        Write-Line 'suspend skipped (WOWFC_NO_SLEEP test seam)'
    } else {
        $ok = [WowfcPower]::Suspend()
        Write-Line "resumed (SetSuspendState returned $ok)"
    }
} catch {
    Write-Line "STAYING AWAKE: unexpected error: $($_.Exception.Message)"
}

exit 0
