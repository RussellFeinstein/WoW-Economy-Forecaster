# M11. How this system fails

Part III. Failure, history, operations. No lab; the fixes already shipped.

## Why this module exists

On 2026-04-15 at 07:16:01 an hourly run crashed without deleting
`data/db/.hourly.lock`. The wrapper read that lock as "previous run still
active", logged SKIPPED, and exited 0.

It did that 1,933 times. Windows Task Scheduler recorded success every single
time. The daily forecast task kept succeeding too, producing forecasts from
features frozen at the last real ingest. Nothing surfaced for 96 days, and the
last good market data was 105 days old by the time ingestion came back.

Nothing about this is specific to auction houses, or to Windows, or to
forecasting. A guard whose uncertain branch resolves toward doing nothing
quietly, plus a skip modeled as success, plus a monitor that was written but
never scheduled. That combination is available in every system anyone has ever
operated, which makes this the most portable thing in the track.

## The idea to hold onto

Two sentences, and everything in the module is a consequence of one of them.

**A skip that exits 0 is invisible at every level anyone actually monitors.**
Correct and quiet loses to loud. Task Scheduler's Last Run Result was truthful
about what the script returned and useless about what the system was doing.

**A pipeline that clamps to the newest data cannot tell fresh from frozen.**
Every component was behaving correctly given the data it was handed. Internal
consistency is not liveness. Freshness has to be checked against the clock,
somewhere that runs whether or not new data arrived.

From those two, one design rule the repo now applies deliberately: an alarm
whose healthy state is reachable by the failure it watches for is decorative.
That is why the cloud gap guard's floor stays at 20 captured hours when the
failure mode delivers 11, why a missing backup counts as a stale backup, and why
an unverifiable lock age is treated as a leaked lock.

## Read this first

The repo is the textbook. Read these before drilling:

- [`docs/postmortem-2026-04-lock-outage.md`](../../docs/postmortem-2026-04-lock-outage.md)
  All of it, including the timeline. Read the root cause chain as four separate
  failures and ask which one produced the green light. Then read the timeline's
  first two rows against the data impact section and notice what does not
  reconcile.
- [`scripts/run_hourly.bat`](../../scripts/run_hourly.bat)
  The comment block and the overlap guard. Work out exactly which conditions
  reach the takeover branch, including the ones that are not about the lock's
  age at all.
- [`wow_forecaster/reporting/health.py`](../../wow_forecaster/reporting/health.py)
  `HealthReport.has_failures` and the four checks feeding it. Note which checks
  are opt-in and why, and why the `OSError` path on the lock stat passes rather
  than fails.
- [`wow_forecaster/pipeline/forecast.py`](../../wow_forecaster/pipeline/forecast.py)
  The `Freshness gate` docstring section and `StaleDataError`. This is the
  second seam; [`scripts/run_daily.bat`](../../scripts/run_daily.bat) is the
  first.
- [`wow_forecaster/ingestion/cloud_sync.py`](../../wow_forecaster/ingestion/cloud_sync.py)
  The `hourly_lock` docstring. Same lock file as the bat, same takeover
  threshold, one deliberate difference, and the docstring says why.
- [`wow_forecaster/ingestion/cloud_fetch.py`](../../wow_forecaster/ingestion/cloud_fetch.py)
  `GUARD_MIN_HOURS_DEFAULT` and `evaluate_gap_guard`, plus the trigger-model
  section of [`docs/cloud-capture.md`](../../docs/cloud-capture.md).

## What you should be able to do afterwards

- Trace how a leaked lock file plus an exit-0 skip becomes 105 days of silence,
  naming all four links in the chain.
- State the dead-man principle in one sentence and point at two places this repo
  applies it and one place it does not.
- Say why the catch-up sync waits and then fails loudly where the hourly wrapper
  skips, in terms of what each skip costs.
- Name the two seams enforcing the freshness gate, and say why the thresholds
  are 4 hours at one surface and 26 at another.
- Given today's guards, say which one would have gone red first on day one of
  the wedge, and roughly when.

## A note on what this module is not

It is not a list of things that were fixed. Every guard here exists because
something specific went wrong, and each one encodes a cost comparison rather
than a best practice: take a lock over because a double run is cheaper than a
dead pipeline, wait on the catch-up lock because a lost night is dearer than a
lost hour, block forecasts at 26 hours but alarm at 4.

When you carry these forward, carry the comparisons. The thresholds are local
and the reasoning is not.
