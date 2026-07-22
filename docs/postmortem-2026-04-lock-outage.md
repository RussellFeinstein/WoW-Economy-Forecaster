# Postmortem: the 2026-04-15 lock leak and the 96-day silent ingestion outage

Written 2026-07-22, one day after the restore. Tracking issue: [#2](https://github.com/RussellFeinstein/WoW-Economy-Forecaster/issues/2). Restore runbook and execution record: [#1](https://github.com/RussellFeinstein/WoW-Economy-Forecaster/issues/1).

## Summary

A crashed hourly run leaked its lock file on 2026-04-15. Every hourly run for the next 96 days saw the lock, logged SKIPPED, and exited 0, so Windows Task Scheduler recorded 1,933 consecutive successes while zero market data arrived. The daily forecast task kept succeeding the whole time, generating forecasts from features frozen at the last real ingest. Nothing surfaced until a full repo review on 2026-07-12. Ingestion was restored 2026-07-21 02:43Z, 105 days after the last good ingest.

## Timeline

| When | What |
|---|---|
| 2026-04-07 | Last successful hourly ingest. |
| 2026-04-15 07:16:01 | A run crashed without deleting `data/db/.hourly.lock` (a process-tree kill leaves a .bat no cleanup path). The wedge begins. |
| 2026-04-15 to 2026-07-12 | 1,933 hourly runs see the lock, log SKIPPED, exit 0. Task Scheduler shows green. The daily forecast keeps "succeeding" on frozen features. |
| 2026-07-12 | State-of-project review discovers the outage. Diagnosis also finds that the wedged lock is the only thing stopping the next run's retention pruner from deleting the surviving observation history, so the lock is left in place and the restore is planned as a runbook (#1). |
| 2026-07-15 to 2026-07-20 | Hardening lands before the lock is touched: stale-lock takeover (#3, v2.4.9), daily freshness gate (#12, v2.5.0), scheduled health check with visible alerting (#4, v2.6.0), lock-age and retention sentinels (#5, v2.7.0), committed task registration (#6, v2.7.1), wake-to-run (#40, v2.7.2). Cloud capture (#42) goes live 2026-07-20 21:02Z as an off-machine backstop. |
| 2026-07-20/21 | Runbook #1 executes: rollups backed up and partially backfilled, evidence captured, both observation tables dropped and rebuilt (a corrupt page from a 2026-07-20 hard crash made row-wise pruning unsafe), database compacted 78 GB to 105 MB, lock deleted, first manual run green, all tasks re-enabled and observed green overnight. |
| 2026-07-22 | Day-one gap verification (#11) runs clean; results below. |

## Root cause chain

Four independent failures had to line up, and all four did:

1. **The overlap guard had no staleness handling.** `run_hourly.bat` treated any existing lock as "previous run still active." A crash-leaked lock is indistinguishable from a running job under that rule, and crash-leaked locks are inevitable: the .bat has no cleanup path when its process tree is killed.
2. **The skip path exited 0.** Skipping because of the lock was treated as a normal outcome, so the scheduler-level view (Last Run Result) showed success 1,933 times. The only place the word SKIPPED appeared was a log file nobody was reading.
3. **The health check existed but was never scheduled.** `check-data-health` shipped in v2.1.0 and would have flagged stale data within hours. It was never registered as a task, so it caught nothing. Monitoring that is not scheduled is documentation.
4. **The daily forecast could not notice.** The feature date spine clamps to the newest observation, so training and forecasting kept running on the frozen window without an error. Forecast CSVs kept appearing daily, which read as a healthy system.

A fifth factor made the eventual restore harder: the retention pruner deletes raw rows older than 30 days on every hourly run. By discovery time, simply removing the lock would have let the next run mass-delete the only surviving in-database history. The outage was protecting the data from the cleanup job.

## Data impact

- **Hourly market data 2026-04-08 through 2026-07-20 is permanently lost**, minus what cloud capture collected from 2026-07-20 21:02Z. Blizzard's commodities endpoint serves only the current snapshot; there is no history API to backfill from.
- **Daily rollups:** backfill recovered 12 of the 18 missing dates (2026-03-20 through 2026-04-02, plus 2026-02-24), taking durable coverage from 22 to 34 dates. 2026-04-03 through 2026-04-06 were past the retention window in the raw data and are unrecoverable. Every recovered date was certified against at least two independent sources after two hardware-induced corruption incidents during the backfill (full forensics in the evidence tree).
- **About 90 days of forecasts were generated from frozen features.** They are now partially measurable: as those forecasts mature against post-restore actuals, day-one live MAE reads 2,456g at the 7d horizon and 2,077g at 28d (n=20 archetype forecasts). No baseline ratio is available (the surviving backtest rows carry no fold results for these horizons; the real baseline arrives with M1).
- **The database file** was rebuilt from 78 GB to 105 MB. The raw table's contents were all past retention and scheduled for deletion regardless; a corrupt page from the 2026-07-20 hard crash sat in that same table and was eliminated by the drop-and-rebuild.

## What the restored system verified (issue #11, day one)

The feature pipeline had never seen a 90-day hole. Day-one checks all passed: the date spine fills the gap with all-None rows (20 archetypes x 149 days, exact), lag and rolling windows return None rather than fabricated values when they reach into the gap, the z-score cold restart fell back to batch stats with a day-one outlier rate of 3.4 to 4.5 percent (under the 5 percent line), and `check-data-health --lookback-days 100` reports HEALTHY with all ~98 gap dates listed. One structural note for future outages: the drift detector's 30-day baseline restarts empty, so drift detection is blind until about 2026-08-20; it degrades safely (no drift reported, no exceptions). Three smaller defects were found and filed during verification: #70, #71, #72.

## What would have caught it on day one

Either of two things, both cheap:

- **A scheduled health check.** The check existed; a task registration was the missing piece. It now runs every 6 hours with a persistent on-screen alert and a durable exit code (#4, #5, #6).
- **A loud skip path.** Any non-zero exit on repeated skips, or any skip counter surfacing anywhere a human looks, would have turned day one of the wedge into a red scheduled task instead of 96 days of green.

The fix set covers both, plus the class of failure behind them: the lock now self-heals (age-based takeover at 180 minutes), the daily forecast refuses to run on stale data at two seams, the health check watches the lock age and the pruner's behavior specifically, and capture no longer depends on this one machine at all (cloud snapshots every hour, with a gap guard that counts distinct captured hours and fails loudly).

## Lessons

1. An overlap guard needs staleness handling from day one. Crash-leaked locks are a certainty on a long enough timeline, and self-heal is the only fix that works unattended.
2. A skip that exits 0 is invisible at every level anyone actually monitors. Loud beats correct-but-quiet.
3. Monitoring that is not scheduled catches nothing. Scheduling and alerting are part of shipping a health check, not a follow-up.
4. A pipeline that clamps to the newest data cannot tell "fresh" from "frozen." Freshness has to be checked against the clock, not the data.
5. Before un-wedging a long-stalled pipeline, check what its first resumed run will do to accumulated state. Retention jobs, compactions, and migrations may be poised to destroy exactly what survived.

## References

- Restore runbook and step-by-step execution record: issue #1 (close-out comment, 2026-07-21).
- Gap verification: issue #11 (day-one findings comment, 2026-07-22).
- Evidence tree (local, deliberately untracked): `data/outputs/backups/evidence_2026-07-20/` on the capture machine: health-check output with all three failure flags, run_metadata stage counts, the leaked lock's file stat, the SKIPPED log tail, and the hardware incident note.
- Fix issues: #3, #4, #5, #6, #12, #40 (hardening); #59, #61, #65 (performance and correctness found along the way); #41/#42/#67/#68 (cloud capture backstop).
