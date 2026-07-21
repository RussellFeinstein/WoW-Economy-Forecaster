# Roadmap: from research system to validated, published forecaster

Last updated: 2026-07-21. Work is tracked in [GitHub milestones](https://github.com/RussellFeinstein/WoW-Economy-Forecaster/milestones); this document is the narrative companion. Milestones run in numeric order: M0, then M0.5 (unattended capture), then M1 through M6. Each milestone is scoped to one coherent arc of PRs and ends with a measurable result. The Work order section below is the issue-level sequence, most urgent first; each milestone description on GitHub opens with a numbered list rendered from it.

## Why this roadmap

Two findings drove it, both from the 2026-07-12 state-of-project review:

1. **A silent 96-day ingestion outage.** A crashed run leaked `data/db/.hourly.lock` on 2026-04-15. The hourly wrapper treated any existing lock as "previous run still active", logged SKIPPED, and exited 0 (behavior removed by the stale-lock takeover in [#3](https://github.com/RussellFeinstein/WoW-Economy-Forecaster/issues/3)), so Task Scheduler saw success 1,933 times in a row while no data arrived. The daily forecast task kept succeeding on features frozen at the last ingest (2026-04-07). Details and fixes: milestone M0 and the postmortem (issue [#2](https://github.com/RussellFeinstein/WoW-Economy-Forecaster/issues/2)).
2. **The system has never measured itself.** 300K+ forecasts have been issued and none compared to what prices actually did. Recommendations have never been scored for profit. M1 and M2 close that loop; M3, M4, and M6 make the results visible to other people.

## Milestones

### M0: Restore and harden operations (issues [#1](https://github.com/RussellFeinstein/WoW-Economy-Forecaster/issues/1)-[#12](https://github.com/RussellFeinstein/WoW-Economy-Forecaster/issues/12), [#40](https://github.com/RussellFeinstein/WoW-Economy-Forecaster/issues/40), [#44](https://github.com/RussellFeinstein/WoW-Economy-Forecaster/issues/44), [#46](https://github.com/RussellFeinstein/WoW-Economy-Forecaster/issues/46), [#49](https://github.com/RussellFeinstein/WoW-Economy-Forecaster/issues/49))

Restore ingestion safely (the pruner would delete the surviving history if the lock were simply removed; see the runbook in #1), add age-based lock takeover, schedule `check-data-health` with visible alerting, gate the daily forecast on freshness, fix the failing tests and the CI lint drift (#44), and let the machine sleep between runs with wake-to-run task settings (#40). Everything else depends on this. The long-lived `feature/portfolio-showcase` branch was merged and frozen 2026-07-12 (#10); development now uses short-lived type-prefixed branches per issue (see Branch Workflow in CLAUDE.md).

### M0.5: Unattended capture (issues [#41](https://github.com/RussellFeinstein/WoW-Economy-Forecaster/issues/41)-[#43](https://github.com/RussellFeinstein/WoW-Economy-Forecaster/issues/43))

Capture that does not depend on the desktop. Wake-to-run settings (#40, in M0) let the machine sleep; this milestone takes the machine out of the loop entirely. An hourly cloud job fetches and compresses the commodities snapshot into private object storage with a 30-day lifecycle rule (the ToS deletion requirement becomes infrastructure), and a local `sync-snapshots` command ingests the backlog through the existing pipeline whenever the desktop is next on. Missed hours are otherwise unrecoverable because the API serves only the current snapshot; for that reason the design (#41) and the fetcher (#42) touch nothing local and can start before or alongside the M0 runbook. Only the catch-up path (#43) needs the restored pipeline.

### M1: Model validation and monitoring (issues [#13](https://github.com/RussellFeinstein/WoW-Economy-Forecaster/issues/13)-[#19](https://github.com/RussellFeinstein/WoW-Economy-Forecaster/issues/19))

The keystone. A durable `forecast_realizations` ledger scores every matured forecast against rollup actuals (MAE, MAPE, directional accuracy, interval coverage), backfilled over the Feb-Apr window and updated nightly. On top of it: a walk-forward LightGBM backtest, Diebold-Mariano and Wilcoxon significance tests against the four baselines, an Optuna tuning study, and quantile-regression confidence intervals with measured coverage.

### M2: Paper trading P&L and ranking A/B (issues [#29](https://github.com/RussellFeinstein/WoW-Economy-Forecaster/issues/29)-[#33](https://github.com/RussellFeinstein/WoW-Economy-Forecaster/issues/33))

Does it make gold? A paper-trading simulator executes recommendations with auction-house fees and per-horizon exit rules, backfilled over the Feb-Apr window where actuals exist. P&L and equity curves report against random-portfolio and buy-everything baselines. Scoring weights become named policies, and a time-sliced A/B test (with an offline replay mode, labeled as such) compares ranking policies with paired significance tests. This runs directly after M1 for two reasons: the live A/B (#33) needs weeks of wall clock to mature, so its clock starts early and runs while M3 and M4 proceed, and the make-gold answer should exist before infrastructure and dashboards are built to showcase it.

### M3: Analytics warehouse, PostgreSQL + dbt (issues [#20](https://github.com/RussellFeinstein/WoW-Economy-Forecaster/issues/20)-[#25](https://github.com/RussellFeinstein/WoW-Economy-Forecaster/issues/25))

The analytical layer (rollups, forecasts, recommendations, realizations, paper trades) moves to a local PostgreSQL warehouse through an idempotent, watermark-based `sync-warehouse` ETL with dual-apply verification. dbt models it into star-schema marts with schema tests and generated docs. Aggregate marts publish to Supabase under the free-tier ceiling. Raw observations stay in SQLite: operational store and warehouse are deliberately separate. Coming after M2 means realizations and trade facts land in the marts in one pass instead of being retrofitted.

### M4: BI dashboards (issues [#26](https://github.com/RussellFeinstein/WoW-Economy-Forecaster/issues/26)-[#28](https://github.com/RussellFeinstein/WoW-Economy-Forecaster/issues/28))

A Power BI report (text-based .pbip project, DAX measures for accuracy and recommendation performance) over the warehouse, then a Tableau Public workbook from the same star schema. Prototyping can start from the existing `export-bi-bundle` CSVs before M3 completes. Accuracy, P&L, and A/B results all exist by this point, so the report covers them from its first version.

### M5: Event impact study (issues [#34](https://github.com/RussellFeinstein/WoW-Economy-Forecaster/issues/34)-[#36](https://github.com/RussellFeinstein/WoW-Economy-Forecaster/issues/36))

Interrupted time-series and difference-in-differences designs measure how patch events move prices, using the `is_known_at()` guard against look-ahead bias, with placebo checks and a reproducible notebook. Independent of M2-M4; can run any time after M0.

### M6: Publish (issues [#37](https://github.com/RussellFeinstein/WoW-Economy-Forecaster/issues/37)-[#39](https://github.com/RussellFeinstein/WoW-Economy-Forecaster/issues/39))

A public Streamlit dashboard on Community Cloud reading the Supabase marts, a README case study with architecture and measured results, and a profile refresh.

## Work order

Most urgent first. Milestone numbers match this sequence; within milestones, follow the issue order given here rather than raw issue numbers.

1. **Stop the permanent loss (M0.5 front, plus one insurance step).** #41 design, then #42 cloud fetcher. From the fetcher's first green run, no more hours are lost for good. Pull one runbook step forward now: dump `daily_rollup_archetype` and `daily_rollup_item` to `data/outputs/backups/` (read-only, minutes), so an accidental lock deletion can no longer destroy the only durable history.
2. **Green CI (M0).** #46 first: repo governance (PR-only merges, delete-after-merge, milestone work-order lists) so every change from here on ships through a gated PR. Then #44: CI currently fails at the ruff step before tests run (`ruff>=0.4` floats to releases with rules the code predates), so nothing else is visible. Then #7, #8, and #49 (the CI-only failures that surfaced once pytest became reachable), so the hardening changes that follow are validated by a trustworthy suite. (Done: #46 and #44 on 2026-07-15, #7/#8/#49 on 2026-07-16; the full suite is green in CI on 3.11 and 3.12.)
3. **Harden before restoring (M0).** #3 stale-lock takeover, #12 daily freshness gate, #4 scheduled health check, #5 lock-age and retention checks, #6 commit setup_tasks.bat and register tasks, #40 wake timers. All of it lands before the lock is touched, so the restored system is born hardened. (Done: #3 on 2026-07-16; the WoWForecaster-Hourly task was disabled the same day, pulling the hourly half of runbook step 1 forward. #12 on 2026-07-19: the daily forecast is now gated on freshness at both the batch and ForecastStage seams, with a dashboard alert banner. #4 on 2026-07-19: run_healthcheck.bat writes a durable alert JSON and raises a once-per-24h red console window on health failure; Task Scheduler registration follows with #6. #5 on 2026-07-19: check-data-health now flags a stale hourly lock (older than the 180-minute takeover threshold) and a retention violation (oldest raw row past raw_snapshot_days + 2), and exits 1 on either, so the daily gate and the health alert inherit both checks. #6 on 2026-07-19: setup_tasks.bat committed with silent execution via run_silent.vbs, the hourly :16 phase pinned, the WoWForecaster-HealthCheck task registered every 6h at :45, and Disabled-state preservation across re-runs so the parked hourly task cannot be re-enabled by accident. #40 on 2026-07-20: all three tasks registered with wake-to-run through a settings round-trip that keeps the parked hourly task Disabled, plus a warn-only power-plan wake-timer check, so the machine may sleep between runs once ingestion is restored.)
4. **Restore (M0, then M0.5).** #1, the runbook, in its documented order. Then #43 catch-up ingestion drains the cloud backlog into the DB; capture is now desktop-independent end to end. (Done: #1 executed 2026-07-20/21. Ingestion restored 2026-07-21 02:43Z after 105 days; rollup coverage 22 -> 34 dates, every date certified against independent sources after two hardware-induced corruption events during execution; DB rebuilt 78 GB -> 105 MB; all three tasks re-enabled and observed green overnight. Close-out record on #1.)
5. **Close out M0.** #11 gap verification (passive, over the following days), #59 health-check query indexes (deferred from #5/#6, re-scoped after the #1 rebuild shrank the DB), #61 orchestrator rollup UTC date anchor (found during the #1 restore: the rollup step targets the local date while observations are stamped UTC; self-heals daily but lags intraday), #2 postmortem, #9 gitignore, #45 README badge. (#10, the merge to main, was pulled forward and done on 2026-07-12 when the umbrella-branch model was retired.)
6. **Prove the forecasts (M1).** #13 realization ledger first (roughly 305K matured forecasts become scoreable the day it lands), then #14, #15, #16, #17, #18, #19.
7. **Prove the gold (M2).** #29 simulator, #32 ScoringPolicy extraction, then start #33 so the live A/B clock runs in the background, then #30 backfill P&L and #31 baselines.
8. **Warehouse (M3).** #20-#25 in order, while the A/B matures.
9. **BI (M4).** #26, then #27 (Power BI, done deeply). #28 Tableau follows later by design.
10. **Close the loop.** Analyze #33 once enough pairs mature. M5 (#34-#36) formally sits here but is the designated filler: it needs only rollups and events, so pull it forward whenever work is blocked on wall clock.
11. **Publish (M6).** #37, #38, #39, with all measured numbers in hand.

When the lowest open milestone's remaining issues are waiting on wall clock (#11 verification days, #33 A/B maturation), advance to the next milestone and circle back.

## Dependency graph

```
M0 (gates everything)
 +-> M0.5 (unattended capture; #41-#42 may even precede the M0 runbook)
 +-> M1 (realizations ledger; scores 300K+ matured forecasts immediately)
 |    +-> M2 (paper trading uses realizations; #33's A/B clock runs in the background)
 |         +-> M3 (warehouse: realizations and trades are the best marts)
 |              +-> M4 (BI dashboards; prototype may start earlier off CSV bundle)
 +-> M5 (event study; independent filler, parallel any time after M0)
 +-> M6 (publish; needs M3 marts and M1/M2 numbers)
```

## Standing risks

| Risk | Mitigation |
|---|---|
| Pruner deletes >30-day rows on the first un-wedged run; rollups are the only durable daily history and have known gaps | Retired 2026-07-21: the #1 runbook backed up, backfilled, and certified the rollups before the lock was touched; ingestion is live again |
| Post-gap model behavior: drift baseline empty for ~30 days, item forecasts need 14 fresh days, retrain spans a 90-day hole | Verification checklist in #11; limitations documented in the postmortem |
| Supabase free tier caps the DB at 500 MB | Archetype-grain marts only, with a size guard (#25) |
| Public deployment requires a public repo | The repo is already public; keep the standing secrets audit, read-only cloud key, local DB never committed (#37) |
| DB grows ~14 GB/month raw once ingest resumes | 30-day prune caps steady state; retention sentinel in health checks (#5) |
| Capture requires the desktop awake; slept or powered-off hours are unrecoverable | Wake timers in M0 (#40); cloud capture and catch-up ingestion in M0.5 (#41-#43), startable before the runbook |

## Backlog (not scheduled)

Live news ingestion and `extract_wow_events()`; governance cooldown wiring (orchestrator never passes `last_call_at` to preflight); EU region expansion; top_n_per_category V2 remainder; data/outputs retention policy; drift-baseline seeding from rollups.
