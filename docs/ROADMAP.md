# Roadmap: from research system to validated, published forecaster

Last updated: 2026-09-20. The order is the pinned tracking issue [#167](https://github.com/RussellFeinstein/WoW-Economy-Forecaster/issues/167), scope is the [GitHub milestones](https://github.com/RussellFeinstein/WoW-Economy-Forecaster/milestones), and this document is the narrative companion. Milestones run in numeric order: M0, then M0.5 (unattended capture), then M1 through M6. Each milestone is scoped to one coherent arc of PRs and ends with a measurable result. The Work order section below says how the tracker reads and what governed the order before it existed.

## Why this roadmap

Two findings drove it, both from the 2026-07-12 state-of-project review:

1. **A silent 96-day ingestion outage.** A crashed run leaked `data/db/.hourly.lock` on 2026-04-15. The hourly wrapper treated any existing lock as "previous run still active", logged SKIPPED, and exited 0 (behavior removed by the stale-lock takeover in [#3](https://github.com/RussellFeinstein/WoW-Economy-Forecaster/issues/3)), so Task Scheduler saw success 1,933 times in a row while no data arrived. The daily forecast task kept succeeding on features frozen at the last ingest (2026-04-07). Details and fixes: milestone M0 and the postmortem ([postmortem-2026-04-lock-outage.md](postmortem-2026-04-lock-outage.md), issue [#2](https://github.com/RussellFeinstein/WoW-Economy-Forecaster/issues/2)).
2. **The system has never measured itself.** 300K+ forecasts have been issued and none compared to what prices actually did. Recommendations have never been scored for profit. M1 and M2 close that loop; M3, M4, and M6 make the results visible to other people.

## Milestones

### M0: Restore and harden operations (issues [#1](https://github.com/RussellFeinstein/WoW-Economy-Forecaster/issues/1)-[#12](https://github.com/RussellFeinstein/WoW-Economy-Forecaster/issues/12), [#40](https://github.com/RussellFeinstein/WoW-Economy-Forecaster/issues/40), [#44](https://github.com/RussellFeinstein/WoW-Economy-Forecaster/issues/44), [#46](https://github.com/RussellFeinstein/WoW-Economy-Forecaster/issues/46), [#49](https://github.com/RussellFeinstein/WoW-Economy-Forecaster/issues/49), [#80](https://github.com/RussellFeinstein/WoW-Economy-Forecaster/issues/80), [#104](https://github.com/RussellFeinstein/WoW-Economy-Forecaster/issues/104)-[#106](https://github.com/RussellFeinstein/WoW-Economy-Forecaster/issues/106), [#117](https://github.com/RussellFeinstein/WoW-Economy-Forecaster/issues/117), [#123](https://github.com/RussellFeinstein/WoW-Economy-Forecaster/issues/123), [#125](https://github.com/RussellFeinstein/WoW-Economy-Forecaster/issues/125)-[#126](https://github.com/RussellFeinstein/WoW-Economy-Forecaster/issues/126), [#136](https://github.com/RussellFeinstein/WoW-Economy-Forecaster/issues/136), [#107](https://github.com/RussellFeinstein/WoW-Economy-Forecaster/issues/107), [#164](https://github.com/RussellFeinstein/WoW-Economy-Forecaster/issues/164)-[#165](https://github.com/RussellFeinstein/WoW-Economy-Forecaster/issues/165))

Restore ingestion safely (the pruner would delete the surviving history if the lock were simply removed; see the runbook in #1), add age-based lock takeover, schedule `check-data-health` with visible alerting, gate the daily forecast on freshness, fix the failing tests and the CI lint drift (#44), and let the machine sleep between runs with wake-to-run task settings (#40). Everything else depends on this. The long-lived `feature/portfolio-showcase` branch was merged and frozen 2026-07-12 (#10); development now uses short-lived type-prefixed branches per issue (see Branch Workflow in CLAUDE.md).

Late in the milestone a verification tier was added ([#104](https://github.com/RussellFeinstein/WoW-Economy-Forecaster/issues/104) off-box backup verification in CI, [#105](https://github.com/RussellFeinstein/WoW-Economy-Forecaster/issues/105) a durable integrity scope for `check-data-health`, [#106](https://github.com/RussellFeinstein/WoW-Economy-Forecaster/issues/106) an incident runbook). It exists because of a circularity: the machine that builds the nightly backups has a documented memory-corruption history, so checking those backups on that same machine proves nothing. Verification therefore runs on a CI runner, and the local integrity check is scoped to the durable tables so it costs seconds rather than a 25-minute full-database scan. These three were filed and shipped without a milestone and were attached to M0 retroactively on 2026-07-29.

### M0.5: Unattended capture (issues [#41](https://github.com/RussellFeinstein/WoW-Economy-Forecaster/issues/41)-[#43](https://github.com/RussellFeinstein/WoW-Economy-Forecaster/issues/43), [#67](https://github.com/RussellFeinstein/WoW-Economy-Forecaster/issues/67), [#68](https://github.com/RussellFeinstein/WoW-Economy-Forecaster/issues/68), [#78](https://github.com/RussellFeinstein/WoW-Economy-Forecaster/issues/78), [#83](https://github.com/RussellFeinstein/WoW-Economy-Forecaster/issues/83), [#86](https://github.com/RussellFeinstein/WoW-Economy-Forecaster/issues/86), [#95](https://github.com/RussellFeinstein/WoW-Economy-Forecaster/issues/95), [#97](https://github.com/RussellFeinstein/WoW-Economy-Forecaster/issues/97), [#113](https://github.com/RussellFeinstein/WoW-Economy-Forecaster/issues/113), [#124](https://github.com/RussellFeinstein/WoW-Economy-Forecaster/issues/124))

Capture that does not depend on the desktop. Wake-to-run settings (#40, in M0) let the machine sleep; this milestone takes the machine out of the loop entirely. An hourly cloud job fetches and compresses the commodities snapshot into private object storage with a 30-day lifecycle rule (the ToS deletion requirement becomes infrastructure), and a local `sync-snapshots` command ingests the backlog through the existing pipeline whenever the desktop is next on. Missed hours are otherwise unrecoverable because the API serves only the current snapshot; for that reason the design (#41) and the fetcher (#42) touch nothing local and can start before or alongside the M0 runbook. Only the catch-up path (#43) needs the restored pipeline.

### M1: Model validation and monitoring (issues [#13](https://github.com/RussellFeinstein/WoW-Economy-Forecaster/issues/13)-[#19](https://github.com/RussellFeinstein/WoW-Economy-Forecaster/issues/19), [#70](https://github.com/RussellFeinstein/WoW-Economy-Forecaster/issues/70), [#71](https://github.com/RussellFeinstein/WoW-Economy-Forecaster/issues/71), [#100](https://github.com/RussellFeinstein/WoW-Economy-Forecaster/issues/100), [#101](https://github.com/RussellFeinstein/WoW-Economy-Forecaster/issues/101), [#129](https://github.com/RussellFeinstein/WoW-Economy-Forecaster/issues/129), [#131](https://github.com/RussellFeinstein/WoW-Economy-Forecaster/issues/131))

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

The order lives on the tracker, not here: **[#167](https://github.com/RussellFeinstein/WoW-Economy-Forecaster/issues/167)**, pinned, one table per milestone in the shape the global tracking-issue rules set out (`# | Issue | Flag | Runs | Needs | What`). It holds only what is not built yet; a row leaves when its issue closes, and every change to the order is a dated comment there. Row 1 of M0A is what gets built next. This section carried the issue-level sequence from 2026-07-12 to 2026-09-20 (v2.14.26 is the last revision with it, in git), and each milestone description carried a numbered copy from v2.14.18 (#46); both were retired on 2026-09-20 because two copies of one order drifted against each other every time the tracker moved.

Two principles from that period still govern how the tracker reads. First, milestones run in numeric order and each area heading on the tracker says whether it runs beside the ones above (its own files, buildable in whichever order it is ready) or behind them (a dependency, or the same functions and tables); M5 is the designated filler, because it needs only rollups and events, so it is pulled forward whenever the rows ahead of it wait on wall clock or an operator step. Second, when a milestone's remaining rows are waiting on a date, work advances to the next area and circles back; the tracker's Needs column names the wait, and #146 is what will announce a date once it passes, since #11, #42 and #143 all sat past theirs unread until 2026-09-20.

Issues that are built and waiting on a date carry the `waiting: wall clock` label, and each one's body opens with the earliest date it can be checked. The label is what makes the state legible from the issue list: without it an issue sitting on a date looks identical to one sitting on unstarted work. It applies to work that is finished except for an acceptance item that needs a date to arrive, never to work that has not been built yet, which is why #33 does not carry it even though its A/B test needs weeks of wall clock to mature.

That label is one of 19 (the count read 21 until 2026-09-20, when the two Dependabot defaults it was counting were deleted with #143). The rest of the taxonomy (the `type:` and `area:` namespaces, the other state labels, and what is deliberately left out of them) is normative in CLAUDE.md under Issue and PR labels; this section keeps only the reasoning specific to wall clock. Every issue and PR was labelled in one backfill pass on 2026-07-30, and new PRs label themselves from the branch prefix and changed paths.

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
| Capture requires the desktop awake; slept or powered-off hours are unrecoverable | Wake timers in M0 (#40) and an explicit return to sleep (#78); cloud capture and catch-up ingestion in M0.5 (#41-#43), startable before the runbook |
| The only machine running this is unstable under sustained load. Recurring bugchecks, two data-corruption events during the #1 runbook (the second after a clean `mdsched` pass), so a clean standard memory test does not clear it | Everything durable is backed up off-machine nightly (#80) and verified on a CI runner rather than on the box that wrote it (#104), because checking a backup on the machine suspected of corrupting it is circular. Local integrity checks are scoped to the durable tables so they run in seconds (#105), and an incident runbook distinguishes real disk damage from a transient in-memory read before anything is restored (#106). Standing operating rule: cross-verify the output of any large index build, VACUUM, or bulk copy against an independent source before deriving writes from it |

## Backlog (not scheduled)

Unscheduled ideas live in the Not filed yet section of the tracker, [#167](https://github.com/RussellFeinstein/WoW-Economy-Forecaster/issues/167), and each is filed as an issue when it is picked up, joining the order with an order-change comment there. The scheduled `waiting: wall clock` nudge left that list on 2026-08-05 as #146, once #11 had passed its stated check date with nothing surfacing it and #143 became a third carrier of the label; both halves of its revisit condition had fired.
