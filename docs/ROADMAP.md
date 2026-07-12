# Roadmap: from research system to validated, published forecaster

Last updated: 2026-07-12. Work is tracked in [GitHub milestones](https://github.com/RussellFeinstein/WoW-Economy-Forecaster/milestones); this document is the narrative companion. Milestones run in numeric order: M0, then M0.5 (unattended capture), then M1 through M6. Each milestone is scoped to one coherent arc of PRs and ends with a measurable result.

## Why this roadmap

Two findings drove it, both from the 2026-07-12 state-of-project review:

1. **A silent 96-day ingestion outage.** A crashed run leaked `data/db/.hourly.lock` on 2026-04-15. The hourly wrapper treats any existing lock as "previous run still active", logs SKIPPED, and exits 0, so Task Scheduler saw success 1,933 times in a row while no data arrived. The daily forecast task kept succeeding on features frozen at the last ingest (2026-04-07). Details and fixes: milestone M0 and the postmortem (issue [#2](https://github.com/RussellFeinstein/WoW-Economy-Forecaster/issues/2)).
2. **The system has never measured itself.** 300K+ forecasts have been issued and none compared to what prices actually did. Recommendations have never been scored for profit. M1 and M4 close that loop; M2, M3, and M6 make the results visible to other people.

## Milestones

### M0: Restore and harden operations (issues [#1](https://github.com/RussellFeinstein/WoW-Economy-Forecaster/issues/1)-[#12](https://github.com/RussellFeinstein/WoW-Economy-Forecaster/issues/12), [#40](https://github.com/RussellFeinstein/WoW-Economy-Forecaster/issues/40))

Restore ingestion safely (the pruner would delete the surviving history if the lock were simply removed; see the runbook in #1), add age-based lock takeover, schedule `check-data-health` with visible alerting, gate the daily forecast on freshness, fix the 9 failing tests, let the machine sleep between runs with wake-to-run task settings (#40), and merge the long-lived `feature/portfolio-showcase` branch. Everything else depends on this.

### M0.5: Unattended capture (issues [#41](https://github.com/RussellFeinstein/WoW-Economy-Forecaster/issues/41)-[#43](https://github.com/RussellFeinstein/WoW-Economy-Forecaster/issues/43))

Capture that does not depend on the desktop. Wake-to-run settings (#40, in M0) let the machine sleep; this milestone takes the machine out of the loop entirely. An hourly cloud job fetches and compresses the commodities snapshot into private object storage with a 30-day lifecycle rule (the ToS deletion requirement becomes infrastructure), and a local `sync-snapshots` command ingests the backlog through the existing pipeline whenever the desktop is next on. Missed hours are otherwise unrecoverable because the API serves only the current snapshot; for that reason the design (#41) and the fetcher (#42) touch nothing local and can start before or alongside the M0 runbook. Only the catch-up path (#43) needs the restored pipeline.

### M1: Model validation and monitoring (issues [#13](https://github.com/RussellFeinstein/WoW-Economy-Forecaster/issues/13)-[#19](https://github.com/RussellFeinstein/WoW-Economy-Forecaster/issues/19))

The keystone. A durable `forecast_realizations` ledger scores every matured forecast against rollup actuals (MAE, MAPE, directional accuracy, interval coverage), backfilled over the Feb-Apr window and updated nightly. On top of it: a walk-forward LightGBM backtest, Diebold-Mariano and Wilcoxon significance tests against the four baselines, an Optuna tuning study, and quantile-regression confidence intervals with measured coverage.

### M2: Analytics warehouse, PostgreSQL + dbt (issues [#20](https://github.com/RussellFeinstein/WoW-Economy-Forecaster/issues/20)-[#25](https://github.com/RussellFeinstein/WoW-Economy-Forecaster/issues/25))

The analytical layer (rollups, forecasts, recommendations, realizations) moves to a local PostgreSQL warehouse through an idempotent, watermark-based `sync-warehouse` ETL with dual-apply verification. dbt models it into star-schema marts with schema tests and generated docs. Aggregate marts publish to Supabase under the free-tier ceiling. Raw observations stay in SQLite: operational store and warehouse are deliberately separate.

### M3: BI dashboards (issues [#26](https://github.com/RussellFeinstein/WoW-Economy-Forecaster/issues/26)-[#28](https://github.com/RussellFeinstein/WoW-Economy-Forecaster/issues/28))

A Power BI report (text-based .pbip project, DAX measures for accuracy and recommendation performance) over the warehouse, then a Tableau Public workbook from the same star schema. Prototyping can start from the existing `export-bi-bundle` CSVs before M2 completes.

### M4: Paper trading P&L and ranking A/B (issues [#29](https://github.com/RussellFeinstein/WoW-Economy-Forecaster/issues/29)-[#33](https://github.com/RussellFeinstein/WoW-Economy-Forecaster/issues/33))

Does it make gold? A paper-trading simulator executes recommendations with auction-house fees and per-horizon exit rules, backfilled over the Feb-Apr window where actuals exist. P&L and equity curves report against random-portfolio and buy-everything baselines. Scoring weights become named policies, and a time-sliced A/B test (with an offline replay mode, labeled as such) compares ranking policies with paired significance tests.

### M5: Event impact study (issues [#34](https://github.com/RussellFeinstein/WoW-Economy-Forecaster/issues/34)-[#36](https://github.com/RussellFeinstein/WoW-Economy-Forecaster/issues/36))

Interrupted time-series and difference-in-differences designs measure how patch events move prices, using the `is_known_at()` guard against look-ahead bias, with placebo checks and a reproducible notebook. Independent of M2-M4; can run any time after M0.

### M6: Publish (issues [#37](https://github.com/RussellFeinstein/WoW-Economy-Forecaster/issues/37)-[#39](https://github.com/RussellFeinstein/WoW-Economy-Forecaster/issues/39))

A public Streamlit dashboard on Community Cloud reading the Supabase marts, a README case study with architecture and measured results, and a profile refresh.

## Dependency graph

```
M0 (gates everything)
 +-> M0.5 (unattended capture; #41-#42 may even precede the M0 runbook)
 +-> M1 (realizations backfill works off rollups immediately)
 |    +-> M2 (warehouse: realizations are the best mart)
 |    |    +-> M3 (BI dashboards; prototype may start earlier off CSV bundle)
 |    +-> M4 (paper trading uses realizations)
 +-> M5 (event study; independent, parallel any time after M0)
 +-> M6 (publish; needs M2 marts and M1/M4 numbers)
```

## Standing risks

| Risk | Mitigation |
|---|---|
| Pruner deletes >30-day rows on the first un-wedged run; rollups are the only durable daily history and have known gaps | Runbook order in #1: back up and backfill rollups before anything else; the lock stays in place until then |
| Post-gap model behavior: drift baseline empty for ~30 days, item forecasts need 14 fresh days, retrain spans a 90-day hole | Verification checklist in #11; limitations documented in the postmortem |
| Supabase free tier caps the DB at 500 MB | Archetype-grain marts only, with a size guard (#25) |
| Public deployment requires a public repo | Secrets audit before flipping; read-only cloud key; local DB never committed (#37) |
| DB grows ~14 GB/month raw once ingest resumes | 30-day prune caps steady state; retention sentinel in health checks (#5) |
| Capture requires the desktop awake; slept or powered-off hours are unrecoverable | Wake timers in M0 (#40); cloud capture and catch-up ingestion in M0.5 (#41-#43), startable before the runbook |

## Backlog (not scheduled)

Live news ingestion and `extract_wow_events()`; governance cooldown wiring (orchestrator never passes `last_call_at` to preflight); EU region expansion; top_n_per_category V2 remainder; data/outputs retention policy; drift-baseline seeding from rollups.
