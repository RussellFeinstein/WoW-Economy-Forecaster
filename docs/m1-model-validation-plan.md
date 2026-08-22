<!-- Promoted from ~/.claude/plans/begin-work-on-milestone-async-sundae.md on 2026-07-29 -->

# M1 (Model validation and monitoring): audit + forward plan

## Context

M0A is done except #11 (wall-clock verification tail through early August) and M0B is drained (#43 evidence posted, closing is Russell's call; #86 unblocked but separate). Per the ROADMAP work-order rule, the lowest milestone's remaining issues are waiting on wall clock, so work advances to M1. M1 is the keystone: the system has issued 345K+ forecasts and never scored one. This plan audits the 11 open M1 issues (#13, #100, #71, #14, #15, #70, #101, #16, #17, #18, #19) against the project's current standing and lays out the execution sequence.

## Audit: current standing (verified 2026-07-28)

### Data standing (read-only queries against the live DB)

- forecast_outputs: 345,819 rows; 318,295 matured (target_date <= today); by horizon 1d 113,004 / 7d 103,920 / 28d 101,371.
- 98% of forecasts are item-level (item_id set): 338,859 item vs 6,960 archetype. Issue #13's ledger must treat the item join as the primary path, not an afterthought.
- Rollup actuals coverage: 42 distinct dates in both rollup tables (2026-02: 3, 03: 29, 04: 2, 07: 8). The Feb-Apr certified window plus the 8 post-restore days.
- **Joinable actuals today: ~155K, not the ~305K in issue #13's text.** 153,912 matured item forecasts have a daily_rollup_item row at target_date; only 1,400 of 6,440 matured archetype forecasts have archetype actuals. The remaining ~163K matured forecasts score realized_source='missing' (mostly targets landing in the 2026-04-08..07-20 gap). The issue's coverage-report acceptance already anticipates this; the 305K number in the body is the matured count, not the scoreable count.
- 28d horizon coverage in the historical window is near zero by construction: March-issued 28d forecasts target April (2 rollup dates exist). Honest per-horizon coverage stats matter.
- model_health_snapshots: 0 rows, which confirms #70 (FK failure has silently dropped every health persistence since the feature shipped).
- backtest_fold_results: 8 rows total (2 per baseline model), which confirms #101's baseline is practically nonexistent as well as wrongly pooled.
- model_metadata: 28d model has validation_mae=None (consistent with #100's split problem: a 14-day validation window cannot contain any 28d targets).

### Issue-text drift found

- **#13 says "Migration 0009"**: 0009 was consumed by the health-check indexes (v2.7.4, issue #59) and 0010 by the normalized obs_id FK index (v2.14.20, issue #149). The ledger migration is **0011**. Comment on #13 at PR time.
- **#13's "~305K matured forecasts have actuals"**: the actual scoreable-today count is ~155K (see above). This does not change the design; it changes the expected coverage report.

### Code standing: ML/training/backtest (verified by Explore agent, file:line evidence)

- **#100 confirmed in full.** trainer.py:80-92 splits on obs_date string comparison, once, outside the per-horizon loop. No purge/embargo exists anywhere in the package (grep is clean). Labels are pure forward lookups (lag_rolling.py:150-153), so the 28d model's validation is fully contaminated. Early stopping uses the leaked val set (lgbm_model.py:201-217). The short-data fallback (trainer.py:93-102) is worse than described: dataset_builder sorts Parquet rows by (archetype_id, obs_date), so the 80/20 row-index split cuts **by archetype**, not by time.
- **#71 confirmed.** lgbm_model.py:152-153 filters on null target only; the date spine deliberately creates all-null feature rows (daily_agg.py:279-293) that pick up real forward targets. dataset_builder.py has no feature-null filter.
- **#16 inputs verified.** `generate_walk_forward_splits(start_date, end_date, window_days, step_days, horizon_days)` at splits.py:76; four baselines in backtest/models.py (last_value, rolling_mean, day_of_week, simple_volatility); backtest_runs/backtest_fold_results schemas match; report-backtest reads them at cli.py:1084-1263.
  - **Bonus defect found**: report-backtest always shows DirAcc = N/A. cli.py:1218 sets `last_known_price=None`, and compute_metrics derives directional accuracy only from last_known_price (metrics.py:165-182); the persisted direction_correct column is never read. #16's "report-backtest works unchanged" acceptance would inherit this. Fix it or file it alongside #16.
- **#19 inputs verified.** No quantile support exists (lgbm_model.py:176-187 hardcodes regression_l1); CI heuristic in cold_start.py:72-129; blending happens before CI computation (predictor.py:131-150); drift widening post-hoc (predictor.py:152-159); ci_quality at predictor.py:161.
- **Feb-Apr dataset build from rollups works.** daily_agg.py:171-231: a per-realm boolean fast path reads daily_rollup_archetype only (never the pruned normalized table) and clamps the spine to rollup MIN/MAX. So #16's backtest over the Feb-Apr window is feasible today from the certified rollups. Caveats: wow_events must be non-empty (dataset_builder.py:424-430), and gap dates inside the window become all-null spine rows, so #71's fix directly protects the backtest's training folds too.
- **forecast_outputs has no forecast-origin column**: only target_date + created_at (schema.py:261-279). The realization ledger derives the anchor date as date(created_at) or target_date minus horizon; anchor_price_gold in the #13 sketch covers direction scoring. Also, horizon is the TEXT `forecast_horizon` ("1d"/"7d"/"28d"), not an integer.
- 98% of forecasts being item-level meets a pipeline asymmetry: `daily_rollup_item` is not consumed by the feature pipeline at all today (viz/dashboard/backup only), so the ledger's item join is a new consumer. Its `_pos` aggregate columns (price_sum_pos / price_obs_count_pos) are the right actuals source for item realizations.

### Code standing: monitoring/pipeline (verified by Explore agent, file:line evidence)

- **#70 confirmed, and wider than the issue text**: `check-drift` (cli.py:1914) has the identical hardcoded `run_id=0`, not just evaluate-live-forecast (cli.py:1790). Both are swallowed in reporter.py (:212-216, :255-260). **`model_health_snapshots.run_id` is ALREADY nullable** (schema.py:397, `INTEGER REFERENCES run_metadata(run_id)`, no NOT NULL), so #70's option 1 (pass None for ad-hoc runs) requires no migration at all. `persist_health_to_db` has no orchestrated caller anywhere: the table has never had a writer that could succeed, and zero tests cover it.
- **#101 confirmed exactly** (all three defects: drift.py:608-617, pipeline/backtest.py:174-186 inside the per-horizon loop, drift.py:282-283 returning NONE for None). Sharper: backtest horizons `[1, 3]` share only h=1 with the product horizons `[1, 7, 28]`, so every health horizon misses. And `live_dir_acc` is structurally always NULL because the anchor price is not stored (health.py:132-134), which the ledger's anchor_price_gold fixes for free.
- **compute_health_summary** (health.py:69-202) joins forecast_outputs to the prunable normalized table, the exact thing #15 replaces with the ledger.
- **run_daily.bat**: 3 fail-fast steps (health gate -> build-datasets -> run-daily-forecast), each `call "%WOWFC%" <cmd> >> logs\daily.log 2>&1` plus an errorlevel capture. #15's addition is a *fourth* step (the issue says "step 3", which is stale text), non-fatal per the issue.
- **Migrations**: a single module wow_forecaster/db/migrations.py with a registry dict; the highest is 0010. New tables go in BOTH schema.py (fresh DBs) and migrations.py (existing DBs). The ledger migration is `0011_forecast_realizations`.
- **RunMetadata**: VALID_PIPELINE_STAGES (models/meta.py:23-26) has no evaluate/health value; the real precedent for a CLI command getting a genuine run_id is instantiating a PipelineStage and calling .run() (base.py:85-90, insert at base.py:169).
- **Learning-track tie-ins**: lab-01 is #100, lab-02 is #16 (it prescribes the `[1,3]` to `[1,7,28]` backtest horizon change), lab-04 is #13. Doing the issues completes the labs' backing work.

## Execution sequence (one issue per PR, per branch discipline)

Order follows the ROADMAP work-order and the M1 milestone description (verified in sync). Dependencies and wall-clock realities annotated.

| PR | Issue | Branch | Bump | Depends on | Notes |
|----|-------|--------|------|-----------|-------|
| 1 | #13 ledger | feat/13-forecast-realizations-ledger | minor | none | Migration 0011. Keystone: unblocks #14, #101, #15, #19 evidence, and M2. Detailed design below. |
| 2 | #100 purge/embargo | fix/100-purge-embargo-training-split | patch | none | Also fixes the short-data fallback (audit: it splits by archetype, not time). Lab-01 backing work. |
| 3 | #71 gap-row filter | fix/71-gap-rows-training | patch | none | Require price_mean non-null at the fit-time filter (lgbm_model.py:152), the issue's "simplest precise rule". Also protects #16's folds. |
| 4 | #14 report-accuracy | feat/14-report-accuracy | minor | #13 | CLI + 9th dashboard tab + viz/charts/accuracy.py. |
| 5 | #70 run_id fix | fix/70-health-snapshot-run-id | patch | none | Pass run_id=None; the column is already nullable, so NO migration is needed. Fix check-drift's identical bug (cli.py:1914) in the same PR (same defect, same line pattern). |
| 6 | #101 drift baseline | fix/101-error-drift-baseline | patch | #13 | Baseline from the ledger (named model, matching horizon, MAPE/skill-score normalized); drift None becomes unknown. |
| 7 | #15 schedule monitoring | feat/15-scheduled-monitoring | minor | #13, #70, #101 | run_daily.bat gains a 4th step (the issue text says "step 3", which is stale); compute_health_summary refitted to read the ledger. Acceptance is two consecutive scheduled runs, so it needs wall clock; start PR 8 while it matures. |
| 8 | #16 ml backtest harness | feat/16-lightgbm-backtest-loop | minor | #100, #71 (cleanliness) | ml_evaluator.py + backtest-ml CLI; writes to the existing backtest tables as model_name=lgbm_global. Lab-02 backing work. |
| 9 | #17 significance | feat/17-backtest-significance | minor | #16 | scipy is already present via scikit-learn. |
| 10 | #18 Optuna study | feat/18-optuna-tuning | minor | #16, #17 | Compute is trivial at current scale (the training matrix is archetype-grain, ~840 rows today). |
| 11 | #19 quantile CIs | feat/19-quantile-ci | minor | #13, #16 | Live within_ci evidence needs fresh matured forecasts (see wall-clock notes). |

### Wall-clock and data realities

- **28d horizon is barely validatable in the historical window**: March-issued 28d forecasts target April, which has 2 rollup dates. Fresh-era 28d realizations start maturing around 2026-08-18 (28 days after the 2026-07-21 restore). #19 and #17's 28d evidence lean on 1d/7d first; report per-horizon coverage honestly rather than waiting.
- Fresh 7d realizations start 2026-07-28 (restore + 7); 1d realizations have been accruing since 07-22.
- #15's acceptance (two consecutive scheduled daily runs) and #19's live coverage are the milestone's wall-clock items, the same advance-and-circle-back rule as #11/#33.
- Ledger backfill is a single-pass ~318K-row insert into a new small table, not a multi-GB job. Still, per the rex-desktop standing rule, verify with count queries plus an idempotent re-run (which the acceptance already requires), and do not run it concurrent with an integrity scan.

### GitHub hygiene at execution time

- Comment on #13: the migration number is 0011 (0009 was taken by #59, 0010 by #149); the scoreable-today count is ~155K of 318K matured, and the rest are realized_source='missing'.
- Comment on #70 at PR: check-drift shares the bug, and run_id is already nullable so option 1 needs no migration.
- File a new small issue for the report-backtest DirAcc defect (always N/A; cli.py:1218 plus metrics.py:165-182). Fix it in or before PR 8, since #16's acceptance leans on report-backtest.
- Per-PR Documentation Sync: CHANGELOG under [Unreleased]; repo CLAUDE.md (table count 23 to 24 and "migrations end at 0011" in PR 1; CLI command count; layer summaries; test counts); README for new user-facing commands; milestone work-order list updates when issues close.

## Session scope (per Russell, 2026-07-28)

The session that produced this plan delivered audit plus plan only. #13 implementation starts in the following session. The PR 1 design below is written to be executed cold from this file.

## PR 1 (#13) detailed design

Branch `feat/13-forecast-realizations-ledger`. Minor bump at stamp time (new table plus new CLI command); the number is assigned at PR-open, never pinned here.

> **PINNED FOR DISCUSSION (Russell, 2026-07-28).** The `forecast_realizations` table schema below is a proposal, not a settled decision. Before writing any code, walk through the DDL with Russell column by column. The design choices most worth arguing about, each with the reasoning that produced them:
>
> 1. **Denormalized copies** (realm_slug, forecast_horizon, target_date, model_slug, archetype_id, item_id all duplicated from forecast_outputs). Proposed because ForecastOutput is frozen so copies cannot drift, and #14/#101 then slice the ledger without joining 345K forecast rows. The alternative is a thin ledger (forecast_id plus measurements only) that joins for every slice. The trade is ~6 extra columns x 318K rows of storage against join-free reporting.
> 2. **`anchor_date` and `anchor_price_gold` as stored columns** rather than derived at query time. They exist to make direction scoring possible at all (health.py:132-134 shows directional accuracy is structurally NULL today precisely because the anchor price was never stored).
> 3. **`matured_at` as the batch key** with no SQL default, so `GROUP BY matured_at` answers "what did the 2026-08-01 run score". The alternative is a separate realization_runs table.
> 4. **The two CHECK constraints** (realized_source enum; missing if and only if NULL price). Cheap invariants, but they make the 'missing' semantics load-bearing in the schema rather than in code.
> 5. **Storing 'missing' rows at all** (see the design rule below): the one genuinely two-sided call, and the one that most changes the table's row count and meaning.
> 6. **Three indexes** proposed off #14's and #101's expected query shapes, which are not written yet. Possibly one too many at this row count.

### Schema (migration `0011_forecast_realizations` plus identical DDL in schema.py, the model_health_snapshots dual-site pattern)

```sql
CREATE TABLE IF NOT EXISTS forecast_realizations (
    realization_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    forecast_id         INTEGER NOT NULL UNIQUE REFERENCES forecast_outputs(forecast_id),
    realm_slug          TEXT    NOT NULL,
    forecast_horizon    TEXT    NOT NULL,           -- '1d'|'7d'|'28d' (copied; forecasts are frozen, zero drift risk)
    target_date         TEXT    NOT NULL,
    anchor_date         TEXT    NOT NULL,           -- target_date minus horizon days (see anchor rule)
    model_slug          TEXT    NOT NULL,
    archetype_id        INTEGER,
    item_id             INTEGER,
    realized_price_gold REAL,                       -- NULL iff realized_source='missing'
    realized_source     TEXT    NOT NULL
        CHECK (realized_source IN ('rollup_archetype','rollup_item','missing')),
    anchor_price_gold   REAL,                       -- NULL when no rollup at anchor_date
    abs_error           REAL,
    pct_error           REAL,                       -- abs/MAX(realized, 0.01), backtest reporter.py convention
    within_ci           INTEGER,
    direction_predicted INTEGER,                    -- 1/-1; NULL when anchor/realized NULL or realized==anchor
    direction_actual    INTEGER,
    direction_correct   INTEGER,
    matured_at          TEXT    NOT NULL,           -- one invocation-wide UTC stamp = queryable batch key
    CHECK ((realized_source = 'missing') = (realized_price_gold IS NULL))
);
CREATE INDEX idx_realizations_horizon_date ON forecast_realizations(forecast_horizon, target_date);
CREATE INDEX idx_realizations_model ON forecast_realizations(model_slug, forecast_horizon, target_date);
CREATE INDEX idx_realizations_archetype ON forecast_realizations(archetype_id, target_date) WHERE archetype_id IS NOT NULL;
```

Denormalized slicing axes are copied from forecast_outputs on purpose: ForecastOutput is frozen so duplication cannot drift, and #14/#101 then query the ledger without per-row joins (the same flat shape as backtest_fold_results). Error and direction conventions copy backtest/reporter.py:92-108 exactly so #101's live-vs-backtest comparison is apples-to-apples.

### Key design rules

- **Anchor date = `target_date - horizon_days`** (module constant `HORIZON_DAYS = {"1d":1,"7d":7,"28d":28}`, SQL CASE). This is the exact inverse of predictor.py:114 (`target_date = base_date + horizon`). `date(created_at)` is rejected: it is an insert timestamp and diverges across UTC midnight. Unknown horizon values are skipped and counted (`skipped_unknown_horizon`), never mis-anchored.
- **Maturity** is `target_date <= as_of - grace_days` (grace default 2: rollups for recent dates can lag a sync-snapshots drain, and without grace, fresh forecasts get prematurely frozen as 'missing').
- **Realized price aggregates**: archetype is `price_sum/price_obs_count` (identical to the training feature price_mean, daily_agg.py:109); item is `price_sum_pos/price_obs_count_pos` (the plain sums include zero-price placeholder rows, and the `_pos` columns exist to exclude them). Count guards (`> 0`) demote degenerate rollup rows to 'missing'. Anchor price uses the same aggregate as its branch, taken from the rollup at anchor_date; if absent, anchor and direction fields are NULL while realized/abs_error still populate.
- **Grain rule**: item forecasts (item_id NOT NULL) join rollup_item only; archetype forecasts join rollup_archetype only. Item wins when both ids are set. No cross-grain fallback.
- **'missing' rows ARE inserted** (the issue mandates honest in-table coverage; the Feb-Apr gap is permanent so most can never fill; and it avoids re-evaluating ~163K dead forecasts every run). Late-arriving rollups are handled by `--rescore-missing`: in one transaction, DELETE missing rows (window-filtered) then re-run the three inserts. Realized rows are never touched.
- **Idempotency** is an anti-join on the ledger (`LEFT JOIN ... WHERE fr.forecast_id IS NULL`) plus `INSERT OR IGNORE` on UNIQUE(forecast_id), belt and braces. Three set-based INSERT...SELECT statements (item, archetype, missing-via-NOT-EXISTS) run in one transaction; no batching is needed (all index lookups, and the obs tables are never referenced, which a static test enforces).

### Module and CLI

- New `wow_forecaster/reporting/realization.py`: SQL constants; frozen dataclasses `CoverageRow` and `RealizationUpdateResult`; `update_realizations(conn, *, as_of, realm_slug=None, since=None, until=None, grace_days=2, rescore_missing=False, dry_run=False, now_utc=None)`; `coverage_by_horizon(conn, ...)` (reused by #14 later).
- `update-realizations` Typer command in cli.py: `--realm` (default None meaning all, because a scoring ledger should be complete by default), `--as-of`, `--since/--until`, `--grace-days`, `--rescore-missing`, `--dry-run`, `--db-path`, `--config`. Exit 0 on success including zero inserts (cron-friendly); exit 1 on bad dates, an inverted window, or a missing DB. Prints a per-horizon coverage table plus `inserted this run: N`. No `--export` (this is a mutation command; exports belong to #14).
- **Lightweight CLI, not a PipelineStage**: provenance is already complete (forecast_id reaches forecast_outputs.run_id, and matured_at identifies the batch); a stage would widen VALID_PIPELINE_STAGES and mint run rows on daily no-ops. Revisit as a thin wrapper if #15's orchestration needs it.

### Tests (TDD, red first)

- `tests/test_db/test_migration_0011.py` (mirror test_migration_0009.py): registry entry; apply_schema creates the table and indexes; upgrade path via run_migrations on a pre-0011 DB; ALL_TABLE_NAMES parity; UNIQUE enforcement; CHECK rejections; EXPLAIN QUERY PLAN pins for the coverage and model-slicing queries (assert seek terms where the 07-22 lesson applies).
- `tests/test_reporting/test_realization.py` (fixtures via apply_schema, NEVER hand-rolled DDL, per the 2026-07-19 lesson): maturity/grace/window/realm selection; item join math; archetype join math; item-wins; no cross-grain fallback; degenerate rollup becomes missing; missing NULL shape; abs/pct error including the realized=0 guard; within_ci boundary inclusive; direction semantics including realized==anchor giving NULL and absent-anchor giving NULL; anchor ignores created_at (a row whose created_at date differs from target minus horizon); unknown-horizon skip; idempotent re-run (realized AND missing); rescore_missing fills and preserves realized rows; dry_run counts match the subsequent real run and write nothing; coverage_by_horizon math; a static no-`market_observations` guard on the module SQL. Plus one test exercising the production default clock/as_of shape (2026-07-27 injectable-seam lesson: at least one test per seam with the real default).
- `tests/test_cli/`: add to the test_cli_smoke.py parametrized help list (exit 0 plus "Usage:" only); dry-run banner; end-to-end insert-then-zero on a tmp init-db DB; bad-date and inverted-window exit 1. Red-phase caution: count expected failures (2026-07-28 lesson: Typer exits 2 for unknown options too).

### Implementation order

1. Branch off latest main. 2. Write failing tests. 3. schema.py DDL plus ALL_TABLE_NAMES plus docstring count 23 to 24. 4. migrations.py `0011_forecast_realizations`. 5. realization.py module. 6. cli.py command. 7. Full suite green (`pytest --tb=no`, no extra -q; lint `ruff check wow_forecaster/ tests/`). 8. Docs: CHANGELOG [Unreleased]/Added; README CLI section; repo CLAUDE.md (24 tables, migrations end at 0011, command count, M1 layer note, test count). 9. Live acceptance on rex-desktop (below). 10. Stamp commit plus PR `(vX.Y.Z)` plus comment the audit corrections on #13.

### Live acceptance (production DB, rex-desktop)

- `update-realizations --dry-run` first (counts preview, no writes); avoid the :16 hourly window and any integrity scan (07-28 lock lesson).
- Real run: expect ~318K rows minus grace exclusions; ~153.9K rollup_item plus ~1.4K rollup_archetype realized, remainder missing; per-horizon totals near 113,004 / 103,920 / 101,371 (numbers measured 2026-07-28, so they will have grown).
- Immediate re-run: `inserted this run: 0`.
- Coverage over `--since 2026-02-26 --until 2026-04-15`: realized rows cluster on the 42 rollup dates.
- Spot SQL: realized_source GROUP BY; within_ci rate by horizon sanity-checked against confidence_pct 0.80 (expect poor coverage, which IS the #19 motivation, so record the number).

## Verification (whole plan)

- PR 1 is verified by its live acceptance above plus green CI.
- Each subsequent PR carries its issue's acceptance test. The audit findings section above lists the file:line evidence to re-check against HEAD at implementation time (this plan is a snapshot of 2026-07-28).
- Milestone-level: M1 is done when report-accuracy publishes measured MAE/MAPE/coverage numbers, the daily pipeline updates the ledger unattended, LightGBM has significance-tested backtest results against all four baselines, and CI coverage is measured from the ledger.
