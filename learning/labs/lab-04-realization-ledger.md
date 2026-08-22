# Lab 04. Build the forecast realization ledger

Module: M15. Real work on a real branch, shipped through a PR like anything else.

## What you are building

The repo has issued roughly 305,000 forecasts into `forecast_outputs` and has
never once checked one against the price that actually printed. Every row in
that table carries a `target_date`, a `predicted_price_gold`, a
`confidence_lower`/`confidence_upper` band, and either an `archetype_id` or an
`item_id` at a `realm_slug`. The actuals to score them against already exist:
`daily_rollup_archetype` and `daily_rollup_item` hold a mean price per
key per `obs_date` (`price_sum / price_obs_count`).

A forecast is *matured* when its `target_date` is in the past and a rollup row
exists for its key on that date. This lab builds `forecast_realizations`, a
durable ledger with one row per matured forecast, carrying the actual, the
absolute error, the percentage error, whether the predicted direction was
right, and whether the actual landed inside the confidence band. It backfills
over the whole window where actuals exist, so the day it lands the ~305K
already-issued forecasts that have matured become scoreable in one pass, and it
runs nightly thereafter to score the newly matured tail.

Four metrics per row:

| metric | definition |
|---|---|
| absolute error | `abs(predicted - actual)` |
| percentage error | `abs(predicted - actual) / max(actual, MAPE_EPSILON)`, excluded when `actual < MAPE_EPSILON` |
| directional correct | did `sign(predicted - origin_actual)` match `sign(target_actual - origin_actual)` |
| interval covered | `confidence_lower <= actual <= confidence_upper` |

Directional accuracy is the one with a trap in it. The reference is not the
predicted price and not the target-date actual. It is the actual price at the
forecast's *origin* date, `target_date - horizon_days`, read from the same
rollup table. A forecast predicts a move away from where the price was when it
was issued; scoring direction against anything else measures nothing. Reuse
`backtest/metrics.py`: it already has `MAPE_EPSILON = 0.01` and `_direction()`,
and reinventing either is how the two diverge later.

## Before you write any code

This is milestone M1 issue [#13](https://github.com/RussellFeinstein/WoW-Economy-Forecaster/issues/13),
the keystone of that milestone. Everything downstream (the significance tests in
#14 through #17, and PLAN.md Phase 5's fix to the error-drift baseline) reads
from this ledger instead of from `backtest_fold_results`. The issue already
exists, so do not file a new one. Cut the branch from the latest main:

```
git checkout main && git pull --ff-only
git checkout -b feat/13-forecast-realizations-ledger
```

## Write the failing test first

Put it in `tests/test_monitoring/test_realizations.py`. Seed a handful of
`forecast_outputs` rows and matching `daily_rollup_archetype` rows by hand,
then call the scoring entry point against a fixed `now`.

The assertions that catch the thing:

> A forecast whose `target_date` is at or after `now`, or whose key has no
> rollup row on `target_date`, produces no ledger row at all. It is not scored
> with `actual = NULL` or `actual = 0`.

> For a matured forecast, `direction_correct` is computed against the rollup
> actual at `target_date - horizon_days`, not against `predicted_price_gold`.
> Construct a case where the prediction moved the right way relative to the
> origin price but the wrong way relative to nothing, and assert it scores
> correct only under the origin reference.

The assertion that does **not** catch it, and that you may be tempted to write:

> After scoring, `forecast_realizations` has more than zero rows and every
> `abs_error` is non-negative.

That passes the instant anything gets inserted, including a build that scores
unmatured forecasts against a zero actual or that uses the predicted price as
its own directional reference. `abs_error >= 0` is true of `abs()` by
construction and tests nothing.

Also assert idempotence: scoring the same window twice leaves the row count and
the values unchanged. The nightly run re-scans a trailing window and must upsert
on the forecast's identity, not append.

Confirm these fail (or error, since the module does not exist yet) before you
write the implementation.

## The implementation

1. **Schema.** Add `forecast_realizations` to `wow_forecaster/db/schema.py`
   (its own `_DDL_*` constant plus registration, so a fresh DB gets it via
   `apply_schema()`), and add `migration_0012_add_forecast_realizations` to
   `wow_forecaster/db/migrations.py` registered in `MIGRATIONS` (so existing DBs
   get it). Migrations end at 0011 today; this is 0012. Follow the 0002 pattern,
   which puts the same table in both places. Grain: one row per `forecast_id`.
   Columns: `forecast_id` (FK to `forecast_outputs`, UNIQUE), `archetype_id`,
   `item_id`, `realm_slug`, `horizon_days`, `target_date`, `origin_date`,
   `predicted_price`, `actual_price`, `origin_actual_price`, `abs_error`,
   `pct_error`, `direction_correct`, `interval_covered`, `scored_at`. The UNIQUE
   on `forecast_id` is what makes re-scoring an upsert.

2. **The module.** New file `wow_forecaster/monitoring/realizations.py`, next to
   `drift.py` and `health.py`, since PLAN.md Phase 5 makes those its consumers.
   A `score_realizations(conn, *, now, since=None)` entry point that mirrors the
   shape of `durable_backup.run_backup`: select matured-but-unscored forecasts,
   join each to its rollup actual and its origin-date actual, compute the four
   metrics via the reused `metrics.py` helpers, upsert into the ledger.

3. **The maturity query.** Select from `forecast_outputs` where `target_date`
   sorts before the bare-date form of `now` (compare the raw text column to a
   `"YYYY-MM-DD"` cutoff so the existing `idx_forecast_archetype_date` can serve
   it; do not wrap `target_date` in `DATE()`, which defeats the index, the exact
   trap #59 fixed elsewhere). Join to `daily_rollup_archetype` on
   `(archetype_id, realm_slug, obs_date = target_date)` for archetype forecasts
   and `daily_rollup_item` on `(item_id, realm_slug, obs_date = target_date)`
   for item forecasts. `forecast_horizon` is stored as `"1d"/"7d"/"28d"`; map it
   to an integer `horizon_days` for the origin-date arithmetic.

4. **Origin actual.** Second rollup lookup at `obs_date = target_date - horizon`.
   It can be missing even when the target actual is present, because the origin
   is 1, 7, or 28 days earlier and the coverage window has holes. When origin is
   missing, still write the row with the error and coverage metrics but leave
   `direction_correct` NULL. Do not drop the whole row for a missing origin, and
   do not silently default `direction_correct` to 0.

5. **Backfill vs nightly.** Same function, different `since`. Backfill passes no
   `since` and scans everything; the nightly run passes a trailing window
   (`now - max_horizon - a grace day`). Both paths upsert, so the overlap is
   free. Log the count scored per horizon: a backfill that scores 305K rows and
   a nightly that scores a few hundred are both normal, and a nightly that
   suddenly scores zero is the signal you want visible, given this repo's history
   with silent gaps.

6. **CLI.** Add a `score-realizations` command (`--since`, `--dry-run`) to
   `cli.py`, wired to the same entry point. This is the surface a scheduled task
   or the orchestrator calls.

## Do not

- Do not score against `backtest_fold_results`. That table is the four naive
  baselines on a different partition and different horizons (1 and 3, not 1, 7,
  28). The whole point of this ledger is that it scores the *production*
  forecasts against real actuals. PLAN.md DS-3 is the standing evidence of what
  conflating the two costs.
- Do not use the predicted price as the directional reference. It is always
  "correct" against itself and produces a directional accuracy of 1.0 that means
  nothing.
- Do not treat a missing rollup actual as a zero price or an actual of NULL that
  still gets a row. No actual means not matured means no row.
- Do not wrap `target_date` or `obs_date` in `DATE()` in the maturity or join
  predicates. Compare the raw text columns to bare-date cutoffs so the existing
  indexes seek.
- Do not add a second copy of the MAPE or direction logic. Import from
  `backtest/metrics.py`.
- Do not fold the Phase 5 drift-baseline fix into this PR. This lab builds the
  ledger. Rewiring `drift.py` and `health.py` to read from it is a separate
  concern and a separate PR.

## Finish line

- The new tests fail against a tree without the module and pass after it lands,
  including the maturity-exclusion case, the origin-reference directional case,
  and idempotence.
- The targeted suites are green: `pytest -q tests/test_monitoring/
  tests/test_db/ tests/test_cli/`. Do not run the full suite here.
- `ruff check wow_forecaster/ tests/` is clean.
- `CHANGELOG.md` gets an entry under `[Unreleased]` in user terms: a durable
  ledger that scores every matured forecast against actuals on error, percentage
  error, direction, and interval coverage, backfilled over the window where
  actuals exist. Work commits carry no version bump; the entry stays under
  `[Unreleased]` and the stamp commit at PR open moves it under the version
  header. The PR ships to main through a pull request, since branch protection
  blocks direct pushes.
- Documentation sync: the new module gets its entry in the matching path-scoped
  rules file under `.claude/rules/` (modeling.md for a forecasting module), the
  migration count line in the repo `CLAUDE.md` moves from "end at 0011" to
  "end at 0012", and the schema table count updates. The `LESSONS.md`
  directional-reference note belongs here if the origin-reference trap bites
  during implementation.
- Version: minor. This adds a new table, a new module, and a new CLI command,
  all surface a consumer must learn about. Not a patch.

## What to expect from the result

The backfill scores on the order of 305K rows in one run. The first aggregate
you will want is MAE and directional accuracy per horizon, and the interesting
read is the comparison across horizons: expect the 1d model to look decent and
the 28d model to look worse, because a 28-day-ahead price is genuinely harder.

Read an ambiguous outcome carefully. If directional accuracy sits near 0.5, the
model is no better than a coin flip on direction, which is a real and reportable
finding, not a bug in the ledger. If interval coverage at the stated 80 percent
band comes in far from 0.80, that is the heuristic confidence interval in
`cold_start.py` being exposed, which PLAN.md already flags as the weakest part of
the modeling story. And if the numbers look suspiciously good, check first
whether directional correctness is being scored against the origin actual or has
quietly reverted to the predicted price as its reference. A ledger that says the
model is excellent is the outcome to distrust most, because until this lab
nothing had ever told the truth about these forecasts at all.
