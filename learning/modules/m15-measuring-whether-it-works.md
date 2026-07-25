# M15. Measuring whether it works

Part IV. Proving it and shipping it. Prereq: M05, M06.

## Why this module exists

Over 300,000 forecasts sit in `forecast_outputs`. Not one has been compared to
what the price actually did.

That is not because nobody thought about monitoring. There is a drift checker
that runs every hour, a live health evaluator with a CLI command behind it, and a
`model_health_snapshots` table with the right columns in it. The machinery is
there. It just never closes the loop, and auditing why is more instructive than
the fix.

The pipeline monitoring in this repo is genuinely strong: coverage gaps, a stale
lock sentinel, a retention sentinel, backup freshness, a dead-man alarm on the
cloud capture. All of it answers "did the plumbing run". The one thing that would
answer "were the predictions any good" is the one thing nobody scheduled.

## The idea to hold onto

A monitor that cannot compute its metric must say so.

```
mae_ratio = None    ->  health.py returns "unknown"     correct
mae_ratio = None    ->  drift.py returns DriftLevel.NONE  wrong
```

`NONE` is not an inert label. It flows into the adaptive policy, which returns an
uncertainty multiplier of 1.0 and no retrain advisory, and that multiplier is
persisted and read back by the next forecast run to size its confidence
intervals. So a monitor that measured nothing does not merely stay quiet: it
issues a positive all-clear and keeps the intervals at their narrowest.

The conditions that produce `None` (no backtest has run, the horizons do not
match, no forecast has matured, the actuals were pruned) are the conditions a
neglected system sits in for months. The monitor is quietest exactly when it is
blindest.

The project already states this principle somewhere else. The cloud capture gap
guard keeps its floor at 20 distinct hours on purpose, because a floor the
failure mode can satisfy hides the failure. Same rule, applied to a different
alarm, by the same author, a few weeks apart.

## Read this first

The repo is the textbook. Read these before drilling:

- [`wow_forecaster/monitoring/drift.py`](../../wow_forecaster/monitoring/drift.py)
  The module docstring first: three detection modes, and a good argument for a
  z-score of means over PSI. Then `_classify_error_drift`, the `baseline_mae`
  query inside `check_error_drift`, and the join condition above it. Note what
  `run_all` passes for `horizon_days`.
- [`wow_forecaster/monitoring/health.py`](../../wow_forecaster/monitoring/health.py)
  The same computation, one file over, with the `unknown` case handled correctly.
  Read the comment in the live loop explaining why directional accuracy is
  skipped, then look at which fields of `ModelHealthSummary` can never be
  populated.
- [`wow_forecaster/monitoring/adaptive.py`](../../wow_forecaster/monitoring/adaptive.py)
  Short. This is where a drift verdict becomes a number that changes a user-facing
  recommendation, which is what makes a wrong `NONE` expensive.
- [`wow_forecaster/backtest/metrics.py`](../../wow_forecaster/backtest/metrics.py)
  The metric rationale block. Why MAPE exists (cross-archetype comparison), how
  directional accuracy is defined, and what it does with ties.
- [`wow_forecaster/governance/pruner.py`](../../wow_forecaster/governance/pruner.py)
  The first twenty lines. Normalized observations are pruned with their raw
  parents at 30 days, and the rollups are not. That single fact decides which
  table the realization ledger has to read.
- [`wow_forecaster/ml/cold_start.py`](../../wow_forecaster/ml/cold_start.py)
  The uncertainty note. The intervals are heuristic, they are labelled 80
  percent, and nothing has ever checked whether they cover 80 percent.
- [`PLAN.md`](../../PLAN.md)
  The DS-3 finding and Phase 5. This is the honest version of everything above.
- [`docs/ROADMAP.md`](../../docs/ROADMAP.md)
  M1, issue #13. The ledger, and why it is the keystone that M2, M3, and M16 all
  hang off.

## What you should be able to do afterwards

- State the realization ledger's grain and both conditions for maturity.
- List the four defects in the pooled error-drift baseline from memory.
- Explain, with a worked two-archetype example, why a pooled absolute gold MAE
  moves with the archetype mix.
- Say what `unknown` buys over `none`, in terms of what the pipeline does next
  rather than in terms of semantics.
- Name the metrics the ledger unlocks on day one, including the one that is a
  single boolean per row.
- Answer "how do you know it works?" honestly, in ninety seconds.

## A note on order

M16 is the significance testing module and it needs this one first. A paired test
operates on per-observation error differences, so a ledger that stores only
aggregate means gives it nothing to work with. That is why q12 spends its time on
the grain and the uniqueness constraint rather than on the metric formulas: the
formulas are easy, and the schema decision is the one that is expensive to change
later.

The fix order in the plan is deliberate too. Prediction monitoring is Phase 5,
after the ledger lands, because fixing the four baseline defects on their own
would produce a cleaner number that still answers the wrong question. Degradation
is a statement about one model's error changing over time, and that needs the
model's own scored history, which does not exist yet.
