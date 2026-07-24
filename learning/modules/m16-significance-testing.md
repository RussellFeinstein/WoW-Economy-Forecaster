# M16. Is the difference real?

Part IV. Proving it and shipping it. Prereqs: M05, M06.

## Why this module exists

Every accuracy claim in this repo is unsupported, and not because the numbers are
bad. They have never been compared to anything. `backtest/models.py` states the
bar out loud, "If an ML model cannot beat ALL of these baselines, it is not ready
for use," and that comparison has never been run. M06 showed the training holdout
leaks; this module is about the tool you reach for once the leak is fixed and a
matched model error series finally exists: a test that says whether one forecaster
is really better than another, or whether you are looking at noise.

The subject is almost entirely prospective. Search `wow_forecaster/` for
Diebold-Mariano or Wilcoxon and you find nothing. `metrics.py` computes MAE, RMSE,
MAPE, and directional accuracy, and stops. The significance layer is M1 issue #17.
So this module teaches the method, not existing code, and half its value is being
able to say precisely why "the model looks good" is a claim the repo cannot yet
back.

## The idea to hold onto

A lower average error does not establish that one model is better. The comparison
has to be **paired** and then **tested**.

```
d_t = error(model, t) - error(baseline, t)      one difference per fold
```

Both models predict the same test dates, so the shared difficulty of each day
cancels in `d_t`. Then you ask one question of that series: is its center zero?
Diebold-Mariano asks it of the mean, Wilcoxon of the median, and they fail
differently, which is why you run both.

## Read this first

The repo is the textbook. Read these before drilling:

- [`wow_forecaster/backtest/metrics.py`](../../wow_forecaster/backtest/metrics.py)
  The metric definitions and the error list construction. Note that
  `compute_metrics` collapses folds into one MAE per call. That is the wrong grain
  for a paired test, and knowing why is the point.
- [`wow_forecaster/backtest/evaluator.py`](../../wow_forecaster/backtest/evaluator.py)
  The loop that emits one PredictionRecord per fold, series, and model. This is
  where the paired structure the test needs is actually produced.
- [`wow_forecaster/backtest/models.py`](../../wow_forecaster/backtest/models.py)
  The four baselines and the acceptance bar. Each baseline poses a specific
  hypothesis; a paired test is how you would answer it with evidence.
- [`wow_forecaster/db/schema.py`](../../wow_forecaster/db/schema.py)
  The `backtest_fold_results` table. Per-fold `abs_error` keyed by model and
  horizon is the raw material for pairing, and it holds the four baselines only.
- [`docs/ROADMAP.md`](../../docs/ROADMAP.md)
  The M1 section. The significance tests are named there and nowhere in the code.
- [`PLAN.md`](../../PLAN.md)
  Audit findings DS-1 and DS-2. These are the preconditions: a significance test
  inherits the validity of the numbers it is fed, and today those numbers leak and
  are unmatched.

## What you should be able to do afterwards

- Explain why forecast comparison is paired and what pairing buys in variance and
  power.
- State what Diebold-Mariano tests, why it uses a HAC variance, and what
  stationarity assumption it rests on.
- Say when to prefer Wilcoxon signed-rank, what it tests instead, and what it
  gives up.
- Work the multiple-comparison arithmetic across 3 horizons and 4 baselines, and
  explain why "beats all" is a self-correcting conjunction while cherry-picking is
  not.
- Separate a significant difference from one that matters, and name P&L and
  directional accuracy as the metrics that answer the second question.

## A note on order of operations

Do not reach for these tests first. A DM p-value on a leaked, unmatched holdout is
a precise answer to the wrong question, and it reads as more trustworthy than a
rough estimate on a sound one, which makes it worse than nothing. Fix the split
(M06's lab), put LightGBM in the backtest loop, align `backtest.horizons_days` up
to `[1, 7, 28]`, and only then does a significance test measure what its math
claims. Until then, the useful move is to run the pipeline on the baselines
against each other: it validates the whole join-difference-test path on real data
and answers a real question (does mean-reversion beat the random walk?) while you
wait for a model worth testing.
