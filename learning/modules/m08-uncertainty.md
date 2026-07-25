# M08. Uncertainty

Part II. Features, models, statistics. Prereq: M07.

## Why this module exists

Every row in `forecast_outputs` carries `confidence_lower`, `confidence_upper`
and `confidence_pct = 0.80`. Around 300 thousand of those forecasts have already
matured, meaning the price they predicted is now a fact sitting in the rollup
tables.

Not one of them has ever been checked against that fact.

So the system has been shipping a probability claim for its whole life and has no
evidence for it in either direction. `cold_start.py` is honest about this in its
own docstring, which is more than most codebases manage, and PLAN.md's CEILING
section calls the interval the weakest part of the modeling story. This module is
about understanding exactly what the claim is made of, and about the difference
between reporting an interval and reporting coverage.

## The idea to hold onto

An interval is a claim about a frequency, and a frequency can only be measured
after the fact:

```
stated:    confidence_pct = 0.80
means:     over many forecasts, ~80% of realized prices land inside their interval
measured:  count(lower <= actual <= upper) / count(matured forecasts)
```

Nothing about how an interval is constructed can tell you whether that holds. You
build it, you wait, you count.

What this system builds instead is a z value times a 7-day rolling standard
deviation of the price level, widened for cold start, floored at 5 percent and
capped at 10x. Every question in this module comes from comparing those two
paragraphs.

## Read this first

The repo is the textbook. Read these before drilling:

- [`wow_forecaster/ml/cold_start.py`](../../wow_forecaster/ml/cold_start.py)
  The whole module docstring, then `compute_confidence_interval` line by line.
  Count the floors. Note what the function signature does not take.
- [`wow_forecaster/ml/predictor.py`](../../wow_forecaster/ml/predictor.py)
  The horizon loop and the drift-widening block below the CI call. Ask what
  `center` is, and whether it equals the prediction.
- [`wow_forecaster/monitoring/adaptive.py`](../../wow_forecaster/monitoring/adaptive.py)
  The policy table and the argument for each multiplier. Then grep the scorer for
  the multiplier and see where the docstring's account and the code part company.
- [`wow_forecaster/recommendations/scorer.py`](../../wow_forecaster/recommendations/scorer.py)
  The `uncertainty_penalty` block and `determine_risk_level`. This is where a
  miscalibrated interval stops being a reporting problem and becomes a wrong risk
  label on a trade.
- [`wow_forecaster/backtest/metrics.py`](../../wow_forecaster/backtest/metrics.py)
  Read `PredictionRecord` and `BacktestMetrics` and notice what is absent. The
  harness cannot compute coverage even if you asked it to.
- [`PLAN.md`](../../PLAN.md)
  The CEILING section on conformal intervals, and DS-3 on the drift baseline that
  feeds the uncertainty multiplier.
- [`docs/ROADMAP.md`](../../docs/ROADMAP.md)
  M1. The realization ledger scores interval coverage; quantile regression comes
  after it, on purpose.

## What you should be able to do afterwards

- Trace a point prediction to a stored interval, naming all seven steps.
- Define empirical coverage, and say what undercoverage and overcoverage each
  cost you.
- Say why the 1d and 28d intervals come out the same width, and what they should
  scale with instead.
- Explain what the 5 percent floor and the 10x cap were fixing, and what they
  broke.
- Say what quantile regression buys, what conformal prediction guarantees, and
  why measuring comes before either.

## A note on being fair to the code

None of this is sloppiness. The interval was built early, when there were no
matured forecasts to score against and a heuristic was the only option available.
The floor and the cap were added because degenerate bounds were showing up in
reports, which is a real problem with a real fix. Every step was reasonable at
the time it was taken.

What accumulated is a chain of reasonable steps with no measurement anywhere
along it, and a stated confidence level that has quietly become a column value
rather than a claim. That is the failure mode worth learning, because it does not
announce itself: the pipeline is green, the intervals look plausible, and the
dashboard renders.

The cure is not cleverer intervals. It is one join between forecasts and actuals.
