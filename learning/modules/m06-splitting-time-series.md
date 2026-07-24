# M06. Splitting time series without lying to yourself

Part II. Features, models, statistics. Prereq: M05. Lab: `lab-01-purge-embargo`.

## Why this module exists

Two components in this repo disagree about what a train/test split means.

`backtest/splits.py` puts the boundary on the **label** date. Its guarantee is
`test_date > train_end`, which is a statement about the date being predicted.

`ml/trainer.py` puts the boundary on the **feature** date. It sorts rows by
`obs_date` and holds out the last 14 calendar days.

For a single-step target those are the same split. For a multi-horizon target
they are not, and the difference is that every validation label the 28-day model
is scored against is a price it already trained on.

Only one of the two was ever audited for leakage. That is the finding, and it was
visible in the source without running anything.

## The idea to hold onto

A forecasting row is two points in time at once:

```
obs_date = T                 <- when the features were observed
target   = price(T + h)      <- the date being predicted
```

A split defined over rows, or over feature dates alone, does not pin down where
the labels fall. A split defined over the predicted date does.

Every question in this module is a consequence of that one sentence.

## Read this first

The repo is the textbook. Read these before drilling:

- [`wow_forecaster/backtest/splits.py`](../../wow_forecaster/backtest/splits.py)
  Read the whole module docstring. It is the correct version of the idea, and it
  argues its own design choices (rolling versus expanding window, why leakage
  prevention is structural rather than asserted).
- [`wow_forecaster/ml/trainer.py`](../../wow_forecaster/ml/trainer.py)
  The `Time-based validation split` block, plus the 80/20 fallback below it. Note
  what the docstring claims and compare it against what the code does.
- [`wow_forecaster/ml/lgbm_model.py`](../../wow_forecaster/ml/lgbm_model.py)
  The `fit` method's watchlist and early-stopping callback. This is where the
  leak stops being a reporting problem and starts changing the model.
- [`LESSONS.md`](../../LESSONS.md)
  The 2026-07-24 entry, written when this was found.
- [`config/default.toml`](../../config/default.toml)
  Compare `backtest.horizons_days` against `features.target_horizons_days`. The
  mismatch is a config-level tell that the comparison could never have run.

## What you should be able to do afterwards

- State the split-boundary rule for a forecasting target in one sentence.
- Given a validation window and a horizon list, compute which horizons leak and
  by how many days.
- Define purging and embargo, and say what each one fixes and what it costs.
- Explain the second-order damage through early stopping.
- Give the ninety-second version out loud, including what it invalidates.

## Then do the lab

`wowfc learn lab lab-01-purge-embargo`

The lab is the fix. Writing the failing test first is the point of it: a test
asserting on `obs_date` passes today and would greenlight the exact bug.

## A note on what this does and does not fix

Purging the split makes the reported number honest. It does not make the model
good. `price_mean` is both a feature and the basis of the target, so a model that
learned nothing beyond "predict roughly today's price" posts a respectable MAE
either way, and nothing in the system can currently tell that apart from skill,
because the model has never been run against the baselines.

That comparison is M07's lab. This module is the prerequisite for it being worth
running.
