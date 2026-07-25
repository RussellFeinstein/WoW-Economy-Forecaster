# M07. The model

Part II. Features, models, statistics. Prereq: M06.

## Why this module exists

The modeling choices in this repo are argued, in prose, in the files that make
them. `lgbm_model.py` opens with a comparison against XGBoost, Prophet, and an
LSTM. `feature_selector.py` explains why each excluded column is excluded.
`trainer.py` states why the model is global rather than per-archetype. That is
better than most portfolio repos manage and it is the reason this module can be
questions rather than a lecture.

What none of those docstrings do is say which parts of the argument are about
this data and which are generic, or what the choices cost. A model-choice
paragraph that would read identically on someone else's tabular problem is not
an argument about your problem. Half of this module is separating those.

The other half is what the argument does not cover: an L1 objective quietly
changes what the model estimates, missing values are a modeling decision rather
than a convenience, and identity can walk back in through the feature list after
being shown the door as a column.

## The idea to hold onto

The model has no idea which series a row came from.

```
archetype_id      excluded, deliberately
realm_slug        excluded, one model per realm instead
obs_date          excluded, temporal signal via day_of_week / week_of_year
```

What remains is a description of market state: price level, lags, rolling
statistics, momentum, event context, category. The booster maps market state to
next price. That is the only reason a model trained entirely on TWW can be
pointed at a Midnight archetype it has never seen.

Every design decision in this module either protects that property or quietly
undermines it, and telling those apart is the work.

## Read this first

The repo is the textbook. Read these before drilling:

- [`wow_forecaster/ml/lgbm_model.py`](../../wow_forecaster/ml/lgbm_model.py)
  The whole module docstring, then `fit()`. Note which comparison arguments cite
  a property of this data and which cite a property of the model family. In
  `fit()`, read the `lgb_params` dict slowly: the objective is the decision with
  the longest reach.
- [`wow_forecaster/ml/feature_selector.py`](../../wow_forecaster/ml/feature_selector.py)
  The excluded-columns list, the three encoding dicts, and
  `CATEGORICAL_FEATURE_COLS`. Ask of every encoding whether two distinct states
  collapse onto the same code.
- [`wow_forecaster/ml/trainer.py`](../../wow_forecaster/ml/trainer.py)
  The global-model design note, and then follow `val_metrics` all the way into
  `_register_model` and see which of the three computed metrics survives the
  trip.
- [`wow_forecaster/ml/predictor.py`](../../wow_forecaster/ml/predictor.py)
  The inference path: `run_inference()` calls `predict()` for every row. Then go
  back to `lgbm_model.py` and compare what `predict()` returns against what
  `_evaluate()` scored. They are not the same function.
- [`config/default.toml`](../../config/default.toml)
  The `[model]` block. Every hyperparameter in the booster is here, and each one
  is a bias-variance position someone took without testing it.
- [`PLAN.md`](../../PLAN.md)
  The DS-3 paragraph on pooled absolute-gold MAE. The objection lands on more
  than the drift baseline.

## What you should be able to do afterwards

- Explain gradient boosting, including what each tree is fit to, and survive the
  follow-up.
- Give the LightGBM argument against Prophet, an LSTM, and ARIMA in terms of
  this data rather than in general.
- Say why one global model beats one model per archetype here, and name what
  pooling costs.
- Say what an L1 objective estimates and where that sits awkwardly against the
  confidence interval built around it.
- Distinguish gain from split-count importance, and say what each one hides.
- Predict what the code does at the seams: an empty validation set, a missing
  inference column, a negative categorical code.

## A note on what is not established

Every hyperparameter in `[model]` is a default that has never been swept. The
global-versus-per-archetype choice is well argued and untested. L1 versus L2 has
never been compared on this data. And the one holdout that would test the
transfer premise directly, holding out entire archetypes rather than trailing
dates, does not exist anywhere in the repo.

None of that makes the choices wrong. It makes them undefended, which is a
different claim and the one to make out loud. M06 is why the current numbers do
not settle it; M15 and M16 are where it gets settled properly.
