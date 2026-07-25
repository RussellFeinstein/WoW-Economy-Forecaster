# Lab 02. Put LightGBM in the backtest

Module: M07. Real work on a real branch, shipped through a PR like anything else.
Issue: [#16](https://github.com/RussellFeinstein/WoW-Economy-Forecaster/issues/16).

## What you are building

`BacktestStage` in `wow_forecaster/pipeline/backtest.py` calls
`all_baseline_models()` and nothing else, so the walk-forward machinery in
`wow_forecaster/backtest/evaluator.py` compares four naive baselines against each
other and never against the model the product actually ships. Grep
`wow_forecaster/backtest/` for `LightGBMForecaster` and you get zero hits. The
production booster is judged only on the single validation holdout inside
`trainer.py`, which is a different partition, a different aggregation, and a
different population from the backtest. The two numbers cannot be compared, so
there is no answer to the one question that matters: does the model beat a random
walk.

Two things have to change for the comparison to exist at all.

First, `backtest.horizons_days = [1, 3]` in `config/default.toml`, while the
product forecasts `[1, 7, 28]` (`features.target_horizons_days`, and
`forecast.horizons`). Horizon 3 is not even a real product horizon:
`TARGET_COL_MAP` in `feature_selector.py` has keys 1, 7, 28 and nothing else. The
backtest side is the wrong one, so raise it to `[1, 7, 28]`.

Second, and this is the substance of the lab, `LightGBMForecaster` does not fit
the evaluator's model protocol. The two interfaces disagree:

| | baseline models | `LightGBMForecaster` |
|---|---|---|
| fit signature | `fit(rows)` | `fit(train_rows, val_rows, feature_cols, categorical_cols, target_col)` |
| what fit sees | one series, `price_mean` history | pre-encoded rows with a target column |
| predict signature | `predict(horizon_days) -> float \| None` | `predict(rows) -> list[float \| None]` |
| scope | fit per series inside the evaluator loop | one global model across all series |

The evaluator loops `(archetype_id, realm_slug)`, calls `model.fit(train_rows)`
once per series per fold, then `model.predict(fold.horizon_days)`. Baselines
carry no horizon into `fit` and no feature vector at all. LightGBM needs both a
horizon (to know which target to learn) and the full 40-column encoded feature
matrix. You reconcile that with an adapter that presents the minimal
`fit(rows)` / `predict(horizon_days)` protocol on the outside and drives a real
`LightGBMForecaster` on the inside. The model docstring in `backtest/models.py`
says this out loud: "This protocol is intentionally minimal so ML models can
implement the same interface later."

## Before you write any code

Cut the branch from the latest main. Adding a model to the comparison and a new
row to `report-backtest` is new surface, so this is `feat/`:

```
git checkout main && git pull --ff-only
git checkout -b feat/16-lightgbm-backtest-loop
```

Read Lab 01 first if you have not. Its purge is the same leakage boundary you are
about to hit inside the adapter, from the other direction.

## The adapter, and the honest scope

Put the adapter in `wow_forecaster/backtest/models.py`, next to the baselines.
Call it `LightGBMBacktestModel`, give it `name = "lightgbm"`.

`fit(rows)` receives one series's training rows, already filtered by the evaluator
to `obs_date in [fold.train_start, fold.train_end]`. The horizon is not known
until `predict`, so `fit` just stores the encoded rows. `predict(horizon_days)`
is where the work happens:

1. Build labels from within the training rows only. For a row at date `T`, the
   label is `price_mean` at `T + horizon_days`, looked up from the same series's
   rows. Any row whose `T + horizon_days` has no partner inside the training
   window is dropped, because its label would have to come from `test_date` or
   later. This is the Lab 01 purge again: no training label may reference a date
   after `train_end`. It is what keeps the backtest number honest.
2. If fewer than the forecaster's floor of labeled rows survive (it raises below
   10), return `None`. `compute_metrics` already drops `None` predictions, the
   same as a baseline that returns `None` on a thin series.
3. Fit a `LightGBMForecaster(horizon_days=horizon_days)` on the surviving
   `(encoded_row, label)` pairs, pass `TRAINING_FEATURE_COLS` and
   `CATEGORICAL_FEATURE_COLS` from `feature_selector.py`, then predict the feature
   vector of the last training row. Memoize per horizon so a repeated
   `predict(h)` does not refit.

The evaluator constructs a fresh model list per horizon (`all_baseline_models()`
is called inside `for h in _horizons`), so per-horizon memoization is trivial and
no cross-horizon state leaks.

State the scope limit plainly in the docstring and the PR, because it is the
difference between an honest artifact and a misleading one. `lgbm_model.py`
argues at length for ONE global model across all archetype-realm series. This
adapter fits a booster per series on a handful of rows. It is not the production
model and it will overfit. What the lightgbm row measures is "does gradient
boosting on this single series beat the naive baselines on the same folds, the
same test dates, and the same metric." That is a real advance over the leaked
holdout, and it is not the same claim as "the production global model wins." The
global-model backtest is the rigorous M1 version (Diebold-Mariano and Wilcoxon,
issue [#17](https://github.com/RussellFeinstein/WoW-Economy-Forecaster/issues/17)),
and it stays there. Do not let this PR imply it did that work.

## Write the failing test first

Two tests, both in `tests/test_backtest/test_models.py`. The lightgbm ones need
the `lightgbm` dependency, which the dev extra already installs.

The unit test that catches the real thing: fit the adapter on a series with a
clear monotonic trend, call `predict(7)`, and assert it returns a finite float
from within the trained range. Then the leakage assertion, which is the one that
matters: construct a series where the only way to hit the target MAE is to have
trained on a label dated after `train_end`, and assert the adapter's labeled-row
count excludes it. If you cannot observe the labeled set directly, assert that a
row added strictly after the last training date changes no prediction.

The wiring test: build synthetic folds and feature rows, run `run_backtest` with
the adapter in the model list, and assert `slice_by_model_and_horizon` produces a
`("lightgbm", h)` key with `n_evaluated > 0`. That is what proves the row will
actually render in `report-backtest`.

The tempting assertion that does **not** earn its place: asserting the adapter
class exists and exposes `name == "lightgbm"`. It passes the moment you write the
class and tells you nothing about labels, leakage, or wiring. An adapter that
trains on zero labeled rows, or on a leaked label, sails straight through it.

Confirm both real assertions fail against the current tree before you touch the
implementation.

## Wiring it in

`pipeline/backtest.py` is the single execution path. The CLI `backtest` command
(`cli.py` around line 1060) runs `BacktestStage.run()`; its `--dry-run` branch
lists models for display only.

- Keep `all_baseline_models()` returning exactly four. Do not add lightgbm to it.
  `test_all_baseline_models_returns_four_instances` pins that set on purpose, and
  "baselines" should stay the four baselines. Add a small separate constructor
  (for example `all_models_with_lightgbm()`) or append the adapter where the
  stage builds its model list, at `models = all_baseline_models()` inside the
  horizon loop.
- Update the `--dry-run` model list in `cli.py` too, so the dry-run echo names
  lightgbm. A dry run that lies about what the real run does is its own small bug.
- In `config/default.toml`, set `horizons_days = [1, 7, 28]` and fix the trailing
  comment that still explains "3d". In `config.py`, change the `BacktestConfig`
  default `horizons_days: list[int] = [1, 3]` to `[1, 7, 28]` so code and config
  agree.

## Do not

- Do not shrink `features.target_horizons_days` to `[1, 3]` to make the horizons
  line up. That is Lab 01's warning from the other side: it deletes the product's
  real outputs to dodge a measurement gap. Raise the backtest side.
- Do not build labels from the evaluator's `price_lookup` on `test_date`, or from
  any date past `train_end`. That is the exact leak the backtest exists to avoid,
  and it would hand back a flattering number that means nothing.
- Do not describe the per-series adapter as the production model. Name the gap.
- Do not retune hyperparameters here. One concern per PR, and the honest number
  from stock config is the deliverable.
- Do not touch `backtest/splits.py` or the leakage structure of the evaluator
  loop. Both are correct.
- Do not special-case the 28d horizon out because current history is short.
  `window_days = 30`, `step_days = 7`, horizon 28: on thin data this may produce
  few folds or none, so the 28d lightgbm row may be sparse or absent. That is an
  honest fact about the data, not a bug to hide.

## Finish line

- The two new tests fail against the current tree and pass after the change.
- The full suite is green: `.venv/Scripts/python.exe -m pytest -q`.
- `ruff check wow_forecaster/ tests/` is clean.
- `CHANGELOG.md` has an entry under `[Unreleased]`, in user terms: the backtest
  now scores the LightGBM model against the four baselines at horizons 1, 7 and
  28, so there is a like-for-like comparison for the first time. Note the config
  horizon change.
- Documentation sync: update the backtest layer note in this repo's `CLAUDE.md`
  (it still says `horizons_days=[1,3]` and describes only baselines). If the
  README quotes the backtest horizons, fix it there too. `PLAN.md` records this as
  DS-2 and as Phase 1 backtest work; reference it, do not restate it.
- Version: minor. This adds a model to a visible comparison and changes the
  default backtest horizons, both of which a reader of `report-backtest` sees.
  Work commits carry no bump and their lines sit under `[Unreleased]`; a separate
  stamp commit at PR open moves them under the version header and bumps
  `pyproject.toml` once, with the `(vX.Y.Z)` suffix on the PR title. It ships
  through a PR to main, because branch protection blocks direct pushes.

## What to expect from the result

Run it and read `report-backtest`:

```
.venv/Scripts/python.exe -m wow_forecaster backtest --horizons 1,7,28
.venv/Scripts/python.exe -m wow_forecaster report-backtest
```

You want a `lightgbm` row beside `last_value`, `rolling_mean`, `day_of_week`, and
`simple_volatility` at each horizon that produced folds.

Now read it honestly. `price_mean` is both a feature and the basis of every
target, so a model that learns "predict roughly today's price" posts a
respectable MAE while being, functionally, the `last_value` baseline. So:

- If lightgbm lands right on top of `last_value`, that is not a win. It is the
  most likely outcome for a per-series booster on thin data, and it is the signal
  that the model has learned the random walk and nothing more.
- If lightgbm is clearly worse than `last_value`, that is also informative and
  entirely plausible: 40 features on 14 to 30 rows overfits. It is an argument for
  the global model, which is exactly the follow-up.
- If lightgbm beats every baseline at a horizon, be suspicious before you are
  pleased. Re-check that no label crossed `train_end`. A leak reads as a win.

The deliverable is the comparison existing and being trustworthy, not the model
winning. Whatever the number says, it is the first version of it that is not
comparing two incomparable things.

## Reflection, before you close the issue

Add a `LESSONS.md`-style note: the protocol comment in `backtest/models.py`
promised ML models would slot in "later," and "later" sat unclaimed long enough
that the comparison the module's own docstring demands ("if an ML model cannot
beat ALL of these baselines, it is not ready") was never run. What cheap check
surfaces a stated-but-never-exercised contract like that before it goes stale?
