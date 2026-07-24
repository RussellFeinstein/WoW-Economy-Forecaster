# M05. Baselines and what good means

Part II. Features, models, statistics. Prereq: M04.

## Why this module exists

`backtest/models.py` states the acceptance bar in one sentence: an ML model that
cannot beat all four baselines is not ready for use. That sentence is the best
line of methodology writing in the repo, and it has never been evaluated.

The harness exists. The four baselines exist. The metrics layer exists, with five
slicing dimensions. What is missing is the run. `BacktestStage` instantiates the
baselines and nothing else, and `backtest.horizons_days = [1, 3]` against
`features.target_horizons_days = [1, 7, 28]` meant two of the three production
horizons had no baseline to be compared against even in principle.

So this module is not "learn what MAE is". It is: know what each number in a
results table can and cannot tell you, so that when the comparison does run you
can read it without fooling yourself.

## The idea to hold onto

A metric is a question, and four metrics are four different questions:

```
MAE                   how far off, in gold, on the records we scored
RMSE / MAE            how much the size of the errors varies
MAPE                  how far off in percent, comparable across price levels
directional accuracy  did we call the direction, on days there was one
```

None of them is the metric. A model can win on MAE by being good at the few
expensive archetypes, win on MAPE by being good at the many cheap ones, and lose
on direction while doing either. Which one decides is a product question: this
system emits buy, hold and sell, so direction is not a tiebreaker.

The same applies to the baselines. Each one is a hypothesis about the market, so a
baseline that loses still tells you something (no exploitable mean reversion at 7
days) and a baseline that wins tells you what to build next.

## Read this first

The repo is the textbook. Read these before drilling:

- [`wow_forecaster/backtest/models.py`](../../wow_forecaster/backtest/models.py)
  The whole opening docstring, then the four classes. Read what each baseline
  claims about the market, then check the `predict()` body against the claim.
  Compare `RollingMeanModel` and `SimpleVolatilityModel` line by line.
- [`wow_forecaster/backtest/metrics.py`](../../wow_forecaster/backtest/metrics.py)
  The metric-rationale docstring, then `compute_metrics()`. Look closely at the
  `mape_terms` filter, at which side of the directional comparison ties are
  dropped from, and at the difference between `n_predictions` and `n_evaluated`.
- [`wow_forecaster/backtest/evaluator.py`](../../wow_forecaster/backtest/evaluator.py)
  Specifically how `last_known_price` is populated. Hold that next to
  `LastValueModel.fit()` and work out what the random-walk baseline's directional
  accuracy has to be.
- [`wow_forecaster/ml/lgbm_model.py`](../../wow_forecaster/ml/lgbm_model.py)
  The `_evaluate()` method, which is the second metrics implementation and the
  one that produces `model_metadata.validation_mae`. Also the `objective` line:
  an L1 loss estimates the conditional median, and this is a market with spikes.
- [`config/default.toml`](../../config/default.toml)
  The `[backtest]` block against the `[features]` horizons.
- [`PLAN.md`](../../PLAN.md)
  Audit findings DS-2 and DS-3. DS-3 is this module's metric lesson showing up in
  production monitoring: a pooled absolute-gold reference with no model filter.

## What you should be able to do afterwards

- Name each baseline and the hypothesis it tests, and say what a loss means.
- Say why `RMSE > MAE` carries no information and what the ratio does carry.
- Explain why MAE cannot be pooled across archetypes and MAPE can, then name
  MAPE's two failure modes (near-zero actuals, asymmetry).
- Explain why ties leave the directional denominator, and why the asymmetry in
  how that filter is applied forces `last_value` to score exactly 0.0.
- State the acceptance bar from memory, and list what a comparison must hold
  constant for it to mean anything.

## A note on what this module does not settle

Nothing here makes the model good. It makes a future results table readable, which
is the prerequisite for finding out. The comparison itself is M07's lab: put
`LightGBMForecaster` behind the existing `fit`/`predict` protocol, align the
horizon lists, and read four numbers per horizon for the first time.

M06 is the other prerequisite. A clean comparison against a leaked training split
would still be a comparison between one honest number and one dishonest one.
