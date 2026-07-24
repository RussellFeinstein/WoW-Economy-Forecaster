# Lessons

Approaches in this repo that were tried, turned out wrong, and had to be
corrected. Kept because the reasoning that produced each mistake was plausible
at the time, and writing down why it was wrong is cheaper than rediscovering it.

New entries go at the bottom, in the form `## YYYY-MM-DD - <title>` with
**Tried**, **Broke**, and **Lesson**.

Related: [docs/postmortem-2026-04-lock-outage.md](docs/postmortem-2026-04-lock-outage.md)
covers the operational outage in depth and is not duplicated here.
[PLAN.md](PLAN.md) carries the audit that produced the first two entries.

---

## 2026-07-24 - A time-based train/validation split is not enough for a multi-horizon target

**Tried:** Split the training Parquet by `obs_date` and hold out the last
`validation_split_days` calendar days as validation. The docstring in
`ml/trainer.py` states the reasoning: "Time-based validation split: last
`validation_split_days` calendar days. NEVER random, random splits on
time-series create look-ahead bias." That reasoning is correct and the split
does prevent the failure it names.

**Broke:** It prevents feature leakage and misses label leakage. Each training
row at date T carries the label `price(T + h)`. With a 14-day validation window
and horizons of 1, 7, and 28 days, training labels run to `split_date + h`. The
28-day model's training labels therefore cover the entire validation window and
14 days beyond it, so every validation label is a price the model already saw
during training. The 7-day model leaks half the window. Because that same
validation set drives LightGBM early stopping, the stopping round is chosen
against the contaminated signal too, and the `validation_mae` recorded in
`model_metadata` reads better than it should.

**Lesson:** For a forecasting target, the split boundary belongs on the *label*
date, not the feature date. Drop training rows whose target date falls at or
after the validation start (purging), then add an embargo of `h` days so
adjacent rows do not share overlapping label windows. `backtest/splits.py` got
this right independently: its structural guarantee is `test_date > train_end`,
which is a statement about labels. Two components in the same repo disagreed
about what a split means, and only one of them was audited for it.

---

## 2026-07-24 - Building the baseline harness is not the same as running the comparison

**Tried:** Build a walk-forward backtest with four baseline models chosen to each
test a specific hypothesis (random walk, mean reversion, weekly seasonality,
volatility persistence), a metrics layer with MAE, RMSE, MAPE, and directional
accuracy, and five slicing dimensions. `backtest/models.py` states the acceptance
bar directly: "If an ML model cannot beat ALL of these baselines, it is not ready
for use." The model interface was deliberately kept minimal so an ML model could
implement it later.

**Broke:** Later never arrived. `BacktestStage` calls `all_baseline_models()` and
nothing else, so the harness spent every run comparing four naive baselines
against each other while the production LightGBM model was judged on a single
holdout split. The two sets of numbers are not comparable: different partitions,
different aggregation, different populations. The horizons did not even match, at
1 and 3 days for backtests against 1, 7, and 28 in production. Meanwhile
`price_mean` is both a model feature and the basis of the target, so a model that
learned nothing beyond "predict roughly today's price" would post a respectable
MAE while being exactly the `last_value` baseline, and nothing in the system
could tell those two apart.

**Lesson:** A baseline that the candidate model is never actually run against is
not a baseline, it is a fixture. When a docstring states an acceptance bar, wire
a test or a report that fails when the bar goes unmeasured, otherwise the stated
standard and the enforced standard drift apart silently. The tell here was
visible in config the whole time: `backtest.horizons_days` and
`features.target_horizons_days` held different values, which no comparison could
have survived.
