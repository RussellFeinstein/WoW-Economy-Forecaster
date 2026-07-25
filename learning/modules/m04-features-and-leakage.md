# M04. Features and the leakage boundary

Part II. Features, models, statistics. Prereq: M03.

## Why this module exists

This is the module where the repo gets it right, which is why it has to come
before M06, where the repo gets it wrong.

Feature leakage is the failure everybody has heard of, and this codebase treats
it seriously. `lag_rolling.py` writes down its own boundary. `event_features.py`
stacks three independent guards against look-ahead on event knowledge.
`feature_selector.py` states, per column, why it is excluded. The target columns
look forward on purpose and three separate mechanisms keep them out of the model.
That is more discipline than most portfolio repos carry, and it is the strongest
prose in this one.

It is also incomplete, in a way that is only visible if you stop reading module by
module. `lag_rolling.py` proves its claim about `lag_rolling.py`. The dataset is
assembled from six modules, and the archetype and transfer columns are computed
once from present-day database state and stamped onto every historical row of the
series. `is_cold_start` on a row dated February encodes how many observations that
series eventually accumulated. Nobody wrote that down, because nobody owns the
assembled row.

## The idea to hold onto

A leakage guarantee proved per module is not a leakage guarantee for the pipeline.

Every feature in this dataset passes through one of three regimes:

```
lag / rolling / momentum   reads obs_date and earlier        correct, documented
event                      reads events announced by obs_date correct, three guards
archetype / transfer       reads the database as of build time  looks backward from now
target                     reads obs_date + h                 deliberate, it is the label
```

Three of those four are defensible. The third is the one to find on your own next
time.

## Read this first

The repo is the textbook. Read these before drilling:

- [`wow_forecaster/features/registry.py`](../../wow_forecaster/features/registry.py)
  The whole file, as a list. It is the single declaration of every column on
  disk. Note `is_target`, note `requires_history_days`, and count the groups.
- [`wow_forecaster/features/lag_rolling.py`](../../wow_forecaster/features/lag_rolling.py)
  The `Leakage notes` and `Missing data handling` blocks, then the rolling window
  loop. Check where the window boundary actually sits relative to `obs_date`.
- [`wow_forecaster/ml/feature_selector.py`](../../wow_forecaster/ml/feature_selector.py)
  The `Excluded columns` docstring first, then the three encoding dicts. Compare
  each one against the enum it claims to mirror.
- [`wow_forecaster/models/event.py`](../../wow_forecaster/models/event.py)
  `is_known_at()`, and why `announced_at` is a separate column from `start_date`.
  This pair is also the identification backbone for the M18 event study.
- [`wow_forecaster/features/event_features.py`](../../wow_forecaster/features/event_features.py)
  The three-layer leakage argument at the top. Ask which layer filters anything
  today and which are tripwires for a future edit.
- [`wow_forecaster/features/dataset_builder.py`](../../wow_forecaster/features/dataset_builder.py)
  How the registry becomes two files, and what `write_inference_parquet` keeps.
- [`wow_forecaster/features/archetype_features.py`](../../wow_forecaster/features/archetype_features.py)
  The `Why static columns?` block. Read it twice: the justification is true and
  the question it answers is not the question that matters.

## What you should be able to do afterwards

- Say what separates the 48-column training file, the 45-column inference file,
  and the 40-column model input list, and why none is a subset of another.
- State the leakage boundary for a lag or rolling feature in one sentence, and
  defend including today's own price in a rolling mean.
- Explain what `is_known_at()` guards against, why `None` means unknown, and why
  an announcement date and an effective date have to be separate columns.
- Say what `features_hash` proves, what it cannot detect, and which forecasts do
  not carry one.
- Find a feature that reads present-day state and is stamped onto historical
  rows, without being told which one.

## A note on what this module is worth

Nothing here is broken enough to stop a run. The registry docstring is off by
three, a column-count guard cannot fire, a severity level encodes to the same
value as no event at all, and a cold-start flag tests one of the two conditions it
documents. Every one of them passes CI today.

That is the point. The failures in this layer are quiet by nature: a null column,
a constant flag, a merged category. They cost accuracy that nobody attributes to
them, and they surface only when someone reads a feature definition against its
consumers instead of against its docstring. Do that once here, on
`is_cold_start`, and follow it all the way to the TSM export string. It is the
habit this module is actually teaching.
