# M09. Cold start and transfer

Part II. Features, models, statistics. Prereq: M08.

## Why this module exists

This is the premise the project is named after. A new expansion ships with no
price history, so a forecast for a Midnight archetype has to borrow from the
analogous TWW archetype. Everything else in the repo is downstream of that idea
working.

It is built. It is tested. It is documented. It has never run.

Nothing in the codebase writes a row into `archetype_mappings`. There is a
repository class with a working `insert()` and no caller, no CLI command, and no
seed file. The table is empty, so `_fetch_cold_start_blend_data` returns `{}`,
so `run_inference` gets `cold_start_blend=None`, so no forecast has ever been
blended.

That is the honest headline, and it is the same shape as the finding in M06:
the mechanism is correct and the operation never happened.

## The idea to hold onto

Two moves, and they are halves of one design.

```
strip identity   archetype_id is excluded from the feature set, so a
                 never-seen archetype arrives as a familiar point in
                 feature space instead of an unseen category value

reinject level   the model can no longer tell you that THIS archetype
                 sits above its category, so the blend supplies a level
                 from the mapped source archetype's recent price
```

The blend itself is shrinkage toward an anchor:

```
blended = conf * model_prediction + (1 - conf) * source_price
```

`conf` weights the **model**, not the anchor. High confidence in the mapping
means less anchoring, which is the opposite of most people's first guess.

Both endpoints are complete forecasters. `conf = 0` is the pure analogy
forecast. `conf = 1` is the pure model. Nothing in this repo has ever scored
them against each other, which means the weight in the middle is unjustified by
construction rather than by accident.

## Read this first

The repo is the textbook. Read these before drilling:

- [`wow_forecaster/ml/cold_start.py`](../../wow_forecaster/ml/cold_start.py)
  The whole module docstring, which is the clearest statement of the design
  anywhere in the repo. Then read the three functions against it: the blend, the
  interval, and the slug. Pay attention to where `transfer_confidence` is used
  twice in opposite-looking ways, and to the parenthetical explaining the
  re-clamp.
- [`wow_forecaster/ml/predictor.py`](../../wow_forecaster/ml/predictor.py)
  The inner loop of `run_inference`. Which values come off the Parquet row and
  which come from the blend dict. That distinction is the whole of q11 and q12.
- [`wow_forecaster/pipeline/forecast.py`](../../wow_forecaster/pipeline/forecast.py)
  `_fetch_cold_start_blend_data`. Note which archetype id the result is keyed by.
- [`wow_forecaster/features/archetype_features.py`](../../wow_forecaster/features/archetype_features.py)
  `load_archetype_metadata` (note the join column) and
  `compute_archetype_features` (note what `is_cold_start` actually evaluates,
  and which argument is never read).
- [`wow_forecaster/taxonomy/archetype_taxonomy.py`](../../wow_forecaster/taxonomy/archetype_taxonomy.py)
  and [`wow_forecaster/models/archetype.py`](../../wow_forecaster/models/archetype.py)
  Why behavior rather than item identity, and why `mapping_rationale` is a
  required field.
- [`wow_forecaster/ml/feature_selector.py`](../../wow_forecaster/ml/feature_selector.py)
  The excluded-columns block. One line explains why identity is not a feature.

## What you should be able to do afterwards

- Write the blend formula and say what happens at conf = 0, conf = 1, and a
  missing source price.
- Explain both places `transfer_confidence` enters a forecast, in which
  direction, and why the two values are read from different queries.
- Compute a cold-start interval by hand and predict its `ci_quality` label.
- Say why `mapping_rationale` is required and where that guard stops.
- Explain what the `_cold` and `_transfer` slug suffixes record, and why a
  blended forecast gets the wrong one today.
- Name this design in standard terms (output-level domain adaptation, shrinkage
  toward an anchor) and say what a principled blend weight would be estimated
  from.

## A note on defects in this module

There are four, and they are worth separating by kind, because the fixes are
different sizes.

**Missing data, not code.** `archetype_mappings` has no writer. A seed file plus
a CLI command that loads it through `ArchetypeMapping` fixes it, and the
validator finally runs on a real path.

**Wiring.** The transfer feature columns join on `source_archetype_id` while the
blend dict is keyed by target. Even with mappings loaded, a Midnight forecast
would be blended at the right weight while being labelled `_cold` and widened by
the maximum penalty.

**A default that hides a missing value.** `is_cold_start` reads a count with
`.get(id, 0)`, so an archetype absent from the target-expansion counts is
indistinguishable from one with zero observations. Every TWW-only archetype is
flagged cold start, which makes the training feature nearly constant and triples
every published interval.

**A comment describing an impossible condition.** The re-clamp parenthetical
compares five percent of a number against ten times the same number. The real
triggers are elsewhere, and both of them collapse the interval to a single
point that `classify_ci_quality` then reports as "good".

None of these were found by a failing test. All four are visible by reading two
files side by side and asking which end of an edge each one indexes.

## Where this goes next

M10 turns forecasts into recommendations, and it inherits every one of these.
`ci_quality` gates the TSM export. CI width divided by predicted price is the
uncertainty penalty in the score. And `compute_score` multiplies that
penalty by another 1.5 when a row is cold start with a null or low transfer
confidence, which per this module is currently every row. A defect in the
transfer layer does not stay in the transfer layer.
