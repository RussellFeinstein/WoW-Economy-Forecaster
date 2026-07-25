# M18. Causal questions

Part IV. Proving it and shipping it. Prereq: M16.

## Why this module exists

"Will consumables be more expensive in 28 days?" and "did the season-start patch
make consumables more expensive?" sound like the same question about the same
series. They are not. The first is a forecast: it predicts the future that
actually happens. The second is causal: it compares what happened against a world
where the patch never shipped, a world you never observe.

The forecaster in this repo answers the first question and cannot answer the
second, no matter how low its MAE gets. M5 is the milestone that answers the
second, and it does so with interrupted time series and difference-in-differences.
This module is the statistics you need before that milestone is worth starting.

There is a second reason. The event-feature layer already in the code looks
data-driven and is not. Every impact magnitude the model consumes was typed into
`config/events/` by hand. The system is currently telling the forecaster the
answer to the causal question instead of measuring it. Knowing that is the point.

## The idea to hold onto

A causal effect is a difference against a counterfactual:

```
effect = observed_outcome  -  outcome_that_would_have_happened_without_the_event
                              ^ never observed; you have to estimate it
```

Everything in this module is a method for estimating that second term honestly.
ITS estimates it by extrapolating the pre-event trend. DiD estimates it by
borrowing a control series that felt the same background shocks. A placebo check
tries to prove your estimate is an artifact. And the two-column event schema
(`announced_at` datetime vs `start_date` date) is what lets you place the
intervention in time without folding anticipation into your baseline.

## Read this first

The repo is the textbook. Read these before drilling:

- [`wow_forecaster/models/event.py`](../../wow_forecaster/models/event.py)
  The two dates. `announced_at` (when the information went public) and
  `start_date` (when the effect mechanism begins) are separate on purpose. Read
  `is_known_at()` and note that it is a forecasting guard, not a causal tool.
- [`wow_forecaster/features/event_features.py`](../../wow_forecaster/features/event_features.py)
  The three leakage layers, the end-of-day `as_of` boundary, and the
  "Assumptions & simplifications" block. The `scope = GLOBAL` note is what forces
  the control group to be a category rather than a realm.
- [`docs/events.md`](../../docs/events.md)
  How events and impacts are seeded. Read the impact table: `impact_direction`,
  `typical_magnitude`, and the negative `lag_days` that encode pre-event run-up.
  These are the hand-authored values M5 is meant to replace.
- [`docs/ROADMAP.md`](../../docs/ROADMAP.md)
  The M5 description: ITS and DiD against look-ahead bias, with placebo checks.
- [`PLAN.md`](../../PLAN.md)
  The ceiling item on event effects: measured causal effects fed back as priors,
  replacing the hand-authored impact records.

## What you should be able to do afterwards

- Say what makes "did the patch move prices" causal, and why forecast accuracy
  cannot answer it.
- Set up an interrupted time series design for one patch event, and name the one
  threat it cannot rule out alone.
- State the parallel trends assumption for a category-vs-category DiD here, and
  give two ways to probe it.
- Explain what a failed placebo check falsifies.
- Explain why `announced_at` and `start_date` must be separate columns, and why
  dating an ITS at the effective date attenuates the estimate when markets
  anticipate.

## A note on what is and is not built

None of this is implemented yet. M5 is the designated filler milestone: it needs
only the rollups and the event calendar, so it gets pulled forward whenever other
work is blocked on wall clock. The event features exist and are leakage-safe, but
the magnitudes in them are researcher guesses. Treat `event_impact_magnitude` as a
labeled hypothesis until M5 has measured the real thing.
