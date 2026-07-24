# M10. Forecasts to decisions

Part II. Features, models, statistics. Prereq: M09.

## Why this module exists

Everything before this module produces a number: a predicted price and an
interval around it. Nobody trades a predicted price. What ships to the user is a
ranked list of archetypes with a BUY, a risk label and a sentence of reasoning,
and the distance between those two things is a weighted sum with about
twenty-five constants in it.

Every one of those constants was chosen by hand. Not tuned, not fitted, not
compared against an alternative. Written down once and never revisited, because
nothing in the repo has ever scored a recommendation against what the price
actually did.

That is not a scandal. It is the normal state of a decision layer built before
the evaluation layer, and the modeling structure underneath it is better than
most naive versions: direction and confidence are kept on separate axes, the
volatility proxy is scale-free, and thin markets are excluded rather than
discounted. The point of this module is to know exactly where the arbitrary
numbers are, so that when M2 turns them into a named policy you know what you
are testing.

## The idea to hold onto

A forecast is an estimate. A decision is a choice made under a loss function.
The score formula is the loss function, written implicitly:

```
total = 0.35*opportunity + 0.20*liquidity - 0.20*volatility
      + 0.15*event_boost - 0.10*uncertainty
```

Read those weights as prices. A point of expected ROI costs seven times what a
point of interval width costs. An event nobody has written an impact record for
is worth a free bonus. Depth above 1,000 units is worth nothing. Those are all
claims about what matters, and none of them was measured.

Second thing to hold onto: this pipeline takes a maximum. Best horizon per
archetype, then best archetypes per category. An argmax over noisy estimates
selects for upward error as well as for quality, so the selected set realizes
worse than it scored. Nothing currently measures that gap.

## Read this first

The repo is the textbook. Read these before drilling:

- [`wow_forecaster/recommendations/scorer.py`](../../wow_forecaster/recommendations/scorer.py)
  The whole module docstring, then the whole file. It is short. Count the
  literals as you go, and watch where `uncertainty_penalty` diverges from
  `uncertainty_pct`.
- [`wow_forecaster/recommendations/ranker.py`](../../wow_forecaster/recommendations/ranker.py)
  `top_n_per_category` for the de-duplication and tie-break rules, and
  `build_recommendation_outputs` for what actually reaches the database. The
  planned-improvements block at the top is a to-do list the author wrote for
  himself and is the shortest statement of what this layer is missing.
- [`wow_forecaster/recommendations/crafting_advisor.py`](../../wow_forecaster/recommendations/crafting_advisor.py)
  A second, independent decision layer with different conventions: a hard volume
  gate, a saturating volume multiplier, six timing windows, and a four-level
  price ladder. Compare its choices against the scorer's.
- [`config/default.toml`](../../config/default.toml)
  The `[model]` and `[crafting]` blocks. Note which of the two decision layers
  exposes its constants and which does not.
- [`docs/ROADMAP.md`](../../docs/ROADMAP.md)
  M1 and M2. #13, #29, #31, #32 and #33 are, in order, everything that would
  turn this module's content from a hypothesis into a result.

## What you should be able to do afterwards

- Write the five-component formula from memory and say what each component is a
  proxy for and what raw column it comes from.
- Compute a score, an action and a risk level by hand from a row of inputs.
- Explain why risk is a separate axis from the action, and name the one place
  the two deliberately merge.
- Say what the volume gate excludes, and show with arithmetic why a
  multiplicative penalty could not replace it.
- List the hand-chosen constants and say how many are reachable from config.
- Name the winner's curse in this pipeline and point at where it enters.

## A note on what would settle any of it

Four steps, in dependency order: the `forecast_realizations` ledger (M1 #13),
a paper-trading simulator with auction-house fees (M2 #29), random-portfolio and
buy-everything baselines to compare it against (M2 #31), then a named
`ScoringPolicy` (#32) and a time-sliced A/B between two of them (#33).

Until that runs, the correct sentence about this layer is the one to practice
saying out loud: the structure is defensible, the constants are guesses, and no
part of it has been shown to beat picking archetypes at random.
