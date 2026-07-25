# M17. Does it make gold?

Part IV. Proving it and shipping it. Prereqs: M10, M16. Lab: none yet (M2, issues #29-#33).

## Why this module exists

The system has issued more than three hundred thousand forecasts and produced
daily buy and sell recommendations for months. It has never once been checked for
whether acting on them makes gold.

That is not an oversight to be embarrassed about. Issuing a recommendation and
scoring it for profit are two separate pieces of work, and only the first was
built. This module is about the second: what it takes to answer "does it make
gold?" honestly, and why the answer is not the same as "is the forecast accurate?"

## The idea to hold onto

Accuracy and profit are different questions over different populations.

```
MAE      averages error over every archetype, every day, equally weighted
profit   counts only the rows the policy traded, weighted by position size,
         net of the auction-house cut
```

A model can win on the first and lose on the second. It can be smoother on average
yet wrong on the handful of high-conviction rows that became trades, and those are
the only rows that touch the P&L. Every question in this module is a consequence of
that gap.

## Read this first

The repo is the textbook. Read these before drilling:

- [`wow_forecaster/recommendations/scorer.py`](../../wow_forecaster/recommendations/scorer.py)
  The whole module docstring, then `ScoreComponents.total`, `compute_score`, and
  `determine_action`. This is the artifact a paper-trading loop would execute. Note
  two things: the five weights in the total are hand-set literals, and the
  opportunity component is buy-only, with sell logic pushed into the action rather
  than the score.
- [`docs/ROADMAP.md`](../../docs/ROADMAP.md)
  The `M2: Paper trading P&L and ranking A/B` section, the second driving finding
  in `Why this roadmap` ("Recommendations have never been scored for profit"), and
  the M2 line in the `Work order`. This is where the simulator, the two
  comparators, the equity curve, and the live-versus-replay A/B are specified.

## What you should be able to do afterwards

- Give three distinct reasons a more accurate model can be less profitable.
- Say what random-portfolio and buy-everything each control for, and why both are
  needed rather than either alone.
- Explain what an equity curve shows that a total return hides: drawdown,
  path-dependence, whether the return was one trade or many.
- Explain why a live A/B needs wall clock, and state one thing offline replay can
  claim and one thing it cannot.
- Locate the five hand-chosen scoring weights and say what turning them into a
  named `ScoringPolicy` (issue #32) makes possible.

## A note on what exists and what does not

The scorer is real and runs daily. The paper-trading simulator, the comparators,
the equity curve, and the A/B are all M2 work that has not landed. So this module
teaches a design you can read but not yet execute. That is deliberate: M16 gave you
the significance machinery, M10 gave you the scorer, and this module is where the
two meet the question a WoW gold-maker actually cares about. The lab lands with
M2.

Hold onto the honest version of the answer. Right now the correct response to "does
it make gold?" is "we have not measured it", and the value is in knowing exactly
what you would build to change that, and what would count as a real yes: beating
both comparators, confirmed forward on wall clock, not just in a replay of one past
window.
