# M01. The market and the problem

Part I. The domain and the data. No prereq. No lab.

## Why this module exists

The README opens with what the system does. It never says why the problem is
hard, and the hard parts are not modeling choices. They are structural facts
about the market and about the API, and every design decision downstream is
downstream of them.

Four of those facts, in the order they bite:

1. **Cold start across expansions.** The expansion you care about has no price
   history, and its item IDs did not exist when the transfer map was written.
2. **A 30-day retention wall.** The API terms require raw data to be deleted
   within 30 days, and the commodities endpoint serves only the current
   snapshot. There is no historical endpoint to ask.
3. **Thin per-archetype series.** Archetypes do not trade every day. The date
   spine is mostly fill, and a 28-day label is missing whenever that future date
   has no observation.
4. **Event shocks.** Prices step on scheduled dates (launch, season start, race
   to world first, patch day) rather than drifting.

Add the one PLAN.md folds into its framing rather than listing: the commodity
auction house is region-wide, so each archetype has exactly one cross-sectional
unit and there is no untreated series to compare against.

## The idea to hold onto

The unit of prediction is not an item.

```
grain   = (archetype_id, realm_slug, obs_date)
feature = price_mean          <- mean non-outlier min-buyout price, in gold
label   = price_mean(d + h)   <- target_price_1d / 7d / 28d
```

An archetype is an economic behavior group, not a bag of items that happen to
look alike. That choice is what makes transfer across an expansion boundary
expressible at all, and it is also what makes every item-level number in the
system a rescale of an archetype-level forecast rather than a model of its own.

## Read this first

The repo is the textbook. Read these before drilling:

- [`wow_forecaster/taxonomy/archetype_taxonomy.py`](../../wow_forecaster/taxonomy/archetype_taxonomy.py)
  The whole module docstring, then the enum members with their economic
  annotations. Note the three rules `CATEGORY_TAG_MAP` enforces, and count the
  tags yourself.
- [`wow_forecaster/models/archetype.py`](../../wow_forecaster/models/archetype.py)
  Why item-to-item mapping is impossible rather than merely inconvenient, and
  what `transfer_confidence` and the required `mapping_rationale` are for.
- [`wow_forecaster/taxonomy/event_taxonomy.py`](../../wow_forecaster/taxonomy/event_taxonomy.py)
  Three orthogonal dimensions plus `ImpactDirection`. Read the severity bands as
  what they are: hand-authored priors with percentages attached.
- [`wow_forecaster/governance/pruner.py`](../../wow_forecaster/governance/pruner.py)
  The docstring is the clearest statement in the repo of what the retention wall
  deletes and what survives it.
- [`config/default.toml`](../../config/default.toml)
  The `[realms]`, `[expansions]` and `[retention]` blocks. Three short sections
  that encode three of the constraints.
- [`PLAN.md`](../../PLAN.md)
  The "Data science legibility" table. It names all four constraints in one line
  and says plainly that none of them reaches the README.
- [`README.md`](../../README.md)
  The Transfer Learning Architecture section, and the counts. One of them is
  wrong, which is the point of one of the find questions.

## What you should be able to do afterwards

- State the forecast target precisely: grain, column, and how the value at
  horizon h is produced.
- Give the two separate reasons item-to-item transfer cannot work, one before
  the target expansion ships and one after.
- Name the four structural constraints and a concrete consequence of each.
- Say what "region-wide commodity AH" implies for `realm_slug`, `faction`, and
  what can be identified from the data.
- Answer "it is a price forecaster, what is hard about that?" in ninety seconds
  without reaching for generic time-series talk.

## A note on what this module does not settle

Whether the archetype grouping is economically correct is an empirical claim,
and nothing in this repo measures it. The taxonomy asserts that items inside one
archetype move together and items across archetypes do not. There is no test of
within-archetype versus between-archetype price correlation anywhere, and adding
one would be a genuinely good piece of work.

Hold that thought through Part II. Most of what looks like a modeling question
later turns out to be a question about whether this grain was the right one.
