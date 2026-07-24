# Lab 01. Purge and embargo the training split

Module: M06. Real work on a real branch, shipped through a PR like anything else.

## What you are fixing

`train_models()` in `wow_forecaster/ml/trainer.py` splits the training Parquet by
`obs_date` and holds out the last `validation_split_days` calendar days. A
training row at date T carries the label `price(T + h)`, so training labels reach
`split_date + h`. With `validation_split_days = 14` and horizons `[1, 7, 28]`:

| horizon | training labels reach | validation days contaminated |
|---|---|---|
| 1d  | split_date + 1  | 1 of 14 |
| 7d  | split_date + 7  | 7 of 14 |
| 28d | split_date + 28 | 14 of 14, plus 14 days past the window |

The same validation set drives LightGBM early stopping, so the stopping round is
also chosen against the contaminated signal. The model is mis-sized, not just
mis-scored.

## Before you write any code

**File the issue.** There is no issue for this yet, and filing it is part of the
lab. It belongs in milestone M1 (model validation and monitoring), next to #16.
The body should state the defect, the per-horizon contamination table above, the
second-order early-stopping effect, and the fix. Link `LESSONS.md` and the
`PLAN.md` DS-1 finding rather than restating them.

Then cut the branch from the latest main:

```
git checkout main && git pull --ff-only
git checkout -b fix/<issue>-purge-embargo-training-split
```

## Write the failing test first

This matters more here than usual, because the obvious assertion passes today.

Put it in `tests/test_ml/test_trainer.py`. Parameterize over horizons: an
assertion that only covers 1d will look almost fine and tell you nothing.

The assertion that catches the bug:

> For each horizon `h`, no training row's target date (`obs_date + h`) falls at
> or after the validation start.

The assertion that does **not** catch it, and that you may be tempted to write:

> `max(obs_date)` over training rows `< min(obs_date)` over validation rows.

That is the feature-date split, which is already correct. It passes now and would
greenlight the exact defect.

Confirm the new test fails against unmodified `trainer.py` before you touch the
implementation. If it passes, the assertion is wrong, not the code.

Also expect to **update** an existing invariant: any test asserting that training
and validation row counts sum to the input row count must stop passing, because
purging deliberately drops rows. Change it rather than deleting it.

## The fix

In the date-split branch of `train_models()`:

1. Keep the existing `val_split_date` computation as the validation boundary.
2. Per horizon, drop training rows whose target date falls at or after the
   validation start. The split is now per horizon, because the purge boundary
   depends on `h`. This is the structural change: today one split feeds every
   horizon's model, and it cannot stay that way.
3. Apply an embargo of `h` days before the purge boundary, so rolling and lag
   features that span the cut do not carry validation-period information across.
4. Log the retained row count per horizon. At `h = 28` on a short history the
   purge is severe, and a silent 90-percent reduction is exactly the kind of thing
   this repo has been bitten by before. Make it visible.

Leave the 80/20 fallback branch alone for now, but log a warning that names the
limitation. It is the cold-start path, it inherits the leak, and it additionally
has no temporal guarantee at all. Fixing it properly is a separate concern and
deserves its own issue rather than being smuggled into this PR.

## Do not

- Do not shrink `features.target_horizons_days` to `[1, 3]` to match the backtest
  config. That deletes the product's most useful outputs to make a measurement
  problem disappear. The horizon lists do need to agree, but by raising the
  backtest side, which is Lab 02.
- Do not touch `backtest/splits.py`. It is already correct.
- Do not retune hyperparameters in this PR. One concern per PR, and the honest
  post-fix number is the deliverable.

## Finish line

- The new per-horizon test fails against unmodified `trainer.py` and passes after
  the fix.
- The full suite is green: `pytest -q`.
- `ruff check wow_forecaster/ tests/` is clean.
- `CHANGELOG.md` has an entry under `[Unreleased]` describing the defect in user
  terms, not implementation terms.
- Documentation sync: `LESSONS.md` gets a line noting the fix landed. The README's
  Known Limitations entry for this defect, if it exists by then, is updated rather
  than removed, because the accuracy numbers still have not been validated against
  a baseline.
- Version: patch. This corrects wrong behavior and adds no new surface.

## What to expect from the number

`validation_mae` for the 28d model should get **worse**. That is the fix working.

If it barely changes, do not read that as reassurance. `price_mean` is both a
feature and the basis of the target, so a model close to the `last_value` baseline
scores similarly whether or not it saw the labels. A flat result is evidence worth
chasing, and chasing it is Lab 02.

## Reflection, before you close the issue

Add a `LESSONS.md`-style note answering: what would have caught this at the time?
The specific answer here is more useful than the general one. Two components in
the same repo implemented the same concept and disagreed, and the disagreement was
readable in the source. What cheap check surfaces that class of thing?
