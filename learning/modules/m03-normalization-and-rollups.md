# M03. Normalization, rollups, and the date spine

Part I. The domain and the data. Prereq: M02.

## Why this module exists

Between a raw API row and a feature vector sit three transformations, and each one
makes a decision that the layers above it cannot see.

Normalization decides which observations count. An `is_outlier` flag does not
annotate a row, it removes it: both rollup tables and every daily aggregate carry
`is_outlier = 0`, so a flagged observation is gone from the series entirely. That
flag comes from a rolling z-score whose baseline is built from two observations
minimum, computed with the naive variance identity, over a window anchored on the
wall clock rather than on the row being scored.

The date spine decides what "a day with no data" means. It emits a row rather
than skipping one, which is the right call for positional lag features and the
reason a 90-day hole showed up as 149 explicit empty rows instead of a silently
shortened series. It also clamps to the newest observation, which is why 96 days
of dead ingestion produced a forecast CSV every morning and no error anywhere.

The rollup decides what is durable. Raw and normalized observations are deleted at
30 days under the API terms of service. The daily rollups are not, so they are the
only history this project will ever have past a month.

## The idea to hold onto

Every layer here turns an absence into a value, and the choice of value is the
design:

```
missing price      -> price_gold = 0.0        (NOT NULL column, sentinel)
day with no data   -> spine row, obs_count 0  (explicit, not skipped)
no prior history   -> z_score = None          (not 0.0)
requested range    -> clamped to data extent  (not padded)
```

Three of those four are defensible. The fourth is root cause four in the
postmortem. Being able to say which is which, and why, is the point of the module.

## Read this first

The repo is the textbook. Read these before drilling:

- [`wow_forecaster/pipeline/normalize.py`](../../wow_forecaster/pipeline/normalize.py)
  The whole module docstring, then `_fetch_rolling_stats` and `_normalize_batch`.
  Watch three things: what the baseline is computed over, what happens when there
  is no history, and what the window is anchored to.
- [`wow_forecaster/features/daily_agg.py`](../../wow_forecaster/features/daily_agg.py)
  The four numbered design choices at the top, then `fetch_daily_agg` step by
  step. Compare the docstring's account of the date range against what steps 1
  and 5 actually do.
- [`wow_forecaster/db/rollup.py`](../../wow_forecaster/db/rollup.py)
  The module docstring is short and every sentence in it is load-bearing. Note
  which columns the tables store instead of a mean, and why.
- [`wow_forecaster/pipeline/orchestrator.py`](../../wow_forecaster/pipeline/orchestrator.py)
  Step 3.5 only. Read the comment above the two-date loop, not just the loop.
- [`docs/postmortem-2026-04-lock-outage.md`](../../docs/postmortem-2026-04-lock-outage.md)
  Root cause four and lesson four. Then the day-one verification paragraph, which
  is the only place the spine's behavior over a real 90-day gap was measured.
- [`config/default.toml`](../../config/default.toml)
  The `[pipeline]` block. Four values, and three of them set the behavior this
  module is about.

## What you should be able to do afterwards

- Say what the rolling z-score is computed against, at what grain, and what
  happens to the very first run against an empty table.
- Explain why `is_outlier` is a deletion rather than an annotation, and trace what
  a genuine level shift does to the baseline and how long it takes to recover.
- Explain why `DATE(observed_at) = ?` cannot use an index, what replaces it, and
  why the row set is unchanged.
- Describe the UTC anchor defect and argue for the previous-date upsert against
  someone who wants to remove it.
- Given a requested date range and a data extent, predict the exact shape of what
  `fetch_daily_agg` returns.

## A note on what is still open

Two of the questions here point at filed, unfixed work. Issue #71 is the one to
carry forward: spine rows with every feature null and a real forward target reach
the training matrix, because the fit-time filter only checks the target. Roughly
3 percent of usable rows per horizon in the measured case, and the condition
recurs in miniature every time a sparse archetype misses a day.

It is the mirror image of the leakage check that did get run. The gap verification
confirmed no row carried a target pointing into the gap. Nobody checked rows
inside the gap carrying targets pointing out of it. Notice the shape: an asymmetric
check is how a whole class of defects survives a passing audit.

M04 picks up directly from here, at the boundary between what the feature layer is
allowed to see and when.
