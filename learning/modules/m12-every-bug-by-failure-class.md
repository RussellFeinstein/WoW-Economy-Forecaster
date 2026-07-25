# M12. Every bug, by failure class

Part III. Failure, history, operations. Prereq: M11. No lab.

## Why this module exists

Twenty-two fix entries in `CHANGELOG.md`, sorted by date, read as twenty-two
unrelated problems. Sorted by what went wrong, they are five problems that kept
happening.

Chronology teaches nothing here. The dates say when someone noticed. The classes
say where to look next time, which is the only part that transfers to a codebase
you have never seen.

## The five classes

| class | the shape | an instance |
|---|---|---|
| Silent substitution | a default, a zero, or a skip stands in for missing data | event features all zero on an empty `wow_events` table |
| Config-to-code drift | a config value stops matching what reads it | `backtest.horizons_days = [1, 3]` against `target_horizons_days = [1, 7, 28]` |
| Documentation drift | a stated count stops matching the thing counted | the README's 37 model features, against an actual 40 |
| Contention under load | correct at fixture scale, wrong at production volume | a 4.2 GB write-ahead log nothing ever checkpointed |
| Fixture and clock drift | the test's world diverges from production | a hand-rolled `ingestion_snapshots` fixture with invented columns |

## The idea to hold onto

Every instance in all five classes shares one property:

```
the failure produces a plausible output instead of an error
```

A zeroed feature column is a valid column. A leaked-lock skip is a successful
exit. A stale count is readable prose. A five-second timeout is a working default
until the table grows. A passing test is a passing test.

Nothing generates a signal at the moment it goes wrong. So none of these is found
by reading one thing carefully. They are found by comparing two things that
should agree: config against code, doc against code, fixture against schema,
`drift.py` against `health.py`, rows ingested against rollup rows written.

## Read this first

The repo is the textbook. Read these before drilling:

- [`CHANGELOG.md`](../../CHANGELOG.md)
  Read every `### Fixed` block, ignoring the dates. Try to sort them yourself
  before looking at the table above. The v2.3.3 and v2.3.4 entries are the whole
  contention class in one release pair.
- [`PLAN.md`](../../PLAN.md)
  The Audit findings section, especially DS-3. It is the cleanest live instance
  of silent substitution in the repo: `_classify_error_drift(None)` returns
  `NONE`, so "cannot compute" reports as "no drift".
- [`LESSONS.md`](../../LESSONS.md)
  Both entries. The second one names the general habit: when a docstring states
  an acceptance bar, wire something that fails when the bar goes unmeasured.
- [`docs/postmortem-2026-04-lock-outage.md`](../../docs/postmortem-2026-04-lock-outage.md)
  The Root cause chain and Lessons sections. Four failures had to line up and
  three of them are silent substitution wearing different clothes.
- [`wow_forecaster/features/dataset_builder.py`](../../wow_forecaster/features/dataset_builder.py)
  The preflight at the top of `build_feature_datasets()`. This is what the fix
  for silent substitution looks like: raise at the one place that knows why, and
  name the command that fixes it.
- [`wow_forecaster/reporting/health.py`](../../wow_forecaster/reporting/health.py)
  The comments around the coverage queries. Two of them explain why a predicate
  is written the way it is, which is the contention class documented at the point
  of the fix.

## What you should be able to do afterwards

- Name the five classes and give an instance of each from this repo.
- Say why substituting a default is worse than raising, in terms of where the
  failure surfaces and what it costs.
- Argue why a config-to-code mismatch is a correctness bug rather than untidiness,
  and name the three remedies for it in order of strength.
- Find a live documentation drift here without being told where, and verify it
  against code rather than against another document.
- Explain why `DATE(observed_at) >= ?` cannot use an index while
  `observed_at >= ?` can, and why both return the same rows.
- Run the one-hour sweep on an unfamiliar repo and say what each step turns up.

## A note on the taxonomy

Five classes, and they are not disjoint. The #61 rollup bug is a wrong calendar
substituted silently and a test fixture that only ever ran where local time and
UTC agreed. It sits in two classes and both remedies applied.

That is the taxonomy working, not failing. It is a search strategy, not a
partition. Given one instance, the useful move is to go looking for its relatives
in the same class, and an instance that belongs to two classes means two places
to search.

One last thing worth carrying out of here: every guard this repo has against
these classes was written after the corresponding failure, never before it. The
query-plan tests exist because of #59. The `apply_schema()` fixture rule exists
because of #12. The injectable clocks exist because eight tests broke when the
calendar moved. Reading the class list as a prospective checklist rather than a
retrospective one is the only way it saves you anything.
