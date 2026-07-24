# M13. The git history as a design record

Part III. Failure, history, operations. No lab: the reading is the work.

## Why this module exists

Seventy releases sit in `CHANGELOG.md`, and about forty of them were written
under a set of working rules that no longer apply. Direct commits to main. A
version bump on every commit. One long-lived branch carrying the whole portfolio
push. All three were retired in a single week in July, and the reasons are
recorded nowhere except the changelog entries and two sections of `CLAUDE.md`.

This module treats that history as evidence rather than trivia. A version number,
a branch name, and a merge policy are design decisions with tradeoffs, and this
repo changed all three under pressure, which means the before state, the trigger,
and the after state are all visible in one file.

There is a second reason. The changelog has a three-month hole in it, and the
hole is the outage. That is the most useful thing in the file, because it is the
one fact a changelog can never tell you on its own: a quiet stretch and a dead
system look identical from the outside.

## The idea to hold onto

Process rules are downstream of each other.

```
merges go through PRs        ->  the PR is the unit of change
the PR is the unit of change ->  one version per PR, not one per commit
one version per PR           ->  work commits log under [Unreleased]
```

The branching change and the versioning change shipped in the same release
(v2.4.4) because the second follows from the first. Reading them as two separate
improvements loses the causality, which is the part that transfers to the next
project.

## Read this first

The repo is the textbook. Read these before drilling:

- [`CHANGELOG.md`](../../CHANGELOG.md)
  Read it end to end, oldest first. Watch three things: where the version-header
  separator switches from an em dash to a hyphen, where the dates jump from April
  to July, and where the entries stop being one line and start being paragraphs
  with measurements in them.
- [`CLAUDE.md`](../../CLAUDE.md)
  The Branch Workflow and Versioning sections. Four rules each, and every one of
  them is a response to something in the changelog above.
- [`docs/ROADMAP.md`](../../docs/ROADMAP.md)
  The Work order section. It states that issue order beats issue number, and the
  milestones were renumbered twice to match it.
- [`PLAN.md`](../../PLAN.md)
  The closing section on the relationship to the roadmap. One-directional
  references, on purpose.
- [`LESSONS.md`](../../LESSONS.md)
  Both entries are the same failure class as this module's: a stated standard and
  an enforced standard drifting apart with nothing to catch it.

## What you should be able to do afterwards

- Sketch the ten eras and say where the two discontinuities are.
- Explain why per-commit version bumps were retired and what one stamp commit
  does that a bump per commit could not.
- Say what `feature/portfolio-showcase` was for, what it spanned, and what
  replaced it.
- State what a branch protection ruleset with no bypass actors buys that a
  written convention does not.
- Read an entry and predict its bump class from the rule rather than from the
  size of the change.
- Say what a changelog cannot tell you, using the April to July gap as the case.

## A note on the honest part

Every process rule in this repo arrived after the failure it prevents. The lock
takeover exists because a lock leaked. The freshness gate exists because ninety
days of forecasts ran on frozen features. The ruleset exists because a solo
maintainer will always be the one in a hurry.

That is not a criticism, and it is not a story to dress up in an interview. It is
the normal shape of operational learning, and the only thing that separates a
project that learned from one that did not is whether the reasoning got written
down while it was still fresh. The entries after the gap are long for that
reason. A short entry would have kept the change and thrown away the argument.
