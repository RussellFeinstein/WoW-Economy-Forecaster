# M20. The path forward

Part IV. Proving it and shipping it. Prereq: the rest of Part IV. No lab.

## Why this module exists

The repo has two planning documents, and they do not say the same thing on
purpose.

`docs/ROADMAP.md` owns the research arc: milestones M0 through M6, every GitHub
milestone, every issue number. `PLAN.md` owns the lifecycle and legibility arc:
orchestration, serving, packaging, infrastructure as code, and the three
methodology defects the 2026-07-24 audit found. They are parallel tracks over the
same project, and the interlock between them runs one way only.

A roadmap is easy to read and easy to nod along to. It is much harder to say why
each milestone sits where it does, and the interesting calls are the ones that
break the obvious order. This module is about those calls. If you can reconstruct
the dependency graph and argue three non-obvious placements, you understand the
plan rather than just remembering its headings.

## The idea to hold onto

Sequencing is an argument, not a list. Every placement here answers a "why not
earlier" or "why not later" question:

- CI/CD comes late because it is already built. An early phase would buy a badge
  while real defects stayed live.
- M2 (paper trading) precedes M3 (the warehouse) so realizations and trade facts
  land in the marts in one pass instead of being retrofitted.
- M5 (the event study) is the designated filler, because it depends on nothing in
  the main chain and can absorb weeks blocked on wall clock.
- The ceiling names what it excludes (Kubernetes, multi-node training, a
  distributed scheduler) as loudly as what it includes, because at 100 MB of state
  the restraint is the credential.

Read the plan asking "why is this here and not three phases earlier," and the
whole document turns from a list into a set of decisions.

## Read this first

The repo is the textbook. Read these before drilling:

- [`docs/ROADMAP.md`](../../docs/ROADMAP.md)
  The full milestone descriptions, the Work order section (issue-level sequence,
  most urgent first), and the Dependency graph block. Note the standing rule at
  the end of the work order about advancing when the lowest milestone is blocked
  on wall clock.
- [`PLAN.md`](../../PLAN.md)
  Read the Phases section, then the Sequencing rationale, then the CEILING. The
  sequencing rationale is where the two non-obvious calls are argued out loud. The
  CEILING is where each ambitious item names the weaker thing it would replace,
  and where the exclusion paragraph says what to leave out and why.
- [`PLAN.md` OPEN DECISIONS](../../PLAN.md)
  Skim OD-1 through OD-4. OD-2 (MLflow) is worth reading closely: its case rests on
  a live bug, the write-only `model_metadata` registry that the serving path
  ignores in favor of an mtime glob.

## What you should be able to do afterwards

- Reconstruct the milestone dependency graph from memory: the gate, the branches,
  the M1-M4 chain, where M5 sits, what M6 waits on.
- Explain why CI/CD is a slice of a late phase rather than an early one, and name
  the sibling concern (monitoring) that gets the same treatment.
- Explain why paper trading precedes the warehouse, in the data-modeling terms the
  roadmap uses.
- Name three ceiling items and say what current mechanism each would replace.
- Say what the ceiling deliberately excludes and give the scale argument for it.

## A closing note

This is the last module in the track, and it is the one that ties the failures of
Part III to the future of Part IV. The 105-day outage is why M0 gates everything.
The invalid split and the never-run baseline comparison are why Phase 1 leads and
why M1 is the keystone. The plan is not a fresh start; it is the audit findings
turned into an ordered set of moves. Reading it that way is the point of finishing
here.
