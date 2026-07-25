# M19. The lifecycle gap

Part IV. Proving it and shipping it. Prereq: M16.

## Why this module exists

The forecasting, backtesting, drift, and feature work in this repo is real. What
is thinner is the operational shell around it, and reading the repo as a
production system means telling the two apart honestly: what is built and works,
what is built and unused, what is built and broken, and what is simply absent.

The sharpest single example is model selection. Three separate pieces of code
answer the question "which model do I use," and they do not share a source of
truth:

- The serving path globs the artifact directory and sorts by filesystem
  modification time.
- `report-feature-importance` sorts the same glob and takes the last filename,
  trusting the ISO date suffix to make lexicographic order chronological.
- `model_metadata.is_active` is a flag the training step sets on purpose. It is
  the authoritative answer, and it is the one nothing reads.

That is the same class of finding as M06's two disagreeing splits: a load-bearing
concept implemented several incompatible ways, visible in the source without
running anything, with the correct version sitting unused.

## The idea to hold onto

"Present" is not one bucket. A system can be present-and-strong (pipeline health
monitoring here), present-and-unused (the model registry), or present-and-broken
(prediction monitoring's pooled drift baseline). A portfolio audit that collapses
all three into "we have monitoring" misses exactly the things a reviewer catches.
The same discipline separates the one coupling that blocks the roadmap (Windows
Task Scheduler is the orchestration layer) from the four that only annoy.

## Read this first

The repo is the textbook. Read these before drilling:

- [`PLAN.md`](../../PLAN.md)
  The whole document, but especially OPEN DECISIONS (the four tool choices), the
  MLOps story table, and the Coupling section. This is the audit, and it names
  the recommendation and the argument for each open decision from this codebase
  specifically, not in general.
- [`wow_forecaster/pipeline/forecast.py`](../../wow_forecaster/pipeline/forecast.py)
  The serving path. See how it loads a model: `find_latest_model_artifact`, an
  mtime glob, with no reference to `is_active`.
- [`wow_forecaster/ml/trainer.py`](../../wow_forecaster/ml/trainer.py)
  `_register_model` writes `is_active` (and flips the prior row to 0), and
  `find_latest_model_artifact` sorts the `.pkl` glob by `st_mtime`. The writer and
  one of the readers, side by side.
- [`wow_forecaster/models/meta.py`](../../wow_forecaster/models/meta.py)
  The `ModelMetadata` docstring states the intent of `is_active` out loud:
  "the currently active production model." Compare that promise against what
  serving does.
- [`wow_forecaster/config.py`](../../wow_forecaster/config.py)
  `auto_retrain_on_critical` is defined here and set in `config/default.toml`.
  Grep for where it is read. It is not.

## What you should be able to do afterwards

- Name the three model-selection strategies, say where each lives, and say which
  one should win and why it does not.
- Sort the lifecycle concerns into present-strong, present-unused, present-broken,
  and absent, with the reason PLAN.md gives for each.
- State each of the four open tool decisions with its recommendation and the
  codebase-specific argument, including the "what would change my mind" for at
  least two of them.
- Explain why Windows Task Scheduler is the one blocking coupling, and why the
  4,508-line `cli.py` is not.

## A note on what this module is and is not

This is a legibility module, not a fix. Nothing here asks you to wire `is_active`
into serving or stand up Dagster. It asks you to read the repo the way a hiring
manager would and to say, precisely, what is missing and what is merely
disconnected. That precision is the deliverable: "the registry is unused" is a
finding; "we should add MLOps" is not. The fixes are the later phases of PLAN.md,
and this module is the map of which ones matter and in what order.
