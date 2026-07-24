# Plan: lifecycle and legibility

## Context

This repo is a portfolio artifact aimed at two audiences: a data science hiring
manager judging modeling and evaluation, and an MLOps hiring manager judging
whether it could run in production. The forecasting, backtesting, drift, and
feature work are real and stay central. What is missing is the operational shell
around them, and the evidence that the modeling works.

An audit on 2026-07-24 found three things that outrank every planned feature.
The model has never been compared to a baseline, and the comparison the code
appears to make is invalid. There is no visible result anywhere: no committed
chart, no notebook output, no number in the README, and a Streamlit demo that
shows "no data" on every tab. And nothing here runs outside one Windows box.

This document is a parallel track to [docs/ROADMAP.md](docs/ROADMAP.md), not a
replacement. ROADMAP milestones M0 through M6 own the research arc and the
GitHub issue numbering. This plan owns the lifecycle and legibility arc, and
names the exact points where the two interlock.

---

## OPEN DECISIONS

Four tool choices below. Each has a recommendation and the reasoning as it
applies to this codebase, not in general. None is implemented. Confirm or
override before the phase that needs it starts.

### OD-1: Orchestrator. Recommend Dagster.

The pipeline is already a well-formed DAG. `PipelineStage` in
[wow_forecaster/pipeline/base.py](wow_forecaster/pipeline/base.py) already does
run tracking, status, timing, and error capture, and `HourlyOrchestrator` already
declares stage order. So an orchestrator does not buy structure here. It buys
retries, backfill, a scheduler that is not Windows Task Scheduler, and a UI.

Dagster fits because its software-defined-asset model maps almost one to one
onto what this pipeline already produces: `daily_rollup_*` tables, the training
and inference Parquets with their JSON manifests, the model `.pkl`, then
`forecast_outputs` and `recommendation_outputs`. The repo already thinks in
artifacts with manifests and provenance hashes. The asset lineage view is also
the single artifact that reads well to both audiences at once: the DS reader
sees which dataset version produced which model, the ops reader sees lineage.

What would change my mind:

- Pick Prefect if the migration must take days rather than a week or two. Prefect
  decorates existing functions and leaves the stage classes intact.
- Pick neither if adding a long-running daemon to this box is unacceptable. The
  honest no-framework path is to containerize the existing `HourlyOrchestrator`
  and let cron inside the image drive it. Operationally sound, weaker as a
  portfolio signal, and it still delivers the portability that later phases need.

### OD-2: Experiment tracking and model registry. Recommend MLflow, local backend.

There is already a partial registry: the `model_metadata` table carries slug,
version, hyperparameters, training date range, validation MAE and RMSE,
`artifact_path`, and `is_active`. It is write-only. Nothing reads it back.
[wow_forecaster/pipeline/forecast.py:206](wow_forecaster/pipeline/forecast.py)
picks the model by filesystem mtime glob, and
[wow_forecaster/cli.py:3620](wow_forecaster/cli.py) picks it lexicographically.
Three selection strategies that can disagree, and the authoritative one is unused.

MLflow with a SQLite backend store and a local artifact root fits because: the
repo already runs SQLite, so tracking is a file and not a service;
`train_models()` already computes exactly what wants logging (params from
`ModelConfig`, metrics from `val_metrics`, artifact path, dataset version), so
instrumentation is roughly a dozen lines at one call site; and the registry's
stage transitions replace the `is_active` flag the serving path ignores, which
fixes a live bug rather than adding ceremony. ROADMAP M1 issue #18 adds an
Optuna study, which produces many runs that are painful to compare without a UI.

What would change my mind:

- Weights and Biases has a better UI but is a hosted dependency that moves data
  off-box, which cuts against the local-first design.
- If M1 #18 gets dropped, the case weakens enough that wiring `is_active`
  properly into the serving path is sufficient and costs nothing.

### OD-3: Serving and deployment target. Recommend FastAPI in the pipeline image.

Two endpoints with different semantics, and the plan should say so out loud. A
read path serving already-computed forecasts and recommendations out of
`forecast_outputs`, which is what anyone actually wants. And a `/predict` path
taking a 40-column feature vector directly against the pickled booster, which
demonstrates the model contract. Real-time prediction from raw market state is
not offered, because the feature vector comes from a batch-built Parquet and
there is no online feature path. Saying that plainly is better than faking it.

Deploy as the same container image as the pipeline, running on the existing box,
with the image also pushed to a registry so "does this run anywhere" has a
demonstrable answer.

What would change my mind: if you want a live clickable URL, this is cheaper
than it looks. `backup-durable-db` already produces a drop-in restorable SQLite
file of every durable table at roughly 31 MB gzipped, excluding the two large
observation tables. That artifact is exactly what a public read-only API or
dashboard needs. Fly.io or Cloud Run free tier over a restored durable backup is
a real option and would make the demo reachable from a resume link.

### OD-4: Infrastructure as code. Recommend Terraform, Cloudflare and GitHub providers.

Scope it to infrastructure that already exists and no further: two R2 buckets
(snapshots with the 30-day lifecycle rule, backups with no expiry), the
Cloudflare Worker in [cloud-trigger/](cloud-trigger/) with its cron trigger,
GitHub Actions repo secrets, and the branch protection ruleset. Roughly 100
lines. The useful detail is that the Blizzard API ToS section 2.r retention
requirement becomes a declared lifecycle rule rather than a promise in a doc.

What would change my mind: if the container work ends up being the whole
deployment story, a `docker-compose.yml` plus a documented bootstrap script may
carry more weight than Terraform describing three cloud resources. Use OpenTofu
instead if the Terraform license is a concern.

---

## Audit findings (2026-07-24)

### Inventory

| Concern | Where |
|---|---|
| Entry points | `wow-forecaster` / `wowfc` Typer CLI, 41 commands, [wow_forecaster/cli.py](wow_forecaster/cli.py) at 4,508 lines |
| Training | `TrainStage` to `train_models()` in [wow_forecaster/ml/trainer.py](wow_forecaster/ml/trainer.py). Runs unconditionally every day via `run-daily-forecast` |
| Inference | `ForecastStage` to `run_inference()` in [wow_forecaster/ml/predictor.py](wow_forecaster/ml/predictor.py), batch only, writes `forecast_outputs` |
| Model artifacts | joblib `.pkl` plus JSON sidecar in `data/outputs/model_artifacts/` (gitignored) |
| Config | TOML via `load_config()`, `config/default.toml` plus gitignored `config/local.toml` |
| Secrets | `.env` via python-dotenv, `.env.example` documents every key, none committed. Clean |
| Scheduling | Windows Task Scheduler, four tasks registered by `scripts/setup_tasks.bat`, driven by `.bat` wrappers |
| Tests | 71 files, ~19,000 lines, 1,481 passing. 34 Windows-only script tests skip on CI |
| Dashboard | `dashboard/` sits outside the installable package, reads output files and SQLite directly. No training or inference logic in it |
| CI | ruff plus pytest on 3.11 and 3.12, PR-gated, branch protection, Dependabot automerge |

### Data science story

Three methodology defects. None is fixed yet.

**DS-1. Train and validation label windows overlap by the forecast horizon.**
[wow_forecaster/ml/trainer.py:88-97](wow_forecaster/ml/trainer.py) splits rows by
`obs_date` alone. A training row at date T carries the label `price(T + h)`. With
`validation_split_days = 14` and horizons 1, 7, and 28, the training labels reach
`split_date + h`. For the 28-day model that covers the entire 14-day validation
window and 14 days past it, so every validation label is a price the model
already saw as a training label. The 7-day model leaks half the window. That same
leaked validation set drives early stopping in
[wow_forecaster/ml/lgbm_model.py:212-217](wow_forecaster/ml/lgbm_model.py), so the
stopping round is chosen against a contaminated signal, and the reported
`validation_mae` in `model_metadata` is optimistic. The fix is purging: drop
training rows whose target date falls at or after the validation start, plus an
embargo of `h` days. The walk-forward split generator in
[wow_forecaster/backtest/splits.py](wow_forecaster/backtest/splits.py) is correct
and does not have this problem. Only the LightGBM training split does.

**DS-2. LightGBM is never backtested.** `BacktestStage` calls
`all_baseline_models()` and nothing else
([wow_forecaster/pipeline/backtest.py:160](wow_forecaster/pipeline/backtest.py),
and the same at [wow_forecaster/cli.py:1033](wow_forecaster/cli.py)). A grep for
`LightGBMForecaster` across `wow_forecaster/backtest/` returns nothing. So the
walk-forward machinery evaluates four naive baselines against each other, and the
production model is judged only on the single leaked holdout from DS-1. The two
numbers are not comparable: different partitions, different aggregation,
different populations. `backtest/models.py` states the bar out loud, "If an ML
model cannot beat ALL of these baselines, it is not ready for use," and that
comparison is never run. Compounding it, `backtest.horizons_days = [1, 3]` while
production forecasts 1, 7, and 28, so even the horizons do not line up.

This matters more than it looks. `price_mean` is both a feature and the basis of
the target, so a model that learns "predict roughly today's price" will post a
respectable MAE while being exactly the `last_value` baseline. Without the
comparison there is no way to tell those apart.

**DS-3. The error-drift baseline does not track model degradation.** Both
[wow_forecaster/monitoring/drift.py:608-617](wow_forecaster/monitoring/drift.py)
and [wow_forecaster/monitoring/health.py:160-165](wow_forecaster/monitoring/health.py)
compute `baseline_mae` as `AVG(abs_error)` over `backtest_fold_results` with no
`model_name` filter, so the reference is a pooled blend of all four baselines
from whichever backtest ran last, with no staleness check. It is also an absolute
gold MAE pooled across archetypes whose price levels differ by orders of
magnitude, so the ratio moves when the archetype mix changes rather than when the
model degrades. And because backtest horizons are 1 and 3 while forecasts are 1,
7, and 28, the 7d and 28d ratios can never be computed at all.

The failure mode is worse in `drift.py` than in `health.py`.
`_classify_error_drift(None)` returns `DriftLevel.NONE`, so "cannot compute"
reports as "no drift." `health.py` returns `unknown` for the same condition,
which is right. This contradicts a principle the project already states
elsewhere: the cloud capture gap guard keeps its floor at 20 on purpose, because
a floor the failure mode can satisfy hides the failure. The fix is to make drift
match health, not to invent anything.

### Data science legibility

| Question | Answer |
|---|---|
| Is the problem framing stated, including why this is hard? | Partly. The README opens with what the system does, not why forecasting it is non-trivial. The genuinely interesting constraints (cold start across expansions, the 30-day ToS retention wall capping evaluation history, thin per-archetype series, event shocks) are scattered across module docstrings, which are excellent, and absent from the README |
| Are backtest results visible artifacts? | No. `docs/images/` is empty. `generate-charts` writes to `data/outputs/charts/`, which is gitignored, so charts can never land in the repo. The README contains no metric, no table, no plot |
| Is there a baseline comparison? | No, per DS-2 |
| Is feature engineering documented? | Yes, and well. `feature_selector.py` documents why each column is excluded, `lag_rolling.py` documents the leakage boundary, `lgbm_model.py` argues LightGBM against XGBoost, Prophet, and LSTM. This is the strongest prose in the repo and none of it surfaces in the README |
| Is there a model card? | No |
| Does the Streamlit app work as a demo? | No. `_DEFAULT_REALMS = ["area-52", "illidan", "stormrage", "tichondrius"]` at [dashboard/app.py:82](dashboard/app.py), but `config/default.toml` sets `realms.defaults = ["us"]` and the commodity AH has been region-wide since 9.2.7. None of the four selectable realms has data, so every tab shows "No data available." The docstring advertises a `--realm` override that is never parsed |

Also: all three notebooks have zero stored outputs across all 48 code cells, while
`.gitignore` states "Notebooks in notebooks/ are committed for portfolio display."
They currently display nothing.

Doc drift found along the way: README says 37 model features (actual 40), 23
tables in one place and 21 in another (actual 23), "1,200+ tests" in a badge and
"1,400+" in the body (actual 1,481), 39 CLI commands (actual 41).

### MLOps story

| Lifecycle concern | State |
|---|---|
| Orchestration and scheduling | **Partial.** Real stage abstraction with provenance, run metadata, and config snapshots. Driven by Windows `.bat` and Task Scheduler with genuinely thoughtful hardening (lock takeover, wake-to-run, silent execution, state preservation). Windows-only, so an MLOps reader cannot run any of it |
| Retraining trigger | **Absent as designed.** Retrains from scratch daily regardless of need, and `_register_model` promotes the new model unconditionally. No champion-challenger, no promotion gate. `auto_retrain_on_critical` exists and is off |
| Experiment tracking | **Absent.** No run comparison, no param sweep history |
| Model registry | **Partial and unused.** `model_metadata` has the right columns including `is_active`. Serving ignores it and globs by mtime |
| Packaging and reproducibility | **Weak.** No Dockerfile, no lockfile, all deps floating `>=`. `dashboard/requirements.txt` duplicates dependency info and can drift from `pyproject.toml`. Good partial credit: `features_hash` on every forecast, `config_snapshot` on every run, dataset manifests |
| Data versioning | **Partial.** Parquet datasets carry JSON manifests. No DVC or equivalent, and the 30-day ToS window means raw data is deliberately not retained |
| CI/CD | **Present and decent.** ruff plus pytest on 3.11 and 3.12, `fail-fast: false`, PR-gated with branch protection and no bypass actors, Dependabot automerge gated on CI. Missing: coverage reporting, image build, artifact publishing |
| Model serving | **Absent.** No API of any kind |
| Pipeline health monitoring | **Present and strong.** `check-data-health` with coverage, gaps, freshness, stale-lock and retention sentinels. Scheduled health check with a durable alert JSON and a once-per-24h alert window. Backup staleness check. Cloud capture gap guard doubling as a dead-man alarm |
| Prediction monitoring | **Present but broken.** See DS-3 |
| Infrastructure as code | **Absent.** Real infrastructure exists (two R2 buckets, a lifecycle rule, a Worker with a cron and a PAT, GitHub secrets, a branch protection ruleset) and none of it is declared |

Worth crediting explicitly, because it is unusual in a portfolio repo: the cloud
capture design in [docs/cloud-capture.md](docs/cloud-capture.md) diagnosed a
GitHub Actions cron delivery ceiling, disproved its own first fix, reverted it,
and moved to an external Cloudflare Worker trigger with the GitHub schedule
thinned to a single fallback that doubles as a dead-man alarm. Plus a postmortem
for a 105-day silent outage. That is the best ops material here and the README
does not mention it.

### Coupling

Nothing catastrophic. The dashboard does no training or inference, which is the
worst version of this problem and it is absent. Real items, in order:

1. **`.bat` and Task Scheduler are the orchestration layer.** This is the one
   blocking coupling. It gates containers, serving, IaC, and any claim that this
   runs in production.
2. **Dashboard hardcodes paths and realms instead of reading config.**
   [dashboard/app.py:80-87](dashboard/app.py) says the values "mirror
   config/default.toml." They no longer do, which is the demo bug above.
3. **Model selection by filesystem mtime.** Ties serving to a local filesystem
   layout and bypasses the registry.
4. **`cli.py` at 4,508 lines.** Not blocking: the stage classes are already
   importable without Typer, so serving can call them directly. Note it, do not
   schedule it.
5. **Non-injectable clocks in about 20 modules.** The project already knows the
   pattern and applies it where it matters (`run(now=...)` in the rollup step,
   `load_ingest_age_hours(now=...)`). Extend opportunistically, not as a sweep.

Where decoupling also helps the DS story: once training is invocable as a
parameterized job rather than a CLI subcommand wired to config defaults, rerunning
an experiment with a different split or feature set stops being a config edit and
becomes an argument. That is what makes DS-1 and DS-2 cheap to iterate on rather
than one-shot fixes.

---

## Proposed structure

Minimal. The README does the steering; the layout barely moves.

```
docs/
  model-card.md          NEW  assumptions, failure modes, data caveats
  results/               NEW  committed metrics tables, backtest summaries
  images/                     currently empty; charts land here
  cli.md                 NEW  the CLI reference moved out of README
serving/                 NEW  FastAPI app, sibling to dashboard/
deploy/                  NEW  Dockerfile, compose, terraform/
notebooks/                    same files, executed with outputs committed
wow_forecaster/               unchanged
```

`generate-charts` needs an output path that is not gitignored. Either add a
`--output-dir` defaulting to `docs/images/` or add an explicit publish step.

### README outline

Currently 851 lines, roughly 70 percent CLI reference, zero results, zero images.
Target is that both readers find their story above the fold.

```
# WoW Economy Forecaster
badges

One paragraph: what it forecasts, for whom, and the honest state of validation.

## The problem                        <- DS reader hooks here
   Why this is not trivial: cold start across expansions, the 30-day
   retention wall, thin per-archetype series, event shocks.

## Results                            <- entirely missing today
   Table: model vs 4 baselines, per horizon, MAE / MAPE / directional.
   Chart: walk-forward MAE across folds.
   Chart: forecast vs actual with intervals.
   A plain statement of what is and is not established yet.

## How it works
   6 to 8 lines plus the existing mermaid diagram.
   Links to the model card and the feature documentation.

## Running it                         <- MLOps reader hooks here
   docker compose up. The API. The dashboard. Three commands, not thirty.

## Operations
   Schedule, monitoring, the dead-man alarm, backup and restore,
   cloud capture redundancy. Link the postmortem.

## Known limitations                  <- the DS-1 and DS-2 disclosure
## Development / CLI reference        -> link out to docs/cli.md
```

---

## Phases

Sequence set 2026-07-24. DS legibility leads because it is days of work, it is
what applications going out now depend on, and two of its items are live defects
on a public repo. See "Sequencing rationale" for the argument against putting
CI/CD and monitoring earlier.

### Phase 1: DS legibility and the methodology fix (week 1). Story: DS, with ops spillover.

The only phase that changes what a reader sees this month.

- Purge and embargo the training split in `trainer.py`. Drop training rows whose
  target date falls at or after the validation start, plus an `h`-day embargo.
  Regression test asserting no training label date falls inside the validation
  window, per horizon.
- Add `LightGBMForecaster` to the backtest loop behind the existing
  `fit`/`predict` protocol in `backtest/models.py`. The protocol was designed for
  this ("This protocol is intentionally minimal so ML models can implement the
  same interface later"). Align `backtest.horizons_days` with
  `features.target_horizons_days` so the comparison is possible at all.
- Fix the Streamlit realm list to read `config.realms.defaults`, or at minimum
  include `us`. Parse the documented `--realm` override or delete the claim.
  Also `page_icon="data/raw/snapshots"` at app.py:60 is a stray path.
- Point `generate-charts` at `docs/images/`, commit the charts.
- Execute the three notebooks and commit outputs, or render them to `docs/` and
  link. Either way they must display something.
- Write `docs/model-card.md`: assumptions, known failure modes, data caveats
  (the 30-day wall, the 2026 April to July gap, cold-start blending, heuristic
  intervals), and the intended use boundary.
- Restructure the README per the outline above. Move the CLI reference to
  `docs/cli.md`. Fix the four drifted counts.
- Add Known Limitations naming DS-1 and DS-2 in plain terms, before the fix
  lands, and let the fix commit reference it.

Interlock: this overlaps ROADMAP M1 issues #14 through #17. The backtest work
here is the minimum honest version. M1's Diebold-Mariano and Wilcoxon tests are
the rigorous version and stay in M1.

### Phase 2: Orchestration and portability (weeks 2 and 3). Story: MLOps, with DS spillover.

The portability unblock. Everything below depends on it.

- Stand up the orchestrator chosen in OD-1 over the existing stage classes. Do
  not rewrite the stages. Wrap them.
- Model the pipeline as assets or tasks matching what already exists: rollups,
  training Parquet, inference Parquet, model artifact, forecasts, recommendations.
- Port the four `.bat` schedules. Keep the hardening behavior that was hard-won:
  lock takeover semantics, the freshness gate, health check alerting. Those are
  requirements, not implementation details.
- Keep Task Scheduler working until the replacement has run green for a week. Do
  not cut over on faith on a box with known instability.
- DS spillover: training becomes a parameterized job, so rerunning an experiment
  with a different split becomes an argument rather than a config edit.

### Phase 3: Serving (week 4). Story: MLOps, with DS spillover.

- FastAPI in `serving/`. Read endpoints over `forecast_outputs` and
  `recommendation_outputs`. A `/predict` endpoint taking the 40-column feature
  vector. `/health` reusing `collect_health_report()`.
- Fix model selection while here: read `model_metadata.is_active` instead of the
  mtime glob, and reconcile the third lexicographic path in `cli.py`.
- Wire OD-2 tracking at the `train_models()` call site.
- DS spillover: an OpenAPI schema is a published, machine-readable statement of
  the feature contract, which is a better artifact than the prose list.

### Phase 4: Packaging and reproducibility (week 5). Story: both.

- Dockerfile for the pipeline plus API. Compose file adding the dashboard.
- Lockfile. Pin the floating `>=` bounds. Collapse `dashboard/requirements.txt`
  into the extras in `pyproject.toml` so there is one dependency source.
- CI/CD folds in here rather than standing alone: extend the existing workflow to
  build the image, add a coverage gate, publish the artifact. There was nothing
  meaningful to add to CI before a container and a lockfile existed for it to act
  on.

### Phase 5: Prediction monitoring (week 6). Story: both. Depends on ROADMAP M1 #13.

Scheduled after M1's realization ledger because the fix needs it.

- Replace the pooled baseline. Compare the deployed model against a named
  baseline at a matching horizon, sourced from the realization ledger rather than
  from whatever backtest ran last.
- Normalize the comparison. Absolute gold MAE pooled across archetypes moves with
  the archetype mix. Use MAPE or a per-series skill score against `last_value`.
- Make `drift.py` return `unknown` where it currently returns `NONE`, matching
  `health.py`.
- Add a promotion gate: a newly trained model is compared to the incumbent before
  `is_active` moves.

### Phase 6: Infrastructure as code (week 7). Story: MLOps.

- Terraform per OD-4, scoped to the R2 buckets, the lifecycle rule, the Worker
  and its cron, GitHub secrets, and the branch protection ruleset.
- The ToS retention requirement becomes a declared lifecycle rule.
- No new infrastructure invented to have something to describe.

---

## Sequencing rationale

Two points where the obvious order is wrong for this codebase.

**CI/CD and monitoring do not deserve early phases, because both are largely
built.** CI runs ruff and pytest on two Python versions, PR-gated, behind branch
protection with no bypass actors, with Dependabot automerge gated on green CI.
Pipeline monitoring has coverage and gap detection, a stale-lock sentinel, a
retention sentinel, a scheduled health check with durable alerting, backup
staleness, and a cloud gap guard doubling as a dead-man alarm. Putting them in
weeks 3 and 4 would buy a coverage badge, while the leak and the broken demo
stayed live on a public repo for a month. So CI/CD is a slice of Phase 4, where
it has something to act on, and monitoring is Phase 5, where its dependency
exists.

**This system is not single-box, and the truth is a better story.** Capture runs
on GitHub Actions triggered by a Cloudflare Worker cron into R2. Durability is a
second R2 bucket. The desktop is neither the source of truth for snapshots nor
the only copy of durable state. The accurate framing is single-box for training
and serving, already distributed for capture and durability, and that is what the
README should say. It has a practical consequence too: the container work should
treat the image as the unit of deployment rather than hardcoding the box, because
this machine has documented systemic instability and the ops story should not
assume it stays up.

---

## CEILING

What this looks like built without regard to job search timelines. Named
honestly, including where the ambitious version is worse.

- **Conformal prediction replacing the heuristic intervals.** The current CI in
  `cold_start.py` is a heuristic with a 5-percent floor and a 10x cap. It is the
  weakest part of the modeling story. M1 #19 adds quantile regression; the
  ceiling is conformal intervals with measured empirical coverage, so the stated
  80 percent means 80 percent.
- **Hierarchical forecasting with reconciliation.** Item, archetype, and category
  form a real hierarchy. Today items and archetypes are forecast independently
  and stitched together by a trend-ratio heuristic in `crafting_advisor`. MinT or
  OLS reconciliation is the principled version and would make item-level
  forecasts coherent with their parents by construction.
- **An online feature store, so `/predict` is meaningful.** The rollup tables are
  already an offline store. An online store keyed by archetype with latest
  features would let the API serve genuine real-time predictions instead of a
  feature-vector echo.
- **Event effects estimated causally and fed back as priors.** M5 measures event
  impact with interrupted time series and difference-in-differences. The ceiling
  closes the loop: measured effects become model inputs rather than hand-authored
  impact records in `config/events/`.
- **Escaping the 30-day retention wall.** This is the binding data constraint.
  The ToS permits 30 days of raw retention, so long-horizon evaluation is
  structurally limited no matter how good the engineering is. The ceiling is a
  derived-aggregate archive designed from the start to be ToS-clean and to
  outlive the raw window, plus an independent historical source for backfill.
- **Panel data across regions.** EU and KR plus per-realm BoE auctions turn one
  set of series into a panel and open cross-sectional methods that a single
  region cannot support.
- **End-to-end queryable lineage.** Every forecast traceable to dataset version,
  model version, code SHA, and config hash. The pieces exist (`features_hash`,
  `config_snapshot`, `run_metadata`, dataset manifests) and are not joinable.
- **Shadow deployment and automated promotion.** A challenger scoring alongside
  the champion against the realization ledger, promoted automatically on a
  significance test rather than on the calendar.

What the ceiling should **not** include: Kubernetes, multi-node training, or a
distributed scheduler. Durable state is around 100 MB and the largest table is
under 10 GB. Single box for compute is the correct architecture, and reaching for
cluster infrastructure would be a portfolio tell in the wrong direction.

---

## Verification

- Phase 1 methodology fix: a test asserting that for each horizon, no training
  row's target date falls at or after the validation start. It must fail against
  today's `trainer.py`.
- Phase 1 backtest: `wow-forecaster backtest --horizons 1,7,28` produces a
  `summary.csv` containing a `lightgbm` row alongside the four baselines, and
  `report-backtest` renders it.
- Phase 1 demo: `streamlit run dashboard/app.py` shows populated tabs without
  editing any file.
- Phase 1 legibility: opening the README on GitHub shows a results table and at
  least one chart without scrolling past the fold.
- Phase 2: the full daily pipeline runs to completion on Linux in CI or a
  container, with no `.bat` involved.
- Phase 3: `curl` against `/health` and a forecast read endpoint returns real
  data from a restored durable backup.
- Phase 4: `docker compose up` from a clean clone reaches a working dashboard and
  API with no host Python.
- Phase 5: a deliberately degraded model raises the drift level, and a missing
  baseline reports `unknown` rather than `none`.
- Phase 6: `terraform plan` against live infrastructure reports no changes.

---

## Relationship to the roadmap

[docs/ROADMAP.md](docs/ROADMAP.md) and its GitHub milestones stay the source of
truth for the research arc and for issue numbering. This plan names M1 issues
where the two overlap and does not renumber or restate them. The interlock is
one-directional on purpose: the roadmap does not reference this document, so
neither goes stale when the other moves. Reconcile them deliberately, not as a
side effect.

Overlaps to keep in view:

- Phase 1 backtest work is the minimum version of M1 #14 through #17.
- Phase 5 is blocked on M1 #13, the `forecast_realizations` ledger.
- OD-2's case is strengthened by M1 #18, the Optuna study.
- CEILING's conformal intervals extend M1 #19's quantile regression.
