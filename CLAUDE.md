# WoW Economy Forecaster — Project Instructions

## Project Overview
Local-first WoW AH economy research system. TWW historical data → Midnight transfer learning.
Category/archetype-based transfer (NOT item-to-item). Python, SQLite, Parquet, Typer CLI.

## Virtual Environments
**Always use virtual environments.** Never install packages globally.

## Branch Workflow
- main is the only permanent branch. A branch protection ruleset (main-pr-only, no bypass actors) requires a pull request for every merge and blocks direct pushes, force pushes, and deletion. This applies to admins too.
- Every piece of work gets a short-lived type-prefixed branch cut from the latest main: feat/, fix/, improvement/, docs/, chore/, refactor/, test/ plus a short kebab slug, with the issue number when one exists (e.g. fix/44-ci-ruff-drift).
- One issue or one concern per branch. Ship it by opening a PR to main and merging via the PR with a merge commit (`gh pr merge --merge`). CI runs on the PR before merge.
- The head branch is deleted on merge (delete_branch_on_merge is on); delete the local copy with `git branch -d`. The merge commit and PR record are the durable history. Never `git branch -D` unmerged work without explicit instruction.
- Scope check before every commit: if the work does not match the current branch's type and slug, stop and cut the right branch from main.
- No umbrella or long-lived topic branches. feature/portfolio-showcase (v1.9.0-v2.4.1) was the last; merged 2026-07-12 (issue #10), deleted with the freeze-convention retirement (issue #46).

## Issue and PR labels
21 labels in two namespaces plus a small state set. Every issue carries exactly one `type:` and at least one `area:`; state labels are added only when they apply. Each label's GitHub description carries its own rule, so the meaning cannot drift into someone's head.
- `type:` mirrors the branch prefixes above exactly: `type: feat`, `type: fix`, `type: improvement`, `type: docs`, `type: chore`, `type: refactor`, `type: test`. One vocabulary, so the label predicts the branch name and the scope check has something to check against. `^dependabot/` maps to `type: chore`.
- `area:` names the subsystem: ops (scheduling, locks, health, backup, integrity), capture (ingest, cloud capture, sync, retention), modeling (features, training, backtest, forecast, drift, recommendations, simulation), warehouse, reporting (report-* CLIs, dashboard, viz, BI), analysis (event study, causal, notebooks), repo (CI, test infra, governance, versioning, repo docs), publish (public dashboard, README case study, profile). One per issue by default; a second only where the issue has separate acceptance items in two subsystems (#15, #24, #30).
- Areas overlapping milestones is deliberate (M3 is all warehouse, M4 reporting, M5 analysis, M6 publish). Milestones are chronological arcs that close and stop being lookup tools; areas are permanent. `gh issue list --state all --label "area: capture"` is the query that assembles a subsystem's whole history across milestones, which is the reason closed issues are labeled at all.
- State labels: `waiting: wall clock` (built and waiting on a date, which is at the top of the body; never for unbuilt work, so #33 does not carry it despite needing weeks to mature), `blocked` (cannot start or finish until another issue lands; the first line of the body names it), `needs: operator` (needs a step only Russell can do, such as a credential, an external account, or a physical action, which is the class the no-secrets rule creates; not the same as blocked, since an issue can need that step at implementation time and not be blocked today), `found: audit` (filed by a deliberate audit or verification pass, not a live failure; the body names the pass).
- PRs are labeled automatically by [.github/workflows/labeler.yml](.github/workflows/labeler.yml) from the branch prefix and the changed paths, with rules in [.github/labeler.yml](.github/labeler.yml). On a PR the `area:` label is advisory (the globs are coarse and multi-subsystem PRs pick up several); the issue carries the authoritative one. Fix by hand what the globs get wrong.
- Deliberately absent: priority labels (docs/ROADMAP.md Work order is the ordered source; a second one would drift against it), milestone labels (milestones already exist), status labels (the open branch and PR show that). GitHub's defaults were pruned in the same pass: bug, enhancement, and documentation were renamed into the `type:` namespace, and the six unused contribution-model defaults were deleted (good first issue and help wanted advertise outside PRs this repo does not take; duplicate, invalid, and wontfix are superseded by GitHub's native close reasons).
- Baseline set by the 2026-07-30 backfill: all 70 issues and all 51 PRs labeled.

## Versioning (stamp commits)
- Work commits take no version bump. Their CHANGELOG lines accumulate under `## [Unreleased]`.
- A dedicated stamp commit at PR-open bumps pyproject.toml once and moves the [Unreleased] entries under the `## [X.Y.Z] - YYYY-MM-DD` header. One version per PR; PR titles carry the `(vX.Y.Z)` suffix.
- If two open PRs stamp the same number, the later-to-merge PR re-stamps to the next free number during rebase.
- Dependabot dependency-bump PRs are exempt from stamping and CHANGELOG. Both watched ecosystems are dev-only (ruff and the GitHub Actions workflow pins, per .github/dependabot.yml), so a bump does not change the product and stamping the product version would be misleading. Action bumps are grouped into a single PR (issue #136). These PRs auto-merge on green CI via .github/workflows/dependabot-automerge.yml (CI is a required status check, so a bump that breaks lint or tests never merges); note that the required checks only exercise ci.yml, so an action bump that breaks cloud-snapshot.yml or verify-backup.yml surfaces at the next capture or the 14:00 UTC backup verification rather than at merge time. If a bump surfaces new drift, fix it with a conformance commit pushed onto the Dependabot branch, never a parallel takeover PR.

## Entry points
- [wow_forecaster/cli.py](wow_forecaster/cli.py) - Typer app; every command registers here
- [wow_forecaster/config.py](wow_forecaster/config.py) - AppConfig via load_config(); static config under config/
- [wow_forecaster/db/schema.py](wow_forecaster/db/schema.py) - apply_schema(); migrations in db/migrations.py
- [wow_forecaster/pipeline/base.py](wow_forecaster/pipeline/base.py) - PipelineStage ABC that every stage inherits
- [wow_forecaster/taxonomy/](wow_forecaster/taxonomy/) - archetype and event taxonomies; imports nothing from models/

## Architecture Patterns
- taxonomy/ imports nothing from models/ (no circular imports)
- Models frozen=True except RunMetadata (mutable status)
- Every pipeline run writes RunMetadata with config_snapshot for reproducibility
- WoWEvent.announced_at + is_known_at() = look-ahead bias guard
- Archetype mappings require non-empty mapping_rationale (audit trail)
- RawMarketObservation has NO obs_id field — query DB rows directly when obs_id needed
- IngestStage pre-persists RunMetadata at start of _execute() to get run_id for FK use
- IngestStage uses 3-phase connection pattern: (1) short read connection for FK guard, (2) no connection during HTTP fetch, (3) short write connection for all inserts — avoids holding DB lock during network I/O
- All pipeline get_connection() calls pass config.database.wal_mode + busy_timeout_ms (default 30s)
- run_hourly.bat uses lock file (data/db/.hourly.lock) to prevent overlapping scheduled runs; locks older than 180 minutes are taken over (STALE LOCK TAKEOVER logged, lock deleted, run continues), and an age-check failure also takes over; only a provably fresh lock skips (exit 0)
- ForecastOutput frozen model — use object.__setattr__(fc, "forecast_id", fc_id) after DB insert
- LightGBM v4+ requires numpy arrays — convert list[list[float]] via np.array(..., dtype=np.float64)
- Windows terminal: avoid Unicode arrows in typer.echo() — use ASCII -> instead
- datetime.utcnow() deprecated — use datetime.now(tz=timezone.utc).replace(tzinfo=None)

## Data Sources (Blizzard API only)
- BlizzardClient: LIVE — fetch_commodities() + fetch_connected_realm_auctions() + OAuth2
- Default realm: ["us"] (commodity AH is region-wide since 9.2.7)

## Primary Workflow
```
run-hourly-refresh   # Blizzard API ingest → normalize → drift → provenance
build-datasets       # feature engineering → Parquet
run-daily-forecast   # train → forecast → recommend
```
`import-auctionator` = historical backfill only, not needed for ongoing operation.

## Layer guide
Subsystem detail lives in path-scoped rules files under .claude/rules/, loaded automatically when working with matching files (progressive disclosure). Open one directly when planning cross-cutting work:
- [ingestion-capture.md](.claude/rules/ingestion-capture.md) - ingestion, cloud capture, catch-up drain
- [modeling.md](.claude/rules/modeling.md) - features, backtesting, ML, normalization, recipes
- [ops-health.md](.claude/rules/ops-health.md) - monitoring, health gates, scheduled tasks, sleep-back, durable backup
- [viz.md](.claude/rules/viz.md) - charts, dashboard, BI exports, notebooks
- [governance-events.md](.claude/rules/governance-events.md) - source policies, retention, seed events
- [learning-track.md](.claude/rules/learning-track.md) - learning curriculum, banks, drift-guard anchors
- [testing.md](.claude/rules/testing.md) - test suite layout, counts, platform skips

## Roadmap
Next-phase work (M0 restore/harden ops -> M0.5 unattended capture -> M1 model validation -> M2 paper-trading P&L + ranking A/B -> M3 PostgreSQL+dbt warehouse -> M4 Power BI/Tableau -> M5 event impact study -> M6 publish) is tracked in [docs/ROADMAP.md](docs/ROADMAP.md) and GitHub milestones M0-M6 plus M0.5, which between them carry every open issue. Session protocol: follow the Work order section in docs/ROADMAP.md (milestone numbers match it; within a milestone use its issue sequence, not raw issue numbers). Each milestone description on GitHub opens with a numbered work-order list rendered from ROADMAP.md; when filing, closing, or reordering an issue, update that milestone's list and the matching ROADMAP text **in the same PR as the change that filed or closed it** (a stale list is a doc bug, same as any Documentation Sync miss). The scope word is load-bearing: this rule said "same session" until 2026-07-30, and a session is not a reviewable unit, so filing #117 and opening the PR that fixed it counted as separate acts and the milestone update fell in the gap between them. A PR is where a missing ROADMAP hunk shows up in a diff before merge. When remaining issues are waiting on wall clock (#11, #33), advance to the next milestone and circle back.

## Operational state (hazard retired 2026-07-21)
- **Ingestion restored 2026-07-21 02:43Z after 105 days dead** (leaked `data/db/.hourly.lock` from a 2026-04-15 crash). The issue #1 runbook executed in full on 2026-07-20/21: rollup backfill (coverage 22 -> 34 dates, all certified against independent sources after two hardware-induced corruption events), evidence captured to `data/outputs/backups/evidence_2026-07-20/`, both observation tables dropped and rebuilt (DB 78 GB -> 105 MB via VACUUM INTO; the known corrupt raw page never copied), lock deleted, first run green, all three scheduled tasks re-enabled and observed green (hourly every hour, health 06:45, daily 07:00 with forecasts + recommendations). Close-out record on issue #1.
- Data gap 2026-04-08..2026-07-20 is permanent locally (Blizzard serves current snapshots only); cloud capture (#42) has been collecting hourly to R2 since 2026-07-20 21:02Z and #43 catch-up ingestion (`sync-snapshots`) drains it. The drain is live, not staged: the backlog ran to zero on 2026-07-30 (47 objects, 11.5M records, Jul 25-27 restored to full hourly coverage, acceptance evidence on #43) and it has since been the repair path for wake-failure gaps, two hours on 2026-07-29. Drift detection rebuilds its baseline over ~30 days; item-level forecasts return ~2026-08-03 (14 fresh days); #11 tracks the verification window.
- Machine caution (rex-desktop): systemic instability under sustained multi-GB load; after any large index build / VACUUM / bulk copy on this box, cross-verify outputs against independent sources before trusting them (two corruption events during the runbook, one after a clean mdsched pass).
- Migrations end at 0009 (health-check indexes); new migrations start at 0010.

## What's NOT Implemented Yet
- top_n_per_category V2 (Pareto-frontier, user-profile weighting, blocklist, A/B test support); cross-horizon dedup done in v0.9.1
- Governance: cooldown enforcement not wired — preflight.py has check but orchestrator.py never passes last_call_at
- Live news ingestion: BlizzardNewsClient.fetch_recent_news() exists but IngestStage._fetch_news() always uses fixture mode
- News-to-event: extract_wow_events() not implemented (news items → WoWEvent candidates)

## Known Bugs (unfixed)
- Note: `except Exception` does NOT catch KeyboardInterrupt/SystemExit (those are BaseException subclasses). The global standard pattern `except (KeyboardInterrupt, SystemExit): raise` is redundant here — signals always propagate through `except Exception:` automatically.

## Tests
The full suite must pass before any PR; the Windows-only script tests in tests/test_scripts/ skip on Linux and CI. Counts, fixture guarantees, and platform gotchas: [.claude/rules/testing.md](.claude/rules/testing.md). Lint and test with the exact invocations in .github/workflows/ci.yml, never self-chosen scopes.
