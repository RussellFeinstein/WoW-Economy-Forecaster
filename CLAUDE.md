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
19 labels in two namespaces plus a small state set (7 `type:`, 8 `area:`, 4 state; the count read 21 until 2026-09-20 because it included two Dependabot defaults, removed with #143). Every issue carries exactly one `type:` and at least one `area:`; state labels are added only when they apply. Each label's GitHub description carries its own rule, so the meaning cannot drift into someone's head.
- `type:` mirrors the branch prefixes above exactly: `type: feat`, `type: fix`, `type: improvement`, `type: docs`, `type: chore`, `type: refactor`, `type: test`. One vocabulary, so the label predicts the branch name and the scope check has something to check against. `^dependabot/` maps to `type: chore`, and dependabot.yml names `type: chore` plus `area: repo` directly (#143), so a bump is labelled by both paths rather than by the branch rule alone.
- `area:` names the subsystem: ops (scheduling, locks, health, backup, integrity), capture (ingest, cloud capture, sync, retention), modeling (features, training, backtest, forecast, drift, recommendations, simulation), warehouse, reporting (report-* CLIs, dashboard, viz, BI), analysis (event study, causal, notebooks), repo (CI, test infra, governance, versioning, repo docs), publish (public dashboard, README case study, profile). One per issue by default; a second only where the issue has separate acceptance items in two subsystems (#15, #24, #30).
- Areas overlapping milestones is deliberate (M3 is all warehouse, M4 reporting, M5 analysis, M6 publish). Milestones are chronological arcs that close and stop being lookup tools; areas are permanent. `gh issue list --state all --label "area: capture"` is the query that assembles a subsystem's whole history across milestones, which is the reason closed issues are labeled at all.
- State labels: `waiting: wall clock` (built and waiting on a date, which is at the top of the body; never for unbuilt work, so #33 does not carry it despite needing weeks to mature), `blocked` (cannot start or finish until another issue lands; the first line of the body names it), `needs: operator` (needs a step only Russell can do, such as a credential, an external account, or a physical action, which is the class the no-secrets rule creates; not the same as blocked, since an issue can need that step at implementation time and not be blocked today), `found: audit` (filed by a deliberate audit or verification pass, not a live failure; the body names the pass).
- PRs are labeled automatically by [.github/workflows/labeler.yml](.github/workflows/labeler.yml) from the branch prefix and the changed paths, with rules in [.github/labeler.yml](.github/labeler.yml). On a PR the `area:` label is advisory (the globs are coarse and multi-subsystem PRs pick up several); the issue carries the authoritative one. Fix by hand what the globs get wrong.
- Deliberately absent: priority labels (the build-order tracker #167 is the ordered source; a second one would drift against it), milestone labels (milestones already exist), status labels (the open branch and PR show that). GitHub's defaults were pruned in the same pass: bug, enhancement, and documentation were renamed into the `type:` namespace, and the six unused contribution-model defaults were deleted (good first issue and help wanted advertise outside PRs this repo does not take; duplicate, invalid, and wontfix are superseded by GitHub's native close reasons).
- Baseline set by the 2026-07-30 backfill: all 70 issues and all 51 PRs labeled. The three labels Dependabot minted on its own (`dependencies`, `python`, `github_actions`) were removed from the merged PRs that carried them and deleted on 2026-09-20 (#143), once the 2026-09-01 bump had proven the `labels:` list in dependabot.yml keeps new bumps inside the taxonomy.

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
Next-phase work (M0 restore/harden ops -> M0.5 unattended capture -> M1 model validation -> M2 paper-trading P&L + ranking A/B -> M3 PostgreSQL+dbt warehouse -> M4 Power BI/Tableau -> M5 event impact study -> M6 publish) lives in three places with three jobs. **The order is [#167](https://github.com/RussellFeinstein/WoW-Economy-Forecaster/issues/167), the pinned build-order tracking issue**: one table per milestone in the shape the global rules set out (`# | Issue | Flag | Runs | Needs | What`), holding only what is not built yet, with `live` and `gate` as the only flags and the beside/behind relation in each area heading. [docs/ROADMAP.md](docs/ROADMAP.md) is the narrative: why each milestone exists, the dependency graph, the standing risks. The GitHub milestones M0A, M0B and M1-M6 carry scope and open with `Work order (pinned): #167.`; they carried numbered work-order lists from v2.14.18 (#46) to v2.14.26, and that convention is retired, because a second copy of the order drifted against the first every time (M1's list carried a closed #100 and omitted #107 for eight weeks; #72 and #102 were in their milestones and in no list at all). Session protocol: read row 1 of M0A on #167; that is what gets built next, and the item is re-audited against the repo immediately before its branch. A row leaves the tracker when its issue closes (a body edit in the closing PR, no comment). Filing an issue, moving one, or clearing a Needs posts an order-change comment on #167 (`**Order change, YYYY-MM-DD.**`) as part of the PR that does it, never a line in the body, because a body edit notifies nobody and a comment's date cannot drift. Milestone numbers match the tracker's area order; within an area the row order is the sequence. When the rows ahead wait on wall clock or an operator step, the next area (or M5, the filler) is where work continues.

## Operational state (hazard retired 2026-07-21)
- **Ingestion restored 2026-07-21 02:43Z after 105 days dead** (leaked `data/db/.hourly.lock` from a 2026-04-15 crash). The issue #1 runbook executed in full on 2026-07-20/21: rollup backfill (coverage 22 -> 34 dates, all certified against independent sources after two hardware-induced corruption events), evidence captured to `data/outputs/backups/evidence_2026-07-20/`, both observation tables dropped and rebuilt (DB 78 GB -> 105 MB via VACUUM INTO; the known corrupt raw page never copied), lock deleted, first run green, all three scheduled tasks re-enabled and observed green (hourly every hour, health 06:45, daily 07:00 with forecasts + recommendations). Close-out record on issue #1.
- Data gap 2026-04-08..2026-07-20 is permanent locally (Blizzard serves current snapshots only); cloud capture (#42) has been collecting hourly to R2 since 2026-07-20 21:02Z and #43 catch-up ingestion (`sync-snapshots`) drains it. The drain is live, not staged: the backlog ran to zero on 2026-07-30 (47 objects, 11.5M records, Jul 25-27 restored to full hourly coverage, acceptance evidence on #43) and it has since been the repair path for wake-failure gaps, two hours on 2026-07-29. Drift detection rebuilds its baseline over ~30 days; item-level forecasts return ~2026-08-03 (14 fresh days); #11 tracks the verification window.
- Machine caution (rex-desktop): systemic instability under sustained multi-GB load; after any large index build / VACUUM / bulk copy on this box, cross-verify outputs against independent sources before trusting them (two corruption events during the runbook, one after a clean mdsched pass).
- **Prune stall 2026-08-31 to 2026-09-20 (#155).** Two entries missing from `idx_obs_raw_realm_ingested` made the raw DELETE for the 2026-07-31T06 hour slice fail on every hourly run once it entered the retention window, and the pruner stops at the first failing slice, so about 105M raw rows and their children went past the ToS window, the file grew 98 GB -> 150 GB, and the daily forecast gate (which includes the retention sentinel, #161) skipped every forecast from 09-01. Migration 0012 dropped the index (applied on rex-desktop 2026-09-20, 271 s) and the health probe moved to the normalized table; the first prune afterward cleared the damaged slice in 79 s (evidence on #155), and the 104M-row remainder was cleared the same afternoon by 55 `prune-snapshots` calls between 13:20 and 14:44 (about 1.6M rows and 80 s each); the health check read HEALTHY at 14:47 with the oldest raw row at 30.8 days, and the daily forecast ran at 14:53, the first since 08-31. Two consequences persist: the file stays at 150 GB with 43% of its pages on the freelist for inserts to reuse (15.8M of 36.7M pages; C: had 30 GB free at the time, so no VACUUM INTO is possible and none is wanted on this box), and the hourly run's drift check had grown past the hour on the oversized table, silently halving local capture (#162; cloud capture held the hours; the post-drain timing is still to be read).
- **Every daily forecast run locks the hourly out until #107 lands.** The forecast stage scans the whole normalized table inside its write transaction (23 to 38 minutes), so the 07:16 hourly's IngestStage fails with `database is locked` on every day the daily runs (08-29, 08-30, 08-31, 09-20) and reports `[PARTIAL]` at exit 0 (#164). Cloud capture holds those hours. Found by the 2026-09-20 audit, which also found four single-character corrupt rows in the rollup tables that no integrity check can see (#165), and reopened #100, which had been closed unfixed since 07-30.
- Migrations end at 0012 (dropping the raw (realm_slug, ingested_at) index); new migrations start at 0013.

## What's NOT Implemented Yet
- top_n_per_category V2 (Pareto-frontier, user-profile weighting, blocklist, A/B test support); cross-horizon dedup done in v0.9.1
- Governance: cooldown enforcement not wired — preflight.py has check but orchestrator.py never passes last_call_at
- Live news ingestion: BlizzardNewsClient.fetch_recent_news() exists but IngestStage._fetch_news() always uses fixture mode
- News-to-event: extract_wow_events() not implemented (news items → WoWEvent candidates)

## Known Bugs (unfixed)
- Note: `except Exception` does NOT catch KeyboardInterrupt/SystemExit (those are BaseException subclasses). The global standard pattern `except (KeyboardInterrupt, SystemExit): raise` is redundant here — signals always propagate through `except Exception:` automatically.

## Tests
The full suite must pass before any PR; the Windows-only script tests in tests/test_scripts/ skip on Linux and CI. Counts, fixture guarantees, and platform gotchas: [.claude/rules/testing.md](.claude/rules/testing.md). Lint and test with the exact invocations in .github/workflows/ci.yml, never self-chosen scopes.
