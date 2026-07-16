# Changelog

All notable changes to the WoW Economy Forecaster.

Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
This project uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.4.4] - 2026-07-15

### Changed
- Merges to main now require a pull request: a branch protection ruleset blocks direct pushes, force pushes, and deletion, with no bypass for admins (issue #46)
- Merged branches are deleted instead of kept frozen; the repo auto-deletes head branches on merge, and the four already-merged branches were removed
- Versioning moved to stamp commits: work commits log under Unreleased, and a single stamp commit at PR-open sets the version for the whole PR
- Each GitHub milestone description now opens with a numbered work-order list matching docs/ROADMAP.md

## [2.4.3] - 2026-07-12

### Fixed
- README CI badge now points at the real repository instead of the yourusername placeholder, so it renders actual CI status (issue #45)

## [2.4.2] - 2026-07-12

### Changed
- Development model: work now lands on short-lived type-prefixed branches (feat/, fix/, docs/, chore/) cut from main and merged per issue, recorded as a Branch Workflow section in CLAUDE.md; the long-lived feature/portfolio-showcase branch was merged to main (issue #10) and frozen
- Cloud capture activation shortened: the workflow is on main but disabled by hand, so the remaining steps are the bucket, the six secrets, gh workflow enable, and one manual dispatch (README, docs/cloud-capture.md, CLAUDE.md updated to match)

## [2.4.1] - 2026-07-12

### Added
- Issue #44 filed and slotted first in the green-CI tier of the work order: CI fails at the ruff lint step before tests run because `ruff>=0.4` floats to releases enforcing rules the codebase predates (782 findings under 0.15.2), so pytest results are invisible on GitHub

### Changed
- docs/ROADMAP.md and CLAUDE.md: M0 issue list and work order updated for #44

## [2.4.0] - 2026-07-12

### Added
- Cloud snapshot fetcher (issue #42): `python -m wow_forecaster.ingestion.cloud_fetch` plus a scheduled GitHub Actions workflow capture the hourly commodities snapshot from always-on infrastructure and upload it gzipped to a private S3-compatible bucket, so capture no longer requires the desktop to be on. Reuses the existing Blizzard client and snapshot writer, so cloud objects carry the identical envelope local ingest produces
- Failure paths are loud by design: refuses implausibly small snapshots (default floor 50,000 records), retries fetch and upload, exits 3 when the trailing 24 hours of objects have gaps, and reports missing configuration by variable name only
- `[cloud]` optional dependency group (boto3) for running the fetcher outside the workflow
- README setup section covering the bucket, lifecycle rule, repository secrets, and first-run verification; activation is a manual one-time step for the repository owner

## [2.3.9] - 2026-07-12

### Added
- docs/cloud-capture.md: cloud capture design record (issue #41). GitHub Actions hourly workflow plus a private Cloudflare R2 bucket with a 30-day lifecycle rule; sizing measured from a real snapshot (58.9 MB raw, 2.2 MiB at gzip level 9, 25.7x, ~1.5 GiB per rolling 30-day window); compliance mapping, failure-visibility plan, and the one-time activation checklist

## [2.3.8] - 2026-07-12

### Changed
- Milestones renumbered to match the decisive work order: paper trading P&L and ranking A/B is now M2 (was M4), the PostgreSQL + dbt warehouse is M3 (was M2), and BI dashboards are M4 (was M3). The live A/B test needs weeks of data to mature, so its clock starts right after model validation instead of waiting behind infrastructure work, and the make-gold answer exists before dashboards are built to showcase it
- docs/ROADMAP.md: added a Work order section with the issue-level sequence, most urgent first (stop data loss, green CI, harden, restore, validate, then build outward)

## [2.3.7] - 2026-07-12

### Changed
- Milestone M7 renamed to M0.5 (unattended capture) and moved to run immediately after M0: the design and cloud fetcher (#41, #42) depend on nothing local and stop further unrecoverable data loss, so they no longer wait behind M1-M6; only the catch-up command (#43) needs the restored pipeline

## [2.3.6] - 2026-07-12

### Added
- Milestone M7 (unattended capture) on the roadmap: cloud-hosted hourly snapshot fetcher, private object storage with a 30-day lifecycle rule, and a local catch-up ingestion command, so capture no longer depends on the desktop being on (issues #41-#43)
- M0 issue #40: wake-to-run task settings so the machine can sleep between scheduled runs

### Changed
- docs/ROADMAP.md and CLAUDE.md updated for milestone M7 and the extended issue range (#1-#43)

## [2.3.5] - 2026-07-12

### Added
- docs/ROADMAP.md: next-phase roadmap (M0 restore/harden operations through M6 publish) with dependency graph and risk register; work tracked as GitHub milestones M0-M6 with issues #1-#39

### Changed
- CLAUDE.md: documented the active ingestion outage (lock leaked 2026-04-15, ingestion dead since) and the lock-clearing hazard (orchestrator auto-prune would delete all rows older than 30 days; rollup tables are incomplete), corrected the date-spine description (Python-generated spine over rollup fast path, not a recursive CTE), and noted that migrations end at 0008

## [2.3.4] — 2026-04-07

### Added
- `checkpoint-db` CLI command to force WAL checkpoint when the write-ahead log grows too large
- Automatic WAL checkpoint step in hourly orchestrator pipeline (after prune, before monitoring outputs)

### Fixed
- WAL file growth unbounded (no checkpoint logic existed); 4.2 GB WAL was causing all DB operations to exceed lock timeout

## [2.3.3] — 2026-04-06

### Fixed
- Rollup tables now update during hourly pipeline (was silently failing due to missing `self._conn` attribute in orchestrator)
- IngestStage no longer holds SQLite write lock during Blizzard API HTTP fetch (connection split into read/fetch/write phases)
- All pipeline stages now use config-driven `busy_timeout_ms` instead of hardcoded 5-second default
- Default `busy_timeout_ms` increased from 5s to 30s to handle realistic batch operation contention
- Overlapping hourly pipeline runs prevented via lock file guard in `run_hourly.bat`
- Version regression from v2.3.2 to v2.2.3 corrected (was a typo in previous commit)

## [2.3.2] — 2026-04-05

### Changed
- Migrated `archetype_features.py` to use pre-aggregated rollup tables for faster feature queries

## [2.3.1] — 2026-04-05

### Fixed
- `backfill-rollups` now uses `get_connection` as context manager (was leaking connections)

## [2.3.0] — 2026-04-05

### Added
- Pre-aggregated rollup tables (`archetype_rollups`, `item_rollups`) for 110M-row performance optimization
- `backfill-rollups` CLI command to populate rollup tables from historical data
- Automatic rollup update step in hourly orchestrator pipeline

## [2.2.3] — 2026-04-06

### Added
- Related Projects section in README linking to alt-army-guide (profession setup guide for executing on forecaster recommendations)

## [2.2.2] — 2026-03-20

### Fixed
- Scheduled tasks no longer open a visible cmd.exe window; `setup_tasks.bat` now uses `wscript.exe` + `run_silent.vbs` wrapper for silent execution

### Added
- `scripts/run_silent.vbs` — generic VBS launcher that runs batch files with no console window

## [2.2.0] — 2026-03-19

### Added
- Visualization layer (`wow_forecaster/viz/`) with WoW-themed dark palette, 6 chart modules, and data query interface
- `generate-charts` CLI command for publication-quality static chart generation (matplotlib/seaborn/Plotly)
- `export-bi-bundle` CLI command for Power BI / Tableau star-schema exports (dim + fact tables)
- BI data dictionary generation (DATA_DICTIONARY.md)
- 3 Jupyter analysis notebooks (EDA, Model Development, Backtest Evaluation) for portfolio narrative
- Streamlit dashboard upgraded to 8 tabs (added Backtest Analysis, Feature Insights, Crafting Margins)
- Interactive Plotly forecast chart with CI bands replaces basic Streamlit line chart
- GitHub Actions CI workflow (pytest + ruff on Python 3.11/3.12)
- CHANGELOG.md with retroactive history from v0.0.1 to v2.1.0
- 125 new tests for visualization and BI export modules (1,228 total)

## [2.1.0] — 2026-03-17

### Added
- `check-data-health` CLI command with DB-backed gap detection (days of coverage, calendar-date gaps, last ingest age)

### Fixed
- `run-hourly-refresh` now exits with code 1 on "failed" pipeline status

## [2.0.0] — 2026-03-14

### Added
- `export-tsm` CLI command for TradeSkillMaster paste-import string generation
- `TsmExportRow` dataclass and TSM export pipeline (filters by ROI + ci_quality)

## [1.12.0] — 2026-03-12

### Added
- Extended item-level forecasting to all items with 14+ observation days (previously recipe-linked only)
- `ItemForecastRoi` dataclass and `fetch_item_rois()` for ROI-based item overlays
- `top_items` column in recommendations now prefers ROI-based items over discount-based fallback

## [1.11.0] — 2026-03-10

### Fixed
- CI floor/cap: lower bound floored at 5% of current price, upper bound capped at 10x current price
- Prevents 0.0 lower bounds and absurd upper bounds in confidence intervals

### Added
- `ci_quality` field on ForecastOutput ("good"/"wide"/"unreliable") with DB migration 0007

## [1.10.0] — 2026-03-08

### Changed
- Decoupled risk_level from action — AVOID only issued at CRITICAL uncertainty (>= 95%)
- Risk levels (LOW/MEDIUM/HIGH/CRITICAL) now independent of buy/sell/hold determination

### Added
- `risk_level` column in recommendation_outputs (DB migration 0006)
- `determine_risk_level()` function in scorer.py

## [1.9.0] — 2026-03-06

### Added
- `prune-snapshots` CLI command with `--days N` and `--dry-run` flags
- `SnapshotPruner` deletes raw JSON and market_observations_raw rows past retention period
- HourlyOrchestrator auto-prunes after every successful ingest (non-fatal step 7)

## [1.8.0] — 2026-03-04

### Added
- `report-feature-importance` CLI command showing LightGBM gain/split importance per horizon
- CSV export support for feature importance data

## [1.7.0] — 2026-03-02

### Added
- Cold-start prediction blending via archetype transfer mappings
- Formula: `blended = confidence * model_pred + (1 - confidence) * source_price`
- `_transfer` suffix on model_slug for blended predictions

## [1.6.0] — 2026-02-28

### Added
- Item-level forecast persistence for recipe-linked items
- `forecast_outputs.item_id` now populated for recipe items after each forecast run

## [1.5.0] — 2026-02-20

### Added
- Recipe and crafting advisor system (v1.5.0 — v1.5.7)
- `seed-recipes`, `build-margins`, `report-crafting`, `report-recipe-status` CLI commands
- 6 crafting temporal windows (NOW_NOW through 28D_28D)
- Trend-ratio future price projection for item-level craft cost estimation
- Volume gate and margin compression/expansion detection
- DB tables: recipes, recipe_reagents, crafting_margin_snapshots

## [1.4.0] — 2026-02-12

### Added
- Item-level discount overlay in recommendation pipeline
- `top_item_names`, `top_item_prices`, `top_item_discounts`, `top_item_z_scores` in CSV export

## [1.3.0] — 2026-02-05

### Fixed
- Horizon mismatch ("30d" vs "28d"), dead Literal entries, stale assertions
- Silent event-feature zeroing when wow_events table is empty
- Config.py defaults diverged from default.toml

### Changed
- Archetype_id populated in normalized observations (v1.3.4)
- Numerous dead-code cleanups and documentation sync fixes (v1.3.5 — v1.3.26)

## [1.2.0] — 2026-01-30

### Changed
- Migrated project memory from local files to CLAUDE.md for cross-machine portability

## [1.1.0] — 2026-01-28

### Added
- Rolling z-score normalization with 30-day rolling stats
- Cold-start fallback to batch statistics

### Fixed
- event_boost silent zeroing when event features were all zero

## [1.0.0] — 2026-01-25

### Added
- Automation layer: `SchedulerDaemon` (stdlib-only foreground daemon)
- `start-scheduler` CLI command
- Windows Task Scheduler setup scripts (`scripts/setup_tasks.bat`, `run_hourly.bat`, `run_daily.bat`)

## [0.9.0] — 2026-01-20

### Added
- Seed events system with WoW event calendar and category-level impact records
- Auctionator CSV import pipeline for historical backfill
- Item bootstrapper (9,950 items from Blizzard Item API)
- Per-item discount overlay and cross-horizon archetype deduplication

## [0.8.0] — 2026-01-15

### Added
- Source governance layer with 3 source policies (blizzard_api, blizzard_news_manual, manual_event_csv)
- 3-check preflight system before each ingest
- Ingestion parsing: snapshot records to market_observations_raw

## [0.7.0] — 2026-01-10

### Added
- Reporting and dashboard layer
- 8 `report-*` CLI commands (top-items, forecasts, volatility, drift, status, crafting, recipe-status, feature-importance)
- 5-tab Streamlit dashboard with provenance-aware freshness badges
- CSV/JSON export for Power BI

## [0.6.0] — 2026-01-05

### Added
- Monitoring, drift detection, and hourly orchestration layer
- `HourlyOrchestrator`: 7-step pipeline with adaptive CI widening
- Data drift, error drift, and event-shock detection
- Personal research license prohibiting AH market manipulation

## [0.5.0] — 2025-12-28

### Added
- ML forecasting model layer (LightGBM) with 1d/7d/28d horizons
- 5-component recommendation scoring (opportunity, liquidity, volatility, event_boost, uncertainty)
- `train-model`, `recommend-top-items`, `run-daily-forecast` CLI commands

## [0.4.0] — 2025-12-20

### Added
- Backtesting framework with walk-forward cross-validation
- 4 baseline forecasting models (naive mean, naive last, linear trend, seasonal naive)
- MAE, RMSE, MAPE, directional accuracy metrics

## [0.3.0] — 2025-12-15

### Added
- Feature engineering layer with 48 training / 45 inference columns
- Daily aggregation with recursive CTE date spine
- Dataset builder producing training/inference Parquet files

## [0.2.0] — 2025-12-10

### Added
- Ingestion layer: Blizzard API client, snapshot management, normalization, CSV import

## [0.1.0] — 2025-12-05

### Added
- Project scaffold: taxonomy (ArchetypeCategory, ArchetypeTag, EventType), Pydantic v2 domain models, SQLite DB layer, pipeline stubs, Typer CLI

## [0.0.1] — 2025-12-01

### Added
- Initial repository setup
