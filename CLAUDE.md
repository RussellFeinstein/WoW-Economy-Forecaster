# WoW Economy Forecaster — Project Instructions

## Project Overview
Local-first WoW AH economy research system. TWW historical data → Midnight transfer learning.
Category/archetype-based transfer (NOT item-to-item). Python, SQLite, Parquet, Typer CLI.

## Stack
- Python 3.11+, Pydantic v2 (frozen models), raw sqlite3, Typer CLI
- tomllib (stdlib) + python-dotenv for config
- pyarrow for Parquet (no pandas at scaffold stage)
- pytest for tests
- Entry point: `wow-forecaster` CLI (alias: `wowfc`)

## Virtual Environments
**Always use virtual environments.** Never install packages globally.
```bash
python -m venv .venv
.venv/Scripts/activate  # Windows
pip install -e ".[dev]"
```

## Credentials (.env, gitignored)
```
BLIZZARD_CLIENT_ID=...
BLIZZARD_CLIENT_SECRET=...
```

## Branch Workflow
- main is the only permanent branch. A branch protection ruleset (main-pr-only, no bypass actors) requires a pull request for every merge and blocks direct pushes, force pushes, and deletion. This applies to admins too.
- Every piece of work gets a short-lived type-prefixed branch cut from the latest main: feat/, fix/, improvement/, docs/, chore/, refactor/, test/ plus a short kebab slug, with the issue number when one exists (e.g. fix/44-ci-ruff-drift).
- One issue or one concern per branch. Ship it by opening a PR to main and merging via the PR with a merge commit (`gh pr merge --merge`). CI runs on the PR before merge.
- The head branch is deleted on merge (delete_branch_on_merge is on); delete the local copy with `git branch -d`. The merge commit and PR record are the durable history. Never `git branch -D` unmerged work without explicit instruction.
- Scope check before every commit: if the work does not match the current branch's type and slug, stop and cut the right branch from main.
- No umbrella or long-lived topic branches. feature/portfolio-showcase (v1.9.0-v2.4.1) was the last; merged 2026-07-12 (issue #10), deleted with the freeze-convention retirement (issue #46).

## Versioning (stamp commits)
- Work commits take no version bump. Their CHANGELOG lines accumulate under `## [Unreleased]`.
- A dedicated stamp commit at PR-open bumps pyproject.toml once and moves the [Unreleased] entries under the `## [X.Y.Z] - YYYY-MM-DD` header. One version per PR; PR titles carry the `(vX.Y.Z)` suffix.
- If two open PRs stamp the same number, the later-to-merge PR re-stamps to the next free number during rebase.
- Dependabot dependency-bump PRs are exempt from stamping and CHANGELOG. The pinned tool is dev-only (ruff, per .github/dependabot.yml), so a bump does not change the product and stamping the product version would be misleading. These PRs auto-merge on green CI via .github/workflows/dependabot-automerge.yml (CI is a required status check, so a bump that breaks lint or tests never merges). If a bump surfaces new drift, fix it with a conformance commit pushed onto the Dependabot branch, never a parallel takeover PR.

## Key Files
- [wow_forecaster/taxonomy/archetype_taxonomy.py](wow_forecaster/taxonomy/archetype_taxonomy.py) — ArchetypeTag, ArchetypeCategory, CATEGORY_TAG_MAP
- [wow_forecaster/taxonomy/event_taxonomy.py](wow_forecaster/taxonomy/event_taxonomy.py) — EventType (26), EventScope (4), EventSeverity (5)
- [wow_forecaster/models/event.py](wow_forecaster/models/event.py) — WoWEvent with is_known_at() backtest bias guard
- [wow_forecaster/config.py](wow_forecaster/config.py) — AppConfig via load_config()
- [wow_forecaster/db/schema.py](wow_forecaster/db/schema.py) — 21 tables, apply_schema() idempotent
- [wow_forecaster/pipeline/base.py](wow_forecaster/pipeline/base.py) — PipelineStage ABC
- [wow_forecaster/cli.py](wow_forecaster/cli.py) — Typer app (40 commands)
- [config/default.toml](config/default.toml) — static config
- [config/sources.toml](config/sources.toml) — 3 source policies
- [config/events/tww_events.json](config/events/tww_events.json) — TWW seed events
- [config/events/tww_event_impacts.json](config/events/tww_event_impacts.json) — 56 category-level impact records

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

## Snapshot Layout (disk)
```
data/raw/snapshots/
  blizzard_api/YYYY/MM/DD/realm_{realm}_{ts}Z.json
  blizzard_news/YYYY/MM/DD/news_{ts}Z.json
```
Each file: `{"_meta": {..., "written_at": "..."}, "data": [...]}`

## Layer Summary

### Ingestion (v0.2.0 / v0.8.1)
- [wow_forecaster/ingestion/blizzard_client.py](wow_forecaster/ingestion/blizzard_client.py) — LIVE OAuth2 + fetch_commodities/connected_realm
- [wow_forecaster/ingestion/snapshot.py](wow_forecaster/ingestion/snapshot.py) — build_snapshot_path, save_snapshot, load_snapshot
- [wow_forecaster/ingestion/item_bootstrapper.py](wow_forecaster/ingestion/item_bootstrapper.py) — seeds 9,950 items from Blizzard Item API
- _parse_blizzard_records: faction="neutral"; min_buyout_raw = unit_price>0 else buyout>0 else None; num_auctions=1
- ItemRepository.get_all_item_ids() → set[int] FK guard

### Feature Engineering (v0.3.0 / v0.9.0)
- [wow_forecaster/features/registry.py](wow_forecaster/features/registry.py) — 48 training / 45 inference cols
- [wow_forecaster/features/daily_agg.py](wow_forecaster/features/daily_agg.py) — Python-generated date spine over rollup fast path (recursive CTE replaced in v2.3.x); spine clamps to [data_min, data_max]; JOINs items.archetype_id (backward-compat with pre-v1.3.4 rows + items with no archetype assignment)
- [wow_forecaster/features/dataset_builder.py](wow_forecaster/features/dataset_builder.py) — orchestrates all steps → training/inference Parquet + JSON manifest
- build-datasets end_date default = date.today()+timedelta(days=1) (captures UTC-midnight observations)

### Backtesting (v0.4.0)
- [wow_forecaster/backtest/evaluator.py](wow_forecaster/backtest/evaluator.py) — run_backtest() fold×series×model loop; leakage-free
- BacktestConfig: horizons_days=[1,3], min_train_rows=14
- DB tables: backtest_runs, backtest_fold_results (migration 0002)

### ML + Recommendations (v0.5.0 / v1.10.0 / v1.11.0 / v1.12.0 / v2.0.0)
- [wow_forecaster/ml/feature_selector.py](wow_forecaster/ml/feature_selector.py) — TRAINING_FEATURE_COLS (40)
- [wow_forecaster/ml/lgbm_model.py](wow_forecaster/ml/lgbm_model.py) — LightGBMForecaster: fit/predict/save/load; global cross-archetype model
- ForecastHorizon: 1d/7d/28d; TARGET_COL_MAP = {1: 1d, 7: 7d, 28: 28d}
- Score formula: 0.35×opportunity + 0.20×liquidity − 0.20×volatility + 0.15×event_boost − 0.10×uncertainty
- event_boost clamp: [-100, 100] (negative impacts penalize score)
- top_n_per_category deduplication: best-scoring horizon per archetype_id (tie: shorter wins)
- DB migration 0003: adds score, score_components, category_tag to recommendation_outputs
- Risk levels (v1.10.0): determine_risk_level() in scorer.py — LOW/MEDIUM/HIGH/CRITICAL tiers independent of action; AVOID only at CRITICAL (uncertainty ≥ 95%); risk_level persisted in recommendation_outputs (migration 0006)
- CI floor/cap (v1.11.0): compute_confidence_interval() in cold_start.py accepts current_price; floor = 5% of current, cap = 10× current; prevents 0.0 lower bounds and absurd upper bounds; ci_quality field ("good"/"wide"/"unreliable") on ForecastOutput (migration 0007)
- Item-level forecasting extended (v1.12.0): _generate_item_forecasts() now covers union of recipe-linked items AND all items with ≥14 distinct observation days; ItemForecastRoi dataclass + fetch_item_rois() in item_overlay.py; enrich_with_top_item_rois() in ranker.py; top_items column in recommendations CSV/JSON prefers ROI-based items over discount-based fallback
- TSM export (v2.0.0): export-tsm CLI command; wow_forecaster/reporting/tsm_export.py; TsmExportRow + fetch_tsm_export_items() + build_tsm_import_string() + write_tsm_export(); filters item-level forecasts by ROI >= min_roi_pct and ci_quality='good'; outputs i:XXXXX,... string for TradeSkillMaster paste import

### Monitoring + Orchestration (v0.6.0)
- [wow_forecaster/pipeline/orchestrator.py](wow_forecaster/pipeline/orchestrator.py) — HourlyOrchestrator: 7-step pipeline
- Drift detection: z-score of means per archetype/realm series; outlier rows excluded
- Adaptive CI chain: drift check → uncertainty_mult in drift_check_results → ForecastStage reads it → widens CI
- DB migration 0004: drift_check_results + model_health_snapshots (18 tables + migration 0005 adds 3 more = 21 total)
- Rollup step UTC anchor (v2.7.5, issue #61): step 3.5 anchors on an injectable UTC clock (run(now=...), pruner-style None default) and upserts BOTH the previous and current UTC dates per realm; the previous-date upsert is load-bearing (observations after the last pre-UTC-midnight run only arrive in post-midnight runs, so it is what completes each UTC day's tail) (previously date.today() local: self-healed daily but rollups lagged up to ~4h intraday and a machine down past local midnight silently lost the tail). Both rollup UPSERTs still use a DATE(observed_at)=? predicate (table scan; follow-up issue filed at #61 close-out for the #59-style range rewrite)

### Reporting (v0.7.0 / v2.1.0)
- CLI commands: report-top-items, report-forecasts, report-volatility, report-drift, report-status, check-data-health
- --export PATH writes flat CSV for Power BI (sc_* score component columns)
- [dashboard/app.py](dashboard/app.py) — 5-tab Streamlit UI (optional dep group)
- check-data-health (v2.1.0): wow_forecaster/reporting/health.py; collect_health_report() + format_health_report(); DB-backed gap detection — days of coverage, calendar-date gaps, last ingest age; exits 1 when stale; fix: run-hourly-refresh now exits 1 on "failed" status
- Freshness gate (v2.5.0, issue #12): ForecastStage raises StaleDataError when the newest normalized observation exceeds config.forecast.max_data_age_hours (default 26h; <=0 disables; injectable now param for tests); run_daily.bat runs check-data-health --stale-hours 26 as step 1/3 and exits non-zero on HEALTH ALERT (WOWFC env var = test seam, tests/test_scripts/test_run_daily.py); dashboard shows a red ingestion banner above the tab bar via load_ingest_age_hours()
- Scheduled health check (v2.6.0, issue #4): scripts/run_healthcheck.bat runs check-data-health --stale-hours 4 -> logs/health.log; on failure writes data/outputs/monitoring/health_alert.json (raised_at, exit_code, 20-line log tail; [IO.File]::ReadAllLines not Get-Content because PS 5.1 provider output carries PSPath/ReadCount ETS properties that ConvertTo-Json serializes as objects) and raises a cmd /k alert window at most once per 24h (mtime of health_window_raised.json; unverifiable flag raises anyway, same bias as the stale-lock takeover); healthy run deletes both files (episode reset); exit code always mirrors check-data-health; test seams WOWFC + WOWFC_NO_ALERT_WINDOW (tests/test_scripts/test_run_healthcheck.py); both JSONs covered by the data/outputs/**/*.json gitignore rule; registered by setup_tasks.bat as WoWForecaster-HealthCheck since v2.7.1 (#6)
- Lock + retention checks (v2.7.0, issue #5): collect_health_report() gains lock_path (read-only stat of data/db/.hourly.lock; stale past lock_stale_minutes=180, must match run_hourly.bat STALE_MINUTES) and retention_days (sentinel: MIN(observed_at) in market_observations_raw — observed_at because that is the pruner's deletion criterion — violation past retention_days + RETENTION_GRACE_DAYS=2); HealthReport.has_failures property = is_stale or lock_is_stale or retention_violation drives the CLI exit code; formatter status is three-way ([STALE] / [UNHEALTHY] for non-staleness failures / [HEALTHY]); run_daily.bat + run_healthcheck.bat inherit the new failure modes via exit code with zero bat changes; MIN(observed_at) scanned the full table until v2.7.4 (#59) indexed it
- Health-check query indexes (v2.7.4, issue #59): migration 0009 adds idx_obs_raw_observed + idx_obs_raw_realm_ingested on market_observations_raw (serve the retention sentinel and last-ingest check; the pruner's observed_at range deletes inherit the first); health.py date-range query split into two single-aggregate MIN/MAX queries (SQLite's one-probe min/max optimization needs a bare column and exactly one aggregate per query) and the coverage predicate compares raw observed_at to the bare-date cutoff instead of DATE(observed_at) so idx_obs_norm_realm_outlier_time serves the range (a "YYYY-MM-DD" cutoff sorts before every timestamp on that date, so the row set is unchanged); EXPLAIN QUERY PLAN index use pinned in tests/test_db/test_migration_0009.py; remaining DATE(observed_at) sites in forecast.py / crafting_advisor.py / margin_calculator.py are out of scope (noted on #59)
- Backup freshness check (v2.9.0, issue #80): collect_health_report() gains backup_dir + backup_stale_hours (opt-in like lock_path: None skips); newest data/outputs/backups/durable/durable_*.db.gz stat'd by mtime, backup_is_stale when older than the threshold OR none present; backup_is_stale joins has_failures; check-data-health --backup-stale-hours N (0=off default), run_healthcheck.bat passes 30 while run_daily.bat passes nothing, so a stale backup surfaces via the health alert window but never blocks the forecast gate (see Durable Backup layer)

### Source Governance (v0.8.0 / v1.9.0)
- [config/sources.toml](config/sources.toml) — blizzard_api, blizzard_news_manual, manual_event_csv (3 policies)
- [wow_forecaster/governance/preflight.py](wow_forecaster/governance/preflight.py) — 3-check preflight before each ingest
- [wow_forecaster/governance/pruner.py](wow_forecaster/governance/pruner.py) — SnapshotPruner: deletes raw JSON + market_observations_raw rows > retention_days (API ToS §2.r)
- RetentionConfig in config.py; `[retention] raw_snapshot_days=30` in default.toml
- HourlyOrchestrator calls pruner as step 7 after every successful ingest run (non-fatal)
- CLI: list-sources, validate-source-policies, check-source-freshness, prune-snapshots (--days N, --dry-run)

### Seed Events (v0.9.0)
- build-events must run before build-datasets
- event_category_impacts table: no archetype_id FK, uses category string
- 8 event feature columns (see event_features.py)

### Normalization (v1.1.0)
- Rolling z-score via _fetch_rolling_stats() + _normalize_batch(); falls back to batch stats on cold-start
- config: pipeline.normalize_rolling_days=30
- archetype_id populated via _fetch_archetype_map() since v1.3.4; daily_agg.py JOINs items for backward-compat + unassigned items

### Recipes + Crafting Advisor (v1.5.0)
- [wow_forecaster/recipes/blizzard_recipe_client.py](wow_forecaster/recipes/blizzard_recipe_client.py) — fetch_all_recipes_for_profession(); NormalisedRecipe/NormalisedReagent; required reagents only
- [wow_forecaster/recipes/recipe_seeder.py](wow_forecaster/recipes/recipe_seeder.py) — RecipeSeeder: seed(expansion_slug, professions) → upserts recipes + reagents; rate-limited
- [wow_forecaster/recipes/recipe_repo.py](wow_forecaster/recipes/recipe_repo.py) — RecipeRepository: upsert_recipe/replace_reagents/get_recipes_by_expansion etc.
- [wow_forecaster/recipes/margin_calculator.py](wow_forecaster/recipes/margin_calculator.py) — MarginCalculator.compute_margins(): daily craft cost vs output price → crafting_margin_snapshots
- [wow_forecaster/recommendations/crafting_advisor.py](wow_forecaster/recommendations/crafting_advisor.py) — CraftingWindow(6 windows), build_crafting_opportunities(), rank_crafting_opportunities()
- CraftingWindow: NOW_NOW, NOW_7D, NOW_28D, _7D_7D, _7D_28D, _28D_28D — all (buy≤sell) pairs using 1d/7d/28d forecasts
- Future window price projection (v1.5.7+): trend-ratio scaling — item_forecast = item_current × (archetype_forecast / archetype_rolling_current); preserves intra-archetype item price differentiation; falls back to raw archetype forecast then current price
- Item-level forecasts (v1.6.0 / v1.12.0): ForecastStage._generate_item_forecasts() writes item_id-keyed rows to forecast_outputs (item_id set, archetype_id=None); v1.6.0: recipe-linked items only; v1.12.0: extended to union of recipe items + any item with ≥14 distinct observation days; crafting_advisor._fetch_item_forecasts() prefers these over archetype-level forecasts (priority: item forecast → trend-ratio → archetype forecast → current price)
- forecast_outputs.item_id was previously always NULL; now populated for recipe items after each run-daily-forecast
- Cold-start prediction blending (v1.7.0): ForecastStage._execute() calls _fetch_cold_start_blend_data() to build (source_price, confidence) pairs from archetype_mappings; run_inference() calls cold_start.blend_cold_start_prediction() BEFORE CI computation; blended = confidence × model_pred + (1-confidence) × source_price; model_slug gets _transfer suffix for blended archetypes
- Volume gate: hard filter (quantity_sum_7d < min_volume_units=50 excluded) + volume_score = clamp(qty/500, 0, 1)
- opportunity_score = best_window_margin_pct × volume_score
- Compression/expansion: linear regression slope of margin_pct over last N days; ±0.02/day thresholds
- DB migration 0005: recipes, recipe_reagents, crafting_margin_snapshots (UNIQUE recipe_id+realm+obs_date)
- CLI: seed-recipes (--expansion default=transfer_target, --all), build-margins (--realm, --days), report-crafting (--realm, --top-n, --export), report-recipe-status (--expansion)
- seed-recipes --expansion defaults to transfer_target config value ("midnight"); use --all for first-time full seed

### Automation (v1.0.0 / v2.7.2)
- [wow_forecaster/scheduler.py](wow_forecaster/scheduler.py) — SchedulerDaemon (stdlib only)
- CLI: start-scheduler (foreground daemon)
- [scripts/setup_tasks.bat](scripts/setup_tasks.bat) — registers four tasks, all silent via wscript.exe + run_silent.vbs (vbs waits and propagates the exit code): WoWForecaster-Hourly (hourly, /ST pinned to :16 — daily-collision avoidance + snapshot-edge sampling; also matches the cloud capture's :16 Worker dispatch each hour, #83), WoWForecaster-Daily (07:00), WoWForecaster-HealthCheck (/SC HOURLY /MO 6 /ST 00:45 — 29 min clear of the :16 ingest, done before 07:00), WoWForecaster-Backup (/SC DAILY /ST 07:30 — after the daily forecast so each backup includes that morning's forecasts; issue #80)
- setup_tasks.bat state preservation (v2.7.1, issue #6): schtasks /Create /F recreates tasks ENABLED, so each registration queries the prior state (PS Get-ScheduledTask via `call powershell`, exit 2 = Disabled) and re-disables right after /Create; `if errorlevel 1` threshold = disable-on-uncertainty (call on an unresolvable name yields 1, not 9009); a failed re-disable exits 1 with a disable-manually error; WOWFC_SCHTASKS test seam (tests/test_scripts/test_setup_tasks.py, powershell stubbed via PATH override)
- Wake-to-run (v2.7.2, issue #40): schtasks /Create cannot set WakeToRun, so after each registration setup_tasks.bat flips it by PS fetch-modify-write (`Get-ScheduledTask` -> `.Settings.WakeToRun = $true` -> `Set-ScheduledTask -InputObject`; NEVER New-ScheduledTaskSettingsSet, which resets all settings to defaults including Enabled=true); for a was-disabled task the script re-asserts /Change /DISABLE after the wake-set (belt and braces around the cmdlet round-trip); wake-set failure is fatal (exit 1) and runs after the re-disable so the failure exit leaves the task Disabled; warn-only RTCWAKE power-plan check via WOWFC_POWERCFG seam (AC index must be 0x1; 0x0 Disable and 0x2 Important-only both block schtasks wake timers); machine may sleep between runs, but a powered-off machine does not wake (cloud capture covers that)

### Durable Backup (v2.9.0, issue #80)
- [wow_forecaster/backup/durable_backup.py](wow_forecaster/backup/durable_backup.py) — build_durable_db copies schema from the SOURCE's live sqlite_master (NOT apply_schema: recommendation_outputs carries migration-added columns score/score_components/category_tag/risk_level that schema.py's DDL omits, so INSERT SELECT * into an apply_schema() table would fail on column count; ALTER TABLE ADD COLUMN updates sqlite_master.sql, so it is the true shape); copies data for every table except EXCLUDED_TABLES (the two obs tables, recreated empty so the file is drop-in restorable); reads only the small tables + CREATE text, never scans the 9.7 GB obs tables; no VACUUM; foreign_key_check must pass. run_backup(): build -> gzip(-9) -> prune_local(keep_local) -> upload LAST (so a failed upload still leaves a local .db.gz). upload_backup reuses cloud_fetch._retry, explicit BACKUP_S3_* creds (own scoped token, separate bucket). Key: db_backups/YYYY/MM/DD/durable_<ts>Z.db.gz
- CLI: backup-durable-db (--output-dir/--upload/--no-upload/--keep-local); [backup] config (BackupConfig: output_dir, keep_local=7, upload_enabled, stale_hours=30); BACKUP_S3_* in .env (endpoint/bucket/access_key_id/secret_access_key, region default auto); boto3 from [cloud] extra
- Real snapshot 2026-07-23: 118 MB raw / 31 MB gz, 24 tables, ~718K rows, <6s; forecast_outputs (325K) + daily_rollup_item (333K) dominate, so ~31 MB/day and ~11 GB/yr unbounded (exceeds R2 free tier in <1yr). Each backup is a COMPLETE snapshot, so a bounded bucket lifecycle (100d recommended, ~3 GB) loses no rollups; design + restore steps + retention note: [docs/db-backup.md](docs/db-backup.md)
- Alerting: check-data-health --backup-stale-hours N (opt-in, 0=off default) flags newest .db.gz older than N (or none present) via collect_health_report(backup_dir=...); backup_is_stale joins HealthReport.has_failures; run_healthcheck.bat passes 30; run_daily.bat does NOT, so a stale backup never blocks the forecast freshness gate

### Cloud Capture (v2.4.0, M0.5)
- [wow_forecaster/ingestion/cloud_fetch.py](wow_forecaster/ingestion/cloud_fetch.py) - hourly commodities capture on GitHub Actions (issue #42); reuses BlizzardClient + build_snapshot_path + save_snapshot so cloud objects carry the identical local envelope; gzip -9 (~59 MB raw -> ~2.2 MiB); run via `python -m wow_forecaster.ingestion.cloud_fetch`, env-only config (no dotenv)
- [.github/workflows/cloud-snapshot.yml](.github/workflows/cloud-snapshot.yml) - live since 2026-07-20; installs `pip install --no-deps .` + httpx + boto3, so the cloud_fetch import chain must stay stdlib-light (httpx/boto3 lazy)
- Trigger model (v2.9.1, issue #83): GitHub delivers only ~11 of 24 scheduled firings/day for this repo and cron density does not change it (the cap is on run delivery, not schedule expressions; #67 densification to :16/:36/:56 was disproven and reverted). Primary trigger is a Cloudflare Worker cron ([cloud-trigger/](cloud-trigger/)) POSTing workflow_dispatch at :16/:46 via a fine-grained PAT (GH_PAT Worker secret, Actions read+write, this repo only); dispatch runs bypass the schedule backlog. The yml schedule is thinned to a single :06 fallback that doubles as the dead-man alarm: Worker/token death drops capture to fallback-only, the gap guard falls below 20 distinct hours, runs go red. Guard floor stays 20 on purpose (a floor the failure mode can satisfy hides the failure)
- Bucket keys mirror local layout: `blizzard_api/YYYY/MM/DD/commodities_us_<ts>Z.json.gz`; private R2 bucket, 30-day lifecycle rule = ToS 2.r enforcement
- Exit codes: 0 ok, 1 fetch/sanity/upload failure, 2 missing env (named, never values), 3 uploaded but trailing-24h gap guard tripped (<20 distinct capture hours with history present; bootstrap passes; listing spans three day-prefixes per #68)
- Sanity floor: refuses snapshots <50K records (healthy ~314K); design record + activation checklist (secrets are added by hand, never by agents): [docs/cloud-capture.md](docs/cloud-capture.md)
- Activated 2026-07-20 (bucket + lifecycle + 6 repo secrets in place, workflow enabled)

### Cloud Catch-up Ingestion (issue #43)
- [wow_forecaster/ingestion/cloud_sync.py](wow_forecaster/ingestion/cloud_sync.py) — listing, download, selection, write lock. NO database code: `select_objects_to_ingest()` is a pure function over key names so ordering/dedup are testable without S3 or SQLite. Reuses `cloud_fetch.parse_key_timestamp` + `_retry`; `local_path_for_key()` is the exact inverse of `cloud_fetch.build_object_key`
- [wow_forecaster/pipeline/sync_stage.py](wow_forecaster/pipeline/sync_stage.py) — `SyncSnapshotsStage` (three-phase connections, per-object try/except) + `sync_snapshots()` entry point mirroring `durable_backup.run_backup`
- Selection order (each rule load-bearing, see docs/cloud-capture.md): unparseable -> beyond retention -> already ingested (by snapshot path) -> UTC hour already covered -> one per hour (earliest) -> oldest first -> cap at max_objects_per_run
- **The hour rule is what prevents double-counting**: `fetched_at` is client-side `datetime.now(UTC)` (blizzard_client.py:296), NOT the AH snapshot's own mtime, so the local :16 run and the Worker's :16 dispatch record the same underlying snapshot seconds apart and nothing else dedupes them
- Objects are written to disk VERBATIM (raw gunzipped bytes), not re-serialized, so the cloud `_meta` block survives as provenance; `content_hash` via `compute_hash(envelope)` reproduces what cloud_fetch stored
- `parse_blizzard_records()` extracted to module level in pipeline/ingest.py (method kept as delegate) so both paths share one implementation
- Holds `data/db/.hourly.lock` for the write phase (bulk inserts exceed the 30s busy timeout). Mirrors run_hourly.bat's 180-minute stale takeover but WAITS then fails loudly instead of skipping: a skipped catch-up loses a whole night, and exit-0 skips are what hid the 96-day outage
- Failed objects are never recorded in `ingestion_snapshots`, so the next run retries them; CLI exits 1 when any object failed
- New queries: `MarketObservationRepository.get_covered_hours()` (bare `observed_at` range so `idx_obs_raw_observed` seeks; `substr()` in the SELECT list only) and `IngestionSnapshotRepository.get_ingested_paths_since()` (success = 1 only)
- `VALID_PIPELINE_STAGES` in models/meta.py gained `sync_snapshots`
- CLI: `sync-snapshots` (--since YYYY-MM-DD, --dry-run, --limit N, 0 = no cap); `[cloud_sync]` config; `SNAPSHOT_S3_*` in .env (read-only token, separate from BACKUP_S3_*)

### Visualization & Portfolio (v2.2.0)
- [wow_forecaster/viz/](wow_forecaster/viz/) — publication-quality chart layer (matplotlib/seaborn/Plotly)
- [wow_forecaster/viz/theme.py](wow_forecaster/viz/theme.py) — WoW dark palette, apply_wow_theme(), get_plotly_template()
- [wow_forecaster/viz/data_queries.py](wow_forecaster/viz/data_queries.py) — SQL/file -> pandas DataFrame fetchers
- [wow_forecaster/viz/charts/](wow_forecaster/viz/charts/) — 6 chart modules: forecast, backtest, feature, recommendation, drift, transfer
- [wow_forecaster/reporting/bi_export.py](wow_forecaster/reporting/bi_export.py) — Star-schema dim/fact table exports for Power BI / Tableau
- CLI: generate-charts (--chart-type, --format png|svg|both), export-bi-bundle (--format csv|parquet)
- Optional dep group: `[viz]` (matplotlib, seaborn, plotly, kaleido, pandas); `[dashboard]` now depends on `[viz]`
- Dashboard upgraded to 8 tabs (added Backtest, Feature Insights, Crafting); Plotly interactive forecast chart
- 3 Jupyter analysis notebooks in notebooks/ (EDA, Model Development, Backtest Evaluation)
- GitHub Actions CI workflow (.github/workflows/ci.yml)

## Roadmap
Next-phase work (M0 restore/harden ops -> M0.5 unattended capture -> M1 model validation -> M2 paper-trading P&L + ranking A/B -> M3 PostgreSQL+dbt warehouse -> M4 Power BI/Tableau -> M5 event impact study -> M6 publish) is tracked in [docs/ROADMAP.md](docs/ROADMAP.md) and GitHub milestones M0-M6 plus M0.5 (issues #1-#61). Session protocol: follow the Work order section in docs/ROADMAP.md (milestone numbers match it; within a milestone use its issue sequence, not raw issue numbers). Each milestone description on GitHub opens with a numbered work-order list rendered from ROADMAP.md; when filing, closing, or reordering an issue, update that milestone's list in the same session (a stale list is a doc bug, same as any Documentation Sync miss). When remaining issues are waiting on wall clock (#11, #33), advance to the next milestone and circle back.

## Operational state (hazard retired 2026-07-21)
- **Ingestion restored 2026-07-21 02:43Z after 105 days dead** (leaked `data/db/.hourly.lock` from a 2026-04-15 crash). The issue #1 runbook executed in full on 2026-07-20/21: rollup backfill (coverage 22 -> 34 dates, all certified against independent sources after two hardware-induced corruption events), evidence captured to `data/outputs/backups/evidence_2026-07-20/`, both observation tables dropped and rebuilt (DB 78 GB -> 105 MB via VACUUM INTO; the known corrupt raw page never copied), lock deleted, first run green, all three scheduled tasks re-enabled and observed green (hourly every hour, health 06:45, daily 07:00 with forecasts + recommendations). Close-out record on issue #1.
- Data gap 2026-04-08..2026-07-20 is permanent locally (Blizzard serves current snapshots only); cloud capture (#42) has been collecting hourly to R2 since 2026-07-20 21:02Z and #43 catch-up ingestion (`sync-snapshots`) drains it, dormant until the read-only SNAPSHOT_S3_* token is added to .env on rex-desktop. Drift detection rebuilds its baseline over ~30 days; item-level forecasts return ~2026-08-03 (14 fresh days); #11 tracks the verification window.
- Machine caution (rex-desktop): systemic instability under sustained multi-GB load; after any large index build / VACUUM / bulk copy on this box, cross-verify outputs against independent sources before trusting them (two corruption events during the runbook, one after a clean mdsched pass).
- Migrations end at 0009 (health-check indexes); new migrations start at 0010.

## What's NOT Implemented Yet
- top_n_per_category V2 (Pareto-frontier, user-profile weighting, blocklist, A/B test support); cross-horizon dedup done in v0.9.1
- Governance: cooldown enforcement not wired — preflight.py has check but orchestrator.py never passes last_call_at
- Live news ingestion: BlizzardNewsClient.fetch_recent_news() exists but IngestStage._fetch_news() always uses fixture mode
- News-to-event: extract_wow_events() not implemented (news items → WoWEvent candidates)

## Known Bugs (unfixed)
- Note: `except Exception` does NOT catch KeyboardInterrupt/SystemExit (those are BaseException subclasses). The global standard pattern `except (KeyboardInterrupt, SystemExit): raise` is redundant here — signals always propagate through `except Exception:` automatically.

## Test Count
All 1480 tests passing locally as of 2026-07-23. Thirty-four of them (5 run_hourly.bat lock-guard tests from issue #3, 4 run_daily.bat freshness-gate tests from issue #12, 8 run_healthcheck.bat alerting tests from issue #4, 13 setup_tasks.bat registration + wake-to-run + backup-task tests from issues #6/#40/#80, and 4 run_backup.bat tests from issue #80, all in tests/test_scripts/) are Windows-only and skip on the Linux CI runners, so CI reports 1446 passed + 34 skipped, green on Python 3.11 and 3.12. Issue #80 also adds 15 durable-backup tests (tests/test_backup/) and 9 backup-freshness health-check tests (tests/test_reporting/test_health.py), all cross-platform. Issue #43 adds 72 cross-platform tests: 47 in tests/test_ingestion/test_cloud_sync.py, 14 in tests/test_db/test_cloud_sync_queries.py, 11 in tests/test_cli/test_sync_snapshots_cli.py.
