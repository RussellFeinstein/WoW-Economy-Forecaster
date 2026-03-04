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

## Key Files
- [wow_forecaster/taxonomy/archetype_taxonomy.py](wow_forecaster/taxonomy/archetype_taxonomy.py) — ArchetypeTag, ArchetypeCategory, CATEGORY_TAG_MAP
- [wow_forecaster/taxonomy/event_taxonomy.py](wow_forecaster/taxonomy/event_taxonomy.py) — EventType (26), EventScope (4), EventSeverity (5)
- [wow_forecaster/models/event.py](wow_forecaster/models/event.py) — WoWEvent with is_known_at() backtest bias guard
- [wow_forecaster/config.py](wow_forecaster/config.py) — AppConfig via load_config()
- [wow_forecaster/db/schema.py](wow_forecaster/db/schema.py) — 18 tables, apply_schema() idempotent
- [wow_forecaster/pipeline/base.py](wow_forecaster/pipeline/base.py) — PipelineStage ABC
- [wow_forecaster/cli.py](wow_forecaster/cli.py) — Typer app (26 commands)
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
- [wow_forecaster/features/daily_agg.py](wow_forecaster/features/daily_agg.py) — recursive CTE date spine; JOINs items.archetype_id (backward-compat with pre-v1.3.4 rows + items with no archetype assignment)
- [wow_forecaster/features/dataset_builder.py](wow_forecaster/features/dataset_builder.py) — orchestrates all steps → training/inference Parquet + JSON manifest
- build-datasets end_date default = date.today()+timedelta(days=1) (captures UTC-midnight observations)

### Backtesting (v0.4.0)
- [wow_forecaster/backtest/evaluator.py](wow_forecaster/backtest/evaluator.py) — run_backtest() fold×series×model loop; leakage-free
- BacktestConfig: horizons_days=[1,3], min_train_rows=14
- DB tables: backtest_runs, backtest_fold_results (migration 0002)

### ML + Recommendations (v0.5.0)
- [wow_forecaster/ml/feature_selector.py](wow_forecaster/ml/feature_selector.py) — TRAINING_FEATURE_COLS (40)
- [wow_forecaster/ml/lgbm_model.py](wow_forecaster/ml/lgbm_model.py) — LightGBMForecaster: fit/predict/save/load; global cross-archetype model
- ForecastHorizon: 1d/7d/28d; TARGET_COL_MAP = {1: 1d, 7: 7d, 28: 28d}
- Score formula: 0.35×opportunity + 0.20×liquidity − 0.20×volatility + 0.15×event_boost − 0.10×uncertainty
- event_boost clamp: [-100, 100] (negative impacts penalize score)
- top_n_per_category deduplication: best-scoring horizon per archetype_id (tie: shorter wins)
- DB migration 0003: adds score, score_components, category_tag to recommendation_outputs

### Monitoring + Orchestration (v0.6.0)
- [wow_forecaster/pipeline/orchestrator.py](wow_forecaster/pipeline/orchestrator.py) — HourlyOrchestrator: 7-step pipeline
- Drift detection: z-score of means per archetype/realm series; outlier rows excluded
- Adaptive CI chain: drift check → uncertainty_mult in drift_check_results → ForecastStage reads it → widens CI
- DB migration 0004: drift_check_results + model_health_snapshots (18 tables total)

### Reporting (v0.7.0)
- CLI commands: report-top-items, report-forecasts, report-volatility, report-drift, report-status
- --export PATH writes flat CSV for Power BI (sc_* score component columns)
- [dashboard/app.py](dashboard/app.py) — 5-tab Streamlit UI (optional dep group)

### Source Governance (v0.8.0)
- [config/sources.toml](config/sources.toml) — blizzard_api, blizzard_news_manual, manual_event_csv (3 policies)
- [wow_forecaster/governance/preflight.py](wow_forecaster/governance/preflight.py) — 3-check preflight before each ingest
- CLI: list-sources, validate-source-policies, check-source-freshness

### Seed Events (v0.9.0)
- build-events must run before build-datasets
- event_category_impacts table: no archetype_id FK, uses category string
- 8 event feature columns (see event_features.py)

### Normalization (v1.1.0)
- Rolling z-score via _fetch_rolling_stats() + _normalize_batch(); falls back to batch stats on cold-start
- config: pipeline.normalize_rolling_days=30
- archetype_id populated via _fetch_archetype_map() since v1.3.4; daily_agg.py JOINs items for backward-compat + unassigned items

### Automation (v1.0.0)
- [wow_forecaster/scheduler.py](wow_forecaster/scheduler.py) — SchedulerDaemon (stdlib only)
- CLI: start-scheduler (foreground daemon)
- [scripts/setup_tasks.bat](scripts/setup_tasks.bat) — one-shot Windows Task Scheduler registration

## What's NOT Implemented Yet
- top_n_per_category V2 (Pareto-frontier, user-profile weighting, blocklist, A/B test support); cross-horizon dedup done in v0.9.1
- Governance: cooldown enforcement not wired — preflight.py has check but orchestrator.py never passes last_call_at
- Governance: prune-snapshots via retention.raw_snapshot_days (field modelled, no CLI/deletion logic)
- Live news ingestion: BlizzardNewsClient.fetch_recent_news() exists but IngestStage._fetch_news() always uses fixture mode
- News-to-event: extract_wow_events() not implemented (news items → WoWEvent candidates)

## Known Bugs (unfixed)
- Note: `except Exception` does NOT catch KeyboardInterrupt/SystemExit (those are BaseException subclasses). The global standard pattern `except (KeyboardInterrupt, SystemExit): raise` is redundant here — signals always propagate through `except Exception:` automatically.

## Test Count
866 tests passing
