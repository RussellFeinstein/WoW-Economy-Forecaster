# Changelog

All notable changes to the WoW Economy Forecaster.

Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
This project uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
