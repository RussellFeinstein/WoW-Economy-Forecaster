# WoW Economy Forecaster

A **local-first research system** for forecasting World of Warcraft auction house economy behavior.

Uses historical data from **The War Within (TWW)** to learn economy patterns, then applies category/archetype-based transfer learning to generate price, volatility, and sale-velocity forecasts for **Midnight** as it launches and matures.

---

## Project Status

**v0.7.0 — Reporting + Dashboard Layer**

| Layer | Status |
|---|---|
| v0.1.0 — Scaffold | Models, taxonomy, schema, CLI interfaces |
| v0.2.0 — Ingestion | IngestStage (fixture mode), NormalizeStage, snapshot writer |
| v0.3.0 — Features | 45-column feature Parquet, lag/rolling/event/archetype features |
| v0.4.0 — Backtesting | Walk-forward backtest, baseline models, CSV/DB results |
| v0.5.0 — ML + Recommendations | LightGBM forecaster, scoring formula, buy/sell/hold ranking |
| v0.6.0 — Monitoring | Drift detection, adaptive CI, provenance, hourly orchestration |
| **v0.7.0 — Reporting** | **CLI report commands, export layer, optional Streamlit dashboard** |

---

## Architecture Overview

```
wow_forecaster/
├── taxonomy/        # Pure enums: EventType, ArchetypeCategory, ArchetypeTag
├── models/          # Pydantic v2 domain models (frozen/immutable value objects)
├── db/              # SQLite: connection, schema DDL (17 tables), migrations, repos
├── pipeline/        # 7 stages: ingest, normalize, feature_build, train,
│                    #           forecast, recommend, orchestrator (backtest separate)
├── ingestion/       # Undermine / Blizzard / news clients + snapshot writer
├── features/        # Daily agg, lag/rolling, event features, archetype features,
│                    # dataset builder → 45-col training Parquet
├── backtest/        # Walk-forward splits, baseline models, metrics, reporter
├── ml/              # LightGBM: feature selector, trainer, predictor, cold-start CI
├── recommendations/ # Scorer (5-component formula), ranker, reporter (CSV/JSON)
├── monitoring/      # Drift detection, adaptive policy, health, provenance, reporter
├── reporting/       # Reader (file discovery + freshness), formatters (ASCII tables),
│                    # export (flat CSV/JSON for Power BI)
└── cli.py           # Typer CLI: 21 commands

config/
├── default.toml             # Static defaults (committed)
└── events/tww_events.json   # 15 TWW seed events

data/
├── raw/snapshots/            # Hourly ingestion snapshots (JSON, gitignored)
├── raw/events/               # Manual event CSVs
├── processed/features/       # Training + inference Parquet + manifests
├── outputs/forecasts/        # forecast_{realm}_{date}.csv
├── outputs/recommendations/  # recommendations_{realm}_{date}.csv/.json
├── outputs/model_artifacts/  # LightGBM .pkl + .json artifacts
├── outputs/monitoring/       # drift_status, model_health, provenance JSON
├── outputs/backtest/         # Per-horizon backtest CSV + manifest
├── db/                       # wow_forecaster.db (SQLite)
└── logs/                     # forecaster.log

dashboard/                    # Optional Streamlit analysis UI
├── app.py                    # 5-tab dashboard (Top Picks/Forecasts/Volatility/Health/Status)
├── data_loader.py            # Cached file loaders + DB queries
└── requirements.txt          # streamlit + pandas

tests/
├── test_models/       # Pydantic validation (60 tests)
├── test_taxonomy/     # Taxonomy integrity (30 tests)
├── test_db/           # Schema + repositories (26 tests)
├── test_pipeline/     # Pipeline interfaces (11 tests)
├── test_ingestion/    # Snapshots, event CSV, persistence (73 tests)
├── test_features/     # Feature engineering, no-leakage (53 tests)
├── test_backtest/     # Splits, models, metrics (60 tests)
├── test_ml/           # Feature selector, LightGBM (44 tests)
├── test_recommendations/ # Scorer, ranker (67 tests)
├── test_monitoring/   # Drift, adaptive, orchestrator (73 tests)
└── test_reporting/    # Reader, formatters, export (55 tests)
```

### Key Design Decisions

| Concern | Choice | Why |
|---|---|---|
| Data models | **Pydantic v2** (frozen) | Runtime validation, immutable value objects, clean serialization |
| Database | **Raw sqlite3** (no ORM) | Single-process local tool; SQL stays explicit |
| CLI | **Typer** | Type-annotation driven, auto-help, built on Click |
| Config | **tomllib + python-dotenv** | TOML for static config, .env for secrets |
| ML | **LightGBM** | Fast training, handles mixed types, interpretable feature importances |
| Reporting | **CLI-first + optional Streamlit** | Terminal reports work headlessly; Streamlit is zero-cost when not needed |
| Tests | **pytest** | Standard; 552 tests across 11 groups |

### Transfer Learning Architecture

The system does **not** do naive TWW-item → Midnight-item mapping. Instead:

1. **Archetype taxonomy** (`wow_forecaster/taxonomy/archetype_taxonomy.py`) defines 36 economic behavior tags (e.g. `consumable.flask.stat`, `mat.ore.common`).
2. **TWW items** are mapped to these archetypes.
3. **Models train** on archetype-level time series from TWW.
4. **Archetype mappings** (`archetype_mappings` table) formally map each TWW archetype to its Midnight equivalent with a confidence score and required rationale.
5. As **Midnight data accumulates**, item-level models are trained and the archetype fallback gradually phases out.

### Event-Aware Forecasting

`WoWEvent.is_known_at(as_of: datetime)` is the look-ahead bias guard:
- Returns `False` if `announced_at is None` (conservative default).
- Returns `False` if `as_of < announced_at`.
- Forecasts only incorporate events that were publicly known at forecast time.

---

## Setup

### Requirements

- Python 3.11+
- Git

### Install

```bash
git clone https://github.com/yourusername/WoW-Economy-Forecaster.git
cd WoW-Economy-Forecaster

# Create a virtual environment (recommended)
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

# Install core + dev dependencies
pip install -e ".[dev]"
```

### Initialize

```bash
# Copy and optionally edit the env template
cp .env.example .env

# Initialize the database (creates data/db/wow_forecaster.db)
wow-forecaster init-db

# Validate your config
wow-forecaster validate-config

# Import the TWW seed events (15 events)
wow-forecaster import-events
```

---

## CLI Reference

```
wow-forecaster --help
```

### Database & Config

```bash
wow-forecaster init-db             [--db-path PATH] [--config PATH]
wow-forecaster validate-config     [--config PATH] [--full]
wow-forecaster import-events       [--file PATH] [--dry-run]
```

### Pipeline

```bash
# Hourly: ingest AH snapshots, normalize, drift check
wow-forecaster run-hourly-refresh  [--realm SLUG] [--check-drift/--no-check-drift]

# Daily: train -> forecast -> recommend
wow-forecaster run-daily-forecast  [--realm SLUG] [--skip-train] [--skip-recommend]

# Train only
wow-forecaster train-model         [--realm SLUG ...]

# Forecast + recommend separately
wow-forecaster recommend-top-items [--realm SLUG] [--top-n N] [--forecast-run-id ID]
```

### Feature Datasets

```bash
wow-forecaster build-datasets      [--realm SLUG ...] [--start-date DATE] [--end-date DATE]
wow-forecaster validate-datasets   --manifest PATH [--strict]
```

### Backtesting

```bash
wow-forecaster backtest            --start-date DATE --end-date DATE [--realm SLUG] \
                                   [--window-days N] [--step-days N] [--horizons 1,3]
wow-forecaster report-backtest     [--realm SLUG] [--run-id ID] [--horizon N]
```

### Monitoring

```bash
wow-forecaster check-drift         [--realm SLUG] [--output-json/--no-output-json]
wow-forecaster evaluate-live-forecast [--realm SLUG] [--window-days N]
```

### Reporting (v0.7.0)

All report commands read already-written output files — they never re-run the pipeline.
Every report prints a `[FRESH]` or `[STALE]` banner so you can judge data currency at a glance.

```bash
# Top-N buy/sell/hold recommendations per category
wow-forecaster report-top-items    [--realm SLUG] [--horizon 1d|7d|28d] [--export PATH]

# Full forecast summary sorted by score
wow-forecaster report-forecasts    [--realm SLUG] [--horizon HORIZ] [--top-n N] [--export PATH]

# Volatility watchlist: items with widest CI bands (highest uncertainty)
wow-forecaster report-volatility   [--realm SLUG] [--top-n N] [--export PATH]

# Drift level + model health (retrain flag, MAE ratio per horizon)
wow-forecaster report-drift        [--realm SLUG] [--export PATH]

# Source freshness: per-source snapshot counts, success rates, stale flags
wow-forecaster report-status       [--realm SLUG] [--export PATH]
```

**Common options for all report commands:**
- `--freshness-hours N` — threshold for `[STALE]` banner (default 4 h).
- `--export PATH` — write a flat CSV or JSON to *PATH* for Power BI / Excel.

---

## Reporting Layer

### How Reports Work

The `wow_forecaster/reporting/` module has three components:

| Module | Purpose |
|---|---|
| `reader.py` | File discovery (`find_latest_file`), freshness checks (`check_freshness`), loaders for each report type |
| `formatters.py` | ASCII terminal table formatters — no third-party deps |
| `export.py` | Flat CSV/JSON export helpers; `flatten_recommendations_for_export()` expands nested score components into columns |

File discovery finds the **most recently modified** file matching the expected naming pattern, so reports from previous days are never silently used when a newer file exists.

### Freshness vs Staleness

Every formatter prints a provenance banner:

```
  [FRESH] Generated 1.2h ago
  [STALE] Generated 26.4h ago -- data may not reflect current market
  [AGE UNKNOWN] generated_at not available
```

The `report-status` command also distinguishes **report age** (how old the provenance file is) from **data freshness** (how old the underlying market snapshots are). A recently written provenance file can still report stale data if the Undermine API didn't respond during the last run.

### Export Format

`--export PATH.csv` writes flat rows with no nested dicts, ready for Power BI or pandas:

- Recommendations: `realm_slug, generated_at, run_slug, category, rank, archetype_id, horizon, target_date, current_price, predicted_price, ci_lower, ci_upper, roi_pct, score, action, reasoning, sc_opportunity, sc_liquidity, sc_volatility, sc_event_boost, sc_uncertainty, model_slug`
- Forecasts: same as forecast CSV + `ci_width_gold`, `ci_pct_of_price` derived columns
- Drift: raw JSON pass-through (for programmatic use)

### Validating Report Correctness

Cross-check a CLI report against raw data:

```bash
# 1. Run the pipeline to produce fresh output
wow-forecaster run-daily-forecast --realm area-52

# 2. Read the report
wow-forecaster report-top-items --realm area-52

# 3. Export for cross-checking
wow-forecaster report-forecasts --realm area-52 --export /tmp/fc.csv

# 4. Check DB directly (SQLite)
sqlite3 data/db/wow_forecaster.db \
  "SELECT archetype_id, predicted_price_gold, confidence_lower, confidence_upper \
   FROM forecast_outputs ORDER BY created_at DESC LIMIT 10;"
```

The `report-forecasts` table values should match the `forecast_outputs` DB rows for the same run.

---

## Optional Streamlit Dashboard

The `dashboard/` directory contains an optional local analysis UI that reads the same `data/outputs/` files as the CLI commands.

**Why optional:**
- Adds ~100 MB of deps (streamlit, pandas) not needed for headless pipeline runs.
- All views are also available via the `wow-forecaster report-*` CLI commands.
- Keep it off if you only use the system in scheduled/cron mode.

**Setup:**

```bash
pip install -e ".[dashboard]"
# or: pip install -r dashboard/requirements.txt
streamlit run dashboard/app.py

# Override default realm:
streamlit run dashboard/app.py -- --realm area-52
```

**Five tabs:**

| Tab | Content | Data Source |
|---|---|---|
| Top Picks | Ranked recommendations per category, score breakdown, category/horizon filters | `recommendations_{realm}_{date}.json` |
| Forecasts | Full forecast table with CI width; archetype selector with historical price line + forecast overlay | `forecast_{realm}_{date}.csv` + SQLite |
| Volatility | Items sorted by CI width, CI% bar chart | `forecast_{realm}_{date}.csv` |
| Model Health | Drift level, uncertainty multiplier, MAE ratio per horizon, retrain flag | `drift_status_*.json` + `model_health_*.json` |
| Source Status | Per-source snapshot counts, success rates, data freshness vs report age | `provenance_{realm}_{date}.json` |

**Freshness badges:** Every tab shows a green/orange/red badge (`FRESH` / `STALE` / `NO DATA`). The threshold is configurable in the sidebar.

**Forecast vs Actual chart:** The Forecasts tab queries `market_observations_normalized` from SQLite to render an actual price line, then overlays the forecast point with CI bounds. Nearby WoW events from the `wow_events` table are shown as a reference table below the chart.

---

## Running Tests

```bash
# All tests (552 total across 11 groups)
pytest

# With coverage
pytest --cov=wow_forecaster --cov-report=term-missing

# Specific groups
pytest tests/test_reporting/    # Reader, formatters, export (55 tests)
pytest tests/test_monitoring/   # Drift, adaptive, orchestrator (73 tests)
pytest tests/test_recommendations/ # Scorer, ranker (67 tests)
pytest tests/test_ml/           # LightGBM training and inference (44 tests)
pytest tests/test_features/     # Feature engineering (53 tests)
pytest tests/test_backtest/     # Backtest framework (60 tests)
pytest tests/test_models/       # Pydantic validation (60 tests)
pytest tests/test_db/           # Schema + repositories (26 tests)
pytest tests/test_taxonomy/     # Taxonomy integrity (30 tests)
pytest tests/test_ingestion/    # Ingestion + snapshots (73 tests)
pytest tests/test_pipeline/     # Pipeline interfaces (11 tests)
```

---

## Data Models

### Market Observations

| Model | Storage | Description |
|---|---|---|
| `RawMarketObservation` | SQLite + Parquet | Price data as received (copper int) |
| `NormalizedMarketObservation` | SQLite + Parquet | Gold-converted, z-scored, outlier-flagged |

### Taxonomy

| Model | Description |
|---|---|
| `EventType` | 26 event categories (expansion launch → content drought) |
| `EventScope` | 4 scopes (global, region, realm_cluster, faction) |
| `EventSeverity` | 5 severity levels (critical → negligible) |
| `ArchetypeCategory` | 10 top-level economic categories |
| `ArchetypeTag` | 36 behavior-specific tags with category-prefix validation |

### Core Entities

| Model | Key Field | Description |
|---|---|---|
| `WoWEvent` | `announced_at` | Economy-affecting event with look-ahead bias guard |
| `EconomicArchetype` | `transfer_confidence` | Behavior grouping for cross-expansion transfer |
| `ArchetypeMapping` | `mapping_rationale` (required) | Explicit TWW→Midnight transfer map |
| `ForecastOutput` | `confidence_lower/upper` | Point forecast with CI |
| `RecommendationOutput` | `action`, `priority` | buy/sell/hold/avoid with urgency rank |
| `RunMetadata` | `config_snapshot` | Full audit record for reproducibility |

---

## SQLite Schema (17 tables)

```
item_categories               Item hierarchy (slug-based, expansion-aware)
economic_archetypes           36 behavior tags with transfer_confidence
items                         WoW item registry (item_id → archetype_id FK)
market_observations_raw       Raw AH snapshots (copper, pre-normalization)
market_observations_normalized Gold-converted, z-scored, outlier-flagged
archetype_mappings            TWW → Midnight transfer map (rationale required)
wow_events                    Economy events with announced_at bias guard
event_archetype_impacts       Event × archetype impact direction/magnitude/lag
model_metadata                Trained model registry (slug, type, artifact_path)
run_metadata                  Pipeline execution audit (config_snapshot, status)
forecast_outputs              Point forecasts with CI bounds
recommendation_outputs        buy/sell/hold/avoid with score components
ingestion_snapshots           Snapshot metadata (source, hash, record_count)
backtest_runs                 Backtest invocation metadata (window, folds)
backtest_fold_results         Per-prediction backtest results
drift_check_results           Drift detection results (level, uncertainty_mult)
model_health_snapshots        Live MAE vs backtest baseline per horizon
```

---

## Score Formula

Recommendations are ranked by a 5-component composite score:

```
score = 0.35 × opportunity
      + 0.20 × liquidity
      − 0.20 × volatility
      + 0.15 × event_boost
      − 0.10 × uncertainty
```

Actions:
- **buy** — predicted ROI ≥ 10%
- **sell** — predicted ROI ≤ −10%
- **avoid** — CI width > 80% of price OR coefficient of variation > 80%
- **hold** — otherwise

---

## Event Seed Data

`config/events/tww_events.json` contains 15 real TWW events:

| Slug | Type | Severity |
|---|---|---|
| `tww-launch` | expansion_launch | critical |
| `tww-prepatch-110` | expansion_prepatch | major |
| `tww-rtwf-nerubar-s1` | rtwf | major |
| `tww-s1-start` | season_start | major |
| `tww-patch-1105` | minor_patch | minor |
| `tww-s2-start` | season_start | major |
| `tww-patch-111` | major_patch | major |
| `tww-rtwf-liberation-s2` | rtwf | major |
| `tww-darkmoon-faire-recurring` | holiday_event | minor |
| `tww-winter-veil-2024` | holiday_event | minor |
| `tww-brewfest-2024` | holiday_event | minor |
| `tww-content-drought-s1-end` | content_drought | moderate |
| `tww-s1-end` | season_end | moderate |
| `tww-trading-post-2024-09` | trading_post_reset | negligible |
| `tww-blizzcon-2024` | blizzcon | minor |

---

## Pipeline Stages

| Stage | Class | Status |
|---|---|---|
| Ingest | `IngestStage` | Fixture mode (writes snapshots; TODO: parse → DB rows) |
| Normalize | `NormalizeStage` | Copper→gold, batch z-score, outlier flag |
| Feature Build | `FeatureBuildStage` | 45-col Parquet (lag/rolling/event/archetype features) |
| Train | `TrainStage` | LightGBM cross-archetype model per horizon |
| Forecast | `ForecastStage` | Inference + heuristic CI with adaptive multiplier |
| Recommend | `RecommendStage` | 5-component scoring, top-N per category, CSV/JSON output |
| Backtest | `BacktestStage` | Walk-forward with 4 baseline models |
| Orchestrator | `HourlyOrchestrator` | 7-step hourly pipeline with drift + provenance |

---

## What's Not Implemented Yet

- `IngestStage`: snapshot records → `market_observations_raw` (TODO in ingest.py)
- `NormalizeStage`: rolling z-score (batch only) and archetype_id lookup via item join
- Real HTTP calls in all 3 clients (httpx not yet wired)
- `top_n_per_category` V2 refinements (Pareto, de-duplication, user profiles)
- Adaptive CI: `get_latest_uncertainty_multiplier()` not yet wired into `ForecastStage`

---

## Next Steps

1. **Implement IngestStage record parsing** — snapshot JSON → `RawMarketObservation` → `market_observations_raw` (enables real price history in reports and charts)
2. **Add httpx + real UndermineClient** — live AH data flowing into the pipeline
3. **Wire adaptive CI multiplier** — `get_latest_uncertainty_multiplier()` into `ForecastStage` so drift automatically widens live CIs
4. **top_n_per_category V2** — Pareto-frontier, cross-horizon de-duplication

---

## Configuration

All settings in `config/default.toml`. Override locally with:
1. `config/local.toml` (gitignored) — local path overrides
2. `.env` (gitignored) — secrets and env-specific overrides
3. `WOW_FORECASTER_*` environment variables

Key sections: `[database]`, `[expansions]`, `[realms]`, `[pipeline]`, `[forecast]`, `[features]`, `[model]`, `[monitoring]`, `[backtest]`, `[logging]`.

### Credentials (.env)

```
UNDERMINE_API_KEY=...        # enables live Undermine data
BLIZZARD_CLIENT_ID=...       # enables live Blizzard AH data
BLIZZARD_CLIENT_SECRET=...
```

Without credentials the pipeline runs in **fixture mode** (synthetic sample data).

---

## Contributing / Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run linter
ruff check wow_forecaster/ tests/

# Run all tests with coverage
pytest --cov=wow_forecaster --cov-report=term-missing

# Run only reporting tests
pytest tests/test_reporting/ -v
```
