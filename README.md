# WoW Economy Forecaster

A **local-first research system** for forecasting World of Warcraft auction house economy behavior.

Uses historical data from **The War Within (TWW)** to learn economy patterns, then applies category/archetype-based transfer learning to generate price, volatility, and sale-velocity forecasts for **Midnight** as it launches and matures.

---

## Project Status

**v0.1.0 — Scaffold** — All data models, taxonomy, schema, pipeline interfaces, and CLI are implemented. Pipeline stages are stubs (not yet implemented). No external API integrations yet.

---

## Architecture Overview

```
wow_forecaster/
├── taxonomy/        # Pure enums: EventType, ArchetypeCategory, ArchetypeTag, etc.
├── models/          # Pydantic v2 domain models (frozen/immutable value objects)
├── db/              # SQLite: connection, schema DDL, migrations, repositories
├── pipeline/        # Abstract base + 6 stub stages (ingest→recommend)
├── utils/           # Logging setup, time helpers, expansion epoch math
└── cli.py           # Typer CLI: init-db, validate-config, import-events, etc.

config/
├── default.toml           # Static defaults (committed)
└── events/tww_events.json # TWW economy event seed data (15 events)

data/
├── raw/snapshots/          # Hourly ingestion snapshots (Parquet, gitignored)
├── raw/events/             # Manual event CSVs
├── processed/features/     # Feature-ready Parquet
├── processed/normalized/   # Normalized observations
├── outputs/forecasts/      # Daily forecast outputs
├── outputs/recommendations/ # Recommendation outputs
├── outputs/model_artifacts/ # Saved model files
├── db/                     # SQLite database files
└── logs/                   # Log files

tests/
├── test_models/     # Pydantic validation tests
├── test_taxonomy/   # Taxonomy integrity tests (CATEGORY_TAG_MAP contract)
├── test_db/         # Schema idempotency, FK enforcement, repository round-trips
└── test_pipeline/   # Abstract base and stub stage tests
```

### Key Design Decisions

| Concern | Choice | Why |
|---|---|---|
| Data models | **Pydantic v2** (frozen) | Runtime validation, immutable value objects, clean serialization |
| Database | **Raw sqlite3** (no ORM) | Single-process local tool; SQL stays explicit |
| CLI | **Typer** | Type-annotation driven, auto-help, built on Click |
| Config | **tomllib + python-dotenv** | TOML for static config, .env for secrets |
| Tests | **pytest** | Standard |

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

```bash
wow-forecaster --help

# Initialize SQLite database with full schema
wow-forecaster init-db [--db-path PATH] [--config PATH]

# Validate and print config values
wow-forecaster validate-config [--config PATH] [--full]

# Import events from JSON seed file (upsert semantics)
wow-forecaster import-events [--file PATH] [--dry-run]

# [STUB] Hourly data refresh (ingest + normalize)
wow-forecaster run-hourly-refresh [--realm SLUG] [--dry-run]

# [STUB] Daily forecast pipeline
wow-forecaster run-daily-forecast [--realm SLUG] [--horizon 7d]

# [STUB] Walk-forward backtest
wow-forecaster backtest --start-date 2024-09-10 --end-date 2024-12-01 [--realm SLUG]
```

---

## Running Tests

```bash
# All tests
pytest

# With coverage
pytest --cov=wow_forecaster --cov-report=term-missing

# Specific test groups
pytest tests/test_taxonomy/     # Taxonomy integrity
pytest tests/test_models/       # Pydantic validation
pytest tests/test_db/           # Schema + repositories
pytest tests/test_pipeline/     # Pipeline interfaces
```

Expected: all tests pass with a clean install.

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

## SQLite Schema

12 tables:

```
item_categories             item hierarchy (slug-based, expansion-aware)
economic_archetypes         behavior-based item groupings
items                       WoW item registry (canonical item IDs)
market_observations_raw     raw AH price snapshots (copper)
market_observations_normalized  gold-converted, z-scored observations
archetype_mappings          TWW → Midnight transfer map
wow_events                  economy-affecting events with time windows
event_archetype_impacts     event × archetype impact direction/magnitude
model_metadata              trained model registry
run_metadata                pipeline execution audit log
forecast_outputs            point forecasts with confidence intervals
recommendation_outputs      buy/sell/hold/avoid actions
```

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

## Pipeline Stages (All Stubs)

| Stage | Class | Status |
|---|---|---|
| Ingest | `IngestStage` | Stub — implement `_fetch_source()` |
| Normalize | `NormalizeStage` | Stub — implement copper→gold, z-score |
| Feature Build | `FeatureBuildStage` | Stub — implement rolling windows, event distances |
| Train | `TrainStage` | Stub — implement model selection and training |
| Forecast | `ForecastStage` | Stub — implement inference and CI generation |
| Recommend | `RecommendStage` | Stub — implement ROI scoring and ranking |

Each stage inherits from `PipelineStage` ABC and writes a `RunMetadata` record (with `config_snapshot`) on every execution.

---

## Next Steps

### Prompt 1 — Implement the Normalize Stage
Implement `NormalizeStage._execute()`:
- Fetch unprocessed raw observations via `MarketObservationRepository.get_unprocessed_raw()`.
- Convert copper → gold (`price_gold = min_buyout_raw / 10_000`).
- Compute rolling z-score within (item_id, realm_slug) windows using a configurable window size.
- Flag outliers where `|z_score| > config.pipeline.outlier_z_threshold`.
- Write `NormalizedMarketObservation` records.
- Mark raw observations as processed.
- Add tests for the normalization math and outlier detection.

### Prompt 2 — TSM/Undermine Data Ingestion
Implement the `IngestStage`:
- Define the file format for TSM export CSV/JSON (document schema).
- Implement `_fetch_source()` to read from local TSM export files.
- Add an adapter interface stub at `wow_forecaster/adapters/undermine.py` for future API integration.
- Write a backfill script that processes a directory of historical export files.
- Include tests with sample fixture data.

### Prompt 3 — Feature Engineering Pipeline
Implement `FeatureBuildStage._execute()`:
- Rolling mean/std/min/max for 7, 14, 30, 90-day windows.
- Linear trend slope over each window.
- Event distance features using `WoWEvent.is_known_at()` for bias-safe event encoding.
- Days-since-expansion-launch from `time_utils.days_since_expansion_launch()`.
- Cross-archetype supply pressure features.
- Write output as Parquet files to `data/processed/features/`.
- Add pytest fixtures with synthetic multi-day price series.

---

## Configuration

All settings in `config/default.toml`. Override locally with:
1. `config/local.toml` (gitignored) — local path overrides
2. `.env` (gitignored) — secrets and env-specific overrides
3. `WOW_FORECASTER_*` environment variables

Key config sections: `[database]`, `[expansions]`, `[realms]`, `[pipeline]`, `[forecast]`, `[backtest]`.

---

## Contributing / Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run linter
ruff check wow_forecaster/ tests/

# Run tests with coverage
pytest --cov=wow_forecaster
```
