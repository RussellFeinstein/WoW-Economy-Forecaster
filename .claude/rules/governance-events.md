---
paths:
  - "wow_forecaster/governance/**"
  - "wow_forecaster/events/**"
  - "wow_forecaster/taxonomy/event_taxonomy.py"
  - "wow_forecaster/models/event.py"
  - "config/**"
---

# Source governance and events

Loaded when working with source policies, retention, or seed events. Root context: [CLAUDE.md](../../CLAUDE.md).

## Key files
- [wow_forecaster/taxonomy/event_taxonomy.py](../../wow_forecaster/taxonomy/event_taxonomy.py) — EventType, EventScope, EventSeverity
- [wow_forecaster/models/event.py](../../wow_forecaster/models/event.py) — WoWEvent with is_known_at() backtest bias guard
- [config/default.toml](../../config/default.toml) — static config
- [config/sources.toml](../../config/sources.toml) — source policies
- [config/events/tww_events.json](../../config/events/tww_events.json) — TWW seed events
- [config/events/tww_event_impacts.json](../../config/events/tww_event_impacts.json) — category-level impact records

## Source Governance (v0.8.0 / v1.9.0)
- [config/sources.toml](../../config/sources.toml) — blizzard_api, blizzard_news_manual, manual_event_csv (3 policies)
- [wow_forecaster/governance/preflight.py](../../wow_forecaster/governance/preflight.py) — 3-check preflight before each ingest
- [wow_forecaster/governance/pruner.py](../../wow_forecaster/governance/pruner.py) — SnapshotPruner: deletes raw JSON + market_observations_raw rows > retention_days (API ToS §2.r)
- DB pruning walks half-open hour slices with a commit per slice (issue #149), so an interrupted prune keeps finished slices and the next run resumes from the new oldest row. `hour_slices()` is pure over strings: boundaries are bare `YYYY-MM-DDTHH:00:00` so they compare correctly against both stored timestamp shapes (`+00:00` with microseconds, and `Z`), and comparison stays on the raw column so `idx_obs_raw_observed` seeks. Slices come from `observed_at`, never obs_id order, because the catch-up drain inserts old hours late
- `MAX_ROWS_PER_RUN` (1.5M) bounds delete work per run: the cutoff is a calendar date, so the first run after UTC midnight sees a whole day (~6M rows) become prunable at once, which does not fit the hourly budget. Empty hours cost nothing and do not spend it. `MAX_SLICES_PER_RUN` (2160) is only a runaway guard for an absurd oldest timestamp. Both log when hit
- **Migration 0010 is what makes the prune possible at all.** `market_observations_normalized.obs_id` is the FK to raw and had no index, and connections run `PRAGMA foreign_keys = ON`, so every parent delete scanned the whole child table to prove nothing referenced it. The first prune with a real backlog (11.2M rows against 158M children) ran 24 hours without finishing and wedged ingestion for two days
- RetentionConfig in config.py; `[retention] raw_snapshot_days=30` in default.toml
- HourlyOrchestrator calls pruner as step 7 after every successful ingest run (non-fatal)
- CLI: list-sources, validate-source-policies, check-source-freshness, prune-snapshots (--days N, --dry-run)

## Seed Events (v0.9.0)
- build-events must run before build-datasets
- event_category_impacts table: no archetype_id FK, uses category string
- 8 event feature columns (see event_features.py)
