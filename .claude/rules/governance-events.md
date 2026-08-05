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
- RetentionConfig in config.py; `[retention] raw_snapshot_days=30` in default.toml
- HourlyOrchestrator calls pruner as step 7 after every successful ingest run (non-fatal)
- CLI: list-sources, validate-source-policies, check-source-freshness, prune-snapshots (--days N, --dry-run)

## Seed Events (v0.9.0)
- build-events must run before build-datasets
- event_category_impacts table: no archetype_id FK, uses category string
- 8 event feature columns (see event_features.py)
