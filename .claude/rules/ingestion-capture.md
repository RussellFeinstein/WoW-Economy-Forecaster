---
paths:
  - "wow_forecaster/ingestion/**"
  - "wow_forecaster/pipeline/ingest.py"
  - "wow_forecaster/pipeline/sync_stage.py"
  - ".github/workflows/cloud-snapshot.yml"
  - "cloud-trigger/**"
---

# Ingestion and capture

Loaded when working with ingestion, cloud capture, or the catch-up drain. Root context: [CLAUDE.md](../../CLAUDE.md).

## Ingestion (v0.2.0 / v0.8.1)
- [wow_forecaster/ingestion/blizzard_client.py](../../wow_forecaster/ingestion/blizzard_client.py) — LIVE OAuth2 + fetch_commodities/connected_realm
- [wow_forecaster/ingestion/snapshot.py](../../wow_forecaster/ingestion/snapshot.py) — build_snapshot_path, save_snapshot, load_snapshot
- [wow_forecaster/ingestion/item_bootstrapper.py](../../wow_forecaster/ingestion/item_bootstrapper.py) — seeds 9,950 items from Blizzard Item API
- _parse_blizzard_records: faction="neutral"; min_buyout_raw = unit_price>0 else buyout>0 else None; num_auctions=1
- ItemRepository.get_all_item_ids() → set[int] FK guard

## Cloud Capture (v2.4.0, M0.5)
- [wow_forecaster/ingestion/cloud_fetch.py](../../wow_forecaster/ingestion/cloud_fetch.py) - hourly commodities capture on GitHub Actions (issue #42); reuses BlizzardClient + build_snapshot_path + save_snapshot so cloud objects carry the identical local envelope; gzip -9 (~59 MB raw -> ~2.2 MiB); run via `python -m wow_forecaster.ingestion.cloud_fetch`, env-only config (no dotenv)
- [.github/workflows/cloud-snapshot.yml](../../.github/workflows/cloud-snapshot.yml) - live since 2026-07-20; installs `pip install --no-deps .` + httpx + boto3, so the cloud_fetch import chain must stay stdlib-light (httpx/boto3 lazy)
- Trigger model (v2.9.1, issue #83): GitHub delivers only ~11 of 24 scheduled firings/day for this repo and cron density does not change it (the cap is on run delivery, not schedule expressions; #67 densification to :16/:36/:56 was disproven and reverted). Primary trigger is a Cloudflare Worker cron ([cloud-trigger/](../../cloud-trigger/)) POSTing workflow_dispatch at :16/:46 via a fine-grained PAT (GH_PAT Worker secret, Actions read+write, this repo only); dispatch runs bypass the schedule backlog. The yml schedule is thinned to a single :06 fallback that doubles as the dead-man alarm: Worker/token death drops capture to fallback-only, the gap guard falls below 20 distinct hours, runs go red. Guard floor stays 20 on purpose (a floor the failure mode can satisfy hides the failure)
- Bucket keys mirror local layout: `blizzard_api/YYYY/MM/DD/commodities_us_<ts>Z.json.gz`; private R2 bucket, 30-day lifecycle rule = ToS 2.r enforcement
- Exit codes: 0 ok, 1 fetch/sanity/upload failure, 2 missing env (named, never values), 3 uploaded but trailing-24h gap guard tripped (<20 distinct capture hours with history present; bootstrap passes; listing spans three day-prefixes per #68)
- Sanity floor: refuses snapshots <50K records (healthy ~314K); design record + activation checklist (secrets are added by hand, never by agents): [docs/cloud-capture.md](../../docs/cloud-capture.md)
- Activated 2026-07-20 (bucket + lifecycle + 6 repo secrets in place, workflow enabled)

## Cloud Catch-up Ingestion (v2.10.0, issue #43)
- [wow_forecaster/ingestion/cloud_sync.py](../../wow_forecaster/ingestion/cloud_sync.py) — listing, download, selection, write lock. NO database code: `select_objects_to_ingest()` is a pure function over key names so ordering/dedup are testable without S3 or SQLite. Reuses `cloud_fetch.parse_key_timestamp` + `_retry`; `local_path_for_key()` is the exact inverse of `cloud_fetch.build_object_key`
- [wow_forecaster/pipeline/sync_stage.py](../../wow_forecaster/pipeline/sync_stage.py) — `SyncSnapshotsStage` (three-phase connections, per-object try/except) + `sync_snapshots()` entry point mirroring `durable_backup.run_backup`
- Selection order (each rule load-bearing, see docs/cloud-capture.md): unparseable -> beyond retention -> already ingested (by snapshot path) -> UTC hour already covered -> one per hour (earliest) -> oldest first -> cap at max_objects_per_run
- **The hour rule is what prevents double-counting**: `fetched_at` is client-side `datetime.now(UTC)` (blizzard_client.py:296), NOT the AH snapshot's own mtime, so the local :16 run and the Worker's :16 dispatch record the same underlying snapshot seconds apart and nothing else dedupes them
- Naive-UTC boundary (v2.11.5, issue #95): `_execute()` normalizes `now` and `--since` to naive UTC at the top (utcnow() is aware; the DB, the CLI parse, and the key-timestamp comparisons are all naive), caught on the first real-bucket run; TestNaiveUtcClockNormalization pins the real clock shapes
- Objects are written to disk VERBATIM (raw gunzipped bytes), not re-serialized, so the cloud `_meta` block survives as provenance; `content_hash` via `compute_hash(envelope)` reproduces what cloud_fetch stored
- `parse_blizzard_records()` extracted to module level in pipeline/ingest.py (method kept as delegate) so both paths share one implementation
- Holds `data/db/.hourly.lock` for the write phase (bulk inserts exceed the 30s busy timeout). Mirrors run_hourly.bat's 180-minute stale takeover but WAITS then fails loudly instead of skipping: a skipped catch-up loses a whole night, and exit-0 skips are what hid the 96-day outage
- Failed objects are never recorded in `ingestion_snapshots`, so the next run retries them; CLI exits 1 when any object failed
- New queries: `MarketObservationRepository.get_covered_hours()` (bare `observed_at` range so `idx_obs_raw_observed` seeks; `substr()` in the SELECT list only) and `IngestionSnapshotRepository.get_ingested_paths_since()` (success = 1 only)
- `VALID_PIPELINE_STAGES` in models/meta.py gained `sync_snapshots`
- CLI: `sync-snapshots` (--since YYYY-MM-DD, --dry-run, --limit N, 0 = no cap); `[cloud_sync]` config; `SNAPSHOT_S3_*` in .env (read-only token, separate from BACKUP_S3_*)
