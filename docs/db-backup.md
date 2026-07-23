# Durable database backup

Status: accepted 2026-07-23. Decision record and activation checklist for issue
[#80](https://github.com/RussellFeinstein/WoW-Economy-Forecaster/issues/80). Part of milestone M0A.

## Problem

The forecaster's durable state lives only in the local SQLite file on one desktop:
the daily rollups (`daily_rollup_archetype`, `daily_rollup_item`), forecasts,
recommendations, backtests, drift and health snapshots, and the reference tables.

That state is not reconstructible after the fact. Once a day's underlying raw
observations age past the 30-day retention window, the pruner deletes them
everywhere (local rows and files, plus the cloud-capture bucket's 30-day
lifecycle), so the rollups become the sole surviving record of that day. The
cloud capture (#42) only protects the raw *input* stream, and its lifecycle rule
expires it at 30 days. So the derived history has no off-machine copy at all.

The durable set is small next to the ~9.7 GB live database (a real snapshot on
2026-07-23 was 118 MB uncompressed, 31 MB gzipped, 24 tables, ~718K rows, built in
under 6 seconds), but it sits on one disk on a machine with a documented failure
history (recurring 0x1A bugchecks, two multi-GB SQLite corruption events during
the #1 runbook). A single corruption event takes the only copy.

## Approach

A nightly `.db.gz`: a fresh SQLite file that is a drop-in restore of the durable
state, written locally and uploaded to a separate object store.

### Format: a restorable database, not an export

The backup copies every table's schema straight from the live database's
`sqlite_master`, then copies data for every table except the two per-observation
tables. Those two are recreated empty, so the file opens as a complete database
you can restore by copying it into place, no re-import step.

Two reasons the schema is copied from `sqlite_master` rather than rebuilt with
`apply_schema()`:

- The live `recommendation_outputs` carries migration-added columns (`score`,
  `score_components`, `category_tag`, `risk_level`) that `schema.py`'s base DDL
  does not declare. A backup built from the base DDL would be a narrower table,
  and `INSERT ... SELECT *` would fail on the column count. `ALTER TABLE ADD
  COLUMN` updates the stored `CREATE TABLE` text, so `sqlite_master` is always
  the true current shape. `bi_export.py` works around the same drift with a
  `PRAGMA table_info` probe.
- Copying the live schema means any future table or migration is captured with no
  code change here.

The build never reads the two multi-GB observation tables, only their `CREATE`
text plus the data of the small durable tables. There is no `VACUUM` (the inserts
already produce a compact file). This keeps the job off the sustained multi-GB
read/write path that has corrupted this machine before.

A CSV export was rejected as the backup format: a CSV has no schema, so a
forgotten column silently drops data, and restore becomes a typed re-import
project at the worst possible time. Human-readable exports already exist via
`export-bi-bundle` for the cases that want them.

### Storage: a separate bucket, its own retention

The `.db.gz` goes to its own private R2 bucket, separate from the transient
snapshots bucket, using its own scoped token (`BACKUP_S3_*`). Keeping it separate
matters because the snapshots bucket's 30-day lifecycle would otherwise delete the
backups it is meant to preserve. Retaining derived aggregates past 30 days is
compliant with Blizzard API ToS 2.r, which governs raw API data only. Excluding
the two observation tables' data also means the backup carries no raw per-listing
data at all.

Sizing (measured 2026-07-23): about 31 MB gzipped per day, growing with the
history, dominated by `forecast_outputs` (~325K rows) and `daily_rollup_item`
(~333K rows). Uploaded daily and kept forever, that is roughly 11 GB per year,
which crosses R2's 10 GB free tier inside a year.

Retention recommendation: each daily backup is a **complete** snapshot of the
durable state, so the newest object always holds the full rollup history. Keeping
only a bounded window of daily backups therefore loses no irreplaceable rollups;
it only gives up the ability to restore to an old point in time. A lifecycle rule
on the backup bucket of 100 days keeps storage near 3 GB while preserving every
rollup through the latest snapshot. The one risk a bounded window adds is a backup
task that stops for longer than the window with the local disk also lost; the
scheduled health check (below) catches a stopped task within hours, well inside
any such window. Set the window to taste, or omit the rule entirely and watch
free-tier usage. `keep_local` bounds the local copies independently (7 by default,
about 220 MB).

### Cadence: its own scheduled task

`WoWForecaster-Backup` runs daily at 07:30, after the 07:00 daily forecast, so
each backup includes that morning's fresh forecasts and recommendations. It is a
separate task with its own exit code, so Task Scheduler's Last Run Result is an
independent backup-health signal. `scripts/run_backup.bat` is the wrapper; it
calls `backup-durable-db --upload` and exits with the CLI's code.

### Alerting

Backup staleness surfaces through the existing scheduled health check rather than
a second alert system. `check-data-health --backup-stale-hours N` (opt-in, off by
default) flags the newest backup when it is older than N hours, and
`run_healthcheck.bat` passes the threshold so a stale backup raises the same alert
window a stale ingest does. The check is off by default so a stale backup never
trips the daily forecast's freshness gate.

## Object layout

```
db_backups/YYYY/MM/DD/durable_<YYYYMMDDTHHMMSS>Z.db.gz
```

The `db_backups/` prefix is distinct from the raw snapshots' `blizzard_api/`
prefix, so a bucket lifecycle rule could target one without expiring the other
(not needed here, since the backup bucket has no lifecycle rule at all).

## Configuration

Non-secret settings live in `config/default.toml` under `[backup]`:

| Key | Default | Meaning |
|---|---|---|
| `output_dir` | `data/outputs/backups/durable` | Local `.db.gz` directory |
| `keep_local` | `7` | Most-recent local backups to keep |
| `upload_enabled` | `true` | Default for `--upload` |
| `stale_hours` | `30.0` | Age used by `--backup-stale-hours` |

Secrets live in `.env` (never in git), read from the environment the same way the
cloud fetcher reads its credentials:

| Variable | Meaning |
|---|---|
| `BACKUP_S3_ENDPOINT` | R2 account endpoint URL |
| `BACKUP_S3_BUCKET` | Backup bucket name |
| `BACKUP_S3_ACCESS_KEY_ID` | Scoped token access key |
| `BACKUP_S3_SECRET_ACCESS_KEY` | Scoped token secret |
| `BACKUP_S3_REGION` | Optional signing region (default `auto`) |

Uploading needs boto3, which is in the `[cloud]` extra: `pip install -e ".[cloud]"`.

## Activation checklist (manual, one time)

Steps for the repo owner; none of them belong in git, and no agent handles the
credential values:

1. Create a Cloudflare R2 bucket (private, default settings), for example
   `wow-forecaster-backups`. Add a lifecycle rule of 100 days (safe, since each
   snapshot is complete; see Retention recommendation above); or omit it to keep
   every backup and watch free-tier usage instead.
2. Create an R2 API token scoped to that bucket with read and write access.
3. Add the five `BACKUP_S3_*` variables above to the desktop `.env`.
4. `pip install -e ".[cloud]"` in the project venv if boto3 is not installed.
5. Verify locally without uploading: `wowfc backup-durable-db --no-upload`, then
   open the `.db.gz` (gunzip, then any SQLite tool) and confirm the durable
   tables hold rows and the observation tables are empty.
6. Register the scheduled task: run `scripts/setup_tasks.bat` (adds
   `WoWForecaster-Backup` alongside the existing three tasks).
7. Trigger once by hand: `schtasks /Run /TN "WoWForecaster-Backup"`, then confirm
   an object appears under `db_backups/` at roughly a few MB.

## Restore

The `.db.gz` is a complete database, so restore is a copy:

1. Download the object from the backup bucket.
2. `gunzip` it to `wow_forecaster.db`.
3. Put it at `data/db/wow_forecaster.db` (stop the scheduled tasks first).

The restored database has the durable tables intact and the two observation
tables empty; the next hourly ingest refills the observations. `schema_versions`
is copied, so migrations do not re-run against an already-current schema.

## Failure visibility

- A build, gzip, or upload failure exits non-zero, so Task Scheduler records the
  failure on `WoWForecaster-Backup`.
- The local `.db.gz` is written and old copies pruned before the upload runs, so a
  failed upload still leaves a good local backup.
- `check-data-health --backup-stale-hours 30` (run by `run_healthcheck.bat`)
  raises the alert window when the newest backup goes stale, catching a task that
  has silently stopped running.

## Out of scope, noted for later

- A `restore-durable-db` command (restore is a documented manual copy for now).
- Server-side encryption of the backup object at rest.
