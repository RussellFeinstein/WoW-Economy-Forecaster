# Integrity incidents: transient or real, and what to do in each case

Status: accepted 2026-07-28. Procedure record for issue
[#106](https://github.com/RussellFeinstein/WoW-Economy-Forecaster/issues/106).
Companion to [db-backup.md](db-backup.md) (restore mechanics) and
[cloud-capture.md](cloud-capture.md) (the snapshot bucket that makes the
observation tables rebuildable).

## The problem this document solves

On the machine that hosts the production DB, an integrity check that reports
errors is not automatically evidence of disk damage. The RAM that runs the
check is itself a suspect component (recurring MEMORY_MANAGEMENT bugchecks;
two real SQLite corruption events during the 2026-07 restore runbook), and
transient "database disk image is malformed" reads are already on record:
`scripts/setup_tasks.bat` phases the health check task to :45 specifically
because concurrent readers on the large DB produced them. When a page is read
through faulty memory, the in-memory copy is garbage while the file on disk is
fine, and a rerun reads clean.

The improvised response to a scary check is REINDEX or VACUUM. On this
machine that is exactly backwards: both are sustained multi-GB write jobs, the
class of work that has produced real corruption here. A repair run through
flaky RAM can take a healthy file and write actual damage into it.

## Where a report can come from

- The daily [Verify durable backup](../.github/workflows/verify-backup.yml)
  workflow going red (off-box, healthy RAM: treat as real until shown
  otherwise; start at [db-backup.md](db-backup.md) restore steps).
- `check-data-health --integrity-scope durable` failing with `[CORRUPT]`
  (local read, so the discriminator below applies).
- A manual full `PRAGMA integrity_check` (local read, ~25 minutes on the
  production file; the discriminator applies).
- Application errors: `sqlite3.DatabaseError: database disk image is
  malformed` during a pipeline run (local read; the discriminator applies,
  and one self-healed occurrence is already on record).
- A value that no query should ever have written: a realm slug outside the
  configured set, a date outside the observed range. None of the checks
  above can see this class, because `integrity_check` verifies pages and
  b-trees and never reads a value, and the CI verification runs the same
  pragma. The on-record case (2026-09-20, #165) is four rows in the rollup
  tables written between 07-28 and 09-09, three of them one bit from a
  valid value (`}s` for `us`, `ur` for `us`, `2027-09-09` for
  `2026-09-09`), each with obs_count 1, each sitting beside the correct
  row for the same day. Until #165 lands the only way to find one is a
  domain query by hand; the discriminator applies to it like any other
  local read, and a value that reads the same on a second connection is
  on disk. Repair is a targeted DELETE of the bad rows, never a rebuild.

## The two-pass discriminator

Run the exact same check twice and compare the error lists.

1. First pass: capture the full output to a file. For the durable tables the
   scoped check is fast:
   `.venv\Scripts\wowfc check-data-health --integrity-scope durable > pass1.txt`
   For a full check, use the same command you ran when the errors appeared,
   redirected to a file.
2. Second pass: identical command, `> pass2.txt`.
3. Compare:
   - **Identical non-empty error lists** point at real on-disk damage. A
     deterministic re-read finds the same broken pages. Go to the restore
     paths below.
   - **A clean second pass, or a different error list**, points at transient
     in-memory reads. Take no action against the database. Record the
     incident (date, command, both outputs) as hardware evidence; it belongs
     with the machine's crash history, not in a repair ticket.
4. When the two passes partially overlap, run a third. Errors that repeat
   across passes are real, the rest is noise, and both can be true at once: a
   small amount of genuine damage with transient reads layered on top. Treat
   the repeating subset as real.

A write that fails identically on every scheduled run is the same
discriminator, already run for you. Issue #155 is the worked example: the
retention prune's DELETE for one hour slice raised `database disk image is
malformed` 364 times over twenty days, across reboots and sleep cycles, on the
two rows a single integrity pass had named a month earlier. That is a
deterministic re-read finding the same broken page 364 times, and a fresh
`PRAGMA integrity_check` (an hour or more on the full-size table, starving
ingest while it runs) would have added nothing to it. Record the run count and
the first and last failure timestamps as the verdict instead.

## Repair by dropping, when the damaged structure is an index with one reader

An index can be dropped without reading its entries, so a corrupt index whose
only consumer can be served another way is repaired by deleting it, never by
rebuilding it. That is the smallest write available on this machine, and it
has been the answer twice: `idx_obs_raw_item_time` (#153, no reader at all)
and `idx_obs_raw_realm_ingested` (#155, one reader, the health check's
freshness probe, which moved to the normalized table). Both drops shipped as
migrations with the schema.py DDL removed in the same change, because
`init-db` runs `apply_schema()` before `run_migrations()` and a surviving
`IF NOT EXISTS` line would rebuild the index on every run. The DROP itself
walks the index's pages to free them and writes only freelist trunks, as long
as `PRAGMA secure_delete` is 0 (it is; a 1 would rewrite every freed page).
Check that before the drop, and check the drive: the freed pages stay in the
file for inserts to reuse, and nothing here reclaims them.

## Standing rules

- **Never REINDEX or VACUUM the production DB as a repair step on this
  machine.** There is a restore path for every table; use it. (The one-off
  `VACUUM INTO` during the 2026-07 rebuild was run under runbook conditions
  with independent cross-verification, and still hit two corruption events.)
- **Verify before you restore.** Every candidate backup file gets
  `python -m wow_forecaster.backup.verify <file.db.gz>` before it is copied
  into place. The daily CI run has usually done this already; the local check
  is for the specific file you are about to use.
- **Stop the scheduled tasks before touching the DB file**, and re-enable
  them after:
  `schtasks /Change /TN "WoWForecaster-<Hourly|Daily|HealthCheck|Backup>" /DISABLE`
  (then `/ENABLE`).

## Restore paths by table class

The DB is, structurally, a rebuildable cache plus a small durable core. Every
class has an off-machine source of truth.

**Durable tables** (rollups, forecasts, recommendations, backtests, drift and
health snapshots, reference tables): restore from the newest verified backup
object. This is the documented file copy in
[db-backup.md](db-backup.md#restore): download, verify, gunzip, copy to
`data/db/wow_forecaster.db`. The observation tables come back empty and
refill from ingestion.

**Observation tables** (`market_observations_raw`,
`market_observations_normalized`): rebuildable from the snapshot bucket,
which holds the same 30-day window local retention does. After a durable
restore (or a targeted cleanup), drain the bucket through the live ingest
path:

```
.venv\Scripts\wowfc sync-snapshots --since <30 days ago, YYYY-MM-DD> --limit 0
```

A bare `sync-snapshots` only reaches back `cloud_sync.max_backfill_days`
(3 days by default); the explicit `--since` is what makes it a full rebuild.
Normalization and rollups run inside the same pipeline, so no separate
re-normalization step exists or is needed. Rollup rows for the drained window
are recomputed by upsert; rollup history older than the window is part of the
durable restore, not the drain.

**Raw JSON snapshot files on disk**: the same objects exist in the bucket;
`sync-snapshots` rewrites them verbatim as it ingests.

## After the incident

- Transient verdict: no DB action. The evidence goes to the machine's
  hardware record, and repeated transient incidents strengthen the case for
  the hardware diagnosis path (MemTest86 per-stick isolation) over any
  software response.
- Real-damage verdict: after restoring, run the scoped integrity check once
  (`check-data-health --integrity-scope durable`) and dispatch the
  verify-backup workflow once, so both the local file and the next backup are
  known good.
