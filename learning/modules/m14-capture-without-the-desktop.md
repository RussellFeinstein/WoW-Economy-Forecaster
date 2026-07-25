# M14. Capture without the desktop

Part III. Failure, history, operations. Reads well after M11, the outage module.

## Why this module exists

Everything else in this pipeline has a retry. A corrupt database restores from
backup. A bad forecast retrains. A missed rollup recomputes from observations.

Capture does not. The Blizzard commodities endpoint serves the current snapshot
and nothing else, so an hour nobody captured is a hole in the record forever.
That single property is why the capture layer gets reliability engineering that
no other stage in this repo gets, and why "the desktop was asleep" is a data-loss
event rather than an inconvenience.

The interesting part is not that capture moved to the cloud. It is what happened
next. The obvious runner (a GitHub Actions scheduled workflow) turned out to
deliver about 11 of 24 hourly firings for this repo. The obvious fix (ask three
times an hour instead of once) was implemented, measured, disproved, and
reverted inside two days. The real fix moved the trigger off GitHub entirely and
kept one GitHub firing as a dead-man alarm.

## The idea to hold onto

Two of them, and they are the same idea from opposite sides.

**A monitor must be able to fail differently from the thing it monitors.** The
`:06` schedule survives the Worker dying, and the gap guard's floor of 20 sits
above what a dead Worker can deliver (~11 hours a day). A floor the failure mode
can satisfy hides the failure.

**Two records of the same event are not two events.** `fetched_at` is client-side
wall clock, not the auction house snapshot's own identity. So the desktop's `:16`
run and the Worker's `:16` dispatch produce two rows describing the same market
moment, with different timestamps, different filenames and different content
hashes. The UTC hour rule in `select_objects_to_ingest` is the only thing that
catches it.

## Read this first

The repo is the textbook. Read these before drilling:

- [`docs/cloud-capture.md`](../../docs/cloud-capture.md)
  The whole thing. The `Trigger` decision and the `Selection rules, and why each
  exists` list are the core. Note the rejected alternatives: they carry as much
  reasoning as the accepted ones.
- [`wow_forecaster/ingestion/cloud_fetch.py`](../../wow_forecaster/ingestion/cloud_fetch.py)
  The module docstring's exit-code table, `evaluate_gap_guard` and its bootstrap
  branch, and `list_recent_keys` on why three day prefixes and not two.
- [`wow_forecaster/ingestion/cloud_sync.py`](../../wow_forecaster/ingestion/cloud_sync.py)
  The seven-filter docstring on `select_objects_to_ingest`, and `hourly_lock` on
  why this waits where `run_hourly.bat` skips.
- [`.github/workflows/cloud-snapshot.yml`](../../.github/workflows/cloud-snapshot.yml)
  and [`cloud-trigger/worker.js`](../../cloud-trigger/worker.js)
  Read them together. One `cron:` line and about fifty lines of JavaScript are
  the entire trigger story.
- [`wow_forecaster/backup/durable_backup.py`](../../wow_forecaster/backup/durable_backup.py)
  and [`docs/db-backup.md`](../../docs/db-backup.md)
  The other half of durability: cloud capture protects the raw input for 30 days,
  this protects the derived history for good. Read the `sqlite_master` argument
  closely.

## What you should be able to do afterwards

- State why a missed capture hour is unrecoverable, in one sentence.
- Describe the cron delivery ceiling, the measurement that proved it was
  deterministic, and why densification was a dead lever.
- Explain how a single `:06` firing plus a floor of 20 forms a dead-man alarm,
  and what breaks if you lower the floor.
- Say what `fetched_at` actually records, and trace which of the seven selection
  rules prevents the desktop-versus-cloud double count.
- Explain why the durable backup copies schema from `sqlite_master`, and what
  fails if it does not.

## A note on what has and has not been proven

Cloud capture is live and the Worker is confirmed firing on every slot. The
drain is not. `sync-snapshots` shipped in v2.10.0 with 73 tests, every one of
which stubs S3 rather than reaching it, and the path has never been run against
the real bucket. `docs/cloud-capture.md` says so in bold, which is the right way
to carry it: a green suite proves the selection logic, and nothing about
credentials, permissions, real key shapes or pagination.

The first real proof is one command, `wowfc sync-snapshots --dry-run`, once the
read-only `SNAPSHOT_S3_*` token exists in `.env`. Until then this module is
teaching a design that is complete and unexercised, which is a distinction worth
being able to make out loud.
