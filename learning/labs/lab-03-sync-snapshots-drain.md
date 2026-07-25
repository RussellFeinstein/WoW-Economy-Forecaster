# Lab 03. Drain the cloud bucket for the first time

Module: M14. This one is different from Lab 01. There is no bug to fix and almost
no code to write. The `sync-snapshots` path shipped in v2.10.0 with 73 green
tests and has never once run against the real R2 bucket. Every one of those tests
injects a stub S3 client, so what they prove is that the selection logic is
right, the queries feed it the right sets, and the CLI reports what it did. What
no test can prove is that the credentials resolve, that the bucket name points at
the capture bucket and not the backup bucket, that the read-only token can list,
and that real R2 keys match the parsing regex. This lab is the acceptance run
that closes that gap, issue #43's real proof, and it unblocks #86 (you do not put
an unproven drain on a schedule).

The finish line is concrete: bucket objects reach `market_observations_raw`
through the live ingest path, with a non-zero row count and no double counting.

## What actually happens when you run it

`SyncSnapshotsStage._execute()` in
[wow_forecaster/pipeline/sync_stage.py](../../wow_forecaster/pipeline/sync_stage.py)
lists the bucket day-prefixes from `now - max_backfill_days` to now
(`max_backfill_days = 3` in [config/default.toml](../../config/default.toml)),
reads local coverage, hands the key list to the pure `select_objects_to_ingest()`
in [wow_forecaster/ingestion/cloud_sync.py](../../wow_forecaster/ingestion/cloud_sync.py),
then downloads, writes each snapshot to disk verbatim, replays it through
`parse_blizzard_records()`, normalizes, and rolls up. The whole write phase holds
`data/db/.hourly.lock`.

The load-bearing selection rule is the fourth one: drop any object whose UTC hour
already holds local observations. `fetched_at` is client-side wall clock
(`datetime.now(UTC)` in `blizzard_client.py`), not the auction house snapshot's
own modification time, so the desktop's `:16` hourly run and the Worker's `:16`
dispatch fetch the same underlying snapshot and stamp it with two timestamps
seconds apart. Content hash, key, and local path all differ. The UTC-hour rule is
the only thing that recognizes them as the same hour. If it works, an overlapping
hour is skipped whole. If it silently does not, that hour gets ingested twice and
every row-counting aggregate downstream (rollups, the volume gate, rolling
normalization stats) is quietly wrong for exactly the hours where catch-up
overlapped the desktop, and right everywhere else. That is the failure this lab
is watching for, and it is the hardest kind to notice after the fact.

## Before you touch anything

**The credential is yours to enter, both halves.** I will not generate, read, or
suggest any token value. Here is the exact handoff.

1. Install the read dependency into the project venv:

   ```
   .venv/Scripts/python.exe -m pip install -e ".[cloud]"
   ```

   That pulls in boto3 from the `[cloud]` extra. Without it `make_s3_client()`
   raises a named error telling you to install it, not a stack trace.

2. In the Cloudflare dashboard, create an R2 API token scoped **read-only** to the
   snapshots bucket. Not the capture workflow's write token. Not the durable
   backup's `BACKUP_S3_*` token, which points at a different bucket entirely. The
   desktop only ever reads here, and a drain has no business being able to write
   to the bucket it drains.

3. Open `.env` at the repo root (gitignored) and add these four lines. Fill in
   both the key id and the secret yourself. Do not paste real values into any
   file I can see, into a commit, a PR, or the CHANGELOG:

   ```
   SNAPSHOT_S3_ENDPOINT=https://<account>.r2.cloudflarestorage.com
   SNAPSHOT_S3_BUCKET=<the capture bucket, not the backup bucket>
   SNAPSHOT_S3_ACCESS_KEY_ID=<read-only token key>
   SNAPSHOT_S3_SECRET_ACCESS_KEY=<read-only token secret>
   ```

   `SNAPSHOT_S3_REGION` is optional and defaults to `auto`. The placeholder forms
   already live in `.env.example` under the "Cloud snapshot catch-up" heading;
   copy from there. The names are explicit rather than boto3's bare `AWS_*`
   because a desktop may carry unrelated `AWS_*` credentials, and `resolve_s3_env()`
   in `cloud_sync.py` reads exactly these four names.

**The branch.** The operational steps below (dry run, real drain, verification
queries) need no branch and no commit. They are the acceptance. The only thing
that ships through a PR is the durable artifact: the regression test that turns
"no double count" into a permanent invariant, plus the doc status flips. Cut it
from the latest main once you have run the drain and know the real result:

```
git checkout main && git pull --ff-only
git checkout -b test/43-sync-snapshots-acceptance
```

## The check to run first

Lab 01 told you to write a test that fails before the code is touched. This lab
inverts that. The code is believed correct and has never run for real, so the
discipline here is not "make a red test go green," it is: **predict the numbers
before you run, and pick the check that would actually reveal a double count
rather than the one that merely looks like success.**

The tempting check, the one that tells you nothing:

> The row count in `market_observations_raw` went up, and no error was raised.

A double-counted overlapping hour passes that with flying colors. Rows went up,
nothing crashed, and you have silently poisoned a handful of hours.

The check that catches it, run against the live DB after the drain:

```sql
SELECT realm_slug,
       substr(observed_at, 1, 13) AS utc_hour,
       COUNT(DISTINCT observed_at) AS distinct_capture_instants
FROM   market_observations_raw
WHERE  source = 'blizzard_api'
GROUP  BY realm_slug, utc_hour
HAVING COUNT(DISTINCT observed_at) > 1;
```

This works because each ingested object stamps every one of its rows with the
same `obj.captured_at` value, so the number of distinct `observed_at` values
inside a single UTC hour is the number of snapshots ingested for that hour. It
uses the same `substr(observed_at, 1, 13)` hour key that
`MarketObservationRepository.get_covered_hours()` uses to decide coverage, so the
query and the code agree on what "an hour" is. **Any row this returns is a
double-ingested hour.** A correct drain returns nothing.

Run it once as a baseline before you drain (it should already be empty; the
desktop-only path never double-stamps an hour). Run it again after. The invariant
is that it stays empty.

## The run

1. **Dry run first.** This is the first real proof, per
   [docs/cloud-capture.md](../../docs/cloud-capture.md):

   ```
   .venv/Scripts/python.exe -m wow_forecaster.cli sync-snapshots --dry-run
   ```

   (or `wowfc sync-snapshots --dry-run` if the entry point is on PATH). It lists
   candidates and reports what it would skip, writing nothing. Read the three
   lines it prints: `Objects listed`, `Selected`, `Skipped` with its breakdown.
   If `Objects listed` is 0, stop: the bucket name, prefix, or list permission is
   wrong, and no amount of re-running fixes that. If listed is healthy and it
   reports a `Skipped` breakdown, credentials resolve and real keys parse.

2. **Baseline the DB.** Run the double-count query above and record that it is
   empty. Note the current `MAX(observed_at)` for `source = 'blizzard_api'` so you
   know what "before" looked like.

3. **Real drain, default bounds.**

   ```
   .venv/Scripts/python.exe -m wow_forecaster.cli sync-snapshots
   ```

   The default `max_backfill_days = 3` lists at most 72 to 96 hourly objects and
   selection keeps one per UTC hour, so `max_objects_per_run = 96` does not bite
   and `over_limit` stays 0. Read the full output block: `Selected`,
   `Skipped (hour_covered=..., duplicate_hour=...)`, `Ingested`, `Observations
   inserted`, `Normalized`, `Dates rolled up`. Exit code must be 0 with no
   `[FAIL]` lines.

4. **Post-drain check.** Re-run the double-count query. It must still be empty.

5. **Idempotency check.** Run `sync-snapshots` again immediately. `Selected` must
   be 0 and everything skipped as `already_ingested` or `hour_covered`. A second
   run that ingests anything is a dedup failure.

## Do not

- **Do not open the first real run with a wide `--since`.** `--since 2026-07-05
  --limit 0` on this box pulls hundreds of objects at ~250K records each in one
  pass, which is exactly the sustained multi-gigabyte write pattern that corrupted
  this machine twice during the April outage runbook. `max_backfill_days = 3`
  exists to prevent precisely that. If you need to reach an older gap, do it
  deliberately, bounded with `--limit`, in stages, with someone watching. This is
  the operating rule for rex-desktop, not a suggestion.
- **Do not read a non-zero, increased row count as success.** That is the tempting
  check above. The acceptance is the double-count query staying empty, not the row
  count moving.
- **Do not skip the `--dry-run`.** It is the one cheap step that catches a wrong
  bucket, an unresolved credential, or a token without list permission before you
  hold the lock and start writing.
- **Do not point `SNAPSHOT_S3_BUCKET` at the backup bucket.** It holds no
  commodities objects, so a dry run against it lists nothing and reads exactly
  like a quiet capture bucket. Confirm the name is the capture bucket before you
  trust an empty listing.
- **Do not put the `SNAPSHOT_S3_*` values anywhere but `.env`.** Not
  `config/default.toml`, not a commit, not the CHANGELOG, not the PR body. Never
  echo them to the terminal in a way that lands in a log you commit.
- **Do not reuse the capture workflow's write token or `BACKUP_S3_*`.** Read-only,
  snapshots bucket, its own token.

## Finish line

- `sync-snapshots --dry-run` against the live bucket lists real candidate keys and
  reports a skip breakdown, writing nothing.
- The real drain exits 0 with no failures, and `Observations inserted` is non-zero
  for at least one genuinely uncovered UTC hour (see the next section on how to
  guarantee one exists).
- The double-count query returns zero rows both before and after the drain.
- An immediate re-run selects 0.
- If you shipped the regression test, the targeted file is green:
  `.venv/Scripts/python.exe -m pytest tests/test_ingestion/test_cloud_sync.py -q`.
  Run the full suite (`pytest -q`) only when the working tree is otherwise quiet.
- `.venv/Scripts/python.exe -m ruff check` is clean if you added a test.
- `CHANGELOG.md` gets an entry under `[Unreleased]`, in user terms: the cloud
  catch-up drain ran against the live bucket for the first time and the no-double-
  count invariant was verified. Not "added a query," but what it means.
- Documentation sync. The repo's operational-state note in
  [CLAUDE.md](../../CLAUDE.md) says `sync-snapshots` is "dormant until the
  read-only SNAPSHOT_S3_* token is added to .env on rex-desktop." That sentence is
  now stale and must change to reflect that the path was activated and run. The
  "pending activation" framing in `docs/cloud-capture.md` (the "A green test suite
  is not evidence this works" note and the m14 bank's "never been run for real"
  status line) is likewise stale once this lands; reconcile it in the same session
  so a doc does not keep claiming the drain is unproven after you proved it. The
  machine-local memory page `sync-snapshots-pending-credentials.md` gets retired or
  updated as a close-out, but that lives outside the repo.
- Version: **patch**. Adding a credential and running a command is not a product
  change. The shippable slice is a regression test plus doc status flips, which
  correct and confirm, adding no new CLI surface or config key. If the run turns
  out completely clean and you choose to ship no code at all, that is the
  investigation-outcome case and takes no bump, but flipping the documented
  "dormant" status is itself a change worth a patch stamp. Stamp at PR open and
  move the `[Unreleased]` lines under the version header.

## What to expect, and how to read an ambiguous result

The desktop's own hourly task has been running continuously since the 2026-07-21
restore. So for the last three days, most UTC hours are already covered by the
local path, and the correct behavior of a default drain is a large
`hour_covered` skip count and a **small or zero** `Selected`. That is not a
failure. It means the two capture paths agree and the dedup is doing its job.

This creates the one ambiguity you have to resolve deliberately: **`Selected = 0`
has two readings.**

- If `Objects listed` is also 0 or tiny, the bucket, prefix, or list permission is
  wrong. Fix the credential or the bucket name.
- If `Objects listed` is healthy (dozens per day) but `Selected = 0` with the skip
  breakdown dominated by `hour_covered` and `already_ingested`, the drain is
  working perfectly and the desktop simply had those hours already. This is a pass
  for correctness, but it does not satisfy "non-zero row count," because nothing
  was ingested.

To make the acceptance meaningful you need at least one UTC hour the desktop
genuinely missed, so that an object is actually selected and rows actually land.
Two honest ways to get one:

- The morning after any night the desktop slept or rebooted, the first drain has
  real gaps to fill. This is the natural, zero-risk path: wait for one.
- Or point `--since` at a bounded window you know the desktop was down, capped
  hard with `--limit`, and watch it. A window with a handful of missing hours
  gives you a small, safe `Selected > 0` that lands real observations. Do not
  widen this into a multi-hundred-object backfill on this machine.

Either way, when a real gap is drained: `Selected` matches the number of uncovered
hours, `Observations inserted` is on the order of 250K per selected hour minus any
`Unknown items` skipped on the FK guard (run `bootstrap-items` first if that count
is large), `Dates rolled up` lists the touched UTC dates, and the double-count
query stays empty. That is the whole acceptance.

One more expected wrinkle. If the command exits 1 naming missing `SNAPSHOT_S3_*`
variables, `.env` did not load: check you edited the file at the repo root and
that the four names are spelled exactly as in `.env.example`. The error names
which variables are missing and never their values, by design.

## Reflection, before you close the issue

The interesting note here is not about the drain. It is about the gap this lab
existed to close: a feature shipped complete, with 73 passing tests, and was still
unproven, because every test stubbed the one boundary that could actually be
wrong. Write a short `LESSONS.md`-style note answering: for a path whose whole
risk lives at an external boundary a test suite cannot reach, what is the cheapest
signal that would tell you at a glance whether it has ever run for real? The
answer that generalizes past this repo is worth more than the one specific to R2.
