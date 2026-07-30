# Changelog

All notable changes to the WoW Economy Forecaster.

Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
This project uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.14.5] - 2026-07-30

### Changed
- Roadmap backlog records the gap the `waiting: wall clock` label leaves open: the label says an issue is waiting on a date, but nothing announces when that date arrives, so a labelled issue can sit past its check indefinitely. A workflow that reads the date from the issue body and comments when it passes would close it. Parked rather than built, because two issues carry the label and that is not enough to justify the machinery

## [2.14.4] - 2026-07-30

### Added
- A `waiting: wall clock` label for issues that are finished except for an acceptance item that cannot be checked until a date arrives, applied to #11 (gap verification, next check 2026-08-03) and #42 (cloud fetcher, lifecycle deletion around 2026-08-19). Both issue bodies now open with the earliest date they can be closed. Until now that state was recorded only in the roadmap and the milestone descriptions, so from the issue list a milestone sitting on a date looked the same as one sitting on unstarted work. M0.5 read as three open issues when only one of them was waiting on anyone

### Changed
- #43 closed after a second drain on 2026-07-28 covered an organic gap as well as the staged one. Two hours lost to post-wake crashes were selected and ingested with hour dedup skipping 25 already-covered hours. It had been reopened after a merge keyword auto-closed it ahead of its acceptance, and then left open because the closing call was the maintainer's
- Roadmap records the #43 close and the label convention, including what the label does not mean: it is for work that is built and waiting on a date, never for work that has not been started

## [2.14.3] - 2026-07-30

### Changed
- The milestone-sync rule in CLAUDE.md now scopes to a PR instead of a session. It said to update a milestone's work-order list "in the same session" as filing or closing an issue, and a session is not a reviewable unit: nothing inspects it and nothing diffs it. Filing #117 and opening the PR that fixed it counted as two separate acts, so the milestone update fell in the gap between them and three merges shipped without it. A PR is where a missing roadmap hunk shows up in a diff before anything lands
- Roadmap and both M0 milestone descriptions caught up with the 2026-07-29 merges: #78 recorded as shipped in v2.14.0 rather than open as PR #115, with its overnight acceptance noted as still outstanding because that acceptance does not start on its own (the guard only fires on a run that was itself a wake, and this machine never sleeps unless someone sleeps it); #117 added to the M0 issue list and to the close-out work order with the TkAgg diagnosis that took ten days and four sightings to reach

## [2.14.2] - 2026-07-29

### Fixed
- The test suite no longer renders on a GUI backend (issue #117). Nothing in the repo pinned one, so matplotlib picked TkAgg on Windows and every figure the viz tests build spun up a real Tcl/Tk interpreter. That is 47 figures per full run, none of them ever displayed. Under load the interpreter creation lost a race often enough to fail four different viz tests between 2026-07-19 and 2026-07-29, one per run, each reporting `This probably means that tk wasn't installed properly` and each passing on rerun. `tests/test_viz/conftest.py` now pins Agg, which is what headless CI already falls back to, so local runs match CI instead of diverging from it. This is why CI never reproduced it in any run

## [2.14.1] - 2026-07-29

### Changed
- Roadmap, README and cloud-capture docs caught up with work that shipped without them. Nine issues had been filed, worked and in six cases closed with no milestone attached, so the milestone view understated what was done and the roadmap had no home for a whole verification tier (#104, #105, #106). Those three now sit under M0 with a paragraph explaining why they exist: the machine that builds the nightly backups has a documented memory-corruption history, so verifying those backups on that same machine proves nothing
- The roadmap's standing-risks table gained a row for that hardware instability. It had no entry despite driving three issues and an incident runbook, which made the table read as though the risk was not being tracked
- Roadmap now records that #78 gates #86. #40's close-out described the machine returning to sleep on an idle timeout, which held only while a temporary setting was in force, so the duty-cycle note filed as an unscheduled improvement turned out to be the blocker for the sleep-overnight work
- README's Project Status table ran to v2.2.0 while the version was v2.13.4, leaving cloud capture, the durable backup, catch-up ingestion, the learning track and the verification tier invisible. Extended through v2.14.0
- `docs/cloud-capture.md`'s sizing figures were an April 2026 measurement. Re-measured: commodity snapshots now run 232,449 to 270,196 records (averaging 254,011 over 74 snapshots since 2026-07-25) against the documented ~314,000, and objects run 1.55 to 1.81 MiB gzipped against the documented ~2.2 MiB. The shrink is market-wide and shows on both capture paths, so it is not a capture defect; the sizing conclusions and the 50,000 sanity floor are unaffected
- PLAN.md records that its four tool decisions are all still open, and that OD-1 is the one with a deadline since Phase 2 depends on it

### Added
- `docs/m1-model-validation-plan.md`: the M1 audit and forward plan, verified against the live database and the code on 2026-07-28. It had been written and left untracked

## [2.14.0] - 2026-07-29

### Added
- `scripts/sleep_back.ps1` (issue #78): the scheduled tasks now return the machine to sleep after an unattended run that woke it, instead of leaving it awake until the next trigger. Wake timers (#40) put the box back to sleep only while a temporary 30-minute idle timeout was in force; with AC idle sleep set to Never there was nothing to return it, so a deliberate overnight sleep ended at the first hourly wake. The helper suspends only when all four conditions hold at end of run: the run was itself a wake attributed by name to a WoWForecaster task, no keyboard or mouse input arrived during the run, no other WoWForecaster task is running and no hourly lock is held, and no unacknowledged health alert is waiting on screen. It runs only inside an overnight window, 20:00 to 08:00 by default, adjustable per machine with `WOWFC_SLEEP_FROM_HOUR` and `WOWFC_SLEEP_UNTIL_HOUR`. Anything it cannot evaluate leaves the machine awake, the opposite of the stale-lock and alert-window biases elsewhere, because a wrong sleep interrupts whoever is at the keyboard. Wired into the hourly and health-check wrappers; the 07:00 daily and 07:30 backup runs deliberately never sleep so the machine is up when you arrive

## [2.13.4] - 2026-07-29

### Fixed
- The test suite no longer writes into the production database (issue #113). `TestMissingCredentials` was the one test in `test_sync_snapshots_cli.py` that did not stub the stage, so it built a real `SyncSnapshotsStage` from the production config and `PipelineStage.run()` persisted a failed `sync_snapshots` row to `data/db/wow_forecaster.db` on every suite run. 39 such rows had accumulated since 2026-07-24, indistinguishable from real operational failures when reading `run_metadata`. This is the half of #97 that fix did not cover: it stopped the test reaching the live bucket, not the live database

### Added
- An autouse `isolated_product_db` fixture in `tests/conftest.py` pinning `WOW_FORECASTER_DB_PATH` to `tmp_path` for every test, so no test can reach the real database whether or not its author remembered to override the path. Mirrors the `WOWFC_LEARN_DB` fixture the learning track already uses. `tests/test_db_isolation.py` asserts the guard actually holds, since the failure mode it prevents is silent (the test still passes; only the production database shows the damage)

## [2.13.3] - 2026-07-28

### Changed
- The db-backup activation checklist now names the CI verification token, `github-actions-verify-db-backups-ro`, and records the token naming convention (`<consumer>-<verb>-<bucket>-<access>`); `.env.example`'s `BACKUP_S3_*` block gains a cross-reference noting the same four names also exist as GitHub repository secrets holding a separate read-only token. Prompted by a real mixup: values in `.env` do not reach GitHub Actions

## [2.13.2] - 2026-07-28

### Changed
- PLAN.md Phase 2 now names the desktop's memory instability as a second driver for the orchestration and portability phase, records that the verification tier shipped separately (#104, #105, #106), and pins where the remaining decisions land: VPS hosting choice at phase start, hardware spend only after a free MemTest86 diagnosis, #107 as the restored-backup prerequisite for off-box ML

## [2.13.1] - 2026-07-28

### Added
- `docs/integrity-incidents.md` (issue #106): the procedure for an integrity check that reports errors on the production machine. Two-pass discriminator (identical error lists mean real disk damage, a differing or clean second pass means transient in-memory reads and no DB action), the standing no-REINDEX/no-VACUUM-as-repair rule, verify-before-restore, and restore paths per table class: durable tables from the newest verified backup, observation tables re-drained from the snapshot bucket with `sync-snapshots --since`. Linked from docs/db-backup.md

## [2.13.0] - 2026-07-28

### Added
- `check-data-health --integrity-scope durable` (issue #105): a table-scoped `PRAGMA integrity_check` over every table except the two large observation tables, so page-level corruption in the durable state is caught by the scheduled health check in seconds instead of requiring a 25-minute full-database scan. The scheduled health check (`run_healthcheck.bat`) passes it and raises the existing alert window on a failure; the daily forecast gate does not, so a corrupt table alerts without blocking forecasts. The observation tables are skipped on purpose: they are a rebuildable cache (the snapshot bucket holds their whole 30-day window), and the failure message points at the two-pass transient-read discriminator before any action

## [2.12.0] - 2026-07-28

### Added
- Daily off-box verification of the durable backup (issue #104). A scheduled GitHub Actions workflow restores the newest `db_backups/` object on a CI runner and checks integrity, foreign keys, row-count floors, and no-shrink of the append-only tables against the previous object. A corrupt, incomplete, shrunk, stale, or missing backup turns the run red; nothing skips. Until now the backup was uploaded nightly and never read back, so a backup corrupted at build time would have surfaced only at restore time. Verification runs off the desktop on purpose: the machine that builds the backups has a documented memory-corruption history, so checking there is circular. The same module verifies a downloaded file before a restore: `python -m wow_forecaster.backup.verify <file.db.gz>`. Activation needs four read-only `BACKUP_S3_*` repository secrets (see README and docs/db-backup.md)

## [2.11.8] - 2026-07-27

### Changed
- M1 work order resequenced to carry the four repair issues that accumulated since it was written: #70 and #71 (filed from the #11 day-one verification) and #100 and #101 (filed from the 2026-07-24 portfolio audit). ROADMAP step 6 and the M1 milestone header updated; the lab-01 instructions now point at #100 instead of telling the reader to file it

## [2.11.7] - 2026-07-27

### Changed
- Trimmed three derivable setup blocks from CLAUDE.md (the venv command fence, the credentials variable names, and the snapshot layout tree). `.env.example` and `wow_forecaster/ingestion/snapshot.py` remain the source of truth for the removed content

### Fixed
- `.env.example` updated to name the durable backup bucket `wow-forecaster-db-backups`, matching the bucket as actually provisioned rather than the placeholder name the example carried

## [2.11.6] - 2026-07-27

### Fixed
- The missing-credentials CLI test failed on any machine whose real `.env` carries `SNAPSHOT_S3_*` values (issue #97): `load_config()` re-reads `.env` mid-invocation and refilled the variables the test had deleted, sending a unit test into the real bucket and the production database. The test now stubs out dotenv loading, so it exercises the named-missing-vars path regardless of what the developer's `.env` contains

## [2.11.5] - 2026-07-27

### Fixed
- `sync-snapshots` crashed with a TypeError before reaching S3 on its first run against the real bucket (issue #95, caught during the #43 acceptance attempt). The stage's default clock is timezone-aware while the CLI's `--since`, the stored coverage timestamps, and the key-timestamp comparisons all run on naive UTC, so `--since` raised at the retention clamp and a bare run raised at object selection; the two coverage queries would also have compared aware parameters against naive stored strings silently. The clock and `--since` are now normalized to naive UTC at the stage boundary. The existing 73 catch-up tests inject naive clocks, which is why none of them saw it; three regression tests now run the stage with the clock shapes production actually sees

## [2.11.4] - 2026-07-26

### Fixed
- The README's per-directory test counts, stale in both places they appeared and wrong in eleven of seventeen rows (test_db said 37 against an actual 94, test_reporting 86 against 173), with six groups missing entirely (test_backup, test_dashboard, test_learning, test_viz, and the top-level test_cloud_fetch.py and test_config.py). The counts are gone rather than refreshed: the architecture tree now lists all 23 groups with descriptions and no numbers, the duplicate by-group pytest list is collapsed to a single example, and the only numbers left are the suite total (1,628) and the group count (23), which the tree's own drift history says is the most that manual sweeps can keep honest. Two stale group descriptions also corrected: test_scripts was missing task registration and backup, and several rows adopted the fuller wording from whichever of the two copies had it

## [2.11.3] - 2026-07-26

### Added
- `docs/audit-pr-89.md`: the audit that produced the corrections below. PR #89 was re-checked claim by claim against the code, and the record keeps both halves of the result: every substantive technical finding held up, and the mechanical parts (one figure that was never right, three citations gone stale, three findings the PR understated) did not

### Fixed
- The model feature count and the table count in the docs, the last two of the four drifts `PLAN.md` named and the only two still wrong. `README.md` said 37 input features in two places against an actual 40, and 21 tables in two places against an actual 23, while its own highlights line already said 23. `CLAUDE.md` carried the same 21. The schema listing was short two rows, `daily_rollup_archetype` and `daily_rollup_item`, which is how the listing and its header managed to agree with each other while both disagreed with `schema.py`
- Stale citations in `PLAN.md`. The `learn` sub-app inserted eight lines at the top of `cli.py` after PR #89 merged, so `cli.py:3620` and `cli.py:1033` each landed short of the code they cited and the 4,508-line figure was two releases old. The inventory table had also half-drifted: its command row was refreshed when the learning track landed but its test row still described the repo at merge (71 files, 1,481 passing). Both rows now carry an as-of date, so the next reader can tell which numbers were re-checked and when
- The README size claim in `PLAN.md`, which was never right rather than drifted. "Roughly 70 percent CLI reference" measured nothing: the CLI reference is 325 of 894 lines, about a third. "Zero images" was wrong too, since the architecture diagram renders. The point behind both stands and now says what it means, which is that there is no chart of anything the model actually did

### Changed
- Three `PLAN.md` findings sharpened, each of which understated the defect it described. DS-3 said the 7d and 28d error-drift ratios can never be computed. None of them can: `persist_backtest_run` sits inside the per-horizon loop, so every horizon opens its own `backtest_run_id` and the newest row is always the last horizon of the last backtest, while both baseline queries take that newest run and then filter for horizon 1, which matches nothing. The scheduled drift check therefore computes no baseline at any horizon and reports no drift rather than unknown. DS-1 gained the second leak vector sitting in the same function, an 80/20 fallback split that slices by row index with no date sort at all, which the module docstring's "NEVER random" warning does not cover. And `auto_retrain_on_critical` is not merely off, it has no reader anywhere in the codebase, so setting it true changes nothing

## [2.11.2] - 2026-07-24

### Fixed
- Taxonomy counts in the docs corrected to match the enums. `EventType` is 28, documented in three places as 26, and `ArchetypeTag` is 40, documented in three places as 36. Six stale references fixed across `README.md` (the taxonomy table, the archetype-mapping walkthrough, and the schema listing), `CLAUDE.md` (the key-files line), and `docs/events.md` (the event schema table). The numbers had drifted as tags and event types were added over time without the doc counts being re-swept. The M01 learning question that cited the old README count was rewritten to teach the corrected state, since the drift guard requires a citing question to move with the line it cites; the lesson it carries (a count in prose has no test attached, so it can silently drift) is unchanged and now names the correction as the example

## [2.11.1] - 2026-07-24

### Added
- The learning track's remaining nineteen modules, filling in every part after the M06 pilot that shipped in 2.11.0. 328 new questions across M01 to M05, M07 to M20, each with a one-page lesson, plus the three remaining lab briefs (LightGBM into the backtest loop, the cloud snapshot drain, and the realization ledger). The track now covers the whole system: the domain and the data pipeline, features and the model and the statistics under them, the failure history and operations, and the work still ahead for validation, paper trading, causal inference, and publishing
- `wowfc learn validate --module <id>`, which checks one bank's citations in isolation. Useful while authoring, and load-bearing when several banks are written at once, since an unscoped check fails for everyone the moment one bank is half-written

## [2.11.0] - 2026-07-24

### Added
- `wowfc learn`, a study and assessment track for this codebase, shipped alongside it. Twenty modules across four parts (the domain and the data; features, models, and statistics; failure, history, and operations; proving it and shipping it), question banks with spaced repetition, and labs that are real open work rather than exercises. Runs as a third parallel track: `docs/ROADMAP.md` owns the research arc, `PLAN.md` owns the lifecycle arc, and this one owns understanding. The premise is that the repo is the textbook and the track is the syllabus and the exam, so no module restates what a docstring already explains
- Seven subcommands under the `learn` group: `status` (mastery per module, cards due, lab state), `next` (drill what is due, then new material), `module` (objectives and reading list), `exam` (scored, nothing revealed until the end), `lab` (print a brief, record progress), `validate` (check every citation against current code), and `reset`
- A citation drift guard, which is what makes a hand-authored question bank worth writing. Every question cites a file path plus a verbatim single-line anchor, never a line number, because a line number is wrong the moment a line is inserted above it. The command line resolves the anchor to a current line number at display time, so a citation is always right without a stored number that would go stale. `wowfc learn validate` and `tests/test_learning/test_bank_integrity.py` call the same `check_content()` implementation, so editing a cited line turns CI red until the question is updated
- Module 06 authored in full (18 questions on splitting time series, purging, and embargo) plus its lab brief. The remaining nineteen modules are declared in `learning/curriculum.toml` so the shape of the track is visible, and report as not authored yet until their banks land
- Spaced repetition using an SM-2 variant with four grades and an injectable clock. A passing grade schedules a card at least a day out, so re-running a drill the same day serves no repeat of anything graded good or better, while a failed card stays due and does come back. Re-drilling a module repeatedly in one afternoon does not push cards months out: the last grade of the day replaces the earlier ones rather than compounding on them

### Changed
- Review progress lives in its own SQLite file at `data/learn/progress.db`, gitignored and overridable with `WOWFC_LEARN_DB`. Deliberately not the product database, which is copied into every durable backup and is the upstream source for the analytics warehouse
- `learn` is registered as a Typer sub-app rather than seven more flat commands, since `cli.py` is already 4,500 lines. The learning command module imports typer and stdlib only at import time and defers everything else to command bodies, so the new group costs nothing at startup

### Fixed
- Command and test counts reconciled across `README.md`, `CLAUDE.md`, and `PLAN.md`. The three documents disagreed (39, 40, and 41 commands respectively, against an actual 40), the README architecture tree said 36, and the test badge said 1,200+ while the body said 1,400+

## [2.10.3] - 2026-07-24

### Added
- `PLAN.md`: an audit of the project as a portfolio artifact, plus a sequenced plan for the lifecycle and legibility work that follows from it. Four tool choices (orchestrator, experiment tracking, serving target, infrastructure as code) are recorded as open decisions with reasoning, not adopted. Runs as a parallel track to `docs/ROADMAP.md`, which stays the source of truth for the research arc and the issue numbering
- `LESSONS.md`: repo-level record of approaches that turned out wrong and had to be corrected, seeded with the two modeling findings from the audit

## [2.10.2] - 2026-07-24

### Fixed
- `docs/cloud-capture.md` now carries the local activation checklist for `sync-snapshots` (install the `[cloud]` extra, create a read-only R2 token, add the four `SNAPSHOT_S3_*` variables to `.env`, verify with `--dry-run`). The command's missing-credentials error already pointed at that file, but the file only documented the GitHub Actions repo secrets for the capture side, so the one page the error sent you to was the one page that did not answer it

## [2.10.1] - 2026-07-24

### Fixed
- Roadmap now names issue #86 (sleep overnight and let cloud capture cover it) in the M0.5 issue range and in the work order, where it was described in prose before the issue existed

## [2.10.0] - 2026-07-23

### Added
- `sync-snapshots` command (issue #43): ingests hourly commodities snapshots that the cloud capture wrote to R2 while this machine was asleep or off. The commodities endpoint serves only the current snapshot, so hours missed locally are unrecoverable from the API; draining the bucket is the only way they reach the database. Each object goes through the same ingest path a live fetch uses (same on-disk location, same `_meta` envelope, same item foreign-key guard), followed by normalization and a rollup upsert for every UTC date touched
- Idempotency, so the command is safe to run at any time and re-running is a no-op: objects already ingested are skipped by snapshot path, and UTC hours that already hold observations are skipped outright. The second rule is what prevents double-counting, because the desktop's own hourly run and the cloud capture fetch the same underlying auction-house snapshot seconds apart
- `[cloud_sync]` config section (`max_backfill_days`, `max_objects_per_run`, `lock_wait_seconds`) and `SNAPSHOT_S3_*` read credentials in `.env` (see `.env.example`). The default 3-day lookback and 96-object cap keep the first run after setup from pulling the entire retention window in one pass; whatever a cap leaves behind is reported and picked up by the next run, never silently dropped. `--limit 0` lifts the cap for a large recovery
- `--dry-run`, which lists what would be ingested and writes nothing

### Changed
- `parse_blizzard_records()` moved from a method on `IngestStage` to module level in `wow_forecaster/pipeline/ingest.py`, so the live path and the cloud catch-up path share one implementation. The method remains as a delegate

### Fixed
- `cloud-trigger/.wrangler/`, the Wrangler build cache, is now gitignored

## [2.9.1] - 2026-07-23

### Changed
- Cloud capture is now triggered by an external Cloudflare Worker cron instead of GitHub's own schedule (issue #83). GitHub delivers only about 11 of 24 scheduled cron firings a day for this repo, deterministically, and densifying the cron does not help because the cap is on run delivery, not on schedule expressions, so #67's three-firings-an-hour change is reverted. A Worker on the account that holds the R2 buckets POSTs `workflow_dispatch` at :16 and :46, bypassing the schedule backlog; the GitHub schedule is thinned to a single :06 fallback that doubles as a dead-man alarm (if the Worker or its token dies, capture falls back to about 11 hours a day and the gap guard goes red). The Worker source and deploy steps live in `cloud-trigger/`; it authenticates with a fine-grained personal access token stored as a Worker secret, never in the repo

## [2.9.0] - 2026-07-23

### Added
- Durable-table backup (issue #80): a new `backup-durable-db` command writes a restorable `.db.gz` of every table except the two large per-observation tables (those are recreated empty, so the file is a drop-in restore) and uploads it to a separate, private R2 bucket. The schema is copied from the live database's `sqlite_master`, so migration-added columns and any future tables are captured, and the build never reads the multi-GB observation tables. A real snapshot on 2026-07-23 was 118 MB uncompressed, 31 MB gzipped, 24 tables, built in under 6 seconds. Design record and restore steps: [docs/db-backup.md](docs/db-backup.md)
- A dedicated `WoWForecaster-Backup` scheduled task (daily 07:30, after the 07:00 forecast) via `scripts/run_backup.bat`, registered by `setup_tasks.bat` with the same wake-to-run and disabled-state-preservation handling as the other tasks. Its exit code is an independent backup-health signal
- `check-data-health --backup-stale-hours N` (opt-in, 0 = off) flags the newest durable backup when it is older than N hours; `run_healthcheck.bat` passes 30, so a backup task that has stopped raises the existing health alert window. The check is off by default so a stale backup never blocks the daily forecast freshness gate
- `[backup]` config section (`output_dir`, `keep_local`, `upload_enabled`, `stale_hours`) and `BACKUP_S3_*` credentials in `.env` (see `.env.example`). Uploading needs boto3 from the `[cloud]` extra

## [2.8.3] - 2026-07-23

### Changed
- Roadmap records the #40 wake-to-run acceptance (2026-07-23): six sleep/wake cycles with every wake attributed by name to a WoWForecaster task, an unbroken hourly ingest chain across the window, and both overnight health checks healthy, so the machine can now sleep between scheduled runs without losing hours. Re-sleep takes about 32 minutes rather than the 10 the issue predicted, because the power plan's 30-minute idle timer governs instead of the unattended timeout; that is a duty-cycle question only and is filed as #78

## [2.8.2] - 2026-07-22

### Added
- docs/postmortem-2026-04-lock-outage.md (issue #2): the full account of the 96-day silent ingestion outage. Timeline from the 2026-04-15 lock leak through the 2026-07-21 restore, the four-failure root cause chain (no lock staleness handling, exit-0 skip path, unscheduled health check, date spine clamping to frozen data), data impact (2026-04-08..07-20 hourly data lost, 12 of 18 rollup dates recovered, ~90 days of frozen-feature forecasts now partially measured at 2,456g/2,077g day-one MAE), the day-one #11 verification results, and the fix set. Linked from the README health-check section and the roadmap; the roadmap work order records #2 and the #11 day-one pass as done

## [2.8.1] - 2026-07-22

### Fixed
- .gitignore now covers the dated snapshot layout and the full outputs pile (issue #9): the three data/raw/snapshots rules widen to **/ so blizzard_api/YYYY/MM/DD files match, data/processed and data/outputs gain json/csv rules, and the charts, model_artifacts, and backups directories are ignored wholesale. 570 untracked local artifacts drop out of git status; the three tracked .gitkeep placeholders are unaffected. Contrary to the issue body's assumption there was nothing to git rm --cached: only .gitkeep files were ever tracked under data/outputs

## [2.8.0] - 2026-07-22

### Changed
- The cloud snapshot cron now fires three times per hour (:16/:36/:56) because GitHub drops most single hourly firings outright, 11 of 24 on the schedule's first day (issue #67). Any one firing per hour covers that hour; duplicate snapshots coexist under timestamped keys and the 30-day lifecycle rule bounds storage at roughly 4 GB worst case, inside the R2 free tier
- The gap guard now counts distinct UTC capture hours covered in the trailing 24 hours (floor 20) instead of raw objects, so the guard keeps meaning "hours are being missed" at any cron density: at three firings per hour a gappy day can still hold more than 20 objects. The override env var is now CLOUD_FETCH_GUARD_MIN_HOURS (was CLOUD_FETCH_GUARD_MIN_OBJECTS; set nowhere outside the module and its tests)

## [2.7.9] - 2026-07-22

### Fixed
- Both rollup UPSERTs now compare observed_at as a raw column against a half-open date range instead of DATE(observed_at) = ?, so idx_obs_norm_realm_outlier_time serves them with an index seek (issue #65). The old expression predicate forced a scan of every normalized row for the realm on each hourly run (2 dates x 2 tables since v2.7.5), which was cheap on the freshly rebuilt table but would have become the hottest DATE() site in the codebase as the table regrows toward its 30-day steady state. Query-plan tests pin the observed_at seek terms and equivalence tests assert identical rollup rows against the legacy form on edge timestamps

## [2.7.8] - 2026-07-22

### Fixed
- The cloud-capture gap guard no longer has a false-pass blind spot just after UTC midnight (issue #68): the bucket listing now covers three day-prefixes (today, yesterday, day before yesterday), so objects older than the 24-hour cutoff are always visible and the bootstrap rule only fires on a genuinely empty history. Before this, the window from midnight until yesterday's earliest object crossed the 24-hour boundary read sparse days as bootstrap, which is how the 2026-07-22 00:10Z run passed on a day the two prior runs had failed

## [2.7.7] - 2026-07-22

### Changed
- The M0.5 section and work order in docs/ROADMAP.md now carry #67 (GitHub drops most hourly cloud-capture cron firings, 11 of 24 on the first scheduled day, so the schedule needs densifying and the gap-guard metric should count distinct hours covered) and #68 (the gap guard has a false-pass blind spot just after UTC midnight), both found while diagnosing the first gap-guard trips; they precede #43 in the M0.5 order

## [2.7.6] - 2026-07-21

### Changed
- The M0 close-out list in docs/ROADMAP.md now carries #65 (rewrite the rollup UPSERT date predicates to half-open ranges), filed at the #61 close-out because the two-date upsert doubled per-run recomputes whose DATE() predicate cannot use an index; it is sequenced to land before the normalized table grows back to its 30-day steady state

## [2.7.5] - 2026-07-21

### Fixed
- The hourly rollup step now anchors on the UTC date instead of the machine's local date, and upserts both the previous and current UTC dates each run (issue #61, found during the #1 restore when a 22:43 EDT run wrote 237,983 UTC-stamped rows but zero rollup rows). Evening runs now update the current day's rollups immediately instead of lagging up to ~4 hours after UTC midnight, and the final minutes of each UTC day no longer depend on the machine surviving to the next local day's self-healing run. The orchestrator's run() accepts an injectable clock so tests can pin the incident scenario in any timezone

## [2.7.4] - 2026-07-21

### Changed
- check-data-health no longer reads the entire raw observations table on every run (issue #59). Migration 0009 adds two indexes on market_observations_raw serving the retention sentinel and the last-ingest check, and the coverage queries were rewritten so the existing normalized-table index serves them (raw-column range comparison instead of a DATE() predicate, and one min/max aggregate per query so SQLite answers each with a single index probe). The pruner's retention deletes inherit the observed_at index for free

## [2.7.3] - 2026-07-21

### Changed
- Retired the ACTIVE OPERATIONAL HAZARD section in CLAUDE.md and the pruner risk row in ROADMAP.md: the issue #1 restore runbook executed on 2026-07-20/21 and hourly ingestion is live again after 105 days (restored 2026-07-21 02:43Z, database rebuilt from 78 GB to 105 MB, rollup history certified and expanded from 22 to 34 dates, all three scheduled tasks re-enabled and observed green). CLAUDE.md now carries a short operational-state note, the machine-caution rule for heavy jobs on this hardware, and the recovery timeline; ROADMAP.md marks the restore done and adds #61 (orchestrator rollup UTC date anchor, found during the restore) to the M0 close-out list

## [2.7.2] - 2026-07-20

### Added
- All three scheduled tasks now wake the machine from sleep (issue #40). schtasks /Create cannot set WakeToRun, so setup_tasks.bat flips it on each task after registration through a PowerShell fetch-modify-write that preserves every other setting, including a Disabled state; for a task that was disabled, the script re-asserts the disable afterward rather than trusting the round-trip. A failed wake-set stops the script with exit 1. With this, the machine may sleep between runs without losing capture hours: Task Scheduler wakes it for each trigger and Windows returns it to sleep on the idle timeout. Wake covers sleep (and hibernate on supporting hardware); a powered-off machine does not wake, which is what the cloud capture path (M0.5) is for
- setup_tasks.bat now verifies the active power plan allows wake timers and prints the elevated powercfg commands to fix it when it does not (warn-only: task registration itself needs no elevation, and both Disable and Important Wake Timers Only block Task Scheduler wakes)

## [2.7.1] - 2026-07-19

### Added
- WoWForecaster-HealthCheck scheduled task: setup_tasks.bat now registers run_healthcheck.bat to fire every 6 hours at :45 (00:45/06:45/12:45/18:45), placed 29 minutes clear of the :16 hourly ingest so a health check never reads the database concurrently with ingestion, and finishing before the 07:00 daily task starts (issue #6)

### Changed
- All three scheduled tasks now run hidden via wscript.exe and run_silent.vbs, which waits for the batch and propagates its exit code, so Task Scheduler's Last Run Result stays truthful with no console flash (this registration change dates to v2.2.2 but was never committed)
- The hourly task registration pins /ST to a :16 anchor, so re-running setup_tasks.bat can no longer silently move the capture phase (the :16 minute avoids the daily-task collision, samples away from Blizzard's top-of-hour snapshot refresh, and stays aligned with the cloud capture cron)
- Re-running setup_tasks.bat preserves a task's Disabled state: schtasks /Create /F recreates tasks enabled, so the script queries each task's state first and re-disables it right after registration, failing loudly if the re-disable does not stick. An operator's decision to disable a task (WoWForecaster-Hourly stays disabled until the issue #1 runbook) survives setup re-runs

## [2.7.0] - 2026-07-19

### Added
- check-data-health now detects the two failures behind the 96-day outage (issue #5). A stale-lock check stats data/db/.hourly.lock read-only and flags [STALE LOCK] when the lock is older than 180 minutes, the same threshold that triggers run_hourly.bat's takeover: a lock that old means the hourly pipeline is wedged or crashed mid-run. A retention sentinel reads the oldest market_observations_raw row by observed_at (the pruner's deletion criterion) and flags [RETENTION VIOLATION] when it is older than `[retention] raw_snapshot_days` plus 2 days of grace: rows that old mean the pruner has stopped enforcing the 30-day Blizzard API ToS window. Both surface in the health report (Hourly lock and Oldest raw row lines) with a new [UNHEALTHY] status label when a check fails while the data itself is still fresh

### Changed
- check-data-health now exits 1 on a stale hourly lock or a retention violation, not just on stale data. run_daily.bat's freshness gate and run_healthcheck.bat's alerting consume the exit code, so both inherit the new failure modes without any script change: a wedged lock or dead pruner now blocks the daily forecast and raises the health alert window

## [2.6.0] - 2026-07-19

### Added
- Scheduled health check with visible failure alerting (issue #4). New scripts/run_healthcheck.bat runs `check-data-health --stale-hours 4` and appends output to logs/health.log. On failure it writes data/outputs/monitoring/health_alert.json (timestamp, exit code, last 20 log lines) and raises a persistent red console window titled "WOW FORECASTER: DATA STALE", at most once per 24 hours (tracked by the mtime of health_window_raised.json). A healthy run deletes both files, so the next distinct failure alerts immediately instead of inheriting a half-spent window. The exit code always mirrors check-data-health, keeping Task Scheduler's Last Run Result truthful even when an alert surface fails, and an unverifiable suppression flag raises anyway (skip-on-uncertainty is what made the 96-day outage silent). Task Scheduler registration lands with issue #6. This check is independent of run_daily.bat's 26-hour freshness gate: different log, threshold, and purpose. Verified during acceptance: `start` still raises a visible window when the parent console runs hidden under run_silent.vbs (windowStyle 0), and the vbs propagates the exit code, so #6 can register the task silently without losing the alert

## [2.5.0] - 2026-07-19

### Added
- Data freshness gate on the daily forecast (issue #12). ForecastStage refuses to run when the newest normalized observation is older than the new `[forecast] max_data_age_hours` config key (default 26 hours; 0 disables): it raises StaleDataError and records the run as failed, so manual run-daily-forecast invocations exit 1 on stale data. run_daily.bat now runs `check-data-health --stale-hours 26` before anything else and exits non-zero with HEALTH ALERT ACTIVE logged when the check fails, so Task Scheduler records the failure. This closes the failure mode behind the 96-day outage, where the daily task generated forecasts from frozen features without anyone noticing
- Red ingestion alert banner above every dashboard tab when the newest DB observation is older than 26 hours or the database is empty

### Fixed
- check-data-health no longer crashes with OperationalError against a real database. Its last-ingest query referenced ingestion_snapshots columns (realm_slug, ingested_at) that the production table has never had; the bug survived since v2.1.0 because the health tests hand-rolled a fixture table with the invented shape. The query now reads MAX(ingested_at) from market_observations_raw (only successful ingests insert raw rows), and the health tests build their fixture with apply_schema() so the fixture cannot drift from the real schema again. Without this fix the new freshness gate would have blocked the daily task even after ingestion is restored, because the gate's health check crashed (exit 1) on fresh data too (issue #12)
- dashboard/data_loader.py no longer crashes on import when Streamlit is not installed: the no-streamlit cache fallback did not accept the @_CACHE(ttl=N) form the loaders use

## [2.4.9] - 2026-07-16

### Fixed
- run_hourly.bat no longer treats a leaked lock file as an active run forever. Before skipping, it now checks the lock age with PowerShell (LastWriteTime); a lock older than 180 minutes is deleted, STALE LOCK TAKEOVER is logged to hourly.log, and the run continues. A fresh lock still logs SKIPPED and exits 0. A failed age check also continues the run, because skip-on-uncertainty is what turned one crashed run on 2026-04-15 into a 96-day silent outage, while a rare double run is covered by the SQLite busy timeout. Covered by the repo's first Windows-only tests (tests/test_scripts/), which run the script in an isolated temp tree with no venv so the real pipeline cannot start (issue #3)

## [2.4.8] - 2026-07-16

### Fixed
- check-data-health coverage windows are now anchored to the UTC calendar date instead of the local date, matching the UTC timestamps on observations, and collect_health_report() accepts an injectable as_of date. This fixes the health test that reported a phantom day-zero gap whenever the local date and UTC date disagreed, on the Linux CI runner and near midnight locally (issue #49)
- The two CLI-locator scheduler tests now run both venv layouts (Windows Scripts/*.exe and Linux bin/ suffix-free) on every host by monkeypatching the platform check, instead of assuming the Windows layout and failing on the Linux runner. No production change: the locator already handled both platforms (issue #49)
- With issues #7 and #8, this completes the green-CI tier: the full suite passes in CI on Python 3.11 and 3.12 (issue #49)

## [2.4.7] - 2026-07-16

### Fixed
- The pruner boundary test no longer flakes by time of day: SnapshotPruner.prune() accepts an injectable reference clock (default: current UTC time), the file-fixture tests share that clock with the cutoff, and a new companion test pins the other side of the boundary (a file one day past retention is deleted). Semantics are unchanged: a file dated exactly retention_days ago is kept (issue #8)
- Corrected the pruner module docstring, which claimed normalised observations are never pruned. They are deleted together with their parent raw rows; the durable derived layers (daily rollups, Parquet features, model weights) are what the pruner never touches (issue #8)

## [2.4.6] - 2026-07-16

### Fixed
- Item-forecast tests no longer depend on the wall clock: _generate_item_forecasts() and _fetch_cold_start_blend_data() accept an optional run_date anchor (default: today, so pipeline behavior is unchanged), and the tests pin it to 2026-03-09 beside their fixture prices. This restores the 8 tests that started failing when the calendar moved past their fixed dates, and several neighboring tests that had been passing vacuously on empty query results now exercise their real assertions (issue #7)

## [2.4.5] - 2026-07-15

### Added
- Dependabot config scoped to ruff: new linter releases arrive as monthly PRs gated by CI instead of drifting into the build (issue #44)

### Changed
- Mechanical ruff 0.15.2 conformance sweep across wow_forecaster/ and tests/: modern union syntax for optional types, datetime.UTC alias, unused imports removed, imports sorted (issue #44)
- Manual lint conformance: all long lines rewrapped to the 100-char limit, exception chaining made explicit (raise from), zip calls declare strict=, blind pytest.raises(Exception) narrowed to ValidationError, dead test scaffolding removed (issue #44)
- ruff pinned exactly (==0.15.2) with per-file exemptions for the Typer and sklearn API conventions; UP042 ignored because StrEnum conversion would change str() output persisted to DB and CSV (issue #44)

### Fixed
- CI reaches the pytest step again on both Python versions; the lint step no longer fails on rules that postdate the code (issue #44)
- CI matrix no longer cancels the second Python version when the first fails, so both report full test results (issue #44)

## [2.4.4] - 2026-07-15

### Changed
- Merges to main now require a pull request: a branch protection ruleset blocks direct pushes, force pushes, and deletion, with no bypass for admins (issue #46)
- Merged branches are deleted instead of kept frozen; the repo auto-deletes head branches on merge, and the four already-merged branches were removed
- Versioning moved to stamp commits: work commits log under Unreleased, and a single stamp commit at PR-open sets the version for the whole PR
- Each GitHub milestone description now opens with a numbered work-order list matching docs/ROADMAP.md

## [2.4.3] - 2026-07-12

### Fixed
- README CI badge now points at the real repository instead of the yourusername placeholder, so it renders actual CI status (issue #45)

## [2.4.2] - 2026-07-12

### Changed
- Development model: work now lands on short-lived type-prefixed branches (feat/, fix/, docs/, chore/) cut from main and merged per issue, recorded as a Branch Workflow section in CLAUDE.md; the long-lived feature/portfolio-showcase branch was merged to main (issue #10) and frozen
- Cloud capture activation shortened: the workflow is on main but disabled by hand, so the remaining steps are the bucket, the six secrets, gh workflow enable, and one manual dispatch (README, docs/cloud-capture.md, CLAUDE.md updated to match)

## [2.4.1] - 2026-07-12

### Added
- Issue #44 filed and slotted first in the green-CI tier of the work order: CI fails at the ruff lint step before tests run because `ruff>=0.4` floats to releases enforcing rules the codebase predates (782 findings under 0.15.2), so pytest results are invisible on GitHub

### Changed
- docs/ROADMAP.md and CLAUDE.md: M0 issue list and work order updated for #44

## [2.4.0] - 2026-07-12

### Added
- Cloud snapshot fetcher (issue #42): `python -m wow_forecaster.ingestion.cloud_fetch` plus a scheduled GitHub Actions workflow capture the hourly commodities snapshot from always-on infrastructure and upload it gzipped to a private S3-compatible bucket, so capture no longer requires the desktop to be on. Reuses the existing Blizzard client and snapshot writer, so cloud objects carry the identical envelope local ingest produces
- Failure paths are loud by design: refuses implausibly small snapshots (default floor 50,000 records), retries fetch and upload, exits 3 when the trailing 24 hours of objects have gaps, and reports missing configuration by variable name only
- `[cloud]` optional dependency group (boto3) for running the fetcher outside the workflow
- README setup section covering the bucket, lifecycle rule, repository secrets, and first-run verification; activation is a manual one-time step for the repository owner

## [2.3.9] - 2026-07-12

### Added
- docs/cloud-capture.md: cloud capture design record (issue #41). GitHub Actions hourly workflow plus a private Cloudflare R2 bucket with a 30-day lifecycle rule; sizing measured from a real snapshot (58.9 MB raw, 2.2 MiB at gzip level 9, 25.7x, ~1.5 GiB per rolling 30-day window); compliance mapping, failure-visibility plan, and the one-time activation checklist

## [2.3.8] - 2026-07-12

### Changed
- Milestones renumbered to match the decisive work order: paper trading P&L and ranking A/B is now M2 (was M4), the PostgreSQL + dbt warehouse is M3 (was M2), and BI dashboards are M4 (was M3). The live A/B test needs weeks of data to mature, so its clock starts right after model validation instead of waiting behind infrastructure work, and the make-gold answer exists before dashboards are built to showcase it
- docs/ROADMAP.md: added a Work order section with the issue-level sequence, most urgent first (stop data loss, green CI, harden, restore, validate, then build outward)

## [2.3.7] - 2026-07-12

### Changed
- Milestone M7 renamed to M0.5 (unattended capture) and moved to run immediately after M0: the design and cloud fetcher (#41, #42) depend on nothing local and stop further unrecoverable data loss, so they no longer wait behind M1-M6; only the catch-up command (#43) needs the restored pipeline

## [2.3.6] - 2026-07-12

### Added
- Milestone M7 (unattended capture) on the roadmap: cloud-hosted hourly snapshot fetcher, private object storage with a 30-day lifecycle rule, and a local catch-up ingestion command, so capture no longer depends on the desktop being on (issues #41-#43)
- M0 issue #40: wake-to-run task settings so the machine can sleep between scheduled runs

### Changed
- docs/ROADMAP.md and CLAUDE.md updated for milestone M7 and the extended issue range (#1-#43)

## [2.3.5] - 2026-07-12

### Added
- docs/ROADMAP.md: next-phase roadmap (M0 restore/harden operations through M6 publish) with dependency graph and risk register; work tracked as GitHub milestones M0-M6 with issues #1-#39

### Changed
- CLAUDE.md: documented the active ingestion outage (lock leaked 2026-04-15, ingestion dead since) and the lock-clearing hazard (orchestrator auto-prune would delete all rows older than 30 days; rollup tables are incomplete), corrected the date-spine description (Python-generated spine over rollup fast path, not a recursive CTE), and noted that migrations end at 0008

## [2.3.4] — 2026-04-07

### Added
- `checkpoint-db` CLI command to force WAL checkpoint when the write-ahead log grows too large
- Automatic WAL checkpoint step in hourly orchestrator pipeline (after prune, before monitoring outputs)

### Fixed
- WAL file growth unbounded (no checkpoint logic existed); 4.2 GB WAL was causing all DB operations to exceed lock timeout

## [2.3.3] — 2026-04-06

### Fixed
- Rollup tables now update during hourly pipeline (was silently failing due to missing `self._conn` attribute in orchestrator)
- IngestStage no longer holds SQLite write lock during Blizzard API HTTP fetch (connection split into read/fetch/write phases)
- All pipeline stages now use config-driven `busy_timeout_ms` instead of hardcoded 5-second default
- Default `busy_timeout_ms` increased from 5s to 30s to handle realistic batch operation contention
- Overlapping hourly pipeline runs prevented via lock file guard in `run_hourly.bat`
- Version regression from v2.3.2 to v2.2.3 corrected (was a typo in previous commit)

## [2.3.2] — 2026-04-05

### Changed
- Migrated `archetype_features.py` to use pre-aggregated rollup tables for faster feature queries

## [2.3.1] — 2026-04-05

### Fixed
- `backfill-rollups` now uses `get_connection` as context manager (was leaking connections)

## [2.3.0] — 2026-04-05

### Added
- Pre-aggregated rollup tables (`archetype_rollups`, `item_rollups`) for 110M-row performance optimization
- `backfill-rollups` CLI command to populate rollup tables from historical data
- Automatic rollup update step in hourly orchestrator pipeline

## [2.2.3] — 2026-04-06

### Added
- Related Projects section in README linking to alt-army-guide (profession setup guide for executing on forecaster recommendations)

## [2.2.2] — 2026-03-20

### Fixed
- Scheduled tasks no longer open a visible cmd.exe window; `setup_tasks.bat` now uses `wscript.exe` + `run_silent.vbs` wrapper for silent execution

### Added
- `scripts/run_silent.vbs` — generic VBS launcher that runs batch files with no console window

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
