# Cloud snapshot capture design

Status: accepted 2026-07-12. Decision record for issue [#41](https://github.com/RussellFeinstein/WoW-Economy-Forecaster/issues/41); implemented by [#42](https://github.com/RussellFeinstein/WoW-Economy-Forecaster/issues/42) (fetcher) and consumed by [#43](https://github.com/RussellFeinstein/WoW-Economy-Forecaster/issues/43) (catch-up ingestion). Part of milestone M0.5.

## Problem

The Blizzard commodities endpoint serves only the current snapshot. Any hour that is not captured is gone: there is no history API to backfill from. Capture currently runs on the desktop through Task Scheduler, so a machine that is asleep (until #40 lands), powered off, or mid-reinstall loses hours permanently. The 96-day outage documented in #1 and #2 was a software failure, but the same loss happens whenever the machine is simply not on.

This design moves hourly capture to infrastructure that is always on, at zero monthly cost, without weakening the compliance position (Blizzard API ToS 2.r: raw API data is deleted within 30 days).

## Measured sizing

Numbers from a real snapshot (`commodities_us_20260415T111711Z.json`, the last hour captured before the outage):

| Measure | Value |
|---|---|
| Raw envelope JSON | 58,942,006 bytes (56.2 MiB) |
| Records | 313,695 |
| gzip level 9 | 2,297,647 bytes (2.2 MiB), 25.7x |
| gzip level 6 | 2,444,275 bytes (2.3 MiB), 24.1x |
| Rolling 30-day window at level 9 (720 objects) | ~1.5 GiB |

The envelope is indent-formatted JSON with repeated keys, which is why gzip does so well. At ~1.5 GiB per rolling 30-day window, a 10 GB free tier holds the full window with over 6x headroom. Uncompressed (~42 GB per window) no free tier fits, so the fetcher always compresses.

## Decisions

### Runner: GitHub Actions scheduled workflow

The repo is public, so Actions minutes on standard runners are free. The job needs no state between runs, secrets live in the repository secret store, the workflow file is versioned next to the code it runs, and every run leaves a visible log. Failure notifications are built in.

Accepted caveats, with mitigations under Failure visibility below:

- Cron is best effort, and GitHub delivers only about 11 of 24 hourly firings for this repo, deterministically (measured across three days; #67). Densifying the schedule does not help: the cap is on run delivery, not on schedule expressions. The fix is an external trigger (see Trigger below), with the schedule here kept as a single :06 fallback and alarm. Duplicate snapshots within an hour coexist under timestamped keys and stay inside the lifecycle window (worst case a few objects per hour x ~1.9 MB x 30 days, comfortably inside the free tier).
- Scheduled workflows are disabled after 60 days without repository activity. GitHub emails a warning first, and `gh workflow enable "Cloud snapshot capture"` turns it back on. The repo is under active development through M6, so this is a documented recovery path rather than an expected event.
- Scheduled workflows run only from the default branch. The schedule does not fire until the workflow file lands on `main` (see Activation checklist).

Rejected:

- Supabase pg_cron + Edge Function: Supabase Storage free tier is 1 GB, below the ~1.5 GiB window even compressed, and it adds a second platform for no gain.
- Always-free VM (Oracle Cloud and similar): real cron with no keepalive concern, but it adds an account, OS patching, and credential handling on a box nobody maintains. Too much operational surface for one hourly HTTP call.

### Trigger: external Cloudflare Worker cron (added 2026-07-23, issue #83)

The runner decision above assumed GitHub's own `schedule:` cron would fire the job hourly. It does not. Measured across all runs on three consecutive days, GitHub delivered only about 11 of 24 hourly firings for this repo, in the same dead windows each day, and #67's densification to `16,36,56` changed nothing (07-22 delivered 11 runs from 72 slots). The limiter is on run delivery, not on schedule expressions, so cron density is a dead lever and the 20-distinct-hours gap guard cannot pass on GitHub cron alone.

`workflow_dispatch` runs, by contrast, are created on demand and start within about a second (every activation dispatch on 07-20 fired instantly). So the trigger moves off GitHub's schedule:

- A Cloudflare Worker cron on the account that already holds the R2 buckets POSTs `workflow_dispatch` at :16 and :46. Source and deploy steps: [../cloud-trigger/](../cloud-trigger/). It authenticates with a fine-grained PAT scoped to this repo with the Actions permission set to read and write, stored as an encrypted Worker secret (`GH_PAT`), never in the repo. The PAT is the first credential that grants control *into* GitHub rather than access *out* to a data source; a leak lets an attacker trigger this one workflow and nothing else.
- The GitHub `schedule:` is thinned to a single `:06` firing. It is both a redundant fallback and the dead-man alarm: if the Worker or its PAT dies, capture falls back to GitHub-only delivery (~11 hours a day), the gap guard drops below 20 distinct hours, and the runs go red. The guard floor stays at 20 deliberately; a floor the failure mode can satisfy would hide the failure.

Rejected: a hosted cron service (cron-job.org and similar) POSTing the dispatch directly. It needs no code, but a third-party service would store the GitHub PAT in its config, a larger blast radius than a Cloudflare Worker secret, and it adds another account. Rejected: the Worker doing the full capture itself (Blizzard fetch, gzip, R2 PUT via a native binding). It would eliminate the S3 credentials and GitHub entirely, but it reimplements the Blizzard client and the snapshot envelope in JavaScript (two implementations to keep in sync), and the ~59 MB raw payload is risky to fetch and gzip inside a free Worker's 128 MB memory and CPU-time limits. Keeping "Worker triggers, Actions captures" leaves one Python capture path and stays inside every limit.

### Storage: Cloudflare R2, private bucket

S3-compatible API (boto3 works unchanged), 10 GB-month free storage, object lifecycle rules, and no egress fees. Zero egress matters because the catch-up path (#43) downloads the backlog to the desktop; on R2 that download is free at any size and any frequency.

Backblaze B2 is the fallback: also S3-compatible with lifecycle rules and a 10 GB free tier, with free egress capped at 3x stored bytes per day (still comfortable here). The fetcher speaks the S3 API through a configurable endpoint, so switching providers is a secrets change, not a code change.

### Fetcher: reuse the package, not a parallel implementation

The workflow installs the package without dependencies (`pip install --no-deps .` plus `httpx` and `boto3`) and runs `python -m wow_forecaster.ingestion.cloud_fetch`. The module reuses `BlizzardClient.fetch_commodities()`, `build_snapshot_path()`, and `save_snapshot()`, so cloud objects carry the same envelope the local pipeline writes, by construction rather than by convention.

The import chain for those modules is stdlib-only (httpx is imported lazily inside methods), which is what makes the no-deps install work. If a future refactor adds a heavy import to that chain, the hourly workflow fails loudly and the email says why.

## Object layout and envelope

Bucket keys mirror the local snapshot layout under `data/raw/snapshots/`:

```
blizzard_api/YYYY/MM/DD/commodities_us_<YYYYMMDDTHHMMSS>Z.json.gz
```

Each object is the standard snapshot envelope, gzipped:

```json
{
  "_meta": {
    "source": "blizzard_api",
    "type": "commodities",
    "region": "us",
    "is_fixture": false,
    "run_slug": "gha_<actions_run_id>",
    "fetcher": "cloud",
    "written_at": "2026-07-12T14:16:05Z"
  },
  "data": [ { "item_id": 190396, "realm_slug": "us", "buyout": 0, "bid": 0,
              "unit_price": 1000, "quantity": 25, "time_left": "VERY_LONG" }, ... ]
}
```

`run_slug` carries the Actions run ID for provenance, and `fetcher: "cloud"` lets #43 and audits distinguish cloud objects from desktop captures. Data records have the same seven keys local ingest writes. Catch-up is therefore: download, gunzip, drop the file into the local layout, replay through the existing ingest path.

## Compliance mapping

| Requirement | Mechanism |
|---|---|
| Raw API data deleted within 30 days (ToS 2.r) | Bucket lifecycle rule deletes objects at 30 days. Deletion is infrastructure, not another scheduled job that can die silently. |
| No redistribution of raw API data | The bucket is private. Nothing raw is committed to the (public) repo. |
| Credentials never in the repo | Blizzard and R2 credentials are GitHub Actions secrets, added by hand in repo settings. The fetcher reads them from the environment and never prints values; missing variables are reported by name only. |

## Failure visibility

The outage in #1 happened because a failure path exited 0. Every failure path here is loud:

- A fetch, sanity, or upload failure exits nonzero, the run shows red, and GitHub emails the failure to the last committer of the workflow's cron line.
- A sanity check refuses snapshots with implausibly few records (default minimum 50,000 against a normal ~314,000), so an API brownout cannot quietly store an empty hour.
- After each upload, the run lists the trailing three day-prefixes (today, yesterday, day before yesterday) and fails (exit 3, snapshot already uploaded) when the trailing 24 hours cover fewer than 20 distinct capture hours. Counting distinct hours instead of raw objects keeps the guard meaning "hours are being missed" no matter how many triggers fire per hour (the Worker's :16/:46 plus the :06 fallback; #83), where a gappy day can still hold plenty of objects. A silent cron skip therefore surfaces within an hour, on the next run that does fire. The third prefix exists so the just-after-midnight window still sees objects older than 24 hours; with two prefixes that window misread sparse days as bootstrap (#68).
- Residual blind spots, accepted: an outage of 48+ hours empties all listed prefixes and still looks like bootstrap on resume (the failed runs during it already emailed), and an Actions-platform outage stops the alerting along with the capture. Once #43 lands, the local health check (#5) adds an independent second check: newest-cloud-object age, measured from the desktop.

## Activation checklist (manual, one time)

Steps for the repo owner; none of them belong in git:

1. Create a Cloudflare R2 bucket (private, default settings), for example `wow-forecaster-snapshots`.
2. Add a lifecycle rule on the bucket: delete objects 30 days after creation.
3. Create an R2 API token scoped to that bucket with read and write access.
4. Add six repository secrets: `BLIZZARD_CLIENT_ID`, `BLIZZARD_CLIENT_SECRET`, `SNAPSHOT_S3_ENDPOINT` (the account R2 endpoint URL), `SNAPSHOT_S3_BUCKET`, `SNAPSHOT_S3_ACCESS_KEY_ID`, `SNAPSHOT_S3_SECRET_ACCESS_KEY`.
5. Done 2026-07-12: the workflow reached `main` with the #10 merge and was then disabled by hand so the schedule cannot fire before the secrets exist. After step 4, enable it: `gh workflow enable "Cloud snapshot capture"` (or the Actions tab).
6. Trigger once by hand (Actions tab, Cloud snapshot capture, Run workflow) and confirm the object appears in the bucket at a plausible size (~2.2 MiB).
7. Deploy the trigger Worker (issue #83), so capture no longer depends on GitHub's schedule. Create a fine-grained PAT scoped to this repo with the Actions permission set to read and write. Then from [../cloud-trigger/](../cloud-trigger/): `wrangler secret put GH_PAT` (paste the PAT), `wrangler deploy`. The non-secret config is already in `wrangler.toml`. Fine-grained PATs expire within a year, so set a renewal reminder. Confirm with `wrangler tail` and the Actions tab that dispatch runs appear at :16 and :46.
8. Acceptance per #42/#83: a rolling 24 hours covers at least 20 distinct capture hours and the gap guard exits 0; killing the Worker or its PAT drops delivery to fallback-only and trips the guard red within a day; one forced failure (for example, temporarily rename a secret) produces the failure email; lifecycle deletion verified on a short-TTL test prefix or a manually aged object.

## Out of scope, noted for later

- Capturing the API's `Last-Modified` response header into `_meta` would let #43 dedupe identical snapshots cheaply. `BlizzardClient` discards headers today; not worth touching the shared client in this milestone. Content hashes already give an equivalent, slightly costlier dedupe.
- Per-realm (non-commodity) auctions, EU region, and news capture stay desktop-only until there is a reason to move them.
