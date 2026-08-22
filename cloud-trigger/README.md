# Cloud-capture trigger Worker

A Cloudflare Worker that POSTs `workflow_dispatch` to run the cloud-capture
GitHub workflow ([../.github/workflows/cloud-snapshot.yml](../.github/workflows/cloud-snapshot.yml))
on a reliable cron.

## Why

GitHub Actions delivers only about 11 of 24 scheduled cron firings per day for
this repo, and densifying the cron does not change it: the limiter is on run
delivery, not on schedule expressions (issue #83). Cloudflare cron triggers
fire reliably, and `workflow_dispatch` runs are created on demand and bypass
GitHub's schedule backlog, so this Worker is the primary trigger. GitHub's own
schedule is kept as a single `:06` fallback that also serves as the dead-man
alarm: if this Worker or its token dies, capture drops to the fallback, the
in-run gap guard falls below 20 distinct hours, and the runs go red.

Design record and the full activation checklist:
[../docs/cloud-capture.md](../docs/cloud-capture.md).

## Deploy

Prerequisites: the Cloudflare account that holds the R2 buckets, and `wrangler`
(`npm install -g wrangler`, then `wrangler login`).

```
cd cloud-trigger
wrangler secret put GH_PAT     # paste the fine-grained PAT when prompted
wrangler deploy
```

The non-secret config (owner, repo, workflow, ref) lives in `wrangler.toml`.
`GH_PAT` is a fine-grained Personal Access Token scoped to this repo only, with
the **Actions** permission set to read and write. It is stored as an encrypted
Worker secret and is never committed.

Deploy from `cloud-trigger/`, not the repo root. From the root, wrangler finds
no `wrangler.toml` and fails with a misleading error about static files, and
`wrangler secret put` fails with "Required Worker name missing".

## Token expiry

**Set the token to no expiration.** GitHub's creation UI defaults the
expiration dropdown to 30 days, and accepting that default is how the first
token died: created 2026-07-23, expired 2026-08-22, silently. Nothing chose 30
days, it was just never changed.

No expiration is the right call for this token specifically, because the two
risks are lopsided. A leak lets someone dispatch one workflow on one repo and
nothing else, which is the smallest blast radius the API offers. An expiry, by
contrast, degrades capture without announcing itself and costs hours of market
data that the API cannot serve again.

## Rotating the token

1. Create the replacement at github.com, Settings, Developer settings,
   Personal access tokens, Fine-grained tokens. Repository access: only this
   repo. Permissions: **Actions, read and write**, nothing else. Expiration:
   none.
2. Set it on the Worker. From `cloud-trigger/`:
   ```
   wrangler secret put GH_PAT
   ```
   From anywhere else, name the Worker explicitly:
   ```
   wrangler secret put GH_PAT --name wow-forecaster-cloud-trigger
   ```
   This uploads a new Worker version by itself. No separate `wrangler deploy`
   is needed, and the secret survives later deploys.
3. Confirm a dispatch lands at the next `:16` or `:46`:
   ```
   gh run list --workflow=cloud-snapshot.yml --limit 5
   ```
   The new run should say `workflow_dispatch`, not `schedule`.

## When the trigger dies

The symptom is quiet, because capture does not stop. `workflow_dispatch` runs
disappear from the Actions tab and only the `:06` `schedule` runs remain, so
delivery drops to roughly 11 hours a day and the in-run gap guard goes red
within a day.

Diagnosing it, cheapest first:

- `gh run list --workflow=cloud-snapshot.yml --limit 10` and look at the
  `event` column. All `schedule` and no `workflow_dispatch` is the signature.
- `gh workflow run cloud-snapshot.yml` dispatches by hand using your own
  GitHub credentials rather than the Worker's. If that succeeds, the workflow
  and the repo are fine and the fault is in the Worker or its token. It also
  captures a real snapshot, so it is worth doing regardless.
- `wrangler deployments list --name wow-forecaster-cloud-trigger` shows a
  "Secret Change" entry with a timestamp, which tells you whether a rotation
  actually landed and when.
- `wrangler secret list --name wow-forecaster-cloud-trigger` shows that
  `GH_PAT` exists. It prints names only, never values.
- `wrangler tail wow-forecaster-cloud-trigger` streams the next cron firing.
  Note the Worker name is a positional argument here; `--name` is not accepted
  by `tail`, unlike `secret` and `deployments`. `worker.js` throws on any
  non-2xx, so the log carries the HTTP status and GitHub's response body: 204
  is success, 401 or 403 means the token is missing **Actions: read and
  write** on this repo, and silence at a `:16` or `:46` slot means the cron
  itself is not firing rather than the token being rejected.

## Verify

- `wrangler tail` streams the Worker's invocation logs; each cron firing logs
  the dispatch status (204 on success).
- On GitHub, the Actions tab shows `Cloud snapshot capture` runs appearing at
  :16 and :46, marked `workflow_dispatch` rather than `schedule`.
- Within a day the in-run gap guard should report at least 20 distinct hours
  covered and the runs go green.
