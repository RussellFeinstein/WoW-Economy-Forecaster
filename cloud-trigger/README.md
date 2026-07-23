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
Worker secret and is never committed. Fine-grained PATs expire within a year,
so set a renewal reminder when you create it.

## Verify

- `wrangler tail` streams the Worker's invocation logs; each cron firing logs
  the dispatch status (204 on success).
- On GitHub, the Actions tab shows `Cloud snapshot capture` runs appearing at
  :16 and :46, marked `workflow_dispatch` rather than `schedule`.
- Within a day the in-run gap guard should report at least 20 distinct hours
  covered and the runs go green.
