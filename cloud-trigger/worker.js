/**
 * Cloud-capture trigger Worker.
 *
 * GitHub Actions delivers only about 11 of 24 scheduled cron firings per day
 * for this repo (issue #83). The limiter is on run delivery, not on schedule
 * expressions, so densifying the cron does not help. This Worker runs on
 * Cloudflare cron triggers (which fire reliably and on demand) and POSTs a
 * workflow_dispatch to run cloud-snapshot.yml, bypassing GitHub's schedule
 * backlog. GitHub's own schedule stays as a single :06 fallback and alarm.
 *
 * Config (wrangler.toml [vars]): GH_OWNER, GH_REPO, GH_WORKFLOW, GH_REF.
 * Secret (never committed; set with `wrangler secret put GH_PAT`): GH_PAT,
 * a fine-grained PAT scoped to this repo with the Actions permission set to
 * read and write.
 */

async function dispatchCapture(env) {
  if (!env.GH_PAT) {
    throw new Error("GH_PAT secret is not set; run `wrangler secret put GH_PAT`");
  }
  const owner = env.GH_OWNER;
  const repo = env.GH_REPO;
  const workflow = env.GH_WORKFLOW;
  const ref = env.GH_REF || "main";
  const url = `https://api.github.com/repos/${owner}/${repo}/actions/workflows/${workflow}/dispatches`;

  const resp = await fetch(url, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${env.GH_PAT}`,
      Accept: "application/vnd.github+json",
      "X-GitHub-Api-Version": "2022-11-28",
      "User-Agent": `${owner}-cloud-trigger`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ ref }),
  });

  // A successful workflow_dispatch returns 204 No Content.
  if (!resp.ok) {
    const detail = await resp.text();
    throw new Error(`workflow_dispatch failed: HTTP ${resp.status} ${detail}`);
  }
  return resp.status;
}

export default {
  // Cloudflare cron-trigger entrypoint (schedule set in wrangler.toml). A throw
  // here marks the invocation failed, so a dead PAT or API outage is visible in
  // the Cloudflare dashboard and in `wrangler tail`.
  async scheduled(event, env, _ctx) {
    const status = await dispatchCapture(env);
    console.log(`Dispatched ${env.GH_WORKFLOW} at cron "${event.cron}": HTTP ${status}`);
  },
};
