---
paths:
  - "learning/**"
  - "wow_forecaster/learning/**"
---

# Learning track

Loaded when working with the learning curriculum, question banks, or the drift-guard anchors. Root context: [CLAUDE.md](../../CLAUDE.md).

## Learning Track (v2.11.0)
- Third parallel track alongside docs/ROADMAP.md (research arc) and PLAN.md (lifecycle arc). This one owns understanding; its labs are drawn from the open work in the other two. Do NOT cross-reference it from ROADMAP.md: the ROADMAP/PLAN separation is deliberate and this inherits it
- Split: content is data at the repo root in [learning/](../../learning/) (curriculum.toml, modules/, banks/, labs/), engine is code in [wow_forecaster/learning/](../../wow_forecaster/learning/). Same pattern as config/events/*.json to events/seed_loader.py. TOML because tomllib is stdlib, so zero new dependencies
- `learn` is a Typer sub-app (`app.add_typer`), not more flat commands on cli.py: status, next, module, exam, lab, validate, reset. learning/cli.py imports typer + stdlib only at module level and lazy-imports loader/store inside command bodies, so the group costs nothing at `wowfc` startup
- **The drift guard is the load-bearing idea**: every question cites `source` (path) + `anchor` (verbatim SINGLE-LINE snippet), never a line number, which is wrong the moment a line is inserted above it. The CLI resolves the anchor to a current line at display time (self-healing citation). `integrity.check_content()` is called by BOTH `learn validate` and tests/test_learning/test_bank_integrity.py, so the authoring rule and the enforced rule cannot diverge
- Anchor reads pin `encoding="utf-8"`: the quoted docstrings contain real Unicode (arrows, ≤, ±) and Path.read_text() defaults to cp1252 on Windows and raises. Anchors must be single-line so CRLF vs LF never breaks a match (enforced by a pydantic validator)
- Commit-SHA checks are skipped when `git rev-parse --is-shallow-repository` is true: ci.yml uses actions/checkout@v4 with no fetch-depth, so CI gets a depth-1 clone where no historical SHA resolves
- Progress in its own SQLite file (`data/learn/progress.db`, gitignored, `WOWFC_LEARN_DB` seam), NEVER the product DB: that one is copied into every durable backup and is the M3 warehouse source
- SM-2 variant, 4 grades, injectable clock. ReviewState carries prev_ease/prev_interval_days so a same-day re-grade rewinds and replaces rather than compounding; passing grades schedule >= 1 day out, so re-running a drill serves no repeat of a card graded good or better, while `again` stays due today
- All 20 modules are authored (348 questions total): M06 pilot in v2.11.0, the other 19 filled in by a verified agent fleet in v2.11.1, plus m03-q19 added by issue #123 in v2.14.13 and m13-q19 by issue #136. A bank grows when the code it teaches changes: #123 broke one M03 anchor and put three more answers slightly out of date, and the response was to re-anchor and extend them rather than delete, because the module outlives the defect it was written against. #136 widened Dependabot's scope past ruff, which did not break the m13-q10 anchor but made its answer's "scoped to ruff" wording wrong, so the answer was extended to name the test the exemption actually applies (does a user experience anything different) rather than the single tool that happened to satisfy it. The curriculum still reports "not authored yet" for any declared module without a bank, which is the state a future milestone re-enters by adding a module before its bank; a bank with no declared module is an integrity failure
- Labs are real open work: lab-01 purge/embargo (M06, issue to be filed), lab-02 LightGBM backtest (#16), lab-03 sync-snapshots drain (#43 acceptance), lab-04 realization ledger (#13)
