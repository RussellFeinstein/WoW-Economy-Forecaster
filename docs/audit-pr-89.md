<!-- Promoted from ~/.claude/plans/audit-pr-89-moonlit-lerdorf.md on 2026-07-26 -->

# Audit of PR #89: "Audit the repo as a portfolio artifact and plan the lifecycle work (v2.10.3)"

Merged 2026-07-24 (merge c6b7d68). Docs-only: PLAN.md (528 lines), LESSONS.md (72 lines), CHANGELOG entry, stamp to 2.10.3. Audited 2026-07-26 by verifying every checkable claim against the code (3 parallel verification passes plus direct spot-checks), plus PR process conformance and drift since merge.

Findings below are recorded as found on 2026-07-26. Everything actionable in them was corrected the same day in v2.11.3; see [Remediation](#remediation-approved-and-executed-2026-07-26-v2113) at the end.

## Verdict

**The PR holds up.** Of ~40 checkable claims, none of the substantive technical findings is wrong; most citations were line-exact. The defects found are small and mostly mechanical:

1. **One factual error, already self-caught:** "41 CLI commands" (actual 40), corrected 3 hours after merge by the learning-track commit, with a note admitting the error.
2. **One figure that was never right:** "README is roughly 70 percent CLI reference" (PLAN.md:284). Actual: the CLI Reference section is 325/851 lines = 38%; even the widest command-heavy reading is 57%. Also "zero images" overstates: the README has a rendering Mermaid architecture diagram (no results/charts, which is the real point).
3. **Stale line citations from post-merge drift:** the learn sub-app inserted 8 lines at the top of cli.py, so PLAN.md's `cli.py:3620` -> now 3628, `cli.py:1033` -> now 1041, "4,508 lines" -> 4,516. All other 8 file:line cites still exact.
4. **PLAN.md now internally inconsistent about its era:** the v2.11.0 correction updated the command count in the inventory (line 124) but left the tests row (line 131: "71 files, ~19,000 lines, 1,481 passing"; now 77 / 21,404 / 1628) and the README line count (line 284: 851; now 894).
5. **Doc-drift list half-resolved:** of the four README drifts PLAN.md names, test counts and command count were fixed in v2.11.x; **"37 model features" (README:78, :767; actual 40) and "21 tables" (README:70, :705, and CLAUDE.md:47; actual 23) were still live on main at audit time.** Fixed in v2.11.3, which also added the two `daily_rollup_*` tables the README's schema listing had never enumerated.
6. **Three places the audit UNDERSTATES the problems it found** (verified, new evidence):
   - **DS-3 is worse than stated.** `persist_backtest_run` sits inside the per-horizon loop (pipeline/backtest.py:145,174), so each horizon gets its own `backtest_run_id` and the newest run is always the h=3 run. drift.py:597-604 and health.py:143-150 pick the newest run then query `horizon_days = 1` -> zero rows -> baseline None. So even the 1d ratio is uncomputable: `run_all`'s only error-drift check always degrades to `DriftLevel.NONE` and `evaluate-live-forecast` reports "unknown" for 1d too. PLAN.md's "the 7d and 28d ratios can never be computed" implies 1d works; it does not. (Also: run_all hardcodes `check_error_drift(horizon_days=1)` at drift.py:367; 7/28 are never even requested.)
   - **DS-1 has a second leak vector:** trainer.py:93-97's 80/20 row-count fallback slices rows by index with no date sort (Parquet row-group order), uncovered by the "NEVER random" docstring and unmentioned in the audit.
   - **`auto_retrain_on_critical` is inert, not just off:** zero code consumers anywhere (only config.py:244, default.toml:171). PLAN.md:218 says "exists and is off"; learning/banks/m19.toml already states the sharper truth.
   - (Also, minor: the drift ratio compares live MAE vs AVG(price_gold) from normalized obs against baseline abs_error vs daily-agg price_mean: different actual-price definitions on the two sides, a third defect beyond pooling and horizons.)

**Process conformance: fully clean.** Branch cut from then-latest main; work commit plus dedicated stamp commit; PR title carries (v2.10.3); patch bump correct for docs; CHANGELOG hyphen-form entry; CI green 3.11/3.12; zero em dashes and zero AI-flavored vocabulary in PLAN.md, LESSONS.md, the changelog section, and the PR body; the claimed one-directional ROADMAP interlock is real (zero references from ROADMAP/README to PLAN.md); all referenced issues #13-#19 exist, open, matching descriptions; the postmortem link resolves.

Out-of-scope observations (report only, no action): CHANGELOG.md:341 (v2.1.0, March) uses an em-dash header; it stays until that line is next touched, per the touch-only rule. PR merged with no review; normal for this solo repo. One verification-pass nuance was refuted on direct check: the two model-artifact globs produce identical patterns (cli.py's `h` is already "7d"), so the only selection divergence is mtime vs lexicographic ordering, exactly as PLAN.md said.

## Detailed claim verification (for reference)

- **DS-1 (label leakage):** trainer.py:88-97 splits purely on obs_date (lines 90-92), no purge/embargo; validation_split_days=14 (config.py:166, default.toml:130); horizons [1,7,28]; early stopping consumes the same leaked val set (lgbm_model.py:201-217) and val metrics too (line 232); backtest/splits.py's `test_date > train_end` guarantee confirmed (splits.py:44-47, 114-124). `target_price_{h}d` at date d is literally price_mean at d+h (lag_rolling.py:150-153). All CONFIRMED.
- **DS-2 (never backtested):** backtest.py:160 `all_baseline_models()` only; zero LightGBM refs under backtest/; both models.py quotes verbatim (lines 26, 41-42); horizons [1,3] (default.toml:90) vs [1,7,28] (default.toml:106). CONFIRMED. One framing nit: the cited cli.py call is in the `--dry-run` branch; live CLI runs delegate to BacktestStage, with the same baselines-only conclusion.
- **DS-3 (drift baseline):** pooled AVG(abs_error), no model_name filter, no staleness bound, newest-rowid run selection: confirmed at drift.py:608-619 and health.py:155-166; `_classify_error_drift(None)` -> `DriftLevel.NONE` (drift.py:282-283) vs health's `unknown` (health.py:214-215). CONFIRMED and understated (see verdict item 6).
- **Registry/MLOps:** model_metadata write-only (writes at trainer.py:202/210, forecast_repo.py:244; no reader exists); mtime glob (trainer.py:280-285 via forecast.py:206); lexicographic pick (cli.py:3619/3628); `_register_model` unconditional promotion (trainer.py:168, 190-221); no Dockerfile/lockfile, runtime deps all `>=` (sole pin: dev-only ruff==0.16.0). All CONFIRMED.
- **Dashboard/demo:** `_DEFAULT_REALMS` hardcodes four realms (app.py:82) vs `["us"]` (default.toml:42), and every output file on disk is `_us_`, so all four selectable realms match nothing; phantom `--realm` flag advertised (app.py:41-42) and never parsed; stray `page_icon="data/raw/snapshots"` (app.py:60); "mirror config/default.toml" comment (app.py:79-80). dashboard/requirements.txt is not just duplicative: it omits the viz deps, so its "standalone" path yields a weaker env. All CONFIRMED.
- **Legibility:** docs/images/ empty AND untracked; generate-charts defaults to gitignored data/outputs/charts/ (cli.py:4144-4147, .gitignore:37); notebooks: exactly 48 code cells, 0 outputs; gitignore portfolio comment at line 77. All CONFIRMED.
- **Counts:** features actual 40 (TRAINING_FEATURE_COLS); README's "37" still live. Tables actual 23 domain tables in schema.py (migrations re-declare 7, add none; plus schema_versions = 24 physical, matching the backup doc); README/CLAUDE.md "21" still live. Tests-at-merge figures (71 files / ~19,700 lines / 1,481 / 34 skips) all accurate then; README 851 lines exact at merge.

## Remediation (approved and executed 2026-07-26, v2.11.3)

One small docs PR, branch `docs/pr89-audit-corrections` off latest main, patch stamp v2.11.3. One concern: documentation inaccuracies surfaced by auditing PR #89. No code changes. Every item below landed in that PR except item 4, which is a judgement call left open.

1. **PLAN.md corrections:**
   - Refresh the two stale cli.py citations (3620 -> 3628, 1033 -> 1041) and the cli.py line count (4,508 -> 4,516).
   - Update the inventory tests row (line 131) to current figures, or date-stamp both inventory rows as as-of figures so the table stops half-drifting.
   - Fix the never-correct "roughly 70 percent CLI reference" (-> roughly 40 percent) and soften "zero images" to "no results visuals" in the same sentence area; refresh 851 -> 894.
   - Update the doc-drift paragraph (lines 207-211): test and command counts fixed in v2.11.x; features "37" and tables "21" still live (until item 2 lands, then state all four resolved).
   - Sharpen DS-3 with the per-horizon `backtest_run_id` finding (even the 1d baseline is uncomputable; run_all hardcodes horizon 1) and note the differing actual-price definitions.
   - Add the trainer.py 80/20 index-slice fallback as a second DS-1 leak vector.
   - Sharpen OD-2/MLOps wording: `auto_retrain_on_critical` has no consumer, not merely "off".
2. **Live count drift (the two PLAN.md already names, still wrong today):** README.md:78 and :767 "37" -> 40 features; README.md:70 and :705 "21 tables" -> 23; CLAUDE.md:47 "21 tables" -> 23 and reconcile the "= 21 total" parenthetical in the Monitoring section. (Same shape as PR #92's count-drift fix.)
3. **CHANGELOG** entry under [Unreleased], moved by the stamp commit; the stamp updates pyproject.toml.
4. **Optional, on approval:** comment on issue #15 (model-monitoring scheduling) recording the per-horizon run-id finding so the Phase 5 / M1 fix accounts for it.

**Anchor-guard caution:** learning banks anchor verbatim PLAN.md lines (m01, m07, m08, m13, m18, m19, m20), notably m07's "gold MAE pooled across archetypes whose price levels differ by orders of" (in the DS-3 paragraph being sharpened) and m19's OD/table-row anchors. Any edited anchored line requires updating the bank in the same commit; that is the drift guard working as designed.

## Verification

- `wowfc learn validate` and `pytest tests/test_learning/test_bank_integrity.py` pass after the PLAN.md/README/CLAUDE.md edits (proves no anchor broke, or that broken ones were updated).
- Full `pytest` green locally; CI green on the PR.
- `grep -rn "37 input\|37 cols\|21 tables\|4,508\|1,481" README.md CLAUDE.md PLAN.md` returns nothing after the edits.
- Merge via `gh pr merge --merge` with PR title suffix (v2.11.3).
