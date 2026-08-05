---
paths:
  - "wow_forecaster/features/**"
  - "wow_forecaster/ml/**"
  - "wow_forecaster/backtest/**"
  - "wow_forecaster/recipes/**"
  - "wow_forecaster/recommendations/**"
  - "wow_forecaster/pipeline/normalize.py"
  - "wow_forecaster/pipeline/forecast.py"
---

# Modeling

Loaded when working with features, backtesting, ML, normalization, recipes, or recommendations. Root context: [CLAUDE.md](../../CLAUDE.md).

## Feature Engineering (v0.3.0 / v0.9.0)
- [wow_forecaster/features/registry.py](../../wow_forecaster/features/registry.py) — 48 training / 45 inference cols
- [wow_forecaster/features/daily_agg.py](../../wow_forecaster/features/daily_agg.py) — Python-generated date spine over rollup fast path (recursive CTE replaced in v2.3.x); spine clamps to [data_min, data_max]; JOINs items.archetype_id (backward-compat with pre-v1.3.4 rows + items with no archetype assignment)
- [wow_forecaster/features/dataset_builder.py](../../wow_forecaster/features/dataset_builder.py) — orchestrates all steps → training/inference Parquet + JSON manifest
- build-datasets end_date default = date.today()+timedelta(days=1) (captures UTC-midnight observations)

## Backtesting (v0.4.0)
- [wow_forecaster/backtest/evaluator.py](../../wow_forecaster/backtest/evaluator.py) — run_backtest() fold×series×model loop; leakage-free
- BacktestConfig: horizons_days=[1,3], min_train_rows=14
- DB tables: backtest_runs, backtest_fold_results (migration 0002)

## ML + Recommendations (v0.5.0 / v1.10.0 / v1.11.0 / v1.12.0 / v2.0.0)
- [wow_forecaster/ml/feature_selector.py](../../wow_forecaster/ml/feature_selector.py) — TRAINING_FEATURE_COLS (40)
- [wow_forecaster/ml/lgbm_model.py](../../wow_forecaster/ml/lgbm_model.py) — LightGBMForecaster: fit/predict/save/load; global cross-archetype model
- ForecastHorizon: 1d/7d/28d; TARGET_COL_MAP = {1: 1d, 7: 7d, 28: 28d}
- Score formula: 0.35×opportunity + 0.20×liquidity − 0.20×volatility + 0.15×event_boost − 0.10×uncertainty
- event_boost clamp: [-100, 100] (negative impacts penalize score)
- top_n_per_category deduplication: best-scoring horizon per archetype_id (tie: shorter wins)
- DB migration 0003: adds score, score_components, category_tag to recommendation_outputs
- Risk levels (v1.10.0): determine_risk_level() in scorer.py — LOW/MEDIUM/HIGH/CRITICAL tiers independent of action; AVOID only at CRITICAL (uncertainty ≥ 95%); risk_level persisted in recommendation_outputs (migration 0006)
- CI floor/cap (v1.11.0): compute_confidence_interval() in cold_start.py accepts current_price; floor = 5% of current, cap = 10× current; prevents 0.0 lower bounds and absurd upper bounds; ci_quality field ("good"/"wide"/"unreliable") on ForecastOutput (migration 0007)
- Item-level forecasting extended (v1.12.0): _generate_item_forecasts() now covers union of recipe-linked items AND all items with ≥14 distinct observation days; ItemForecastRoi dataclass + fetch_item_rois() in item_overlay.py; enrich_with_top_item_rois() in ranker.py; top_items column in recommendations CSV/JSON prefers ROI-based items over discount-based fallback
- TSM export (v2.0.0): export-tsm CLI command; wow_forecaster/reporting/tsm_export.py; TsmExportRow + fetch_tsm_export_items() + build_tsm_import_string() + write_tsm_export(); filters item-level forecasts by ROI >= min_roi_pct and ci_quality='good'; outputs i:XXXXX,... string for TradeSkillMaster paste import

## Normalization (v1.1.0)
- Rolling z-score via _fetch_rolling_stats() + _normalize_batch(); falls back to batch stats on cold-start
- Baseline source (v2.14.13, issue #123): _fetch_rolling_stats reads `daily_rollup_item`'s stored partial sums, NOT market_observations_normalized. COUNT/SUM/SUM-of-squares are sufficient statistics for mean and variance, and the rollup is built from the same table under the same `is_outlier = 0` filter, so summing across days is exact rather than approximate (36m39s -> 0.54s on the production DB, same 10,315 pairs; the reduction factor is the ~590 observations per item/realm/day). **The exactness rests on `price_gold` never being NULL** (SUM skips NULLs, COUNT(*) does not): if the write-path coercion at normalize.py ever changes, this becomes a schema change, because the rollup carries no non-NULL price counter (`price_obs_count_pos` is a `> 0` filter, not a NULL filter). The is_outlier filter now lives only in `_UPSERT_ITEM_SQL`; the query never names it. Window edge moved from a timestamp cutoff to a calendar-day boundary (sub-1%, removes the sawtooth) and that shift moves a few rows across the outlier threshold in the forward stream only. Clock injectable (`now=`, pruner-style None default). A covering index was rejected: it shrinks the constant while the scan still grows with the raw table, and building it is a 2-3 GB sustained write on this box
- _check_rollup_freshness() warns when a batch realm's newest rollup date is >1 day old, or the realm is absent while others are present; an entirely empty table logs INFO (cold start, not a fault)
- config: pipeline.normalize_rolling_days=30
- archetype_id populated via _fetch_archetype_map() since v1.3.4; daily_agg.py JOINs items for backward-compat + unassigned items

## Recipes + Crafting Advisor (v1.5.0)
- [wow_forecaster/recipes/blizzard_recipe_client.py](../../wow_forecaster/recipes/blizzard_recipe_client.py) — fetch_all_recipes_for_profession(); NormalisedRecipe/NormalisedReagent; required reagents only
- [wow_forecaster/recipes/recipe_seeder.py](../../wow_forecaster/recipes/recipe_seeder.py) — RecipeSeeder: seed(expansion_slug, professions) → upserts recipes + reagents; rate-limited
- [wow_forecaster/recipes/recipe_repo.py](../../wow_forecaster/recipes/recipe_repo.py) — RecipeRepository: upsert_recipe/replace_reagents/get_recipes_by_expansion etc.
- [wow_forecaster/recipes/margin_calculator.py](../../wow_forecaster/recipes/margin_calculator.py) — MarginCalculator.compute_margins(): daily craft cost vs output price → crafting_margin_snapshots
- [wow_forecaster/recommendations/crafting_advisor.py](../../wow_forecaster/recommendations/crafting_advisor.py) — CraftingWindow(6 windows), build_crafting_opportunities(), rank_crafting_opportunities()
- CraftingWindow: NOW_NOW, NOW_7D, NOW_28D, _7D_7D, _7D_28D, _28D_28D — all (buy≤sell) pairs using 1d/7d/28d forecasts
- Future window price projection (v1.5.7+): trend-ratio scaling — item_forecast = item_current × (archetype_forecast / archetype_rolling_current); preserves intra-archetype item price differentiation; falls back to raw archetype forecast then current price
- Item-level forecasts (v1.6.0 / v1.12.0): ForecastStage._generate_item_forecasts() writes item_id-keyed rows to forecast_outputs (item_id set, archetype_id=None); v1.6.0: recipe-linked items only; v1.12.0: extended to union of recipe items + any item with ≥14 distinct observation days; crafting_advisor._fetch_item_forecasts() prefers these over archetype-level forecasts (priority: item forecast → trend-ratio → archetype forecast → current price)
- forecast_outputs.item_id was previously always NULL; now populated for recipe items after each run-daily-forecast
- Cold-start prediction blending (v1.7.0): ForecastStage._execute() calls _fetch_cold_start_blend_data() to build (source_price, confidence) pairs from archetype_mappings; run_inference() calls cold_start.blend_cold_start_prediction() BEFORE CI computation; blended = confidence × model_pred + (1-confidence) × source_price; model_slug gets _transfer suffix for blended archetypes
- Volume gate: hard filter (quantity_sum_7d < min_volume_units=50 excluded) + volume_score = clamp(qty/500, 0, 1)
- opportunity_score = best_window_margin_pct × volume_score
- Compression/expansion: linear regression slope of margin_pct over last N days; ±0.02/day thresholds
- DB migration 0005: recipes, recipe_reagents, crafting_margin_snapshots (UNIQUE recipe_id+realm+obs_date)
- CLI: seed-recipes (--expansion default=transfer_target, --all), build-margins (--realm, --days), report-crafting (--realm, --top-n, --export), report-recipe-status (--expansion)
- seed-recipes --expansion defaults to transfer_target config value ("midnight"); use --all for first-time full seed
