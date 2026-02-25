"""
Backtest evaluator: apply baseline models over walk-forward folds.

How it works
------------
1. Receive all feature rows for one realm (all archetypes, all dates).
2. Build a price lookup: (archetype_id, realm_slug, obs_date) → price_mean.
   This is a read-only dict; models never see it.
3. Group rows by (archetype_id, realm_slug) for efficient per-fold access.
4. For each fold:
   a. Filter series rows to obs_date in [fold.train_start, fold.train_end].
   b. Skip series with fewer than min_train_rows non-null prices.
   c. For each baseline model:
      - model.fit(series_train_rows)
      - predicted = model.predict(fold.horizon_days)
      - actual = price_lookup[(arch_id, realm, fold.test_date)]  (may be None)
      - Emit a PredictionRecord.
5. Return all PredictionRecords.

Leakage proof
-------------
- train_rows are filtered to obs_date <= fold.train_end.
- test_date > fold.train_end (structural guarantee from split generation).
- The price lookup is used ONLY to retrieve actual_price on test_date.
  It is NOT passed to models; they receive only their train_rows.
- Event window classification (is_event_window) uses the active event set
  for classification purposes only — it is never a model input.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from datetime import date
from typing import Any

from wow_forecaster.backtest.metrics import PredictionRecord
from wow_forecaster.backtest.splits import BacktestFold

log = logging.getLogger(__name__)


def run_backtest(
    feature_rows: list[dict[str, Any]],
    folds: list[BacktestFold],
    models: list[Any],
    archetype_categories: dict[int, str],
    active_event_dates: set[date],
    min_train_rows: int = 14,
) -> list[PredictionRecord]:
    """Evaluate all models over all walk-forward folds.

    Args:
        feature_rows:         All feature rows for the realm, sorted by
                              (archetype_id, realm_slug, obs_date).
        folds:                Walk-forward folds from generate_walk_forward_splits().
        models:               Baseline model instances (will be re-fit each fold).
        archetype_categories: Map archetype_id → category_tag for slicing.
        active_event_dates:   Set of dates when any WoW event is active
                              (for is_event_window classification only).
        min_train_rows:       Minimum non-null price rows required per series
                              before a model will be fit.

    Returns:
        List of PredictionRecord — one per (fold × series × model).
    """
    # Read-only price lookup; models never receive this dict.
    price_lookup: dict[tuple[int, str, date], float | None] = {}
    for r in feature_rows:
        key = (r["archetype_id"], r["realm_slug"], r["obs_date"])
        price_lookup[key] = r.get("price_mean")

    # Group rows by series for O(1) per-fold access.
    series_map: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for r in feature_rows:
        series_map[(r["archetype_id"], r["realm_slug"])].append(r)

    for rows in series_map.values():
        rows.sort(key=lambda r: r["obs_date"])

    all_records: list[PredictionRecord] = []

    for fold in folds:
        log.debug(
            "Fold %d | train=[%s..%s] | test=%s",
            fold.fold_index, fold.train_start, fold.train_end, fold.test_date,
        )

        for (arch_id, realm_slug), series_rows in series_map.items():
            # Partition: training rows are STRICTLY before or at train_end.
            train_rows = [
                r for r in series_rows
                if fold.train_start <= r["obs_date"] <= fold.train_end
            ]

            # Require enough non-null price rows to fit a meaningful model.
            train_prices = [r for r in train_rows if r.get("price_mean") is not None]
            if len(train_prices) < min_train_rows:
                continue

            # Actual price on the test date (may be None — no data that day).
            actual = price_lookup.get((arch_id, realm_slug, fold.test_date))

            # Last known price at train_end (for directional accuracy computation).
            last_known = price_lookup.get((arch_id, realm_slug, fold.train_end))
            if last_known is None:
                for r in reversed(train_rows):
                    if r.get("price_mean") is not None:
                        last_known = float(r["price_mean"])
                        break

            category_tag = archetype_categories.get(arch_id)
            is_event = fold.test_date in active_event_dates

            for model in models:
                model.fit(train_rows)
                predicted = model.predict(fold.horizon_days)
                all_records.append(PredictionRecord(
                    fold_index=fold.fold_index,
                    archetype_id=arch_id,
                    realm_slug=realm_slug,
                    category_tag=category_tag,
                    model_name=model.name,
                    train_end=fold.train_end,
                    test_date=fold.test_date,
                    horizon_days=fold.horizon_days,
                    actual_price=actual,
                    predicted_price=predicted,
                    last_known_price=last_known,
                    is_event_window=is_event,
                ))

    log.info(
        "Backtest complete | folds=%d | series=%d | records=%d",
        len(folds), len(series_map), len(all_records),
    )
    return all_records
