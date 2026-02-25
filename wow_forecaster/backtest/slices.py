"""
Evaluation slicing — break down metrics by different dimensions.

Slices let you understand WHERE and WHY a model is performing well or poorly.

  slice_by_model               → one row per model (which baseline is best?)
  slice_by_model_and_horizon   → (model, horizon) grid (does 1d beat 3d?)
  slice_by_category            → one row per archetype category
  slice_by_archetype           → one row per archetype_id (which items cause errors?)
  slice_by_event_window        → event vs non-event periods

All slicers return dict[key, BacktestMetrics] and reuse compute_metrics().
"""

from __future__ import annotations

from collections import defaultdict

from wow_forecaster.backtest.metrics import BacktestMetrics, PredictionRecord, compute_metrics


def slice_by_model(records: list[PredictionRecord]) -> dict[str, BacktestMetrics]:
    """Aggregate metrics per model name."""
    groups: dict[str, list[PredictionRecord]] = defaultdict(list)
    for r in records:
        groups[r.model_name].append(r)
    return {
        name: compute_metrics(recs, model_name=name, slice_key=name)
        for name, recs in groups.items()
    }


def slice_by_model_and_horizon(
    records: list[PredictionRecord],
) -> dict[tuple[str, int], BacktestMetrics]:
    """Aggregate metrics per (model_name, horizon_days) pair."""
    groups: dict[tuple[str, int], list[PredictionRecord]] = defaultdict(list)
    for r in records:
        groups[(r.model_name, r.horizon_days)].append(r)
    return {
        key: compute_metrics(
            recs,
            model_name=key[0],
            horizon_days=key[1],
            slice_key=f"{key[0]}_{key[1]}d",
        )
        for key, recs in groups.items()
    }


def slice_by_category(records: list[PredictionRecord]) -> dict[str, BacktestMetrics]:
    """Aggregate metrics per archetype category (e.g. 'consumable', 'mat').

    Helps answer: "Is the model better at consumables or crafting materials?"
    """
    groups: dict[str, list[PredictionRecord]] = defaultdict(list)
    for r in records:
        key = r.category_tag or "unknown"
        groups[key].append(r)
    return {
        cat: compute_metrics(recs, slice_key=cat)
        for cat, recs in groups.items()
    }


def slice_by_archetype(records: list[PredictionRecord]) -> dict[int, BacktestMetrics]:
    """Aggregate metrics per archetype_id.

    Helps answer: "Which specific archetype has the worst forecast error?"
    High-error archetypes may need special treatment (event modelling, more data).
    """
    groups: dict[int, list[PredictionRecord]] = defaultdict(list)
    for r in records:
        groups[r.archetype_id].append(r)
    return {
        arch_id: compute_metrics(recs, slice_key=str(arch_id))
        for arch_id, recs in groups.items()
    }


def slice_by_event_window(
    records: list[PredictionRecord],
) -> dict[str, BacktestMetrics]:
    """Split metrics into event-window vs non-event-window observations.

    This is one of the most informative slices for WoW economy forecasting:
    if accuracy drops significantly during events, the model needs explicit
    event-awareness features to remain useful during raid tiers, holidays,
    and content patches.
    """
    event: list[PredictionRecord] = []
    non_event: list[PredictionRecord] = []
    for r in records:
        (event if r.is_event_window else non_event).append(r)

    result: dict[str, BacktestMetrics] = {}
    if event:
        result["event_window"] = compute_metrics(event, slice_key="event_window")
    if non_event:
        result["non_event_window"] = compute_metrics(non_event, slice_key="non_event_window")
    return result
