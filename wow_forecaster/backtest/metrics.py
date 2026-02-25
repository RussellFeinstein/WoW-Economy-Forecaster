"""
Forecast evaluation metrics.

Metric design rationale
-----------------------
MAE (Mean Absolute Error)
  Most intuitive metric for this use case: "on average, we're off by X gold."
  Equally weights all errors.  Best for communicating accuracy to a player:
  "expect to be within ±X gold on average."
  Interpretation: lower is better; 0 is perfect.

RMSE (Root Mean Squared Error)
  Penalizes large errors more than MAE because errors are squared before
  averaging.  Useful for detecting catastrophic mispredictions — a flask
  predicted at 200g but actually 1000g matters more than ten 20g errors.
  Interpretation: RMSE > MAE implies occasional large misses.

MAPE (Mean Absolute Percentage Error)
  Normalizes MAE by the actual price, enabling cross-archetype comparison.
  "Was I off by 5%?" is meaningful whether the item costs 10g or 10,000g.
  Safeguard: actual prices below MAPE_EPSILON are excluded to prevent
  division-by-zero and unstable percentages from near-zero prices.
  Interpretation: 0.05 = 5% average error; comparable across price ranges.

Directional Accuracy
  "Did we predict the right direction of price movement (up/down)?"
  Critical for buy/hold/sell recommendations: a model that gets the
  direction wrong is actively harmful even if the magnitude is acceptable.
  Computed as: sign(predicted - last_known) == sign(actual - last_known).
  Ties (no change in either direction) are excluded from the denominator.
  Interpretation: 0.5 = random; 0.7+ = meaningful directional signal.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date

MAPE_EPSILON = 0.01  # minimum actual price (gold) to include in MAPE


@dataclass(frozen=True)
class PredictionRecord:
    """One prediction-vs-actual comparison for a single fold/model/series.

    Attributes:
        fold_index:       Which fold this came from.
        archetype_id:     Archetype primary key.
        realm_slug:       Realm slug.
        category_tag:     Archetype category for slicing (e.g. 'consumable').
        model_name:       Name of the model that made this prediction.
        train_end:        Last date the model was trained on.
        test_date:        Date being predicted (train_end + horizon_days).
        horizon_days:     How many days ahead the prediction is.
        actual_price:     Actual price_mean on test_date (None = no data).
        predicted_price:  Model's predicted price (None = model abstained).
        last_known_price: price_mean at train_end (for directional accuracy).
        is_event_window:  True if any WoW event is active on test_date.
    """

    fold_index: int
    archetype_id: int
    realm_slug: str
    category_tag: str | None
    model_name: str
    train_end: date
    test_date: date
    horizon_days: int
    actual_price: float | None
    predicted_price: float | None
    last_known_price: float | None
    is_event_window: bool = False


@dataclass(frozen=True)
class BacktestMetrics:
    """Aggregated evaluation metrics over a set of PredictionRecords.

    All float fields are None when there is insufficient data to compute them
    (e.g., n_evaluated == 0).

    Attributes:
        n_predictions:        Total predictions attempted.
        n_evaluated:          Predictions where both actual and predicted are non-null.
        mae:                  Mean absolute error in gold.
        rmse:                 Root mean squared error in gold.
        mape:                 Mean absolute percentage error (0.0–1.0; 0.05 = 5%).
        directional_accuracy: Fraction of correctly predicted price directions.
        n_directional:        Count used for directional accuracy denominator.
        mean_actual:          Mean actual price (sanity check).
        mean_predicted:       Mean predicted price (sanity check).
        model_name:           Source model name (optional label).
        horizon_days:         Forecast horizon (optional label).
        slice_key:            Identifies the evaluation slice (optional label).
    """

    n_predictions: int
    n_evaluated: int
    mae: float | None
    rmse: float | None
    mape: float | None
    directional_accuracy: float | None
    n_directional: int
    mean_actual: float | None
    mean_predicted: float | None
    model_name: str | None = None
    horizon_days: int | None = None
    slice_key: str | None = None


def compute_metrics(
    records: list[PredictionRecord],
    model_name: str | None = None,
    horizon_days: int | None = None,
    slice_key: str | None = None,
) -> BacktestMetrics:
    """Compute all evaluation metrics for a set of PredictionRecords.

    Handles None actual/predicted values gracefully: rows where either
    value is None are excluded from metric computation but counted in
    n_predictions.

    Args:
        records:      The predictions to evaluate.
        model_name:   Label to attach to the returned metrics.
        horizon_days: Label to attach to the returned metrics.
        slice_key:    Label to attach to the returned metrics.

    Returns:
        BacktestMetrics with all computed values.
    """
    n_predictions = len(records)
    evaluated = [
        r for r in records
        if r.actual_price is not None and r.predicted_price is not None
    ]
    n_evaluated = len(evaluated)

    if not evaluated:
        return BacktestMetrics(
            n_predictions=n_predictions,
            n_evaluated=0,
            mae=None, rmse=None, mape=None,
            directional_accuracy=None, n_directional=0,
            mean_actual=None, mean_predicted=None,
            model_name=model_name, horizon_days=horizon_days, slice_key=slice_key,
        )

    actuals   = [r.actual_price    for r in evaluated]   # type: ignore[misc]
    predicted = [r.predicted_price for r in evaluated]   # type: ignore[misc]
    errors    = [a - p for a, p in zip(actuals, predicted)]
    abs_errors = [abs(e) for e in errors]

    mae  = sum(abs_errors) / len(abs_errors)
    rmse = math.sqrt(sum(e * e for e in errors) / len(errors))

    mape_terms = [
        abs(e) / max(a, MAPE_EPSILON)
        for e, a in zip(abs_errors, actuals)
        if a >= MAPE_EPSILON
    ]
    mape = (sum(mape_terms) / len(mape_terms)) if mape_terms else None

    # Directional accuracy: was the predicted direction correct?
    # Only count rows where the actual price actually changed vs last known.
    directional = [
        r for r in evaluated
        if r.last_known_price is not None
        and r.actual_price != r.last_known_price
    ]
    n_directional = len(directional)
    if n_directional > 0:
        correct = sum(
            1 for r in directional
            if (
                _direction(r.predicted_price, r.last_known_price)  # type: ignore[arg-type]
                == _direction(r.actual_price, r.last_known_price)  # type: ignore[arg-type]
                != 0
            )
        )
        dir_acc: float | None = correct / n_directional
    else:
        dir_acc = None

    mean_actual    = sum(actuals)   / len(actuals)
    mean_predicted = sum(predicted) / len(predicted)

    return BacktestMetrics(
        n_predictions=n_predictions,
        n_evaluated=n_evaluated,
        mae=mae,
        rmse=rmse,
        mape=mape,
        directional_accuracy=dir_acc,
        n_directional=n_directional,
        mean_actual=mean_actual,
        mean_predicted=mean_predicted,
        model_name=model_name,
        horizon_days=horizon_days,
        slice_key=slice_key,
    )


def _direction(price: float, reference: float) -> int:
    """Return +1 if price > reference, -1 if price < reference, 0 if equal."""
    if price > reference:
        return 1
    if price < reference:
        return -1
    return 0
