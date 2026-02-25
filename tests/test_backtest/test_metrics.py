"""
Tests for backtest evaluation metrics.

What we test
------------
1. MAE — basic computation with known values.
2. RMSE — verifies squared-error penalty (RMSE >= MAE always).
3. MAPE — percentage error computation and near-zero actual safeguard.
4. Directional accuracy — correct direction counting, tie handling.
5. Empty record handling — all metrics return None gracefully.
6. None actual/predicted — excluded from evaluation but counted in n_predictions.
7. All-correct and all-wrong directional accuracy boundary values.
8. BacktestMetrics label fields (model_name, horizon_days, slice_key).
"""

from __future__ import annotations

from datetime import date

import pytest

from wow_forecaster.backtest.metrics import (
    BacktestMetrics,
    PredictionRecord,
    compute_metrics,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_record(
    actual: float | None,
    predicted: float | None,
    last_known: float | None = None,
    fold: int = 0,
    arch: int = 1,
    model: str = "test_model",
    is_event: bool = False,
) -> PredictionRecord:
    return PredictionRecord(
        fold_index=fold,
        archetype_id=arch,
        realm_slug="area-52",
        category_tag="consumable",
        model_name=model,
        train_end=date(2024, 9, 10),
        test_date=date(2024, 9, 11),
        horizon_days=1,
        actual_price=actual,
        predicted_price=predicted,
        last_known_price=last_known,
        is_event_window=is_event,
    )


# ── MAE ────────────────────────────────────────────────────────────────────────

def test_mae_basic() -> None:
    """MAE = mean of absolute errors."""
    records = [
        _make_record(actual=100.0, predicted=90.0),   # error = 10
        _make_record(actual=200.0, predicted=220.0),  # error = 20
        _make_record(actual=150.0, predicted=150.0),  # error = 0
    ]
    m = compute_metrics(records)
    assert m.mae is not None
    assert abs(m.mae - 10.0) < 1e-6  # (10 + 20 + 0) / 3


def test_mae_perfect_prediction_is_zero() -> None:
    records = [
        _make_record(actual=100.0, predicted=100.0),
        _make_record(actual=500.0, predicted=500.0),
    ]
    m = compute_metrics(records)
    assert m.mae == pytest.approx(0.0)


# ── RMSE ───────────────────────────────────────────────────────────────────────

def test_rmse_penalizes_large_errors_more_than_mae() -> None:
    """RMSE >= MAE; large errors inflate RMSE disproportionately."""
    records = [
        _make_record(actual=100.0, predicted=110.0),   # error = 10
        _make_record(actual=100.0, predicted=200.0),   # error = 100 (large miss)
    ]
    m = compute_metrics(records)
    assert m.rmse is not None and m.mae is not None
    assert m.rmse >= m.mae


def test_rmse_equals_mae_for_equal_errors() -> None:
    """When all errors are identical, RMSE == MAE."""
    records = [
        _make_record(actual=100.0, predicted=90.0),
        _make_record(actual=200.0, predicted=190.0),
    ]
    m = compute_metrics(records)
    assert m.rmse is not None and m.mae is not None
    assert abs(m.rmse - m.mae) < 1e-6


def test_rmse_perfect_is_zero() -> None:
    records = [_make_record(actual=100.0, predicted=100.0)]
    m = compute_metrics(records)
    assert m.rmse == pytest.approx(0.0)


# ── MAPE ───────────────────────────────────────────────────────────────────────

def test_mape_basic() -> None:
    """MAPE = mean of |error| / actual (for actual >= epsilon)."""
    records = [
        _make_record(actual=100.0, predicted=110.0),  # |10| / 100 = 0.10
        _make_record(actual=200.0, predicted=190.0),  # |10| / 200 = 0.05
    ]
    m = compute_metrics(records)
    assert m.mape is not None
    assert abs(m.mape - 0.075) < 1e-6  # (0.10 + 0.05) / 2


def test_mape_excludes_near_zero_actuals() -> None:
    """MAPE skips rows where actual < MAPE_EPSILON to prevent division instability."""
    records = [
        _make_record(actual=0.001, predicted=10.0),   # too small — excluded
        _make_record(actual=100.0, predicted=110.0),  # error = 10%
    ]
    m = compute_metrics(records)
    assert m.mape is not None
    # Only the 100g row contributes: MAPE = 0.10
    assert abs(m.mape - 0.10) < 1e-3


def test_mape_is_none_when_all_actuals_below_epsilon() -> None:
    """Returns None when all actuals are below MAPE_EPSILON."""
    records = [
        _make_record(actual=0.001, predicted=1.0),
        _make_record(actual=0.005, predicted=2.0),
    ]
    m = compute_metrics(records)
    assert m.mape is None


# ── Directional accuracy ───────────────────────────────────────────────────────

def test_directional_accuracy_all_correct() -> None:
    """1.0 when all directions are correctly predicted."""
    # last_known = 100; actual goes up; predicted also goes up
    records = [
        _make_record(actual=120.0, predicted=110.0, last_known=100.0),
        _make_record(actual=130.0, predicted=115.0, last_known=100.0),
    ]
    m = compute_metrics(records)
    assert m.directional_accuracy == pytest.approx(1.0)


def test_directional_accuracy_all_wrong() -> None:
    """0.0 when all directions are incorrectly predicted."""
    # last_known = 100; actual goes up; predicted goes down
    records = [
        _make_record(actual=120.0, predicted=80.0, last_known=100.0),
        _make_record(actual=130.0, predicted=70.0, last_known=100.0),
    ]
    m = compute_metrics(records)
    assert m.directional_accuracy == pytest.approx(0.0)


def test_directional_accuracy_mixed() -> None:
    """0.5 when half the directions are correctly predicted."""
    records = [
        _make_record(actual=120.0, predicted=110.0, last_known=100.0),  # correct: up
        _make_record(actual=120.0, predicted=80.0,  last_known=100.0),  # wrong: up but pred down
    ]
    m = compute_metrics(records)
    assert m.directional_accuracy == pytest.approx(0.5)


def test_directional_accuracy_excludes_unchanged_actuals() -> None:
    """Ties (actual == last_known) are excluded from directional accuracy."""
    records = [
        _make_record(actual=100.0, predicted=110.0, last_known=100.0),  # tie — excluded
        _make_record(actual=120.0, predicted=110.0, last_known=100.0),  # correct: up
    ]
    m = compute_metrics(records)
    assert m.n_directional == 1
    assert m.directional_accuracy == pytest.approx(1.0)


def test_directional_accuracy_none_when_no_last_known() -> None:
    """Returns None when no record has a last_known_price."""
    records = [
        _make_record(actual=120.0, predicted=110.0, last_known=None),
        _make_record(actual=130.0, predicted=115.0, last_known=None),
    ]
    m = compute_metrics(records)
    assert m.directional_accuracy is None
    assert m.n_directional == 0


# ── None handling ──────────────────────────────────────────────────────────────

def test_empty_records_returns_none_metrics() -> None:
    """All metric fields are None when no records are provided."""
    m = compute_metrics([])
    assert m.n_predictions == 0
    assert m.n_evaluated == 0
    assert m.mae is None
    assert m.rmse is None
    assert m.mape is None
    assert m.directional_accuracy is None


def test_none_actual_excluded_from_evaluation() -> None:
    """Records with None actual are counted in n_predictions but not n_evaluated."""
    records = [
        _make_record(actual=None,  predicted=100.0),
        _make_record(actual=200.0, predicted=190.0),
    ]
    m = compute_metrics(records)
    assert m.n_predictions == 2
    assert m.n_evaluated   == 1
    assert m.mae            == pytest.approx(10.0)


def test_none_predicted_excluded_from_evaluation() -> None:
    """Records with None predicted are counted in n_predictions but not n_evaluated."""
    records = [
        _make_record(actual=100.0, predicted=None),
        _make_record(actual=200.0, predicted=190.0),
    ]
    m = compute_metrics(records)
    assert m.n_predictions == 2
    assert m.n_evaluated   == 1


def test_all_none_actuals_returns_zero_evaluated() -> None:
    records = [
        _make_record(actual=None, predicted=100.0),
        _make_record(actual=None, predicted=200.0),
    ]
    m = compute_metrics(records)
    assert m.n_predictions == 2
    assert m.n_evaluated   == 0
    assert m.mae is None


# ── Label fields ───────────────────────────────────────────────────────────────

def test_label_fields_are_passed_through() -> None:
    """model_name, horizon_days, slice_key are attached to BacktestMetrics."""
    records = [_make_record(actual=100.0, predicted=90.0)]
    m = compute_metrics(records, model_name="last_value", horizon_days=1, slice_key="flask")
    assert m.model_name   == "last_value"
    assert m.horizon_days == 1
    assert m.slice_key    == "flask"


# ── Mean actual / predicted ────────────────────────────────────────────────────

def test_mean_actual_and_predicted() -> None:
    records = [
        _make_record(actual=100.0, predicted=90.0),
        _make_record(actual=200.0, predicted=180.0),
    ]
    m = compute_metrics(records)
    assert m.mean_actual    == pytest.approx(150.0)
    assert m.mean_predicted == pytest.approx(135.0)
