"""
Tests for walk-forward split generation.

What we test
------------
1. Basic split generation — correct fold count and structure.
2. Temporal ordering — each fold is strictly later than the previous.
3. No leakage — test_date > train_end for every fold (the key invariant).
4. Training window size — train_start to train_end spans exactly window_days.
5. Step size — consecutive fold origins differ by exactly step_days.
6. Horizon — test_date == train_end + horizon_days for every fold.
7. Edge cases — range too short, minimum valid range, single fold.
8. Parameter validation — invalid window/step/horizon raise ValueError.
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from wow_forecaster.backtest.splits import BacktestFold, generate_walk_forward_splits


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_folds(
    days_available: int = 90,
    window: int = 30,
    step: int = 7,
    horizon: int = 1,
) -> list[BacktestFold]:
    """Generate folds for a date range of `days_available` calendar days."""
    start = date(2024, 9, 1)
    end   = start + timedelta(days=days_available - 1)
    return generate_walk_forward_splits(start, end, window, step, horizon)


# ── Basic structure ────────────────────────────────────────────────────────────

def test_basic_fold_structure() -> None:
    """Each fold has consistent internal structure."""
    folds = _make_folds(days_available=60, window=20, step=7, horizon=1)
    assert len(folds) >= 1
    for f in folds:
        assert f.train_start < f.train_end
        assert f.test_date   > f.train_end
        assert f.horizon_days == 1
        assert isinstance(f.fold_index, int)


def test_fold_count_is_sensible() -> None:
    """Fold count increases as date range grows relative to step size."""
    folds_small = _make_folds(days_available=40,  window=20, step=7, horizon=1)
    folds_large = _make_folds(days_available=120, window=20, step=7, horizon=1)
    assert len(folds_large) > len(folds_small)


def test_fold_indices_are_sequential() -> None:
    """fold_index is 0-based and sequential."""
    folds = _make_folds(days_available=90, window=30, step=7, horizon=1)
    for i, f in enumerate(folds):
        assert f.fold_index == i


# ── Leakage prevention ─────────────────────────────────────────────────────────

def test_no_leakage_test_date_after_train_end() -> None:
    """The fundamental leakage invariant: test_date > train_end for every fold."""
    folds = _make_folds(days_available=120, window=30, step=7, horizon=1)
    assert folds, "Expected at least one fold"
    for f in folds:
        assert f.test_date > f.train_end, (
            f"Fold {f.fold_index}: test_date={f.test_date} is not after train_end={f.train_end}"
        )


def test_no_leakage_with_3day_horizon() -> None:
    """Leakage check holds for 3-day horizon too."""
    folds = _make_folds(days_available=120, window=30, step=7, horizon=3)
    for f in folds:
        assert f.test_date > f.train_end


def test_train_windows_do_not_overlap_test_dates() -> None:
    """No training row date could accidentally be the test date."""
    folds = _make_folds(days_available=120, window=30, step=7, horizon=1)
    for f in folds:
        # The entire training window [train_start, train_end] is before test_date.
        assert f.train_end < f.test_date


# ── Window and step size ───────────────────────────────────────────────────────

def test_training_window_size() -> None:
    """train_start to train_end spans exactly window_days calendar days."""
    window = 30
    folds = generate_walk_forward_splits(
        date(2024, 1, 1), date(2024, 12, 31), window, step_days=7, horizon_days=1
    )
    for f in folds:
        span = (f.train_end - f.train_start).days + 1  # inclusive
        assert span == window, (
            f"Fold {f.fold_index}: window span={span}, expected {window}"
        )


def test_step_size_between_consecutive_folds() -> None:
    """Consecutive fold train_end dates differ by exactly step_days."""
    step = 7
    folds = generate_walk_forward_splits(
        date(2024, 1, 1), date(2024, 12, 31), window_days=30, step_days=step, horizon_days=1
    )
    for i in range(1, len(folds)):
        delta = (folds[i].train_end - folds[i - 1].train_end).days
        assert delta == step, (
            f"Between fold {i-1} and {i}: expected step={step}, got {delta}"
        )


def test_horizon_observed_in_test_date() -> None:
    """test_date == train_end + horizon_days for every fold."""
    for horizon in [1, 3, 7]:
        folds = _make_folds(days_available=120, window=30, step=7, horizon=horizon)
        for f in folds:
            expected = f.train_end + timedelta(days=horizon)
            assert f.test_date == expected, (
                f"Fold {f.fold_index}: test_date={f.test_date}, "
                f"expected train_end+{horizon}d={expected}"
            )


# ── Edge cases ─────────────────────────────────────────────────────────────────

def test_empty_when_range_too_short() -> None:
    """Returns empty list when range cannot fit even one fold + test point."""
    start = date(2024, 1, 1)
    # window=30d, horizon=1d requires 31 days minimum.
    end = start + timedelta(days=29)  # only 30 days: can't fit test_date
    folds = generate_walk_forward_splits(start, end, window_days=30, step_days=7, horizon_days=1)
    assert folds == []


def test_empty_when_end_equals_start() -> None:
    """Returns empty list when end_date == start_date."""
    d = date(2024, 6, 1)
    folds = generate_walk_forward_splits(d, d, window_days=7, step_days=1, horizon_days=1)
    assert folds == []


def test_empty_when_end_before_start() -> None:
    """Returns empty list when end_date < start_date."""
    folds = generate_walk_forward_splits(
        date(2024, 6, 10), date(2024, 6, 1),
        window_days=7, step_days=1, horizon_days=1
    )
    assert folds == []


def test_single_fold_minimum_valid_range() -> None:
    """Exactly one fold is produced for the minimum valid date range."""
    start   = date(2024, 1, 1)
    # window=7, horizon=1 → need at least 8 days, but test_date must be <= end.
    # First cutoff = start + 6 = Jan 7.  test_date = Jan 8.
    # end must be >= Jan 8 for one fold.
    end = start + timedelta(days=7)  # Jan 8 — exactly one fold
    folds = generate_walk_forward_splits(start, end, window_days=7, step_days=7, horizon_days=1)
    assert len(folds) == 1
    assert folds[0].fold_index == 0


def test_all_test_dates_within_data_range() -> None:
    """All test_dates are <= end_date (no fold reaches outside the data range)."""
    start = date(2024, 9, 1)
    end   = date(2024, 12, 1)
    folds = generate_walk_forward_splits(start, end, window_days=30, step_days=7, horizon_days=3)
    for f in folds:
        assert f.test_date <= end, (
            f"Fold {f.fold_index}: test_date={f.test_date} exceeds end_date={end}"
        )


# ── Parameter validation ───────────────────────────────────────────────────────

def test_invalid_window_raises() -> None:
    with pytest.raises(ValueError, match="window_days"):
        generate_walk_forward_splits(
            date(2024, 1, 1), date(2024, 12, 31),
            window_days=0, step_days=7, horizon_days=1
        )


def test_invalid_step_raises() -> None:
    with pytest.raises(ValueError, match="step_days"):
        generate_walk_forward_splits(
            date(2024, 1, 1), date(2024, 12, 31),
            window_days=30, step_days=0, horizon_days=1
        )


def test_invalid_horizon_raises() -> None:
    with pytest.raises(ValueError, match="horizon_days"):
        generate_walk_forward_splits(
            date(2024, 1, 1), date(2024, 12, 31),
            window_days=30, step_days=7, horizon_days=0
        )
