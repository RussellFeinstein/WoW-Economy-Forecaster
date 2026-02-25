"""
Walk-forward (rolling-origin) split generation.

Design
------
Rolling-origin cross-validation works by simulating how a forecasting model
would be deployed in production.  At each "origin" (the fold cutoff date),
we freeze what the model "knows" at that point in time and ask it to predict
the future.

For WoW economy forecasting this is critical because:
- Market regimes change throughout an expansion (RTWF spike, mid-season lull,
  content-patch burst, season-end crash).
- A single train/test split hides these regime changes.
- Walk-forward evaluation reveals whether a model degrades or improves as the
  expansion matures.

Split structure (rolling-origin, fixed training window)
-------------------------------------------------------
Given:
  start_date   = first available data date
  end_date     = last available data date
  window_days  = size of each training window (e.g. 30 days)
  step_days    = how far to advance the origin each fold (e.g. 7 days)
  horizon_days = how many days ahead to forecast (e.g. 1 or 3)

Each fold:
  train_start = cutoff - window_days + 1
  train_end   = cutoff              ← model only "knows" data up to here
  test_date   = cutoff + horizon_days ← what we predict

Cutoff advances from (start_date + window_days - 1) by step_days,
stopping when test_date > end_date.

Why rolling (fixed window) instead of expanding window?
--------------------------------------------------------
Expanding window (train grows from a fixed start) is simpler but means:
- Early data always influences every fold.
- Regime shifts are smoothed away by large historical samples.
Rolling origin better tests: "can the model generalize across regimes?" —
which is the real production question for expansion-to-expansion transfer.

Leakage prevention
------------------
The structural guarantee is: test_date > train_end for every fold.
No test-window information is ever accessible during training.
This is a *design-level* guarantee, not a runtime check.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta


@dataclass(frozen=True)
class BacktestFold:
    """One walk-forward evaluation fold.

    Attributes:
        fold_index:   Zero-based index (for sorting and display).
        train_start:  First date in the training window.
        train_end:    Last date in the training window (the "origin" cutoff).
                      The model must NOT use any data after this date.
        test_date:    The single date being predicted (train_end + horizon_days).
        horizon_days: How many days ahead we forecast.
    """

    fold_index: int
    train_start: date
    train_end: date
    test_date: date
    horizon_days: int


def generate_walk_forward_splits(
    start_date: date,
    end_date: date,
    window_days: int,
    step_days: int,
    horizon_days: int,
) -> list[BacktestFold]:
    """Generate walk-forward evaluation folds.

    Args:
        start_date:   First date of the available data range.
        end_date:     Last date of the available data range.
        window_days:  Number of days in each training window (>= 1).
        step_days:    Number of days to advance between folds (>= 1).
        horizon_days: Number of days ahead to predict (>= 1).

    Returns:
        List of BacktestFold objects, sorted by fold_index.
        Empty list if the date range is too short to form any valid fold.

    Raises:
        ValueError: If any parameter is out of valid range.
    """
    if window_days < 1:
        raise ValueError(f"window_days must be >= 1, got {window_days}")
    if step_days < 1:
        raise ValueError(f"step_days must be >= 1, got {step_days}")
    if horizon_days < 1:
        raise ValueError(f"horizon_days must be >= 1, got {horizon_days}")
    if end_date <= start_date:
        return []

    folds: list[BacktestFold] = []
    # First cutoff is start_date + window_days - 1 (the end of the first full window).
    cutoff = start_date + timedelta(days=window_days - 1)
    fold_index = 0

    while True:
        test_date = cutoff + timedelta(days=horizon_days)
        if test_date > end_date:
            break

        train_start = cutoff - timedelta(days=window_days - 1)
        folds.append(BacktestFold(
            fold_index=fold_index,
            train_start=train_start,
            train_end=cutoff,
            test_date=test_date,
            horizon_days=horizon_days,
        ))
        cutoff += timedelta(days=step_days)
        fold_index += 1

    return folds
