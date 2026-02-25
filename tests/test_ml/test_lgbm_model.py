"""
Tests for wow_forecaster/ml/lgbm_model.py.

What we test
------------
LightGBMForecaster.is_fitted:
  - False before fit(), True after fit().

LightGBMForecaster.fit():
  - Raises ValueError if fewer than 10 training rows.
  - Raises ValueError if no rows have a non-null target column.
  - Returns a dict of validation metrics (mae, rmse, n_val).
  - Returns empty dict when val_rows is empty.

LightGBMForecaster.predict():
  - Returns a list of float >= 0 (non-negative prices).
  - Returns [None] * N if the model is not fitted.
  - Returns the same number of values as input rows.

LightGBMForecaster.save() / load():
  - save() raises RuntimeError on an unfitted model.
  - Round-trip: save() then load() produces a fitted model.
  - Loaded model predicts non-negative values.
  - Loaded model has same horizon_days.

LightGBMForecaster.write_metadata():
  - Writes a valid JSON file with expected keys.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from wow_forecaster.ml.feature_selector import (
    CATEGORICAL_FEATURE_COLS,
    TARGET_COL_MAP,
    TRAINING_FEATURE_COLS,
    encode_row,
)
from wow_forecaster.ml.lgbm_model import LightGBMForecaster


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _synthetic_row(
    price: float = 100.0,
    target: float | None = 110.0,
    target_col: str = "target_price_7d",
) -> dict:
    """Build a minimal, fully-encoded row usable as a training sample."""
    row: dict = {col: price for col in TRAINING_FEATURE_COLS}
    # Override a few categorical/int columns so they are valid
    row["archetype_category_enc"] = 2
    row["event_severity_enc"]     = 0
    row["day_of_week"]            = 1
    row["is_cold_start_int"]      = 0
    row["is_transferable_int"]    = 1
    row["event_active_int"]       = 0
    row["has_transfer_mapping_int"] = 0
    row[target_col] = target
    return row


def _make_rows(
    n: int = 20,
    target_col: str = "target_price_7d",
    include_target: bool = True,
) -> list[dict]:
    """Generate n synthetic training rows."""
    rows = []
    for i in range(n):
        price  = 50.0 + i * 2.0
        target = (price * 1.1) if include_target else None
        rows.append(_synthetic_row(price=price, target=target, target_col=target_col))
    return rows


# ── is_fitted ─────────────────────────────────────────────────────────────────

def test_not_fitted_before_fit():
    m = LightGBMForecaster(horizon_days=7)
    assert not m.is_fitted


def test_fitted_after_fit():
    m = LightGBMForecaster(horizon_days=7, n_estimators=5)
    m.fit(
        train_rows=_make_rows(20),
        val_rows=[],
        feature_cols=TRAINING_FEATURE_COLS,
        categorical_cols=CATEGORICAL_FEATURE_COLS,
        target_col="target_price_7d",
    )
    assert m.is_fitted


# ── fit() ─────────────────────────────────────────────────────────────────────

def test_fit_raises_on_too_few_rows():
    m = LightGBMForecaster(horizon_days=7)
    with pytest.raises(ValueError, match="10 training rows"):
        m.fit(
            train_rows=_make_rows(5),
            val_rows=[],
            feature_cols=TRAINING_FEATURE_COLS,
            categorical_cols=CATEGORICAL_FEATURE_COLS,
            target_col="target_price_7d",
        )


def test_fit_raises_when_no_valid_targets():
    m = LightGBMForecaster(horizon_days=7)
    rows = _make_rows(15, include_target=False)
    with pytest.raises(ValueError, match="non-null target"):
        m.fit(
            train_rows=rows,
            val_rows=[],
            feature_cols=TRAINING_FEATURE_COLS,
            categorical_cols=CATEGORICAL_FEATURE_COLS,
            target_col="target_price_7d",
        )


def test_fit_returns_empty_metrics_without_val():
    m = LightGBMForecaster(horizon_days=7, n_estimators=5)
    metrics = m.fit(
        train_rows=_make_rows(20),
        val_rows=[],
        feature_cols=TRAINING_FEATURE_COLS,
        categorical_cols=CATEGORICAL_FEATURE_COLS,
        target_col="target_price_7d",
    )
    assert metrics == {}


def test_fit_returns_metrics_with_val():
    m = LightGBMForecaster(horizon_days=7, n_estimators=5)
    metrics = m.fit(
        train_rows=_make_rows(20),
        val_rows=_make_rows(5),
        feature_cols=TRAINING_FEATURE_COLS,
        categorical_cols=CATEGORICAL_FEATURE_COLS,
        target_col="target_price_7d",
    )
    assert "mae" in metrics
    assert "rmse" in metrics
    assert "n_val" in metrics
    assert metrics["n_val"] == 5.0
    assert metrics["mae"] >= 0.0
    assert metrics["rmse"] >= 0.0


# ── predict() ─────────────────────────────────────────────────────────────────

def test_predict_before_fit_returns_nones():
    m = LightGBMForecaster(horizon_days=7)
    rows = _make_rows(3)
    preds = m.predict(rows)
    assert preds == [None, None, None]


def test_predict_returns_correct_length():
    m = LightGBMForecaster(horizon_days=7, n_estimators=5)
    m.fit(
        train_rows=_make_rows(20),
        val_rows=[],
        feature_cols=TRAINING_FEATURE_COLS,
        categorical_cols=CATEGORICAL_FEATURE_COLS,
        target_col="target_price_7d",
    )
    rows = _make_rows(4)
    preds = m.predict(rows)
    assert len(preds) == 4


def test_predict_returns_nonnegative():
    m = LightGBMForecaster(horizon_days=7, n_estimators=5)
    m.fit(
        train_rows=_make_rows(20),
        val_rows=[],
        feature_cols=TRAINING_FEATURE_COLS,
        categorical_cols=CATEGORICAL_FEATURE_COLS,
        target_col="target_price_7d",
    )
    rows = _make_rows(10)
    preds = m.predict(rows)
    assert all(isinstance(p, float) for p in preds)
    assert all(p >= 0.0 for p in preds)


def test_predict_empty_rows():
    m = LightGBMForecaster(horizon_days=7, n_estimators=5)
    m.fit(
        train_rows=_make_rows(20),
        val_rows=[],
        feature_cols=TRAINING_FEATURE_COLS,
        categorical_cols=CATEGORICAL_FEATURE_COLS,
        target_col="target_price_7d",
    )
    assert m.predict([]) == []


# ── save() / load() ───────────────────────────────────────────────────────────

def test_save_raises_on_unfitted(tmp_path: Path):
    m = LightGBMForecaster(horizon_days=7)
    with pytest.raises(RuntimeError, match="unfitted"):
        m.save(tmp_path / "model.pkl")


def test_save_load_round_trip(tmp_path: Path):
    m = LightGBMForecaster(horizon_days=7, n_estimators=5)
    m.fit(
        train_rows=_make_rows(20),
        val_rows=[],
        feature_cols=TRAINING_FEATURE_COLS,
        categorical_cols=CATEGORICAL_FEATURE_COLS,
        target_col="target_price_7d",
    )
    artifact = tmp_path / "lgbm_7d.pkl"
    m.save(artifact)

    loaded = LightGBMForecaster.load(artifact)
    assert loaded.is_fitted
    assert loaded.horizon_days == 7


def test_load_raises_on_missing_file(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        LightGBMForecaster.load(tmp_path / "nonexistent.pkl")


def test_loaded_model_predicts_nonnegative(tmp_path: Path):
    m = LightGBMForecaster(horizon_days=7, n_estimators=5)
    m.fit(
        train_rows=_make_rows(20),
        val_rows=[],
        feature_cols=TRAINING_FEATURE_COLS,
        categorical_cols=CATEGORICAL_FEATURE_COLS,
        target_col="target_price_7d",
    )
    artifact = tmp_path / "lgbm_7d.pkl"
    m.save(artifact)

    loaded = LightGBMForecaster.load(artifact)
    preds = loaded.predict(_make_rows(5))
    assert all(p >= 0.0 for p in preds)


# ── write_metadata() ──────────────────────────────────────────────────────────

def test_write_metadata_produces_valid_json(tmp_path: Path):
    m = LightGBMForecaster(horizon_days=7, n_estimators=5)
    m.fit(
        train_rows=_make_rows(20),
        val_rows=[],
        feature_cols=TRAINING_FEATURE_COLS,
        categorical_cols=CATEGORICAL_FEATURE_COLS,
        target_col="target_price_7d",
    )
    meta_path = tmp_path / "lgbm_7d.json"
    m.write_metadata(meta_path, realm_slug="area-52", dataset_version="train_area-52_2024-09-01_2025-01-31.parquet")

    data = json.loads(meta_path.read_text())
    assert data["horizon_days"] == 7
    assert data["realm_slug"] == "area-52"
    assert "feature_columns" in data
    assert "hyperparameters" in data
    assert data["model_type"] == "lightgbm"
