"""
Tests for wow_forecaster/ml/feature_selector.py.

What we test
------------
encode_row():
  - Bool fields are converted to 0/1 integers (event_active, is_cold_start, etc.).
  - String fields are mapped to ordinals via CATEGORY/SEVERITY/IMPACT_ENCODING.
  - Unknown/None strings produce the 0-sentinel.
  - Original raw keys are preserved in the output dict.

to_float():
  - None  -> float("nan")
  - int / float passthrough
  - Non-numeric string -> float("nan")

build_feature_matrix():
  - Returns a list of lists, each inner list has len == len(feature_cols).
  - None values become NaN inside the matrix.
  - Custom feature_cols subset works correctly.
"""

from __future__ import annotations

import math

import pytest

from wow_forecaster.ml.feature_selector import (
    CATEGORY_ENCODING,
    SEVERITY_ENCODING,
    IMPACT_ENCODING,
    TRAINING_FEATURE_COLS,
    build_feature_matrix,
    encode_row,
    to_float,
)


# ── to_float ──────────────────────────────────────────────────────────────────

class TestToFloat:
    def test_none_is_nan(self):
        assert math.isnan(to_float(None))

    def test_int_passthrough(self):
        assert to_float(42) == 42.0

    def test_float_passthrough(self):
        assert to_float(3.14) == pytest.approx(3.14)

    def test_numeric_string(self):
        assert to_float("7.5") == pytest.approx(7.5)

    def test_non_numeric_string_is_nan(self):
        assert math.isnan(to_float("hello"))

    def test_empty_string_is_nan(self):
        assert math.isnan(to_float(""))


# ── encode_row ────────────────────────────────────────────────────────────────

class TestEncodeRow:
    def _base_row(self) -> dict:
        return {
            "price_mean":        100.0,
            "archetype_category": "consumable",
            "event_severity_max": "major",
            "event_archetype_impact": "positive",
            "event_active":       True,
            "is_transferable":    False,
            "is_cold_start":      True,
            "has_transfer_mapping": False,
        }

    def test_bool_event_active_true(self):
        row = self._base_row()
        enc = encode_row(row)
        assert enc["event_active_int"] == 1

    def test_bool_event_active_false(self):
        row = self._base_row()
        row["event_active"] = False
        enc = encode_row(row)
        assert enc["event_active_int"] == 0

    def test_bool_is_cold_start(self):
        row = self._base_row()
        enc = encode_row(row)
        assert enc["is_cold_start_int"] == 1

    def test_bool_is_transferable_false(self):
        row = self._base_row()
        enc = encode_row(row)
        assert enc["is_transferable_int"] == 0

    def test_bool_has_transfer_mapping_false(self):
        row = self._base_row()
        enc = encode_row(row)
        assert enc["has_transfer_mapping_int"] == 0

    def test_category_encoding_consumable(self):
        row = self._base_row()
        enc = encode_row(row)
        assert enc["archetype_category_enc"] == CATEGORY_ENCODING["consumable"]

    def test_category_encoding_mat(self):
        row = self._base_row()
        row["archetype_category"] = "mat"
        enc = encode_row(row)
        assert enc["archetype_category_enc"] == CATEGORY_ENCODING["mat"]

    def test_category_encoding_unknown_is_zero(self):
        row = self._base_row()
        row["archetype_category"] = "nonexistent_category"
        enc = encode_row(row)
        assert enc["archetype_category_enc"] == 0

    def test_category_encoding_none_is_zero(self):
        row = self._base_row()
        row["archetype_category"] = None
        enc = encode_row(row)
        assert enc["archetype_category_enc"] == 0

    def test_severity_encoding_major(self):
        row = self._base_row()
        enc = encode_row(row)
        assert enc["event_severity_enc"] == SEVERITY_ENCODING["major"]

    def test_severity_encoding_none_is_zero(self):
        row = self._base_row()
        row["event_severity_max"] = None
        enc = encode_row(row)
        assert enc["event_severity_enc"] == 0

    def test_impact_encoding_positive(self):
        row = self._base_row()
        enc = encode_row(row)
        assert enc["event_archetype_impact_enc"] == 1

    def test_impact_encoding_negative(self):
        row = self._base_row()
        row["event_archetype_impact"] = "negative"
        enc = encode_row(row)
        assert enc["event_archetype_impact_enc"] == -1

    def test_impact_encoding_neutral_is_zero(self):
        row = self._base_row()
        row["event_archetype_impact"] = "neutral"
        enc = encode_row(row)
        assert enc["event_archetype_impact_enc"] == 0

    def test_original_keys_preserved(self):
        row = self._base_row()
        enc = encode_row(row)
        # Original raw keys must survive
        assert "archetype_category" in enc
        assert enc["archetype_category"] == "consumable"
        assert enc["price_mean"] == 100.0

    def test_missing_keys_default_to_zero(self):
        enc = encode_row({})
        assert enc["event_active_int"] == 0
        assert enc["is_cold_start_int"] == 0
        assert enc["archetype_category_enc"] == 0


# ── build_feature_matrix ──────────────────────────────────────────────────────

class TestBuildFeatureMatrix:
    def _encoded_row(self, price: float = 100.0) -> dict:
        base = {c: price for c in TRAINING_FEATURE_COLS}
        base["archetype_category_enc"] = 2   # consumable
        base["event_severity_enc"]     = 0
        base["day_of_week"]            = 3
        return base

    def test_correct_shape(self):
        rows = [self._encoded_row(i) for i in range(5)]
        matrix = build_feature_matrix(rows, TRAINING_FEATURE_COLS)
        assert len(matrix) == 5
        assert all(len(row) == len(TRAINING_FEATURE_COLS) for row in matrix)

    def test_none_becomes_nan(self):
        row = {c: None for c in TRAINING_FEATURE_COLS}
        matrix = build_feature_matrix([row], TRAINING_FEATURE_COLS)
        assert all(math.isnan(v) for v in matrix[0])

    def test_custom_feature_cols(self):
        row = {"price_mean": 50.0, "obs_count": 10.0}
        matrix = build_feature_matrix([row], ["price_mean", "obs_count"])
        assert matrix == [[50.0, 10.0]]

    def test_empty_rows(self):
        matrix = build_feature_matrix([], TRAINING_FEATURE_COLS)
        assert matrix == []

    def test_values_are_float(self):
        row = self._encoded_row(price=42)
        matrix = build_feature_matrix([row], ["price_mean"])
        assert isinstance(matrix[0][0], float)


# ── TRAINING_FEATURE_COLS sanity ──────────────────────────────────────────────

def test_training_feature_cols_count():
    """Sanity check: TRAINING_FEATURE_COLS has the expected 37 columns."""
    assert len(TRAINING_FEATURE_COLS) == 37


def test_training_feature_cols_no_leakage():
    """Target columns must never appear in TRAINING_FEATURE_COLS."""
    forbidden = {"target_price_1d", "target_price_7d", "target_price_28d",
                 "archetype_id", "realm_slug", "obs_date"}
    overlap = forbidden & set(TRAINING_FEATURE_COLS)
    assert not overlap, f"Leaking columns in TRAINING_FEATURE_COLS: {overlap}"
