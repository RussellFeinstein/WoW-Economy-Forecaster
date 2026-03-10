"""Unit tests for cold_start — blend_cold_start_prediction, compute_confidence_interval,
and classify_ci_quality."""

from __future__ import annotations

import pytest

from wow_forecaster.ml.cold_start import (
    blend_cold_start_prediction,
    classify_ci_quality,
    compute_confidence_interval,
)


class TestBlendColdStartPrediction:
    """Tests for blend_cold_start_prediction()."""

    def test_confidence_one_returns_model_prediction(self):
        """confidence=1.0 → pure model prediction, source price ignored."""
        result = blend_cold_start_prediction(
            model_prediction=100.0,
            source_price=50.0,
            transfer_confidence=1.0,
        )
        assert result == pytest.approx(100.0)

    def test_confidence_zero_returns_source_price(self):
        """confidence=0.0 → pure source price, model prediction ignored."""
        result = blend_cold_start_prediction(
            model_prediction=100.0,
            source_price=50.0,
            transfer_confidence=0.0,
        )
        assert result == pytest.approx(50.0)

    def test_midpoint_confidence_produces_weighted_average(self):
        """confidence=0.5 → equal weight blend."""
        result = blend_cold_start_prediction(
            model_prediction=100.0,
            source_price=50.0,
            transfer_confidence=0.5,
        )
        assert result == pytest.approx(75.0)  # 0.5*100 + 0.5*50

    def test_typical_confidence_weighted_toward_model(self):
        """confidence=0.7 → correct weighted average."""
        result = blend_cold_start_prediction(
            model_prediction=100.0,
            source_price=40.0,
            transfer_confidence=0.7,
        )
        # 0.7*100 + 0.3*40 = 70 + 12 = 82
        assert result == pytest.approx(82.0)

    def test_source_price_zero_returns_model_prediction(self):
        """source_price <= 0 → fall back to model prediction (source unavailable)."""
        result = blend_cold_start_prediction(
            model_prediction=100.0,
            source_price=0.0,
            transfer_confidence=0.5,
        )
        assert result == pytest.approx(100.0)

    def test_source_price_negative_returns_model_prediction(self):
        """Negative source prices are treated as unavailable."""
        result = blend_cold_start_prediction(
            model_prediction=100.0,
            source_price=-10.0,
            transfer_confidence=0.5,
        )
        assert result == pytest.approx(100.0)

    def test_confidence_above_one_clamped_to_model_prediction(self):
        """confidence > 1.0 is clamped to 1.0 → pure model prediction."""
        result = blend_cold_start_prediction(
            model_prediction=100.0,
            source_price=50.0,
            transfer_confidence=2.5,
        )
        assert result == pytest.approx(100.0)

    def test_confidence_below_zero_clamped_to_source_price(self):
        """confidence < 0.0 is clamped to 0.0 → pure source price."""
        result = blend_cold_start_prediction(
            model_prediction=100.0,
            source_price=50.0,
            transfer_confidence=-1.0,
        )
        assert result == pytest.approx(50.0)

    def test_result_is_non_negative(self):
        """Blended result must always be >= 0 even with edge-case inputs."""
        result = blend_cold_start_prediction(
            model_prediction=-500.0,
            source_price=-100.0,
            transfer_confidence=0.5,
        )
        assert result == pytest.approx(0.0)

    def test_model_prediction_zero_with_positive_source(self):
        """Model predicts zero; source anchors the result."""
        result = blend_cold_start_prediction(
            model_prediction=0.0,
            source_price=80.0,
            transfer_confidence=0.6,
        )
        # 0.6*0 + 0.4*80 = 32
        assert result == pytest.approx(32.0)


class TestComputeConfidenceInterval:
    """Tests for compute_confidence_interval() floor/cap logic."""

    def test_no_current_price_returns_unclamped_bounds(self):
        """Without current_price, no floor/cap is applied."""
        lower, upper = compute_confidence_interval(
            predicted=100.0,
            rolling_std_7d=None,
            is_cold_start=False,
            transfer_confidence=None,
        )
        # Default uncertainty frac = 20% of 100 = 20.  Floor = 5% of 100 = 5.
        # ci_half = max(20, 5) = 20 → lower = 80, upper = 120.
        assert lower == pytest.approx(80.0)
        assert upper == pytest.approx(120.0)

    def test_current_price_floor_prevents_zero_lower_bound(self):
        """With small predicted and large ci_half, floor keeps lower >= 5% of current."""
        # predicted=1, rolling_std=5 → ci_half = 1.28*5 = 6.4 → lower = max(0, -5.4) = 0.0
        # current_price=100 → floor = 5.0 → lower should be clamped to 5.0
        lower, upper = compute_confidence_interval(
            predicted=1.0,
            rolling_std_7d=5.0,
            is_cold_start=False,
            transfer_confidence=None,
            current_price=100.0,
        )
        assert lower >= 5.0

    def test_current_price_cap_prevents_absurd_upper_bound(self):
        """Cold-start widening can push upper very high; cap at 10× current_price."""
        # predicted=100, no rolling_std → ci_half = 20, cold-start no transfer → ×3 = 60
        # upper = 100 + 60 = 160. current_price=100 → cap = 1000 → no clip needed.
        # But with transfer_confidence=0.1 → widening = 15 → ci_half = 300 → upper = 400
        # cap at 100*10 = 1000 → upper stays at 400 (no clip needed with current_price=100)
        # Let's use current_price=30 → cap = 300 → upper 400 clips to 300
        lower, upper = compute_confidence_interval(
            predicted=100.0,
            rolling_std_7d=None,
            is_cold_start=True,
            transfer_confidence=0.1,   # widening = 1.5/0.1 = 15× → ci_half = 20*15 = 300
            current_price=30.0,        # cap = 300
        )
        assert upper <= 300.0

    def test_lower_bound_never_exceeds_upper_after_clamping(self):
        """Even in extreme cases (floor > cap), lower is re-clamped to upper."""
        # This pathological case: current_price=1000, predicted=0.01
        # floor = 50, cap = 10000, no conflict here, but let's verify invariant
        lower, upper = compute_confidence_interval(
            predicted=0.01,
            rolling_std_7d=None,
            is_cold_start=True,
            transfer_confidence=None,   # widening = 3×
            current_price=1000.0,
        )
        assert lower <= upper

    def test_floor_applied_when_lower_would_be_zero(self):
        """Verify floor is applied when normal computation yields ci_lower=0."""
        # large std relative to predicted → lower = 0 without floor
        lower, upper = compute_confidence_interval(
            predicted=5.0,
            rolling_std_7d=20.0,  # ci_half = 1.28*20 = 25.6 → lower = max(0, -20.6) = 0
            is_cold_start=False,
            transfer_confidence=None,
            current_price=100.0,  # floor = 5.0
        )
        assert lower >= 5.0

    def test_without_current_price_lower_can_still_be_zero(self):
        """Without current_price, lower can be 0 when std >> predicted."""
        lower, upper = compute_confidence_interval(
            predicted=1.0,
            rolling_std_7d=50.0,
            is_cold_start=False,
            transfer_confidence=None,
            current_price=None,
        )
        assert lower == pytest.approx(0.0)


class TestClassifyCiQuality:
    """Tests for classify_ci_quality()."""

    def test_zero_predicted_returns_unreliable(self):
        """predicted=0 → unreliable (division by zero guard)."""
        assert classify_ci_quality(0.0, 10.0, 0.0) == "unreliable"

    def test_negative_predicted_returns_unreliable(self):
        """Negative predicted is treated as unreliable."""
        assert classify_ci_quality(0.0, 5.0, -1.0) == "unreliable"

    def test_narrow_ci_is_good(self):
        """Width < 50% → good."""
        # lower=90, upper=130, predicted=100 → width=40/100=40% < 50%
        assert classify_ci_quality(90.0, 130.0, 100.0) == "good"

    def test_boundary_49_percent_is_good(self):
        """Width just below 50% → good."""
        assert classify_ci_quality(75.5, 124.5, 100.0) == "good"  # 49/100 = 49%

    def test_boundary_50_percent_is_wide(self):
        """Width exactly 50% → wide."""
        assert classify_ci_quality(75.0, 125.0, 100.0) == "wide"  # 50/100 = 50%

    def test_moderate_ci_is_wide(self):
        """Width in 50–200% range → wide."""
        # lower=50, upper=150, predicted=100 → width=100/100=100%
        assert classify_ci_quality(50.0, 150.0, 100.0) == "wide"

    def test_boundary_just_under_200_percent_is_wide(self):
        """Width just under 200% → wide."""
        assert classify_ci_quality(1.0, 200.0, 100.0) == "wide"  # 199/100 = 199%

    def test_boundary_200_percent_is_unreliable(self):
        """Width = 200% exactly → unreliable."""
        assert classify_ci_quality(0.0, 200.0, 100.0) == "unreliable"  # 200/100 = 200%

    def test_very_wide_ci_is_unreliable(self):
        """Width > 200% → unreliable."""
        # lower=0, upper=500, predicted=100 → width=500/100=500%
        assert classify_ci_quality(0.0, 500.0, 100.0) == "unreliable"

    def test_perfect_ci_is_good(self):
        """Very narrow CI around predicted is good."""
        assert classify_ci_quality(99.0, 101.0, 100.0) == "good"  # 2% width
