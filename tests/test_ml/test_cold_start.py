"""Unit tests for cold_start.blend_cold_start_prediction()."""

from __future__ import annotations

import pytest

from wow_forecaster.ml.cold_start import blend_cold_start_prediction


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
