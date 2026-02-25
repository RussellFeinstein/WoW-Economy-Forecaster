"""Tests for ForecastOutput and RecommendationOutput models."""

from __future__ import annotations

from datetime import date

import pytest
from pydantic import ValidationError

from wow_forecaster.models.forecast import ForecastOutput, RecommendationOutput


class TestForecastOutput:
    def test_valid_construction(self, sample_forecast):
        f = sample_forecast
        assert f.predicted_price_gold == 520.0
        assert f.confidence_lower == 480.0
        assert f.confidence_upper == 580.0
        assert f.forecast_horizon == "7d"

    def test_inverted_confidence_interval_raises(self):
        with pytest.raises(ValidationError, match="confidence_lower"):
            ForecastOutput(
                run_id=1,
                archetype_id=1,
                realm_slug="area-52",
                forecast_horizon="7d",
                target_date=date(2024, 9, 22),
                predicted_price_gold=500.0,
                confidence_lower=600.0,  # higher than upper!
                confidence_upper=400.0,
                confidence_pct=0.80,
                model_slug="stub",
            )

    def test_negative_price_raises(self):
        with pytest.raises(ValidationError, match="non-negative"):
            ForecastOutput(
                run_id=1,
                archetype_id=1,
                realm_slug="area-52",
                forecast_horizon="7d",
                target_date=date(2024, 9, 22),
                predicted_price_gold=-10.0,
                confidence_lower=0.0,
                confidence_upper=50.0,
                confidence_pct=0.80,
                model_slug="stub",
            )

    def test_confidence_pct_zero_raises(self):
        with pytest.raises(ValidationError, match="confidence_pct"):
            ForecastOutput(
                run_id=1,
                archetype_id=1,
                realm_slug="area-52",
                forecast_horizon="7d",
                target_date=date(2024, 9, 22),
                predicted_price_gold=500.0,
                confidence_lower=400.0,
                confidence_upper=600.0,
                confidence_pct=0.0,
                model_slug="stub",
            )

    def test_confidence_pct_one_raises(self):
        with pytest.raises(ValidationError, match="confidence_pct"):
            ForecastOutput(
                run_id=1,
                archetype_id=1,
                realm_slug="area-52",
                forecast_horizon="7d",
                target_date=date(2024, 9, 22),
                predicted_price_gold=500.0,
                confidence_lower=400.0,
                confidence_upper=600.0,
                confidence_pct=1.0,
                model_slug="stub",
            )

    def test_no_archetype_or_item_raises(self):
        with pytest.raises(ValidationError, match="archetype_id or item_id"):
            ForecastOutput(
                run_id=1,
                archetype_id=None,
                item_id=None,
                realm_slug="area-52",
                forecast_horizon="7d",
                target_date=date(2024, 9, 22),
                predicted_price_gold=500.0,
                confidence_lower=400.0,
                confidence_upper=600.0,
                confidence_pct=0.80,
                model_slug="stub",
            )

    def test_item_id_only_is_valid(self):
        f = ForecastOutput(
            run_id=1,
            item_id=12345,
            realm_slug="area-52",
            forecast_horizon="1d",
            target_date=date(2024, 9, 16),
            predicted_price_gold=100.0,
            confidence_lower=80.0,
            confidence_upper=120.0,
            confidence_pct=0.80,
            model_slug="stub",
        )
        assert f.item_id == 12345
        assert f.archetype_id is None

    def test_frozen_immutable(self, sample_forecast):
        with pytest.raises(Exception):
            sample_forecast.predicted_price_gold = 999.0


class TestRecommendationOutput:
    def test_valid_construction(self):
        rec = RecommendationOutput(
            forecast_id=1,
            action="buy",
            reasoning="Flask prices expected to spike before RTWF.",
            priority=2,
        )
        assert rec.action == "buy"
        assert rec.priority == 2

    def test_priority_zero_raises(self):
        with pytest.raises(ValidationError, match="priority"):
            RecommendationOutput(
                forecast_id=1,
                action="buy",
                reasoning="Test.",
                priority=0,
            )

    def test_priority_eleven_raises(self):
        with pytest.raises(ValidationError, match="priority"):
            RecommendationOutput(
                forecast_id=1,
                action="sell",
                reasoning="Test.",
                priority=11,
            )

    def test_priority_boundaries_valid(self):
        for p in (1, 10):
            rec = RecommendationOutput(
                forecast_id=1,
                action="hold",
                reasoning="Boundary test.",
                priority=p,
            )
            assert rec.priority == p

    def test_empty_reasoning_raises(self):
        with pytest.raises(ValidationError, match="reasoning"):
            RecommendationOutput(
                forecast_id=1,
                action="buy",
                reasoning="",
            )

    def test_all_actions_valid(self):
        for action in ("buy", "sell", "hold", "avoid"):
            rec = RecommendationOutput(
                forecast_id=1,
                action=action,
                reasoning=f"Test {action}.",
            )
            assert rec.action == action

    def test_frozen_immutable(self):
        rec = RecommendationOutput(
            forecast_id=1,
            action="buy",
            reasoning="Test.",
        )
        with pytest.raises(Exception):
            rec.action = "sell"
