"""
Tests for wow_forecaster/recommendations/ranker.py.

What we test
------------
build_scored_forecasts():
  - Matches forecasts to inference rows by (archetype_id, realm_slug).
  - Forecasts with no matching inference row are silently skipped.
  - Returns a ScoredForecast with score, action, reasoning, category_tag.
  - Empty forecast list -> empty result.
  - Empty inference list -> empty result (no matches).

top_n_per_category():
  - Returns at most n items per category.
  - Items within each category are sorted by score descending.
  - Ties are broken by archetype_id ascending (deterministic).
  - Empty input -> empty dict.
  - n=0 -> empty lists in all categories.
  - actions filter excludes non-matching action strings.

build_recommendation_outputs():
  - Returns one RecommendationOutput per ScoredForecast with a forecast_id.
  - Skips ScoredForecast entries whose forecast.forecast_id is None.
  - priority is 1-based rank within category.
  - score_components is a non-empty JSON string.
  - category_tag matches the ScoredForecast.category_tag.
"""

from __future__ import annotations

import json
from datetime import date

import pytest

from wow_forecaster.models.forecast import ForecastOutput, RecommendationOutput
from wow_forecaster.recommendations.ranker import (
    ScoredForecast,
    build_recommendation_outputs,
    build_scored_forecasts,
    top_n_per_category,
)
from wow_forecaster.recommendations.scorer import ScoreComponents


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _forecast(
    archetype_id: int = 1,
    realm_slug: str = "area-52",
    forecast_id: int | None = 1,
    predicted_price: float = 110.0,
    horizon: str = "7d",
) -> ForecastOutput:
    return ForecastOutput(
        forecast_id=forecast_id,
        run_id=1,
        archetype_id=archetype_id,
        item_id=None,
        realm_slug=realm_slug,
        forecast_horizon=horizon,
        target_date=date(2025, 2, 1),
        predicted_price_gold=predicted_price,
        confidence_lower=predicted_price * 0.9,
        confidence_upper=predicted_price * 1.1,
        confidence_pct=0.80,
        model_slug="lgbm_7d_area-52",
        features_hash="abc123",
    )


def _inf_row(
    archetype_id: int = 1,
    realm_slug: str = "area-52",
    price_mean: float = 100.0,
    category: str = "consumable",
    quantity_sum: float = 500.0,
    price_roll_std_7d: float = 5.0,
) -> dict:
    return {
        "archetype_id":          archetype_id,
        "realm_slug":            realm_slug,
        "price_mean":            price_mean,
        "archetype_category":    category,
        "quantity_sum":          quantity_sum,
        "auctions_sum":          50.0,
        "price_roll_std_7d":     price_roll_std_7d,
        "event_active":          False,
        "event_days_to_next":    None,
        "event_severity_max":    None,
        "event_archetype_impact": None,
        "is_cold_start":         False,
        "transfer_confidence":   1.0,
    }


def _scored_forecast(
    score: float = 30.0,
    archetype_id: int = 1,
    category: str = "consumable",
    action: str = "buy",
    forecast_id: int | None = 1,
) -> ScoredForecast:
    fc = _forecast(archetype_id=archetype_id, forecast_id=forecast_id)
    components = ScoreComponents(
        opportunity_score=40.0,
        liquidity_score=50.0,
        volatility_penalty=10.0,
        event_boost=0.0,
        uncertainty_penalty=10.0,
        roi=0.10,
        volatility_cv=0.05,
        uncertainty_pct=0.10,
    )
    return ScoredForecast(
        forecast=fc,
        score=score,
        components=components,
        action=action,
        reasoning="Moderate upward forecast: +10.0% expected 7d return",
        category_tag=category,
        archetype_sub_tag=None,
        archetype_id=archetype_id,
        realm_slug="area-52",
        current_price=100.0,
        horizon_days=7,
    )


# ── build_scored_forecasts ────────────────────────────────────────────────────

class TestBuildScoredForecasts:
    def test_empty_forecasts_returns_empty(self):
        result = build_scored_forecasts([], [_inf_row()])
        assert result == []

    def test_empty_inference_rows_returns_empty(self):
        result = build_scored_forecasts([_forecast()], [])
        assert result == []

    def test_matched_forecast_returns_scored(self):
        fc  = _forecast(archetype_id=1, realm_slug="area-52")
        inf = _inf_row(archetype_id=1, realm_slug="area-52")
        result = build_scored_forecasts([fc], [inf])
        assert len(result) == 1
        sf = result[0]
        assert sf.archetype_id == 1
        assert sf.realm_slug == "area-52"
        assert sf.category_tag == "consumable"
        assert isinstance(sf.score, float)
        assert sf.action in {"buy", "sell", "hold", "avoid"}
        assert isinstance(sf.reasoning, str)

    def test_unmatched_forecast_is_skipped(self):
        # archetype_id mismatch
        fc  = _forecast(archetype_id=99)
        inf = _inf_row(archetype_id=1)
        result = build_scored_forecasts([fc], [inf])
        assert result == []

    def test_realm_slug_mismatch_skipped(self):
        fc  = _forecast(archetype_id=1, realm_slug="us-stormrage")
        inf = _inf_row(archetype_id=1, realm_slug="area-52")
        result = build_scored_forecasts([fc], [inf])
        assert result == []

    def test_multiple_forecasts_multiple_matches(self):
        fcs  = [_forecast(archetype_id=i) for i in range(1, 4)]
        infs = [_inf_row(archetype_id=i) for i in range(1, 4)]
        result = build_scored_forecasts(fcs, infs)
        assert len(result) == 3

    def test_partial_matches(self):
        fcs  = [_forecast(archetype_id=1), _forecast(archetype_id=99)]
        infs = [_inf_row(archetype_id=1)]
        result = build_scored_forecasts(fcs, infs)
        assert len(result) == 1
        assert result[0].archetype_id == 1

    def test_positive_roi_gives_buy_action(self):
        fc  = _forecast(predicted_price=150.0)
        inf = _inf_row(price_mean=100.0, price_roll_std_7d=2.0)
        result = build_scored_forecasts([fc], [inf])
        assert result[0].action == "buy"

    def test_category_tag_from_inference_row(self):
        fc  = _forecast(archetype_id=1)
        inf = _inf_row(archetype_id=1, category="mat")
        result = build_scored_forecasts([fc], [inf])
        assert result[0].category_tag == "mat"


# ── top_n_per_category ────────────────────────────────────────────────────────

class TestTopNPerCategory:
    def test_empty_input_returns_empty_dict(self):
        result = top_n_per_category([], n=3)
        assert result == {}

    def test_respects_n_limit(self):
        scored = [_scored_forecast(score=float(i), archetype_id=i) for i in range(1, 6)]
        result = top_n_per_category(scored, n=3)
        assert "consumable" in result
        assert len(result["consumable"]) == 3

    def test_n_larger_than_available_returns_all(self):
        scored = [_scored_forecast(score=10.0, archetype_id=1)]
        result = top_n_per_category(scored, n=5)
        assert len(result["consumable"]) == 1

    def test_sorted_by_score_descending(self):
        scored = [
            _scored_forecast(score=10.0, archetype_id=1),
            _scored_forecast(score=50.0, archetype_id=2),
            _scored_forecast(score=30.0, archetype_id=3),
        ]
        result = top_n_per_category(scored, n=3)
        scores = [sf.score for sf in result["consumable"]]
        assert scores == sorted(scores, reverse=True)

    def test_tie_broken_by_archetype_id_ascending(self):
        scored = [
            _scored_forecast(score=25.0, archetype_id=5),
            _scored_forecast(score=25.0, archetype_id=2),
            _scored_forecast(score=25.0, archetype_id=8),
        ]
        result = top_n_per_category(scored, n=3)
        ids = [sf.archetype_id for sf in result["consumable"]]
        assert ids == [2, 5, 8]

    def test_multiple_categories_separated(self):
        scored = [
            _scored_forecast(score=40.0, archetype_id=1, category="consumable"),
            _scored_forecast(score=35.0, archetype_id=2, category="mat"),
            _scored_forecast(score=30.0, archetype_id=3, category="consumable"),
            _scored_forecast(score=25.0, archetype_id=4, category="mat"),
        ]
        result = top_n_per_category(scored, n=3)
        assert "consumable" in result
        assert "mat" in result
        assert len(result["consumable"]) == 2
        assert len(result["mat"]) == 2

    def test_n_zero_returns_empty_lists(self):
        scored = [_scored_forecast(score=50.0, archetype_id=1)]
        result = top_n_per_category(scored, n=0)
        # Each category present but empty
        assert all(len(v) == 0 for v in result.values())

    def test_actions_filter_excludes_other_actions(self):
        scored = [
            _scored_forecast(score=50.0, archetype_id=1, action="buy"),
            _scored_forecast(score=40.0, archetype_id=2, action="sell"),
            _scored_forecast(score=30.0, archetype_id=3, action="hold"),
        ]
        result = top_n_per_category(scored, n=3, actions=["buy"])
        assert "consumable" in result
        items = result["consumable"]
        assert len(items) == 1
        assert items[0].action == "buy"

    def test_actions_none_includes_all(self):
        scored = [
            _scored_forecast(score=50.0, archetype_id=1, action="buy"),
            _scored_forecast(score=40.0, archetype_id=2, action="avoid"),
        ]
        result = top_n_per_category(scored, n=3, actions=None)
        assert len(result["consumable"]) == 2


# ── build_recommendation_outputs ─────────────────────────────────────────────

class TestBuildRecommendationOutputs:
    def test_empty_input_returns_empty(self):
        result = build_recommendation_outputs({})
        assert result == []

    def test_skips_none_forecast_id(self):
        sf = _scored_forecast(forecast_id=None)
        result = build_recommendation_outputs({"consumable": [sf]})
        assert result == []

    def test_returns_one_output_per_valid_scored(self):
        sfs = [
            _scored_forecast(score=50.0, archetype_id=1, forecast_id=10),
            _scored_forecast(score=40.0, archetype_id=2, forecast_id=11),
        ]
        result = build_recommendation_outputs({"consumable": sfs})
        assert len(result) == 2

    def test_priority_is_one_based_rank(self):
        sfs = [
            _scored_forecast(score=50.0, archetype_id=1, forecast_id=10),
            _scored_forecast(score=40.0, archetype_id=2, forecast_id=11),
            _scored_forecast(score=30.0, archetype_id=3, forecast_id=12),
        ]
        result = build_recommendation_outputs({"consumable": sfs})
        priorities = [r.priority for r in result]
        assert priorities == [1, 2, 3]

    def test_category_tag_set_correctly(self):
        sf = _scored_forecast(category="mat", forecast_id=10)
        result = build_recommendation_outputs({"mat": [sf]})
        assert result[0].category_tag == "mat"

    def test_score_persisted(self):
        sf = _scored_forecast(score=42.5, forecast_id=10)
        result = build_recommendation_outputs({"consumable": [sf]})
        assert result[0].score == pytest.approx(42.5)

    def test_score_components_is_valid_json(self):
        sf = _scored_forecast(forecast_id=10)
        result = build_recommendation_outputs({"consumable": [sf]})
        blob = json.loads(result[0].score_components)
        expected_keys = {
            "opportunity_score", "liquidity_score", "volatility_penalty",
            "event_boost", "uncertainty_penalty", "roi", "volatility_cv",
            "uncertainty_pct",
        }
        assert expected_keys <= blob.keys()

    def test_action_set_correctly(self):
        sf = _scored_forecast(action="sell", forecast_id=10)
        result = build_recommendation_outputs({"consumable": [sf]})
        assert result[0].action == "sell"

    def test_multiple_categories(self):
        top_by_cat = {
            "consumable": [_scored_forecast(score=50.0, archetype_id=1, forecast_id=10)],
            "mat":        [_scored_forecast(score=40.0, archetype_id=2, category="mat", forecast_id=11)],
        }
        result = build_recommendation_outputs(top_by_cat)
        assert len(result) == 2
        cats = {r.category_tag for r in result}
        assert cats == {"consumable", "mat"}
