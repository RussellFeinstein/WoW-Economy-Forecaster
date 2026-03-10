"""
Tests for wow_forecaster/recommendations/scorer.py.

What we test
------------
compute_score():
  - All components are within expected ranges (0-100).
  - opportunity_score is 0 when predicted == current (ROI = 0).
  - opportunity_score grows with increasing ROI.
  - liquidity_score falls back to 10 when both qty and auctions are 0/None.
  - liquidity_score uses quantity_sum when > 0 (1000 units = 100).
  - volatility_penalty is 0 when std is 0 or unknown.
  - event_boost > 0 for an active "positive" event.
  - event_boost has anticipation component when days_to_next <= 7.
  - event_boost is clamped to [-100, 100] (regression: was wrongly documented as 0–100).
  - uncertainty_penalty reflects CI width relative to predicted price.
  - Cold-start with low transfer_confidence widens uncertainty_penalty.

ScoreComponents.total:
  - Follows the weighted formula.
  - Can be negative when penalties dominate.

determine_risk_level():
  - "critical" when uncertainty_pct >= 0.95.
  - "high" when uncertainty_pct >= 0.80 OR volatility_cv >= 0.80.
  - "medium" when uncertainty_pct >= 0.50 OR volatility_cv >= 0.50.
  - "low" otherwise.

determine_action():
  - "avoid" when uncertainty_pct >= 0.95 (critical only; was 0.80).
  - HIGH risk (0.80–0.95) no longer forces avoid — profitable signals surface.
  - "buy" when roi >= 0.10 (even at high uncertainty/volatility).
  - "sell" when roi <= -0.10.
  - "hold" otherwise.
  - Rules applied in priority order (avoid before buy).

build_reasoning():
  - Returns a non-empty string.
  - Contains ROI direction signal.
  - Contains cold-start notice when is_cold_start=True.
"""

from __future__ import annotations

import pytest

from wow_forecaster.recommendations.scorer import (
    ScoreComponents,
    build_reasoning,
    compute_score,
    determine_action,
    determine_risk_level,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _score(
    forecast_price: float = 110.0,
    current_price: float | None = 100.0,
    confidence_lower: float = 100.0,
    confidence_upper: float = 120.0,
    quantity_sum: float | None = 500.0,
    auctions_sum: float | None = None,
    price_roll_std_7d: float | None = 5.0,
    price_mean: float | None = 100.0,
    event_active: bool = False,
    event_days_to_next: float | None = None,
    event_severity_max: str | None = None,
    event_archetype_impact: str | None = None,
    is_cold_start: bool = False,
    transfer_confidence: float | None = 1.0,
) -> ScoreComponents:
    return compute_score(
        forecast_price=forecast_price,
        current_price=current_price,
        confidence_lower=confidence_lower,
        confidence_upper=confidence_upper,
        quantity_sum=quantity_sum,
        auctions_sum=auctions_sum,
        price_roll_std_7d=price_roll_std_7d,
        price_mean=price_mean,
        event_active=event_active,
        event_days_to_next=event_days_to_next,
        event_severity_max=event_severity_max,
        event_archetype_impact=event_archetype_impact,
        is_cold_start=is_cold_start,
        transfer_confidence=transfer_confidence,
    )


# ── compute_score: component ranges ──────────────────────────────────────────

class TestComputeScoreRanges:
    def test_all_components_are_nonnegative(self):
        # event_boost can be negative (negative event impact penalty)
        c = _score()
        assert c.opportunity_score   >= 0.0
        assert c.liquidity_score     >= 0.0
        assert c.volatility_penalty  >= 0.0
        assert c.uncertainty_penalty >= 0.0

    def test_all_components_at_most_100(self):
        c = _score(quantity_sum=9999.0, forecast_price=1000.0, current_price=1.0)
        assert c.opportunity_score   <= 100.0
        assert c.liquidity_score     <= 100.0
        assert c.volatility_penalty  <= 100.0
        assert c.event_boost         <= 100.0
        assert c.uncertainty_penalty <= 100.0


# ── compute_score: opportunity_score ─────────────────────────────────────────

class TestOpportunityScore:
    def test_zero_roi_gives_zero_opportunity(self):
        c = _score(forecast_price=100.0, current_price=100.0)
        assert c.roi == pytest.approx(0.0)
        assert c.opportunity_score == pytest.approx(0.0)

    def test_positive_roi_gives_positive_opportunity(self):
        c = _score(forecast_price=150.0, current_price=100.0)
        assert c.roi == pytest.approx(0.5, rel=1e-3)
        # 50% roi -> 50*200 = 100, clamped to 100
        assert c.opportunity_score == pytest.approx(100.0)

    def test_small_positive_roi(self):
        c = _score(forecast_price=110.0, current_price=100.0)
        # roi = 0.10, score = 0.10 * 200 = 20
        assert c.opportunity_score == pytest.approx(20.0)

    def test_negative_roi_gives_zero_opportunity(self):
        c = _score(forecast_price=80.0, current_price=100.0)
        assert c.roi < 0.0
        assert c.opportunity_score == pytest.approx(0.0)

    def test_none_current_price_falls_back_to_price_mean(self):
        c = _score(forecast_price=110.0, current_price=None, price_mean=100.0)
        assert c.roi == pytest.approx(0.1, rel=1e-3)


# ── compute_score: liquidity_score ───────────────────────────────────────────

class TestLiquidityScore:
    def test_unknown_liquidity_is_10(self):
        c = _score(quantity_sum=None, auctions_sum=None)
        assert c.liquidity_score == pytest.approx(10.0)

    def test_1000_units_gives_100(self):
        c = _score(quantity_sum=1000.0, auctions_sum=None)
        assert c.liquidity_score == pytest.approx(100.0)

    def test_500_units_gives_50(self):
        c = _score(quantity_sum=500.0, auctions_sum=None)
        assert c.liquidity_score == pytest.approx(50.0)

    def test_falls_back_to_auctions_when_qty_zero(self):
        c = _score(quantity_sum=0.0, auctions_sum=50.0)
        assert c.liquidity_score == pytest.approx(50.0)


# ── compute_score: volatility_penalty ────────────────────────────────────────

class TestVolatilityPenalty:
    def test_no_std_gives_zero_penalty(self):
        c = _score(price_roll_std_7d=None)
        assert c.volatility_penalty == pytest.approx(0.0)

    def test_zero_std_gives_zero_penalty(self):
        c = _score(price_roll_std_7d=0.0, price_mean=100.0)
        assert c.volatility_penalty == pytest.approx(0.0)

    def test_cv_50pct_gives_50_penalty(self):
        c = _score(price_roll_std_7d=50.0, price_mean=100.0)
        assert c.volatility_penalty == pytest.approx(50.0)

    def test_cv_exceeds_100pct_clamped(self):
        c = _score(price_roll_std_7d=200.0, price_mean=100.0)
        assert c.volatility_penalty == pytest.approx(100.0)


# ── compute_score: event_boost ───────────────────────────────────────────────

class TestEventBoost:
    def test_no_event_gives_zero_boost(self):
        c = _score(event_active=False, event_days_to_next=None)
        assert c.event_boost == pytest.approx(0.0)

    def test_active_positive_major_event(self):
        c = _score(
            event_active=True,
            event_severity_max="major",
            event_archetype_impact="positive",
        )
        # major base = 30.0
        assert c.event_boost == pytest.approx(30.0)

    def test_active_negative_event_soft_penalty(self):
        c = _score(
            event_active=True,
            event_severity_max="major",
            event_archetype_impact="negative",
        )
        # penalty = -base * 0.5 = -30 * 0.5 = -15 (negative, reduces total score)
        assert c.event_boost == pytest.approx(-15.0)

    def test_anticipation_boost_within_7_days(self):
        c = _score(event_active=False, event_days_to_next=0.0)
        # days_to_next=0 -> boost = 15 * (1 - 0/7) = 15
        assert c.event_boost == pytest.approx(15.0)

    def test_anticipation_boost_beyond_7_days_is_zero(self):
        c = _score(event_active=False, event_days_to_next=8.0)
        assert c.event_boost == pytest.approx(0.0)

    def test_event_boost_clamped_at_positive_100(self):
        """event_boost must never exceed +100, even if severity base is extreme."""
        import wow_forecaster.recommendations.scorer as scorer_mod
        original = scorer_mod._SEVERITY_BOOST.copy()
        scorer_mod._SEVERITY_BOOST["catastrophic"] = 999.0
        try:
            c = _score(
                event_active=True,
                event_severity_max="catastrophic",
                event_archetype_impact="positive",
            )
            assert c.event_boost <= 100.0
        finally:
            scorer_mod._SEVERITY_BOOST.update(original)

    def test_event_boost_clamped_at_negative_100(self):
        """event_boost must never go below -100, even if severity base is extreme."""
        import wow_forecaster.recommendations.scorer as scorer_mod
        original = scorer_mod._SEVERITY_BOOST.copy()
        scorer_mod._SEVERITY_BOOST["catastrophic"] = 999.0
        try:
            c = _score(
                event_active=True,
                event_severity_max="catastrophic",
                event_archetype_impact="negative",
            )
            assert c.event_boost >= -100.0
        finally:
            scorer_mod._SEVERITY_BOOST.update(original)

    def test_event_boost_is_negative_for_negative_impact(self):
        """Regression: event_boost range is [-100, 100], not [0, 100]."""
        c = _score(
            event_active=True,
            event_severity_max="major",
            event_archetype_impact="negative",
        )
        assert c.event_boost < 0.0


# ── compute_score: uncertainty_penalty ───────────────────────────────────────

class TestUncertaintyPenalty:
    def test_narrow_ci_low_penalty(self):
        # CI width = 10% of forecast_price
        c = _score(forecast_price=100.0, confidence_lower=95.0, confidence_upper=105.0)
        assert c.uncertainty_pct == pytest.approx(0.10)
        assert c.uncertainty_penalty == pytest.approx(10.0)

    def test_wide_ci_high_penalty(self):
        c = _score(forecast_price=100.0, confidence_lower=0.0, confidence_upper=100.0)
        assert c.uncertainty_pct == pytest.approx(1.0)
        assert c.uncertainty_penalty == pytest.approx(100.0)

    def test_cold_start_low_confidence_widens_penalty(self):
        normal = _score(
            forecast_price=100.0, confidence_lower=90.0, confidence_upper=110.0,
            is_cold_start=False, transfer_confidence=1.0,
        )
        cold = _score(
            forecast_price=100.0, confidence_lower=90.0, confidence_upper=110.0,
            is_cold_start=True, transfer_confidence=0.1,
        )
        assert cold.uncertainty_penalty > normal.uncertainty_penalty


# ── ScoreComponents field bounds ──────────────────────────────────────────────

class TestScoreComponentsBounds:
    """ScoreComponents.__post_init__ enforces documented field bounds."""

    def _valid_kwargs(self, **overrides):
        base = dict(
            opportunity_score=50.0,
            liquidity_score=50.0,
            volatility_penalty=20.0,
            event_boost=0.0,
            uncertainty_penalty=10.0,
            roi=0.1,
            volatility_cv=0.1,
            uncertainty_pct=0.1,
        )
        base.update(overrides)
        return base

    def test_valid_construction_succeeds(self):
        ScoreComponents(**self._valid_kwargs())  # must not raise

    def test_opportunity_score_above_100_raises(self):
        with pytest.raises(ValueError, match="opportunity_score"):
            ScoreComponents(**self._valid_kwargs(opportunity_score=101.0))

    def test_opportunity_score_below_0_raises(self):
        with pytest.raises(ValueError, match="opportunity_score"):
            ScoreComponents(**self._valid_kwargs(opportunity_score=-0.1))

    def test_liquidity_score_above_100_raises(self):
        with pytest.raises(ValueError, match="liquidity_score"):
            ScoreComponents(**self._valid_kwargs(liquidity_score=100.01))

    def test_volatility_penalty_above_100_raises(self):
        with pytest.raises(ValueError, match="volatility_penalty"):
            ScoreComponents(**self._valid_kwargs(volatility_penalty=100.1))

    def test_uncertainty_penalty_above_100_raises(self):
        with pytest.raises(ValueError, match="uncertainty_penalty"):
            ScoreComponents(**self._valid_kwargs(uncertainty_penalty=100.1))

    def test_event_boost_above_100_raises(self):
        with pytest.raises(ValueError, match="event_boost"):
            ScoreComponents(**self._valid_kwargs(event_boost=100.01))

    def test_event_boost_below_neg_100_raises(self):
        with pytest.raises(ValueError, match="event_boost"):
            ScoreComponents(**self._valid_kwargs(event_boost=-100.01))

    def test_event_boost_at_neg_100_is_valid(self):
        ScoreComponents(**self._valid_kwargs(event_boost=-100.0))  # boundary: valid

    def test_event_boost_at_pos_100_is_valid(self):
        ScoreComponents(**self._valid_kwargs(event_boost=100.0))   # boundary: valid

    def test_raw_fields_are_unconstrained(self):
        """roi, volatility_cv, uncertainty_pct have no bounds — large values are fine."""
        ScoreComponents(**self._valid_kwargs(roi=-9.9, volatility_cv=5.0, uncertainty_pct=3.0))


# ── ScoreComponents.total ─────────────────────────────────────────────────────

class TestScoreComponentsTotal:
    def test_total_formula(self):
        c = ScoreComponents(
            opportunity_score=40.0,
            liquidity_score=60.0,
            volatility_penalty=20.0,
            event_boost=10.0,
            uncertainty_penalty=15.0,
            roi=0.1,
            volatility_cv=0.1,
            uncertainty_pct=0.1,
        )
        expected = 40*0.35 + 60*0.20 - 20*0.20 + 10*0.15 - 15*0.10
        assert c.total == pytest.approx(expected)

    def test_total_can_be_negative(self):
        c = ScoreComponents(
            opportunity_score=0.0,
            liquidity_score=0.0,
            volatility_penalty=100.0,
            event_boost=0.0,
            uncertainty_penalty=100.0,
            roi=-0.5,
            volatility_cv=1.0,
            uncertainty_pct=1.0,
        )
        assert c.total < 0.0


# ── determine_action ──────────────────────────────────────────────────────────

class TestDetermineRiskLevel:
    def test_critical_at_95pct_uncertainty(self):
        assert determine_risk_level(uncertainty_pct=0.95, volatility_cv=0.0) == "critical"

    def test_critical_above_95pct(self):
        assert determine_risk_level(uncertainty_pct=1.0, volatility_cv=0.0) == "critical"

    def test_high_at_80pct_uncertainty(self):
        assert determine_risk_level(uncertainty_pct=0.80, volatility_cv=0.0) == "high"

    def test_high_via_volatility_cv(self):
        assert determine_risk_level(uncertainty_pct=0.10, volatility_cv=0.80) == "high"

    def test_medium_at_50pct_uncertainty(self):
        assert determine_risk_level(uncertainty_pct=0.50, volatility_cv=0.0) == "medium"

    def test_medium_via_volatility_cv(self):
        assert determine_risk_level(uncertainty_pct=0.10, volatility_cv=0.50) == "medium"

    def test_low_below_all_thresholds(self):
        assert determine_risk_level(uncertainty_pct=0.10, volatility_cv=0.10) == "low"

    def test_critical_takes_priority_over_high(self):
        # volatility_cv >= 0.80 alone is HIGH; but uncertainty >= 0.95 is CRITICAL
        assert determine_risk_level(uncertainty_pct=0.95, volatility_cv=0.85) == "critical"

    def test_boundary_just_below_critical(self):
        assert determine_risk_level(uncertainty_pct=0.949, volatility_cv=0.0) == "high"


class TestDetermineAction:
    def test_avoid_critical_uncertainty(self):
        # Only CRITICAL (>= 0.95) triggers avoid now
        assert determine_action(roi=0.20, uncertainty_pct=0.95, volatility_cv=0.10) == "avoid"

    def test_buy_at_high_uncertainty(self):
        # HIGH risk (0.80) no longer suppresses buy signals
        assert determine_action(roi=0.20, uncertainty_pct=0.80, volatility_cv=0.10) == "buy"

    def test_buy_at_high_volatility(self):
        # HIGH volatility no longer suppresses buy signals
        assert determine_action(roi=0.20, uncertainty_pct=0.10, volatility_cv=0.80) == "buy"

    def test_avoid_takes_priority_over_buy_at_critical(self):
        # roi >= 0.10 AND uncertainty >= 0.95 -> avoid wins
        assert determine_action(roi=0.50, uncertainty_pct=0.95, volatility_cv=0.10) == "avoid"

    def test_buy_10pct_roi(self):
        assert determine_action(roi=0.10, uncertainty_pct=0.10, volatility_cv=0.10) == "buy"

    def test_buy_large_roi(self):
        assert determine_action(roi=0.50, uncertainty_pct=0.05, volatility_cv=0.05) == "buy"

    def test_sell_negative_roi(self):
        assert determine_action(roi=-0.10, uncertainty_pct=0.10, volatility_cv=0.10) == "sell"

    def test_hold_small_positive_roi(self):
        assert determine_action(roi=0.05, uncertainty_pct=0.10, volatility_cv=0.10) == "hold"

    def test_hold_small_negative_roi(self):
        assert determine_action(roi=-0.05, uncertainty_pct=0.10, volatility_cv=0.10) == "hold"

    def test_hold_exactly_zero_roi(self):
        assert determine_action(roi=0.0, uncertainty_pct=0.10, volatility_cv=0.10) == "hold"

    def test_boundary_avoid_at_exactly_95pct(self):
        assert determine_action(roi=0.0, uncertainty_pct=0.95, volatility_cv=0.0) == "avoid"

    def test_buy_just_below_avoid_threshold(self):
        # 0.949 uncertainty + positive ROI -> buy (not avoid)
        assert determine_action(roi=0.15, uncertainty_pct=0.949, volatility_cv=0.0) == "buy"


# ── build_reasoning ───────────────────────────────────────────────────────────

class TestBuildReasoning:
    def _components(self, roi: float = 0.10) -> ScoreComponents:
        return ScoreComponents(
            opportunity_score=20.0,
            liquidity_score=50.0,
            volatility_penalty=5.0,
            event_boost=0.0,
            uncertainty_penalty=10.0,
            roi=roi,
            volatility_cv=0.05,
            uncertainty_pct=0.10,
        )

    def test_returns_nonempty_string(self):
        reason = build_reasoning(
            components=self._components(),
            action="buy",
            is_cold_start=False,
            transfer_confidence=None,
            event_active=False,
            event_days_to_next=None,
            event_severity_max=None,
            horizon_days=7,
        )
        assert isinstance(reason, str)
        assert len(reason) > 0

    def test_contains_roi_signal(self):
        reason = build_reasoning(
            components=self._components(roi=0.10),
            action="buy",
            is_cold_start=False,
            transfer_confidence=None,
            event_active=False,
            event_days_to_next=None,
            event_severity_max=None,
            horizon_days=7,
        )
        # "Moderate upward forecast" for 10% roi
        assert "forecast" in reason.lower()

    def test_cold_start_notice_present(self):
        reason = build_reasoning(
            components=self._components(),
            action="hold",
            is_cold_start=True,
            transfer_confidence=None,
            event_active=False,
            event_days_to_next=None,
            event_severity_max=None,
            horizon_days=7,
        )
        assert "cold-start" in reason.lower()

    def test_active_event_mentioned(self):
        reason = build_reasoning(
            components=self._components(),
            action="buy",
            is_cold_start=False,
            transfer_confidence=None,
            event_active=True,
            event_days_to_next=None,
            event_severity_max="major",
            horizon_days=7,
        )
        assert "major" in reason.lower()
