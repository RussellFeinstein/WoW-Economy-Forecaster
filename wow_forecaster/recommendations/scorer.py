"""
Recommendation scoring: converts a ForecastOutput + inference row into a
scored recommendation with action, reasoning, and component breakdown.

Score formula (weighted sum, approximate range 0–100)
------------------------------------------------------
    total = (
        opportunity_score    * 0.35   # buy-side ROI potential
        + liquidity_score    * 0.20   # market depth / ease of trading
        - volatility_penalty * 0.20   # price instability risk
        + event_boost        * 0.15   # event-driven demand signals
        - uncertainty_penalty* 0.10   # CI width / forecast confidence
    )

Component explanations
----------------------
opportunity_score (0–100):
    Expected ROI = (predicted − current) / current.
    50% ROI → score 100. Negative ROI → 0 (buy-only signal; sell logic
    is captured in the action decision, not the score magnitude).
    Formula: clamp(roi * 200, 0, 100).

liquidity_score (0–100):
    Proxy for AH market depth using quantity_sum (units listed).
    1000 units → 100.  Falls back to auctions_sum if quantity unavailable.
    Unknown liquidity: 10 (small baseline, not 0).

volatility_penalty (0–100):
    Coefficient of variation (price_roll_std_7d / price_mean).
    CV = 50% → penalty 50.  Penalises execution risk on volatile archetypes.

event_boost (0–100):
    Active events with "positive" archetype impact add severity-based boost.
    Upcoming events within 7 days add a proportional anticipation boost.
    Negative impacts apply a soft penalty (negative boost * 0.5).

uncertainty_penalty (0–100):
    CI width / predicted_price.
    CI width = 100% of predicted → penalty 100 (max uncertainty).

Action determination (priority order)
--------------------------------------
    1. AVOID : uncertainty_pct >= 0.80  OR  volatility_cv >= 0.80
    2. BUY   : roi >= 0.10
    3. SELL  : roi <= -0.10
    4. HOLD  : all other cases
"""

from __future__ import annotations

from dataclasses import dataclass

# Severity → base event boost mapping
_SEVERITY_BOOST: dict[str, float] = {
    "minor":        5.0,
    "moderate":    15.0,
    "major":       30.0,
    "critical":    50.0,
    "catastrophic": 70.0,
}


@dataclass
class ScoreComponents:
    """All components of a recommendation score.

    Attributes:
        opportunity_score:   0–100, derived from expected ROI.
        liquidity_score:     0–100, derived from quantity / auction depth.
        volatility_penalty:  0–100, derived from price coefficient of variation.
        event_boost:         0–100, derived from active/upcoming events.
        uncertainty_penalty: 0–100, derived from CI width / predicted price.
        roi:                 Raw expected return (can be negative).
        volatility_cv:       Raw CV = price_roll_std_7d / price_mean.
        uncertainty_pct:     Raw CI width / predicted price.
    """

    opportunity_score:   float
    liquidity_score:     float
    volatility_penalty:  float
    event_boost:         float
    uncertainty_penalty: float
    roi:                 float
    volatility_cv:       float
    uncertainty_pct:     float

    @property
    def total(self) -> float:
        """Weighted total score.  Approximate range 0–100 for buy opportunities."""
        return (
            self.opportunity_score    * 0.35
            + self.liquidity_score    * 0.20
            - self.volatility_penalty * 0.20
            + self.event_boost        * 0.15
            - self.uncertainty_penalty* 0.10
        )


def compute_score(
    forecast_price:       float,
    current_price:        float | None,
    confidence_lower:     float,
    confidence_upper:     float,
    quantity_sum:         float | None,
    auctions_sum:         float | None,
    price_roll_std_7d:    float | None,
    price_mean:           float | None,
    event_active:         bool,
    event_days_to_next:   float | None,
    event_severity_max:   str | None,
    event_archetype_impact: str | None,
    is_cold_start:        bool,
    transfer_confidence:  float | None,
) -> ScoreComponents:
    """Compute all recommendation score components for one forecast row.

    Args:
        forecast_price:          Predicted price in gold (central estimate).
        current_price:           Current price_mean from inference row (gold).
        confidence_lower:        Lower CI bound in gold.
        confidence_upper:        Upper CI bound in gold.
        quantity_sum:            Total units listed on AH (depth proxy).
        auctions_sum:            Number of distinct auction listings.
        price_roll_std_7d:       7-day rolling std of price_mean (gold).
        price_mean:              Current mean price (gold) — may equal current_price.
        event_active:            True if any known event is currently active.
        event_days_to_next:      Days until next event start; None if unknown.
        event_severity_max:      Severity string of the currently active event.
        event_archetype_impact:  Impact direction for this archetype from DB.
        is_cold_start:           True if archetype has insufficient history.
        transfer_confidence:     Transfer mapping confidence (0–1); None = none.

    Returns:
        ScoreComponents with all fields populated.
    """
    # ── Opportunity score ─────────────────────────────────────────────────────
    ref_current = current_price or price_mean
    if ref_current is not None and ref_current > 0:
        roi = (forecast_price - ref_current) / ref_current
    else:
        roi = 0.0
    opportunity_score = _clamp(roi * 200.0, 0.0, 100.0)

    # ── Liquidity score ───────────────────────────────────────────────────────
    qty  = quantity_sum  or 0.0
    aucs = auctions_sum  or 0.0
    if qty > 0:
        liquidity_score = _clamp(qty / 10.0, 0.0, 100.0)     # 1000 units = 100
    elif aucs > 0:
        liquidity_score = _clamp(aucs * 1.0, 0.0, 100.0)     # 100 auctions = 100
    else:
        liquidity_score = 10.0                                 # unknown → small baseline

    # ── Volatility penalty ────────────────────────────────────────────────────
    ref_price = price_mean or ref_current or forecast_price
    if price_roll_std_7d is not None and ref_price and ref_price > 0:
        volatility_cv = price_roll_std_7d / ref_price
    else:
        volatility_cv = 0.0
    volatility_penalty = _clamp(volatility_cv * 100.0, 0.0, 100.0)

    # ── Event boost ───────────────────────────────────────────────────────────
    event_boost = 0.0
    if event_active:
        base = _SEVERITY_BOOST.get(event_severity_max or "", 10.0)
        if event_archetype_impact == "positive":
            event_boost = base
        elif event_archetype_impact == "negative":
            event_boost = -base * 0.5          # soft penalty for negative impact
        else:
            event_boost = base * 0.3           # neutral / unknown impact
    elif event_days_to_next is not None and event_days_to_next <= 7:
        # Anticipation boost decays linearly from 15 to 0 over 7 days
        event_boost = 15.0 * (1.0 - event_days_to_next / 7.0)
    event_boost = _clamp(event_boost, 0.0, 100.0)

    # ── Uncertainty penalty ───────────────────────────────────────────────────
    if forecast_price > 0:
        ci_width        = confidence_upper - confidence_lower
        uncertainty_pct = ci_width / forecast_price
    else:
        uncertainty_pct = 1.0
    uncertainty_penalty = _clamp(uncertainty_pct * 100.0, 0.0, 100.0)

    # Cold-start modifier: widen uncertainty penalty when no transfer mapping
    if is_cold_start and (transfer_confidence is None or transfer_confidence < 0.3):
        uncertainty_penalty = _clamp(uncertainty_penalty * 1.5, 0.0, 100.0)

    return ScoreComponents(
        opportunity_score=round(opportunity_score,   2),
        liquidity_score=round(liquidity_score,       2),
        volatility_penalty=round(volatility_penalty, 2),
        event_boost=round(event_boost,               2),
        uncertainty_penalty=round(uncertainty_penalty, 2),
        roi=round(roi,                 4),
        volatility_cv=round(volatility_cv,           4),
        uncertainty_pct=round(uncertainty_pct,       4),
    )


def determine_action(
    roi:             float,
    uncertainty_pct: float,
    volatility_cv:   float,
) -> str:
    """Determine the trading action from score components.

    Rules (evaluated in order — first match wins):
        1. AVOID : uncertainty >= 80%  OR  volatility CV >= 80%
        2. BUY   : roi >= 10%
        3. SELL  : roi <= -10%
        4. HOLD  : everything else

    Returns:
        One of "avoid", "buy", "sell", "hold".
    """
    if uncertainty_pct >= 0.80 or volatility_cv >= 0.80:
        return "avoid"
    if roi >= 0.10:
        return "buy"
    if roi <= -0.10:
        return "sell"
    return "hold"


def build_reasoning(
    components:          ScoreComponents,
    action:              str,
    is_cold_start:       bool,
    transfer_confidence: float | None,
    event_active:        bool,
    event_days_to_next:  float | None,
    event_severity_max:  str | None,
    horizon_days:        int,
) -> str:
    """Assemble a human-readable reasoning string from score components.

    Returns a semicolon-separated list of explanation tokens such as:
        "Strong upward forecast: +23.5% expected 7d return; Active major
        event boosts archetype demand; Narrow CI (8.2% width)"

    Args:
        components:          ScoreComponents from compute_score().
        action:              Action string ("buy", "sell", "hold", "avoid").
        is_cold_start:       Whether this is a cold-start archetype.
        transfer_confidence: Transfer mapping confidence (or None).
        event_active:        Whether an event is currently active.
        event_days_to_next:  Days until next known event.
        event_severity_max:  Severity string of any active event.
        horizon_days:        Forecast horizon in days (for display).

    Returns:
        Non-empty reasoning string.
    """
    reasons: list[str] = []

    # ROI signal
    roi = components.roi
    if roi >= 0.20:
        reasons.append(
            f"Strong upward forecast: +{roi:.1%} expected {horizon_days}d return"
        )
    elif roi >= 0.10:
        reasons.append(
            f"Moderate upward forecast: +{roi:.1%} expected {horizon_days}d return"
        )
    elif roi <= -0.10:
        reasons.append(
            f"Downward forecast: {roi:.1%} expected {horizon_days}d return"
        )
    else:
        reasons.append(f"Flat forecast: {roi:+.1%} expected {horizon_days}d return")

    # Volatility signal
    cv = components.volatility_cv
    if cv < 0.05:
        reasons.append(f"Very stable market (CV {cv:.1%})")
    elif cv > 0.40:
        reasons.append(f"High volatility risk (CV {cv:.1%})")

    # Event signal
    if event_active and event_severity_max:
        reasons.append(f"Active {event_severity_max} event affects archetype demand")
    elif event_days_to_next is not None and event_days_to_next <= 7:
        reasons.append(f"Event in {event_days_to_next:.0f}d — consider positioning early")

    # Uncertainty signal
    unc = components.uncertainty_pct
    if unc < 0.15:
        reasons.append(f"Narrow CI ({unc:.1%} width) — high model confidence")
    elif unc > 0.60:
        reasons.append(f"Wide CI ({unc:.1%} width) — low model confidence")

    # Cold-start signal
    if is_cold_start:
        if transfer_confidence is not None:
            reasons.append(
                f"Cold-start item: transfer weights applied "
                f"(confidence {transfer_confidence:.0%})"
            )
        else:
            reasons.append("Cold-start item: no transfer mapping — use caution")

    return "; ".join(reasons) or "No notable signals detected"


# ── Helper ────────────────────────────────────────────────────────────────────

def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))
