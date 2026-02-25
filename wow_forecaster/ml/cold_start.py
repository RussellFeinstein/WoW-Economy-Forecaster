"""
Confidence-interval widening for cold-start Midnight archetypes.

Cold-start items (is_cold_start=True) lack sufficient price history to
produce well-calibrated predictions.  Two mechanisms handle this:

1. Global model: trained on ALL archetypes including cold-start rows.
   The feature ``is_cold_start_int=1`` signals to LightGBM that the row
   belongs to a thin series.  The model self-regulates via this feature.

2. CI widening (this module): the heuristic confidence interval is
   multiplied by a factor derived from ``transfer_confidence``:

     transfer_confidence = 0.90  → widening = 1.5 / 0.90 ≈ 1.67×
     transfer_confidence = 0.60  → widening = 1.5 / 0.60 = 2.50×
     transfer_confidence = None  → widening = 3.00× (no mapping at all)

   Minimum CI half-width is always ≥ 5% of the predicted price.

V1 limitation
-------------
Full prediction blending (source-archetype prediction × transfer_confidence)
is a planned V2 feature.  It requires source predictions to be available at
inference time as a lookup table — adding complexity not yet warranted.

Uncertainty note
----------------
CIs produced here are HEURISTIC, not model-derived.  They do NOT come from
quantile regression or conformal prediction.  V2 should use LightGBM with
``objective="quantile"`` to generate model-calibrated intervals.
"""

from __future__ import annotations

# z-score for 80% CI two-sided: P(|Z| <= 1.28) ≈ 0.80
_Z_LOOKUP: dict[float, float] = {
    0.50: 0.674,
    0.80: 1.280,
    0.90: 1.645,
    0.95: 1.960,
    0.99: 2.576,
}
_DEFAULT_Z = 1.280

# Fallback CI half-width when rolling_std is unavailable (20% of predicted)
_DEFAULT_UNCERTAINTY_FRAC = 0.20

# Minimum CI half-width as fraction of predicted price (floor = 5%)
_MIN_CI_FRAC = 0.05


def compute_confidence_interval(
    predicted: float,
    rolling_std_7d: float | None,
    is_cold_start: bool,
    transfer_confidence: float | None,
    confidence_pct: float = 0.80,
) -> tuple[float, float]:
    """Compute heuristic confidence interval for a price forecast.

    Args:
        predicted:         Predicted price in gold (central estimate).
        rolling_std_7d:    7-day rolling standard deviation of price_mean
                           in gold.  None if insufficient history.
        is_cold_start:     True when the archetype has < cold_start_threshold
                           observations.
        transfer_confidence: Confidence of the transfer mapping (0.0–1.0).
                           None means no transfer mapping exists.
        confidence_pct:    Target CI level (default 0.80 = 80% two-sided).

    Returns:
        Tuple ``(lower, upper)`` in gold, both non-negative.
    """
    z = _Z_LOOKUP.get(confidence_pct, _DEFAULT_Z)

    if rolling_std_7d is not None and rolling_std_7d > 0:
        ci_half = z * rolling_std_7d
    else:
        ci_half = _DEFAULT_UNCERTAINTY_FRAC * predicted

    if is_cold_start:
        if transfer_confidence is not None and transfer_confidence > 0.0:
            widening = 1.5 / transfer_confidence
        else:
            widening = 3.0
        ci_half *= widening

    # Apply floor
    min_half = _MIN_CI_FRAC * predicted
    ci_half = max(ci_half, min_half)

    lower = max(0.0, predicted - ci_half)
    upper = predicted + ci_half
    return lower, upper


def cold_start_model_slug(
    base_slug: str,
    is_cold_start: bool,
    has_transfer_mapping: bool,
) -> str:
    """Annotate the model slug for provenance tracking.

    Callers can inspect model_slug to distinguish warm forecasts from
    cold-start / transfer forecasts in the forecast_outputs table.

    Args:
        base_slug:          Base slug (e.g. "lgbm_7d_v0.5.0").
        is_cold_start:      True if the archetype is cold-start.
        has_transfer_mapping: True if an archetype_mapping record exists.

    Returns:
        Possibly annotated slug string.
    """
    if is_cold_start and has_transfer_mapping:
        return f"{base_slug}_transfer"
    if is_cold_start:
        return f"{base_slug}_cold"
    return base_slug
