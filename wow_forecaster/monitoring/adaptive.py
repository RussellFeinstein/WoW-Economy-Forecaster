"""
Adaptive update policy for the WoW Economy Forecaster.

Maps drift severity (DriftLevel) to concrete behaviour changes:

  1. uncertainty_multiplier — applied to the uncertainty_penalty component
     in the recommendation scorer.  Widens confidence intervals for all
     recommendations produced under drifted conditions.

  2. retrain_recommended — advisory flag.  Written to drift_check_results
     and surfaced in check-drift CLI output.

  3. auto_retrain — whether to actually trigger a retrain.  Off by default;
     enabled only when ``allow_auto_retrain=True`` (config opt-in) AND drift
     is CRITICAL.

Why these multipliers?
----------------------
  - 1.0  (NONE):     Model performing normally.  No adjustment.
  - 1.25 (LOW):      Mild perturbation.  25% CI widening keeps strong
                     BUY/SELL signals valid while flagging uncertainty.
  - 1.5  (MEDIUM):   Noticeable regime change.  50% widening; borderline
                     BUY/HOLD signals will flip to HOLD, which is correct.
  - 2.0  (HIGH):     Substantial drift.  Doubles uncertainty.  Most
                     borderline recommendations become HOLD/AVOID.
  - 3.0  (CRITICAL): Extreme drift.  Model likely stale.  Almost all
                     recommendations become HOLD or AVOID.  Retrain urgent.

Why auto-retrain is off by default?
--------------------------------------
  Automatic retraining on live data during event shocks (e.g. a patch day)
  risks overfitting to transient price spikes.  A human review of the drift
  report and retrain trigger is safer.  The flag exists for future pipelines
  that have sufficient data-quality guardrails.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AdaptivePolicyResult:
    """Result of evaluating the adaptive policy for a drift level.

    Attributes:
        uncertainty_multiplier: Scale factor to apply to uncertainty_penalty
                                in recommendation scoring (>= 1.0).
        retrain_recommended:    Advisory flag — model should be retrained.
        auto_retrain:           Whether the system should auto-trigger a
                                retrain (requires allow_auto_retrain=True
                                AND DriftLevel.CRITICAL).
    """

    uncertainty_multiplier: float
    retrain_recommended:    bool
    auto_retrain:           bool


# ── Policy table ──────────────────────────────────────────────────────────────
# Keyed by DriftLevel value (string) to avoid a circular import from drift.py.
# evaluate_policy() accepts DriftLevel objects and looks up by .value.

_POLICY: dict[str, tuple[float, bool]] = {
    "none":     (1.00, False),
    "low":      (1.25, False),
    "medium":   (1.50, True),
    "high":     (2.00, True),
    "critical": (3.00, True),
}


def evaluate_policy(
    drift_level,   # DriftLevel — typed loosely to avoid circular import at module level
    allow_auto_retrain: bool = False,
) -> AdaptivePolicyResult:
    """Evaluate the adaptive update policy for a given drift level.

    Args:
        drift_level:        DriftLevel enum value (or string "none"/"low"/…).
        allow_auto_retrain: Whether auto-retrain is permitted.  Off by default.
                            Only takes effect when drift_level is CRITICAL.

    Returns:
        AdaptivePolicyResult with multiplier, retrain advisory, and auto flag.
    """
    key = drift_level.value if hasattr(drift_level, "value") else str(drift_level)
    multiplier, retrain = _POLICY.get(key, (1.0, False))

    # Auto-retrain only when explicitly allowed AND drift is critical
    auto = allow_auto_retrain and (key == "critical")

    return AdaptivePolicyResult(
        uncertainty_multiplier=multiplier,
        retrain_recommended=retrain,
        auto_retrain=auto,
    )
