"""
Tests for the adaptive update policy.

What we test
------------
1. Uncertainty multiplier progression matches documented values.
2. retrain_recommended is False for NONE/LOW, True for MEDIUM/HIGH/CRITICAL.
3. auto_retrain is off by default at all levels, including CRITICAL.
4. auto_retrain is enabled only for CRITICAL when allow_auto_retrain=True.
5. Policy function accepts both DriftLevel enum and plain strings.
6. Unknown drift level falls back to 1.0 multiplier / no retrain.
"""

from __future__ import annotations

import pytest

from wow_forecaster.monitoring.adaptive import AdaptivePolicyResult, evaluate_policy
from wow_forecaster.monitoring.drift import DriftLevel


# ── Uncertainty multiplier ────────────────────────────────────────────────────

class TestUncertaintyMultiplier:
    @pytest.mark.parametrize("level, expected_mult", [
        (DriftLevel.NONE,     1.00),
        (DriftLevel.LOW,      1.25),
        (DriftLevel.MEDIUM,   1.50),
        (DriftLevel.HIGH,     2.00),
        (DriftLevel.CRITICAL, 3.00),
    ])
    def test_multiplier_progression(self, level: DriftLevel, expected_mult: float):
        result = evaluate_policy(level)
        assert result.uncertainty_multiplier == expected_mult

    def test_multiplier_increases_monotonically(self):
        levels = [DriftLevel.NONE, DriftLevel.LOW, DriftLevel.MEDIUM,
                  DriftLevel.HIGH, DriftLevel.CRITICAL]
        mults = [evaluate_policy(lvl).uncertainty_multiplier for lvl in levels]
        assert mults == sorted(mults), "Multipliers must increase monotonically"

    def test_multiplier_always_at_least_one(self):
        for level in DriftLevel:
            result = evaluate_policy(level)
            assert result.uncertainty_multiplier >= 1.0


# ── Retrain recommended ───────────────────────────────────────────────────────

class TestRetrainRecommended:
    @pytest.mark.parametrize("level, expected", [
        (DriftLevel.NONE,     False),
        (DriftLevel.LOW,      False),
        (DriftLevel.MEDIUM,   True),
        (DriftLevel.HIGH,     True),
        (DriftLevel.CRITICAL, True),
    ])
    def test_retrain_flag(self, level: DriftLevel, expected: bool):
        result = evaluate_policy(level)
        assert result.retrain_recommended == expected

    def test_no_retrain_for_low_drift(self):
        assert not evaluate_policy(DriftLevel.NONE).retrain_recommended
        assert not evaluate_policy(DriftLevel.LOW).retrain_recommended

    def test_retrain_for_medium_and_above(self):
        for level in (DriftLevel.MEDIUM, DriftLevel.HIGH, DriftLevel.CRITICAL):
            assert evaluate_policy(level).retrain_recommended


# ── Auto-retrain behaviour ────────────────────────────────────────────────────

class TestAutoRetrain:
    def test_auto_retrain_off_by_default_for_all_levels(self):
        for level in DriftLevel:
            result = evaluate_policy(level)
            assert result.auto_retrain == False, (
                f"auto_retrain should be False by default for {level}"
            )

    def test_auto_retrain_off_by_default_for_critical(self):
        result = evaluate_policy(DriftLevel.CRITICAL)
        assert result.auto_retrain == False

    def test_auto_retrain_enabled_for_critical_when_allowed(self):
        result = evaluate_policy(DriftLevel.CRITICAL, allow_auto_retrain=True)
        assert result.auto_retrain == True

    def test_auto_retrain_not_enabled_below_critical_even_when_allowed(self):
        for level in (DriftLevel.NONE, DriftLevel.LOW, DriftLevel.MEDIUM, DriftLevel.HIGH):
            result = evaluate_policy(level, allow_auto_retrain=True)
            assert result.auto_retrain == False, (
                f"auto_retrain should not trigger for {level} even when allow_auto_retrain=True"
            )


# ── String input compatibility ────────────────────────────────────────────────

class TestStringInput:
    """Policy function should accept both DriftLevel enum and plain strings."""

    def test_string_none(self):
        result = evaluate_policy("none")
        assert result.uncertainty_multiplier == 1.0

    def test_string_critical(self):
        result = evaluate_policy("critical")
        assert result.uncertainty_multiplier == 3.0
        assert result.retrain_recommended == True

    def test_unknown_string_falls_back_gracefully(self):
        result = evaluate_policy("unknown_level")
        assert result.uncertainty_multiplier == 1.0
        assert result.retrain_recommended == False
        assert result.auto_retrain == False


# ── AdaptivePolicyResult type ─────────────────────────────────────────────────

class TestAdaptivePolicyResult:
    def test_result_is_frozen(self):
        result = evaluate_policy(DriftLevel.HIGH)
        with pytest.raises((AttributeError, TypeError)):
            result.uncertainty_multiplier = 99.0

    def test_result_fields_present(self):
        result = evaluate_policy(DriftLevel.MEDIUM)
        assert hasattr(result, "uncertainty_multiplier")
        assert hasattr(result, "retrain_recommended")
        assert hasattr(result, "auto_retrain")
