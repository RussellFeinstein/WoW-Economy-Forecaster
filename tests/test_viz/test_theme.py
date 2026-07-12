"""Tests for wow_forecaster.viz.theme — palette, theme application, Plotly template."""

from __future__ import annotations

import re

from wow_forecaster.viz.theme import (
    ACTION_COLORS,
    CATEGORY_COLORS,
    CI_QUALITY_COLORS,
    DRIFT_COLORS,
    FEATURE_GROUP_COLORS,
    HORIZON_COLORS,
    RISK_COLORS,
    WOW_PALETTE,
    apply_wow_theme,
    get_plotly_template,
)

_HEX_RE = re.compile(r"^#[0-9A-Fa-f]{6}$")


# ── Palette completeness ──────────────────────────────────────────────────────


class TestWoWPalette:
    def test_required_keys_exist(self):
        for key in ("primary", "secondary", "background", "surface", "text",
                     "text_muted", "grid", "accent_blue", "accent_red", "white"):
            assert key in WOW_PALETTE, f"Missing palette key: {key}"

    def test_all_values_are_hex(self):
        for key, val in WOW_PALETTE.items():
            assert _HEX_RE.match(val), f"Palette[{key}]={val!r} is not a valid hex color"


class TestActionColors:
    def test_all_actions_mapped(self):
        for action in ("buy", "sell", "hold", "avoid"):
            assert action in ACTION_COLORS

    def test_all_hex(self):
        for key, val in ACTION_COLORS.items():
            assert _HEX_RE.match(val), f"ACTION_COLORS[{key}] invalid"


class TestHorizonColors:
    def test_all_horizons_mapped(self):
        for h in ("1d", "7d", "28d"):
            assert h in HORIZON_COLORS

    def test_all_hex(self):
        for key, val in HORIZON_COLORS.items():
            assert _HEX_RE.match(val), f"HORIZON_COLORS[{key}] invalid"


class TestCategoryColors:
    def test_all_categories_mapped(self):
        from wow_forecaster.taxonomy.archetype_taxonomy import ArchetypeCategory

        for cat in ArchetypeCategory:
            assert cat.value in CATEGORY_COLORS, f"Missing category color: {cat.value}"

    def test_all_hex(self):
        for key, val in CATEGORY_COLORS.items():
            assert _HEX_RE.match(val), f"CATEGORY_COLORS[{key}] invalid"


class TestCIQualityColors:
    def test_all_qualities_mapped(self):
        for q in ("good", "wide", "unreliable"):
            assert q in CI_QUALITY_COLORS

    def test_all_hex(self):
        for _key, val in CI_QUALITY_COLORS.items():
            assert _HEX_RE.match(val)


class TestRiskColors:
    def test_all_levels_mapped(self):
        for level in ("low", "medium", "high", "critical"):
            assert level in RISK_COLORS

    def test_all_hex(self):
        for _key, val in RISK_COLORS.items():
            assert _HEX_RE.match(val)


class TestDriftColors:
    def test_all_levels_mapped(self):
        for level in ("none", "low", "medium", "high", "critical"):
            assert level in DRIFT_COLORS

    def test_all_hex(self):
        for _key, val in DRIFT_COLORS.items():
            assert _HEX_RE.match(val)


class TestFeatureGroupColors:
    def test_all_groups_mapped(self):
        for g in ("price", "volume", "lag", "rolling", "momentum",
                   "temporal", "event", "archetype", "transfer"):
            assert g in FEATURE_GROUP_COLORS

    def test_all_hex(self):
        for _key, val in FEATURE_GROUP_COLORS.items():
            assert _HEX_RE.match(val)


# ── Theme application ─────────────────────────────────────────────────────────


class TestApplyWoWTheme:
    def test_apply_to_figure(self):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        apply_wow_theme(fig)
        assert fig.get_facecolor() != (1.0, 1.0, 1.0, 1.0)  # not white
        plt.close(fig)

    def test_apply_to_axes(self):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        apply_wow_theme(ax)
        assert ax.get_facecolor() != (1.0, 1.0, 1.0, 1.0)
        plt.close(fig)

    def test_does_not_raise_on_empty_figure(self):
        import matplotlib.pyplot as plt

        fig = plt.figure()
        apply_wow_theme(fig)  # no axes — should not crash
        plt.close(fig)


# ── Plotly template ───────────────────────────────────────────────────────────


class TestGetPlotlyTemplate:
    def test_returns_dict(self):
        template = get_plotly_template()
        assert isinstance(template, dict)

    def test_has_required_keys(self):
        template = get_plotly_template()
        for key in ("paper_bgcolor", "plot_bgcolor", "font", "xaxis", "yaxis",
                     "legend", "colorway"):
            assert key in template, f"Missing Plotly template key: {key}"

    def test_colorway_is_list(self):
        template = get_plotly_template()
        assert isinstance(template["colorway"], list)
        assert len(template["colorway"]) >= 6

    def test_colors_are_hex(self):
        template = get_plotly_template()
        assert _HEX_RE.match(template["paper_bgcolor"])
        assert _HEX_RE.match(template["plot_bgcolor"])
