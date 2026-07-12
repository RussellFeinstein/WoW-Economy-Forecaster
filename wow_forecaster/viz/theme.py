"""
WoW-inspired visual theme for matplotlib and Plotly.

Dark background with gold accent palette — immediately signals domain
expertise while maintaining publication-quality readability.

Usage::

    import matplotlib.pyplot as plt
    from wow_forecaster.viz.theme import apply_wow_theme

    fig, ax = plt.subplots()
    apply_wow_theme(fig)
    ax.plot(x, y)
"""

from __future__ import annotations

from typing import Any

# ── Core palette ──────────────────────────────────────────────────────────────

WOW_PALETTE: dict[str, str] = {
    "primary": "#C9AA71",       # WoW gold
    "secondary": "#8B7355",     # Muted gold
    "background": "#1A1A2E",    # Deep midnight blue
    "surface": "#16213E",       # Slightly lighter surface
    "text": "#E0D5C0",          # Warm parchment
    "text_muted": "#8A8070",    # Subdued text
    "grid": "#2A2A4A",          # Subtle grid lines
    "accent_blue": "#4A9BD9",   # Alliance blue
    "accent_red": "#C41E3A",    # Horde red
    "white": "#F5F0E8",         # Off-white
}

# ── Action colors ─────────────────────────────────────────────────────────────

ACTION_COLORS: dict[str, str] = {
    "buy": "#2ECC71",           # Confident green
    "sell": "#E74C3C",          # Alert red
    "hold": "#F39C12",          # Amber / wait
    "avoid": "#95A5A6",         # Muted gray
}

# ── Horizon colors ────────────────────────────────────────────────────────────

HORIZON_COLORS: dict[str, str] = {
    "1d": "#3498DB",            # Bright blue — immediate
    "7d": "#9B59B6",            # Purple — weekly
    "28d": "#E67E22",           # Orange — monthly
}

# ── Category colors (one per ArchetypeCategory slug) ──────────────────────────

CATEGORY_COLORS: dict[str, str] = {
    "consumable": "#2ECC71",    # Green — flasks, potions
    "mat": "#E67E22",           # Orange — raw materials
    "gear": "#E74C3C",          # Red — equipment
    "enchant": "#9B59B6",       # Purple — enchantments
    "gem": "#1ABC9C",           # Teal — gems
    "prof_tool": "#F1C40F",     # Yellow — profession tools
    "reagent": "#3498DB",       # Blue — reagents
    "trade_good": "#95A5A6",    # Gray — trade goods
    "service": "#8E44AD",       # Dark purple — services
    "collection": "#E91E63",    # Pink — pets/mounts/transmog
}

# ── CI quality colors ─────────────────────────────────────────────────────────

CI_QUALITY_COLORS: dict[str, str] = {
    "good": "#2ECC71",          # Green
    "wide": "#F39C12",          # Orange
    "unreliable": "#E74C3C",    # Red
}

# ── Risk level colors ─────────────────────────────────────────────────────────

RISK_COLORS: dict[str, str] = {
    "low": "#2ECC71",
    "medium": "#F39C12",
    "high": "#E74C3C",
    "critical": "#8E44AD",
}

# ── Drift level colors ───────────────────────────────────────────────────────

DRIFT_COLORS: dict[str, str] = {
    "none": "#2ECC71",
    "low": "#3498DB",
    "medium": "#F39C12",
    "high": "#E74C3C",
    "critical": "#8E44AD",
}

# ── Feature group colors (match FeatureSpec.group slugs) ──────────────────────

FEATURE_GROUP_COLORS: dict[str, str] = {
    "price": "#C9AA71",         # Gold — core price features
    "volume": "#3498DB",        # Blue
    "lag": "#9B59B6",           # Purple
    "rolling": "#E67E22",       # Orange
    "momentum": "#E74C3C",      # Red
    "temporal": "#1ABC9C",      # Teal
    "event": "#F1C40F",         # Yellow
    "archetype": "#2ECC71",     # Green
    "transfer": "#95A5A6",      # Gray
}


def apply_wow_theme(fig_or_ax: Any) -> None:
    """Apply the WoW dark theme to a matplotlib Figure or Axes.

    Sets background colors, text colors, grid style, and font sizes
    suitable for publication-quality output at 300 DPI.

    Args:
        fig_or_ax: A matplotlib ``Figure`` or ``Axes`` instance.
    """
    import matplotlib.pyplot as plt  # noqa: F811

    # Apply rcParams globally for new figures
    plt.rcParams.update({
        "figure.facecolor": WOW_PALETTE["background"],
        "axes.facecolor": WOW_PALETTE["surface"],
        "axes.edgecolor": WOW_PALETTE["grid"],
        "axes.labelcolor": WOW_PALETTE["primary"],
        "axes.titlesize": 14,
        "axes.labelsize": 11,
        "text.color": WOW_PALETTE["text"],
        "xtick.color": WOW_PALETTE["text_muted"],
        "ytick.color": WOW_PALETTE["text_muted"],
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "grid.color": WOW_PALETTE["grid"],
        "grid.alpha": 0.5,
        "grid.linestyle": "--",
        "legend.facecolor": WOW_PALETTE["surface"],
        "legend.edgecolor": WOW_PALETTE["grid"],
        "legend.fontsize": 9,
        "legend.labelcolor": WOW_PALETTE["text"],
        "figure.titlesize": 16,
        "figure.titleweight": "bold",
        "savefig.facecolor": WOW_PALETTE["background"],
        "savefig.edgecolor": "none",
        "savefig.dpi": 300,
        "font.size": 10,
    })

    # Apply to the specific figure/axes if provided
    import matplotlib.figure

    if isinstance(fig_or_ax, matplotlib.figure.Figure):
        fig = fig_or_ax
        fig.set_facecolor(WOW_PALETTE["background"])
        for ax in fig.get_axes():
            _style_axes(ax)
    else:
        _style_axes(fig_or_ax)


def _style_axes(ax: Any) -> None:
    """Apply WoW theme styling to a single Axes."""
    ax.set_facecolor(WOW_PALETTE["surface"])
    ax.tick_params(colors=WOW_PALETTE["text_muted"])
    ax.xaxis.label.set_color(WOW_PALETTE["primary"])
    ax.yaxis.label.set_color(WOW_PALETTE["primary"])
    ax.title.set_color(WOW_PALETTE["text"])
    for spine in ax.spines.values():
        spine.set_color(WOW_PALETTE["grid"])
    ax.grid(True, alpha=0.5, linestyle="--", color=WOW_PALETTE["grid"])


def get_plotly_template() -> dict:
    """Return a Plotly layout template dict matching the WoW dark theme.

    Usage::

        import plotly.graph_objects as go
        from wow_forecaster.viz.theme import get_plotly_template

        fig = go.Figure(layout=get_plotly_template())
    """
    return {
        "paper_bgcolor": WOW_PALETTE["background"],
        "plot_bgcolor": WOW_PALETTE["surface"],
        "font": {
            "color": WOW_PALETTE["text"],
            "size": 12,
        },
        "title": {
            "font": {
                "color": WOW_PALETTE["text"],
                "size": 18,
            },
        },
        "xaxis": {
            "gridcolor": WOW_PALETTE["grid"],
            "zerolinecolor": WOW_PALETTE["grid"],
            "tickfont": {"color": WOW_PALETTE["text_muted"]},
            "title": {"font": {"color": WOW_PALETTE["primary"]}},
        },
        "yaxis": {
            "gridcolor": WOW_PALETTE["grid"],
            "zerolinecolor": WOW_PALETTE["grid"],
            "tickfont": {"color": WOW_PALETTE["text_muted"]},
            "title": {"font": {"color": WOW_PALETTE["primary"]}},
        },
        "legend": {
            "bgcolor": WOW_PALETTE["surface"],
            "bordercolor": WOW_PALETTE["grid"],
            "font": {"color": WOW_PALETTE["text"]},
        },
        "colorway": [
            WOW_PALETTE["primary"],
            WOW_PALETTE["accent_blue"],
            WOW_PALETTE["accent_red"],
            "#2ECC71",
            "#9B59B6",
            "#E67E22",
            "#1ABC9C",
            "#F1C40F",
        ],
    }
