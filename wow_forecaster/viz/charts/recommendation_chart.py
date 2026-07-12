"""
Recommendation scoring visualizations.

Score component tornado charts, action distributions, and ROI vs
uncertainty scatter — shows the scoring system is not a black box.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from wow_forecaster.viz.theme import (
    ACTION_COLORS,
    RISK_COLORS,
    WOW_PALETTE,
    apply_wow_theme,
)
from wow_forecaster.viz.utils import add_watermark, truncate_label


def plot_score_tornado(
    df_recs: pd.DataFrame,
    top_n: int = 10,
) -> plt.Figure:
    """Tornado chart of score components for top-N recommendations.

    Positive components (opportunity, liquidity, event_boost) stack right.
    Negative components (volatility, uncertainty) stack left.

    Score formula: 0.35*opp + 0.20*liq - 0.20*vol + 0.15*evt - 0.10*unc
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    apply_wow_theme(fig)

    if df_recs.empty:
        ax.text(0.5, 0.5, "No recommendation data",
                ha="center", va="center", color=WOW_PALETTE["text_muted"],
                transform=ax.transAxes, fontsize=14)
        return fig

    # Sort by score descending, take top N
    data = df_recs.nlargest(top_n, "score").copy()
    data = data.sort_values("score", ascending=True)  # bottom-to-top

    # Build labels
    labels = []
    for _, row in data.iterrows():
        cat = str(row.get("category", ""))
        aid = str(row.get("archetype_id", ""))
        labels.append(f"{cat} #{aid}")

    y = np.arange(len(labels))

    # Weighted components (matching the score formula)
    components = {
        "Opportunity (+)":  data["sc_opportunity"].astype(float) * 0.35,
        "Liquidity (+)":    data["sc_liquidity"].astype(float) * 0.20,
        "Event Boost (+)":  data["sc_event_boost"].astype(float) * 0.15,
        "Volatility (-)":   data["sc_volatility"].astype(float) * -0.20,
        "Uncertainty (-)":  data["sc_uncertainty"].astype(float) * -0.10,
    }

    colors = {
        "Opportunity (+)":  "#2ECC71",
        "Liquidity (+)":    "#3498DB",
        "Event Boost (+)":  "#F1C40F",
        "Volatility (-)":   "#E74C3C",
        "Uncertainty (-)":  "#95A5A6",
    }

    # Stack bars
    for name, vals in components.items():
        ax.barh(
            y, vals.values, height=0.6,
            color=colors[name], alpha=0.85,
            label=name,
            left=0,  # all start from zero for tornado effect
        )

    ax.set_yticks(y)
    ax.set_yticklabels([truncate_label(lbl, 30) for lbl in labels])
    ax.axvline(0, color=WOW_PALETTE["text_muted"], linewidth=0.8, linestyle="-")
    ax.set_xlabel("Weighted Score Contribution")
    ax.set_title("Recommendation Score Breakdown (Top Items)", fontweight="bold")
    ax.legend(loc="lower right", framealpha=0.8, fontsize=8)
    fig.tight_layout()
    add_watermark(fig)
    return fig


def plot_action_distribution(df_recs: pd.DataFrame) -> plt.Figure:
    """Donut chart of buy/sell/hold/avoid action split."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    apply_wow_theme(fig)

    if df_recs.empty or "action" not in df_recs.columns:
        for ax in axes:
            ax.text(0.5, 0.5, "No data",
                    ha="center", va="center", color=WOW_PALETTE["text_muted"],
                    transform=ax.transAxes, fontsize=14)
        return fig

    # Action distribution
    ax1 = axes[0]
    action_counts = df_recs["action"].value_counts()
    colors = [ACTION_COLORS.get(a, WOW_PALETTE["text_muted"]) for a in action_counts.index]
    wedges, texts, autotexts = ax1.pie(
        action_counts.values, labels=action_counts.index,
        colors=colors, autopct="%1.0f%%",
        pctdistance=0.75, startangle=90,
        wedgeprops={"width": 0.5, "edgecolor": WOW_PALETTE["background"]},
    )
    for t in texts:
        t.set_color(WOW_PALETTE["text"])
    for t in autotexts:
        t.set_color(WOW_PALETTE["background"])
        t.set_fontweight("bold")
    ax1.set_title("Action Distribution", fontweight="bold",
                  color=WOW_PALETTE["text"])

    # Risk level distribution
    ax2 = axes[1]
    if "risk_level" in df_recs.columns:
        risk_counts = df_recs["risk_level"].value_counts()
        risk_order = ["low", "medium", "high", "critical"]
        risk_counts = risk_counts.reindex([r for r in risk_order if r in risk_counts.index])
        risk_colors = [RISK_COLORS.get(r, WOW_PALETTE["text_muted"]) for r in risk_counts.index]
        wedges2, texts2, autotexts2 = ax2.pie(
            risk_counts.values, labels=risk_counts.index,
            colors=risk_colors, autopct="%1.0f%%",
            pctdistance=0.75, startangle=90,
            wedgeprops={"width": 0.5, "edgecolor": WOW_PALETTE["background"]},
        )
        for t in texts2:
            t.set_color(WOW_PALETTE["text"])
        for t in autotexts2:
            t.set_color(WOW_PALETTE["background"])
            t.set_fontweight("bold")
    ax2.set_title("Risk Level Distribution", fontweight="bold",
                  color=WOW_PALETTE["text"])

    fig.suptitle("Recommendation Analytics",
                 fontsize=14, fontweight="bold", color=WOW_PALETTE["text"])
    fig.tight_layout()
    add_watermark(fig)
    return fig


def plot_roi_vs_uncertainty(df_recs: pd.DataFrame) -> plt.Figure:
    """Scatter of ROI% vs uncertainty%, colored by action.

    Shows the rational decision boundary between buy/hold/avoid.
    """
    fig, ax = plt.subplots(figsize=(10, 7))
    apply_wow_theme(fig)

    if df_recs.empty:
        ax.text(0.5, 0.5, "No data",
                ha="center", va="center", color=WOW_PALETTE["text_muted"],
                transform=ax.transAxes, fontsize=14)
        return fig

    data = df_recs.copy()

    # Compute uncertainty % from CI if not directly available
    if "sc_uncertainty" in data.columns:
        uncertainty = data["sc_uncertainty"].astype(float)
    else:
        uncertainty = pd.Series(np.zeros(len(data)))

    roi = (
        data["roi_pct"].astype(float) * 100
        if "roi_pct" in data.columns
        else pd.Series(np.zeros(len(data)))
    )

    # Color by action
    for action in ["buy", "sell", "hold", "avoid"]:
        if "action" in data.columns:
            mask = data["action"] == action
        else:
            mask = pd.Series([True] * len(data))
        if mask.sum() == 0:
            continue
        ax.scatter(
            uncertainty[mask], roi[mask],
            c=ACTION_COLORS.get(action, WOW_PALETTE["text_muted"]),
            alpha=0.6, s=40, label=action.title(),
            edgecolors="none",
        )

    # Decision boundary lines
    ax.axhline(10, color=ACTION_COLORS["buy"], linestyle="--",
               alpha=0.4, linewidth=1, label="ROI=10% (buy threshold)")
    ax.axhline(-10, color=ACTION_COLORS["sell"], linestyle="--",
               alpha=0.4, linewidth=1, label="ROI=-10% (sell threshold)")
    ax.axvline(95, color=ACTION_COLORS["avoid"], linestyle="--",
               alpha=0.4, linewidth=1, label="Uncertainty=95% (avoid)")

    ax.set_xlabel("Uncertainty Score")
    ax.set_ylabel("ROI (%)")
    ax.set_title("ROI vs Uncertainty — Decision Landscape", fontweight="bold")
    ax.legend(loc="upper left", framealpha=0.8, fontsize=8)
    fig.tight_layout()
    add_watermark(fig)
    return fig
