"""
Transfer learning and cold-start visualizations.

Demonstrates the unique archetype-based transfer approach and how
the system handles uncertainty for never-seen-before items.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from wow_forecaster.viz.theme import (
    WOW_PALETTE,
    apply_wow_theme,
)
from wow_forecaster.viz.utils import add_watermark


def plot_ci_width_by_category(
    df: pd.DataFrame,
    split_cold_start: bool = True,
) -> plt.Figure:
    """Box plot of CI width distribution per category, optionally split
    by cold-start vs warm archetypes.

    Args:
        df: Forecast DataFrame with columns: confidence_lower,
            confidence_upper, and optionally category_tag, model_slug.
        split_cold_start: If True and model_slug contains '_transfer',
            splits into cold-start vs warm groups.
    """
    import seaborn as sns

    fig, ax = plt.subplots(figsize=(12, 6))
    apply_wow_theme(fig)

    if df.empty:
        ax.text(0.5, 0.5, "No forecast data",
                ha="center", va="center", color=WOW_PALETTE["text_muted"],
                transform=ax.transAxes, fontsize=14)
        return fig

    data = df.copy()
    ci_upper = data["confidence_upper"].astype(float)
    ci_lower = data["confidence_lower"].astype(float)
    data["ci_width"] = ci_upper - ci_lower

    if split_cold_start and "model_slug" in data.columns:
        data["type"] = data["model_slug"].apply(
            lambda s: "Cold Start" if "_transfer" in str(s) else "Warm"
        )
        hue = "type"
        palette = {"Cold Start": WOW_PALETTE["accent_blue"], "Warm": WOW_PALETTE["primary"]}
    else:
        hue = None
        palette = None

    group_col = "category_tag" if "category_tag" in data.columns else None

    if group_col:
        sns.boxplot(
            data=data, x=group_col, y="ci_width",
            hue=hue, palette=palette,
            ax=ax, fliersize=2,
            boxprops={"alpha": 0.7},
            medianprops={"color": WOW_PALETTE["accent_red"]},
        )
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    else:
        if hue:
            sns.boxplot(
                data=data, x="type", y="ci_width",
                hue="type", palette=palette, legend=False,
                ax=ax, fliersize=2,
                boxprops={"alpha": 0.7},
                medianprops={"color": WOW_PALETTE["accent_red"]},
            )
        else:
            sns.boxplot(
                data=data, y="ci_width", ax=ax, fliersize=2,
                color=WOW_PALETTE["primary"],
                boxprops={"alpha": 0.7},
                medianprops={"color": WOW_PALETTE["accent_red"]},
            )

    ax.set_ylabel("Confidence Interval Width (gold)")
    ax.set_title("Uncertainty by Category — Cold Start vs Warm", fontweight="bold")
    fig.tight_layout()
    add_watermark(fig)
    return fig


def plot_cold_start_blend_diagram() -> plt.Figure:
    """Conceptual diagram illustrating the cold-start prediction blending.

    Shows: model prediction, source price anchor, blended result,
    and how transfer_confidence acts as a mixing weight.

    Formula: blended = confidence * model_pred + (1 - confidence) * source_price
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    apply_wow_theme(fig)

    # Simulate the blending across confidence levels
    confidence = np.linspace(0, 1, 100)
    model_pred = 600  # model's raw prediction
    source_price = 450  # historical source-expansion price anchor

    blended = confidence * model_pred + (1 - confidence) * source_price

    ax.fill_between(confidence, source_price, model_pred,
                    alpha=0.08, color=WOW_PALETTE["primary"])

    ax.plot(confidence, blended, color=WOW_PALETTE["primary"],
            linewidth=2.5, label="Blended Prediction", zorder=3)
    ax.axhline(model_pred, color=WOW_PALETTE["accent_blue"],
               linestyle="--", alpha=0.7, linewidth=1.5,
               label=f"Model Prediction ({model_pred}g)")
    ax.axhline(source_price, color=WOW_PALETTE["accent_red"],
               linestyle="--", alpha=0.7, linewidth=1.5,
               label=f"Source Price Anchor ({source_price}g)")

    # Annotate key points
    for conf, label, marker_y in [
        (0.0, "No confidence\n(pure anchor)", source_price),
        (0.5, "50% confidence\n(equal blend)", (model_pred + source_price) / 2),
        (1.0, "Full confidence\n(pure model)", model_pred),
    ]:
        ax.annotate(
            label,
            xy=(conf, marker_y),
            xytext=(conf, marker_y + 30 if conf < 0.8 else marker_y - 30),
            fontsize=8, color=WOW_PALETTE["text"],
            ha="center",
            arrowprops={"arrowstyle": "->", "color": WOW_PALETTE["text_muted"],
                        "alpha": 0.6},
        )
        ax.plot(conf, marker_y, "o", color=WOW_PALETTE["primary"],
                markersize=8, zorder=4)

    ax.set_xlabel("Transfer Confidence Score")
    ax.set_ylabel("Price (gold)")
    ax.set_title("Cold-Start Prediction Blending", fontweight="bold")
    ax.legend(loc="center left", framealpha=0.8)

    # Formula annotation
    ax.text(
        0.5, 0.05,
        r"$\hat{p} = c \cdot p_{model} + (1 - c) \cdot p_{source}$",
        transform=ax.transAxes, fontsize=11,
        color=WOW_PALETTE["text"], ha="center",
        bbox={"boxstyle": "round,pad=0.4",
              "facecolor": WOW_PALETTE["surface"],
              "edgecolor": WOW_PALETTE["grid"]},
    )

    fig.tight_layout()
    add_watermark(fig)
    return fig
