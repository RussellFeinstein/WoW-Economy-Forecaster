"""
Feature importance and correlation visualizations.

Shows what the model learned — demonstrates feature engineering rigor.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from wow_forecaster.viz.theme import (
    FEATURE_GROUP_COLORS,
    HORIZON_COLORS,
    WOW_PALETTE,
    apply_wow_theme,
)
from wow_forecaster.viz.utils import add_watermark, truncate_label


def _feature_group(feature_name: str) -> str:
    """Infer feature group from name for color coding."""
    prefixes = {
        "price_": "price", "market_value": "price", "historical_value": "price",
        "obs_count": "price",
        "quantity_": "volume", "auctions_": "volume", "is_volume": "volume",
        "price_lag_": "lag",
        "price_roll_": "rolling",
        "price_pct_change_": "momentum",
        "day_of_": "temporal", "week_of_": "temporal", "days_since_": "temporal",
        "event_": "event", "days_until_": "event", "is_pre_event": "event",
        "archetype_": "archetype", "is_transferable": "archetype",
        "is_cold_start": "archetype", "item_count": "archetype",
        "has_transfer": "transfer", "transfer_": "transfer",
    }
    for prefix, group in prefixes.items():
        if feature_name.startswith(prefix):
            return group
    return "price"  # default


def plot_feature_importance(
    df: pd.DataFrame,
    top_n: int = 15,
    importance_type: str = "gain",
) -> plt.Figure:
    """Horizontal bar chart of top-N features colored by group.

    Args:
        df: DataFrame with columns: feature, gain, gain_pct, split, split_pct.
        top_n: Number of features to display.
        importance_type: "gain" or "split".

    Returns:
        matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=(10, 7))
    apply_wow_theme(fig)

    if df.empty:
        ax.text(0.5, 0.5, "No feature importance data",
                ha="center", va="center", color=WOW_PALETTE["text_muted"],
                transform=ax.transAxes, fontsize=14)
        return fig

    col = f"{importance_type}_pct"
    if col not in df.columns:
        col = importance_type

    # Aggregate across horizons if multiple
    agg = df.groupby("feature")[col].mean().sort_values(ascending=True)
    top = agg.tail(top_n)

    colors = [
        FEATURE_GROUP_COLORS.get(_feature_group(f), WOW_PALETTE["primary"])
        for f in top.index
    ]

    ax.barh(
        [truncate_label(f, 30) for f in top.index],
        top.values,
        color=colors,
        alpha=0.85,
        edgecolor=WOW_PALETTE["grid"],
        linewidth=0.5,
    )

    ax.set_xlabel(f"Importance ({importance_type} %)")
    ax.set_title(f"Top {top_n} Features by {importance_type.title()} Importance",
                 fontweight="bold")

    # Legend for feature groups
    seen_groups: set[str] = set()
    legend_patches = []
    import matplotlib.patches as mpatches
    for f in top.index:
        group = _feature_group(f)
        if group not in seen_groups:
            seen_groups.add(group)
            legend_patches.append(
                mpatches.Patch(
                    color=FEATURE_GROUP_COLORS.get(group, WOW_PALETTE["primary"]),
                    label=group,
                )
            )
    if legend_patches:
        ax.legend(handles=legend_patches, loc="lower right", framealpha=0.8)

    fig.tight_layout()
    add_watermark(fig)
    return fig


def plot_importance_by_horizon(
    df: pd.DataFrame,
    top_n: int = 10,
) -> plt.Figure:
    """Faceted comparison of feature importance across horizons.

    Shows how the model's reliance on features shifts from 1d to 28d.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 7), sharey=True)
    apply_wow_theme(fig)

    if df.empty:
        for ax in axes:
            ax.text(0.5, 0.5, "No data",
                    ha="center", va="center", color=WOW_PALETTE["text_muted"],
                    transform=ax.transAxes)
        return fig

    horizons = ["1d", "7d", "28d"]
    for i, h in enumerate(horizons):
        ax = axes[i]
        h_data = df[df["horizon"] == h] if "horizon" in df.columns else df

        if h_data.empty:
            ax.text(0.5, 0.5, f"No {h} data",
                    ha="center", va="center", color=WOW_PALETTE["text_muted"],
                    transform=ax.transAxes)
            continue

        if "gain_pct" in h_data.columns:
            top = h_data.nlargest(top_n, "gain_pct")
        else:
            top = h_data.head(top_n)
        top = top.sort_values("gain_pct", ascending=True)

        colors = [
            FEATURE_GROUP_COLORS.get(_feature_group(f), WOW_PALETTE["primary"])
            for f in top["feature"]
        ]

        ax.barh(
            [truncate_label(f, 25) for f in top["feature"]],
            top["gain_pct"],
            color=colors, alpha=0.85,
        )
        ax.set_title(f"{h} Horizon", fontweight="bold",
                     color=HORIZON_COLORS.get(h, WOW_PALETTE["text"]))
        if i == 0:
            ax.set_ylabel("Feature")

    fig.suptitle("Feature Importance by Forecast Horizon",
                 fontsize=16, fontweight="bold", color=WOW_PALETTE["text"])
    fig.tight_layout()
    add_watermark(fig)
    return fig


def plot_feature_correlation(
    df_features: pd.DataFrame,
    top_n: int = 20,
) -> plt.Figure:
    """Seaborn heatmap of pairwise feature correlations.

    Args:
        df_features: Feature DataFrame (rows = observations, cols = features).
        top_n:       Number of features to include (by variance).
    """
    import seaborn as sns

    fig, ax = plt.subplots(figsize=(12, 10))
    apply_wow_theme(fig)

    if df_features.empty:
        ax.text(0.5, 0.5, "No feature data",
                ha="center", va="center", color=WOW_PALETTE["text_muted"],
                transform=ax.transAxes, fontsize=14)
        return fig

    # Select numeric columns and top-N by variance
    numeric = df_features.select_dtypes(include=[np.number])
    if len(numeric.columns) > top_n:
        variances = numeric.var().sort_values(ascending=False)
        numeric = numeric[variances.head(top_n).index]

    corr = numeric.corr()

    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f",
        cmap="RdBu_r", center=0, vmin=-1, vmax=1,
        ax=ax, square=True,
        linewidths=0.5, linecolor=WOW_PALETTE["grid"],
        cbar_kws={"label": "Pearson Correlation"},
        annot_kws={"fontsize": 7},
    )
    ax.set_title(f"Feature Correlation Matrix (Top {top_n})", fontweight="bold")
    fig.tight_layout()
    add_watermark(fig)
    return fig
