"""
Backtest performance visualizations.

The classic ML validation charts: actual vs predicted scatter, residual
distribution, directional accuracy heatmap, and walk-forward stability.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from wow_forecaster.viz.theme import (
    CATEGORY_COLORS,
    WOW_PALETTE,
    apply_wow_theme,
)
from wow_forecaster.viz.utils import add_watermark


def plot_actual_vs_predicted_scatter(
    df: pd.DataFrame,
    max_points: int = 2000,
) -> plt.Figure:
    """Scatter plot of actual vs predicted prices with y=x reference.

    Args:
        df: DataFrame with columns: actual_price, predicted_price,
            and optionally category_tag for coloring.
        max_points: Downsample if more than this many points.

    Returns:
        matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    apply_wow_theme(fig)

    if df.empty:
        ax.text(0.5, 0.5, "No backtest data available",
                ha="center", va="center", color=WOW_PALETTE["text_muted"],
                transform=ax.transAxes, fontsize=14)
        return fig

    data = df.copy()
    if len(data) > max_points:
        data = data.sample(max_points, random_state=42)

    actual = data["actual_price"].astype(float)
    predicted = data["predicted_price"].astype(float)

    # Color by category if available
    if "category_tag" in data.columns:
        categories = data["category_tag"].unique()
        for cat in categories:
            mask = data["category_tag"] == cat
            color = CATEGORY_COLORS.get(cat, WOW_PALETTE["text_muted"])
            ax.scatter(
                actual[mask], predicted[mask],
                c=color, alpha=0.5, s=20, label=cat, edgecolors="none",
            )
        ax.legend(loc="lower right", framealpha=0.8)
    else:
        ax.scatter(actual, predicted, c=WOW_PALETTE["primary"],
                   alpha=0.5, s=20, edgecolors="none")

    # Perfect prediction line
    lims = [
        min(actual.min(), predicted.min()),
        max(actual.max(), predicted.max()),
    ]
    ax.plot(lims, lims, "--", color=WOW_PALETTE["accent_red"],
            alpha=0.7, linewidth=1.5, label="Perfect Prediction")

    # R-squared annotation
    ss_res = np.sum((actual.values - predicted.values) ** 2)
    ss_tot = np.sum((actual.values - actual.mean()) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    ax.text(
        0.05, 0.92, f"R² = {r_squared:.3f}",
        transform=ax.transAxes, fontsize=12,
        color=WOW_PALETTE["text"],
        bbox={"boxstyle": "round,pad=0.3",
              "facecolor": WOW_PALETTE["surface"],
              "edgecolor": WOW_PALETTE["grid"]},
    )

    ax.set_xlabel("Actual Price (gold)")
    ax.set_ylabel("Predicted Price (gold)")
    ax.set_title("Backtest: Actual vs Predicted", fontweight="bold")
    fig.tight_layout()
    add_watermark(fig)
    return fig


def plot_residual_distribution(df: pd.DataFrame) -> plt.Figure:
    """Histogram/KDE of prediction residuals (actual - predicted).

    Shows bias (center offset from 0) and spread.
    """
    import seaborn as sns

    fig, ax = plt.subplots(figsize=(10, 5))
    apply_wow_theme(fig)

    if df.empty:
        ax.text(0.5, 0.5, "No backtest data available",
                ha="center", va="center", color=WOW_PALETTE["text_muted"],
                transform=ax.transAxes, fontsize=14)
        return fig

    residuals = df["actual_price"].astype(float) - df["predicted_price"].astype(float)

    sns.histplot(residuals, kde=True, ax=ax,
                 color=WOW_PALETTE["primary"], alpha=0.6,
                 edgecolor=WOW_PALETTE["grid"], linewidth=0.5)
    ax.axvline(0, color=WOW_PALETTE["accent_red"], linestyle="--",
               linewidth=1.5, alpha=0.7, label="Zero (unbiased)")
    ax.axvline(residuals.mean(), color=WOW_PALETTE["accent_blue"],
               linestyle="-", linewidth=1.5, alpha=0.7,
               label=f"Mean = {residuals.mean():.1f}g")

    ax.set_xlabel("Residual (Actual - Predicted) in gold")
    ax.set_ylabel("Count")
    ax.set_title("Prediction Residual Distribution", fontweight="bold")
    ax.legend(loc="upper right", framealpha=0.8)
    fig.tight_layout()
    add_watermark(fig)
    return fig


def plot_metrics_by_category(df_summary: pd.DataFrame) -> plt.Figure:
    """Grouped bar chart of MAE/MAPE per category and horizon.

    Args:
        df_summary: DataFrame with columns: category_tag (or model_name),
            mae, mape, and optionally horizon.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    apply_wow_theme(fig)

    if df_summary.empty:
        for ax in axes:
            ax.text(0.5, 0.5, "No summary data",
                    ha="center", va="center", color=WOW_PALETTE["text_muted"],
                    transform=ax.transAxes, fontsize=14)
        return fig

    # Determine grouping column
    group_col = "category_tag" if "category_tag" in df_summary.columns else "model_name"
    groups = df_summary[group_col].unique()

    x = np.arange(len(groups))
    width = 0.35

    # MAE chart
    ax1 = axes[0]
    if "mae" in df_summary.columns:
        mae_vals = [
            df_summary[df_summary[group_col] == g]["mae"].mean()
            for g in groups
        ]
        colors = [CATEGORY_COLORS.get(g, WOW_PALETTE["primary"]) for g in groups]
        ax1.bar(x, mae_vals, width, color=colors, alpha=0.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(groups, rotation=45, ha="right")
    ax1.set_ylabel("MAE (gold)")
    ax1.set_title("Mean Absolute Error by Category", fontweight="bold")

    # MAPE chart
    ax2 = axes[1]
    if "mape" in df_summary.columns:
        mape_vals = [
            df_summary[df_summary[group_col] == g]["mape"].mean() * 100
            for g in groups
        ]
        colors = [CATEGORY_COLORS.get(g, WOW_PALETTE["primary"]) for g in groups]
        ax2.bar(x, mape_vals, width, color=colors, alpha=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(groups, rotation=45, ha="right")
    ax2.set_ylabel("MAPE (%)")
    ax2.set_title("Mean Absolute % Error by Category", fontweight="bold")

    fig.tight_layout()
    add_watermark(fig)
    return fig


def plot_directional_accuracy_heatmap(df: pd.DataFrame) -> plt.Figure:
    """Heatmap of directional accuracy: category x horizon.

    Cell value = fraction of correct up/down predictions.
    Color scale: 50% (random) = red, 70%+ = green.
    """
    import seaborn as sns

    fig, ax = plt.subplots(figsize=(8, 6))
    apply_wow_theme(fig)

    if df.empty or "direction_correct" not in df.columns:
        ax.text(0.5, 0.5, "No directional accuracy data",
                ha="center", va="center", color=WOW_PALETTE["text_muted"],
                transform=ax.transAxes, fontsize=14)
        return fig

    # Determine grouping columns
    cat_col = "category_tag" if "category_tag" in df.columns else "model_name"
    horizon_col = "horizon_days" if "horizon_days" in df.columns else "horizon"

    if horizon_col not in df.columns:
        ax.text(0.5, 0.5, "No horizon column found",
                ha="center", va="center", color=WOW_PALETTE["text_muted"],
                transform=ax.transAxes, fontsize=14)
        return fig

    pivot = df.groupby([cat_col, horizon_col])["direction_correct"].mean().unstack()
    pivot = pivot * 100  # to percentage

    cmap = sns.diverging_palette(10, 130, as_cmap=True)  # red → green
    sns.heatmap(
        pivot, annot=True, fmt=".1f", cmap=cmap,
        center=50, vmin=30, vmax=80,
        ax=ax, cbar_kws={"label": "Accuracy (%)"},
        linewidths=0.5, linecolor=WOW_PALETTE["grid"],
    )
    ax.set_title("Directional Accuracy (%)", fontweight="bold")
    ax.set_ylabel("Category")
    ax.set_xlabel("Horizon")
    fig.tight_layout()
    add_watermark(fig)
    return fig


def plot_walk_forward_mae(df: pd.DataFrame) -> plt.Figure:
    """Line chart of MAE per fold over time, showing model stability."""
    fig, ax = plt.subplots(figsize=(10, 5))
    apply_wow_theme(fig)

    if df.empty or "fold_index" not in df.columns:
        ax.text(0.5, 0.5, "No walk-forward data",
                ha="center", va="center", color=WOW_PALETTE["text_muted"],
                transform=ax.transAxes, fontsize=14)
        return fig

    def _fold_mae(g):
        return (g["actual_price"].astype(float) - g["predicted_price"].astype(float)).abs().mean()

    fold_mae = df.groupby("fold_index").apply(_fold_mae).reset_index()
    fold_mae.columns = ["fold_index", "mae"]

    ax.plot(fold_mae["fold_index"], fold_mae["mae"],
            marker="o", color=WOW_PALETTE["primary"], linewidth=2,
            markersize=6, markerfacecolor=WOW_PALETTE["accent_blue"])
    ax.fill_between(fold_mae["fold_index"], fold_mae["mae"],
                    alpha=0.15, color=WOW_PALETTE["primary"])

    ax.set_xlabel("Fold Index (Time ->)")
    ax.set_ylabel("MAE (gold)")
    ax.set_title("Walk-Forward MAE Over Time", fontweight="bold")
    fig.tight_layout()
    add_watermark(fig)
    return fig
