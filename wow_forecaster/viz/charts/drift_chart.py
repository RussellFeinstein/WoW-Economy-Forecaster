"""
Drift monitoring and model health visualizations.

Shows the monitoring pipeline is real and actively tracking model
performance degradation.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from wow_forecaster.viz.theme import (
    DRIFT_COLORS,
    HORIZON_COLORS,
    WOW_PALETTE,
    apply_wow_theme,
)
from wow_forecaster.viz.utils import add_watermark

# Drift level to numeric for plotting
_DRIFT_LEVEL_ORDER = {"none": 0, "low": 1, "medium": 2, "high": 3, "critical": 4}


def plot_drift_timeline(df_drift: pd.DataFrame) -> plt.Figure:
    """Time series of drift level with color-coded bands.

    Uncertainty multiplier on secondary Y-axis.
    """
    fig, ax1 = plt.subplots(figsize=(12, 5))
    apply_wow_theme(fig)

    if df_drift.empty:
        ax1.text(0.5, 0.5, "No drift history available",
                 ha="center", va="center", color=WOW_PALETTE["text_muted"],
                 transform=ax1.transAxes, fontsize=14)
        return fig

    data = df_drift.copy()
    data["checked_at"] = pd.to_datetime(data["checked_at"])
    data = data.sort_values("checked_at")

    # Drift level as numeric
    data["drift_numeric"] = data["overall_drift_level"].map(_DRIFT_LEVEL_ORDER).fillna(0)

    # Color background bands by drift level
    for i in range(len(data) - 1):
        level = data.iloc[i]["overall_drift_level"]
        color = DRIFT_COLORS.get(level, DRIFT_COLORS["none"])
        ax1.axvspan(
            data.iloc[i]["checked_at"], data.iloc[i + 1]["checked_at"],
            alpha=0.15, color=color,
        )

    # Drift level line
    ax1.step(data["checked_at"], data["drift_numeric"],
             where="post", color=WOW_PALETTE["primary"],
             linewidth=2, label="Drift Level")
    ax1.set_ylabel("Drift Level")
    ax1.set_yticks(list(_DRIFT_LEVEL_ORDER.values()))
    ax1.set_yticklabels(list(_DRIFT_LEVEL_ORDER.keys()))
    ax1.set_ylim(-0.3, 4.3)

    # Uncertainty multiplier on secondary axis
    if "uncertainty_multiplier" in data.columns:
        ax2 = ax1.twinx()
        ax2.plot(data["checked_at"], data["uncertainty_multiplier"].astype(float),
                 color=WOW_PALETTE["accent_blue"], linewidth=1.5,
                 linestyle="--", alpha=0.7, label="Uncertainty Mult.")
        ax2.set_ylabel("Uncertainty Multiplier",
                       color=WOW_PALETTE["accent_blue"])
        ax2.tick_params(axis="y", labelcolor=WOW_PALETTE["accent_blue"])
        ax2.spines["right"].set_color(WOW_PALETTE["accent_blue"])

    ax1.set_xlabel("Date")
    ax1.set_title("Model Drift Over Time", fontweight="bold")
    ax1.legend(loc="upper left", framealpha=0.8)
    fig.tight_layout()
    add_watermark(fig)
    return fig


def plot_mae_ratio_gauge(
    health_data: dict,
    horizons: list[str] | None = None,
) -> plt.Figure:
    """Gauge-style chart per horizon showing MAE ratio vs thresholds.

    Args:
        health_data: Dict with per-horizon health info. Expected keys per
            horizon: mae_ratio, baseline_mae, live_mae.
        horizons: List of horizon strings to display (default: ["1d", "7d", "28d"]).
    """
    if horizons is None:
        horizons = ["1d", "7d", "28d"]

    fig, axes = plt.subplots(1, len(horizons), figsize=(4 * len(horizons), 4))
    apply_wow_theme(fig)

    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    for i, h in enumerate(horizons):
        ax = axes[i]
        h_data = health_data.get(h, {})
        ratio = h_data.get("mae_ratio", 0)

        # Determine color zone
        if ratio < 1.2:
            color = DRIFT_COLORS["none"]
            label = "Healthy"
        elif ratio < 1.5:
            color = DRIFT_COLORS["low"]
            label = "Minor"
        elif ratio < 2.0:
            color = DRIFT_COLORS["medium"]
            label = "Degraded"
        elif ratio < 3.0:
            color = DRIFT_COLORS["high"]
            label = "Poor"
        else:
            color = DRIFT_COLORS["critical"]
            label = "Critical"

        # Draw gauge as a horizontal bar
        ax.barh(0, ratio, height=0.5, color=color, alpha=0.8)
        ax.axvline(1.0, color=WOW_PALETTE["text_muted"], linestyle="--",
                   linewidth=1, alpha=0.5)
        ax.axvline(1.5, color=DRIFT_COLORS["medium"], linestyle=":",
                   linewidth=1, alpha=0.4)
        ax.axvline(3.0, color=DRIFT_COLORS["critical"], linestyle=":",
                   linewidth=1, alpha=0.4)

        ax.set_xlim(0, max(ratio * 1.2, 4))
        ax.set_yticks([])
        ax.set_title(f"{h}", fontweight="bold",
                     color=HORIZON_COLORS.get(h, WOW_PALETTE["text"]))
        ax.text(ratio, 0, f" {ratio:.2f}x ({label})",
                va="center", fontsize=10, color=WOW_PALETTE["text"])

    fig.suptitle("MAE Ratio vs Baseline (lower is better)",
                 fontsize=14, fontweight="bold", color=WOW_PALETTE["text"])
    fig.tight_layout()
    add_watermark(fig)
    return fig
