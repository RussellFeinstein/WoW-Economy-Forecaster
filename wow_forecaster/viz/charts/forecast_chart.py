"""
Forecast vs actual price charts with confidence interval bands.

The hero chart of the portfolio — shows the full data science pipeline:
historical data goes in, model makes predictions, CI conveys uncertainty.
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from wow_forecaster.viz.theme import (
    CI_QUALITY_COLORS,
    HORIZON_COLORS,
    WOW_PALETTE,
    apply_wow_theme,
)
from wow_forecaster.viz.utils import add_watermark, format_gold


def plot_forecast_timeline(
    df_hist: pd.DataFrame,
    df_forecast: pd.DataFrame,
    archetype_name: str = "",
    horizon: str | None = None,
    events: pd.DataFrame | None = None,
) -> plt.Figure:
    """Plot historical prices with forecast point and CI band.

    Args:
        df_hist:     Historical prices with columns: obs_date, avg_price_gold.
        df_forecast: Forecasts with columns: target_date, predicted_price_gold,
                     confidence_lower, confidence_upper, ci_quality,
                     forecast_horizon.
        archetype_name: Display name for the title.
        horizon:     If set, filter forecasts to this horizon only.
        events:      Optional events DataFrame for vertical annotation lines.

    Returns:
        matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    apply_wow_theme(fig)

    # Historical line
    if not df_hist.empty:
        dates = pd.to_datetime(df_hist["obs_date"])
        prices = df_hist["avg_price_gold"].astype(float)
        ax.plot(dates, prices, color=WOW_PALETTE["primary"], linewidth=1.8,
                label="Historical Price", zorder=3)

        # Price range band if min/max available
        if "min_price_gold" in df_hist.columns and "max_price_gold" in df_hist.columns:
            ax.fill_between(
                dates,
                df_hist["min_price_gold"].astype(float),
                df_hist["max_price_gold"].astype(float),
                alpha=0.1, color=WOW_PALETTE["primary"],
                label="Daily Range",
            )

    # Forecasts
    fc = df_forecast.copy()
    if horizon and not fc.empty:
        fc = fc[fc["forecast_horizon"] == horizon]

    if not fc.empty:
        fc["target_date"] = pd.to_datetime(fc["target_date"])
        for _, row in fc.iterrows():
            h = row.get("forecast_horizon", "1d")
            color = HORIZON_COLORS.get(h, WOW_PALETTE["accent_blue"])
            ci_qual = row.get("ci_quality", "good")
            ci_color = CI_QUALITY_COLORS.get(ci_qual, CI_QUALITY_COLORS["good"])

            # CI band as a vertical span
            ax.plot(row["target_date"], row["predicted_price_gold"],
                    marker="D", markersize=10, color=color, zorder=5,
                    label=f"Forecast ({h})")
            ax.vlines(row["target_date"],
                      row["confidence_lower"], row["confidence_upper"],
                      color=ci_color, linewidth=4, alpha=0.5, zorder=4,
                      label=f"CI [{ci_qual}]")

    # Event annotations
    if events is not None and not events.empty:
        for _, ev in events.iterrows():
            ev_date = pd.to_datetime(ev["start_date"])
            ax.axvline(ev_date, color=WOW_PALETTE["accent_red"],
                       linestyle="--", alpha=0.6, linewidth=0.8)
            ax.text(ev_date, ax.get_ylim()[1] * 0.98,
                    f" {ev['display_name']}", fontsize=7,
                    color=WOW_PALETTE["accent_red"], rotation=90,
                    va="top", ha="left", alpha=0.7)

    title = "Price Forecast"
    if archetype_name:
        title = f"{archetype_name} — {title}"
    if horizon:
        title += f" ({horizon})"
    ax.set_title(title, fontsize=14, fontweight="bold",
                 color=WOW_PALETTE["text"])
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (gold)")

    # De-duplicate legend entries
    handles, labels = ax.get_legend_handles_labels()
    seen: dict[str, Any] = {}
    unique_h, unique_l = [], []
    for handle, label in zip(handles, labels, strict=False):
        if label not in seen:
            seen[label] = True
            unique_h.append(handle)
            unique_l.append(label)
    if unique_h:
        ax.legend(unique_h, unique_l, loc="upper left", framealpha=0.8)

    fig.tight_layout()
    add_watermark(fig)
    return fig


def plot_forecast_multi_horizon(
    df_hist: pd.DataFrame,
    df_forecast: pd.DataFrame,
    archetype_name: str = "",
) -> plt.Figure:
    """Plot all horizons overlaid on one chart.

    Same as ``plot_forecast_timeline`` but shows 1d, 7d, 28d together.
    """
    return plot_forecast_timeline(
        df_hist, df_forecast,
        archetype_name=archetype_name,
        horizon=None,
    )


def plot_forecast_timeline_interactive(
    df_hist: pd.DataFrame,
    df_forecast: pd.DataFrame,
    archetype_name: str = "",
    horizon: str | None = None,
) -> Any:
    """Plotly interactive version for Streamlit.

    Returns a plotly.graph_objects.Figure.
    """
    import plotly.graph_objects as go

    from wow_forecaster.viz.theme import get_plotly_template

    layout = get_plotly_template()
    fig = go.Figure(layout=layout)

    # Historical line
    if not df_hist.empty:
        fig.add_trace(go.Scatter(
            x=pd.to_datetime(df_hist["obs_date"]),
            y=df_hist["avg_price_gold"].astype(float),
            mode="lines",
            name="Historical Price",
            line={"color": WOW_PALETTE["primary"], "width": 2},
        ))

    # Forecasts
    fc = df_forecast.copy()
    if horizon and not fc.empty:
        fc = fc[fc["forecast_horizon"] == horizon]

    if not fc.empty:
        fc["target_date"] = pd.to_datetime(fc["target_date"])
        for _, row in fc.iterrows():
            h = row.get("forecast_horizon", "1d")
            color = HORIZON_COLORS.get(h, WOW_PALETTE["accent_blue"])
            ci_qual = row.get("ci_quality", "good")

            # Forecast marker
            fig.add_trace(go.Scatter(
                x=[row["target_date"]],
                y=[row["predicted_price_gold"]],
                mode="markers",
                name=f"Forecast ({h})",
                marker={"color": color, "size": 12, "symbol": "diamond"},
                hovertemplate=(
                    f"<b>Forecast ({h})</b><br>"
                    f"Price: {format_gold(row['predicted_price_gold'])}<br>"
                    f"CI: {format_gold(row['confidence_lower'])} - "
                    f"{format_gold(row['confidence_upper'])}<br>"
                    f"Quality: {ci_qual}"
                    "<extra></extra>"
                ),
            ))

            # CI error bar
            fig.add_trace(go.Scatter(
                x=[row["target_date"], row["target_date"]],
                y=[row["confidence_lower"], row["confidence_upper"]],
                mode="lines",
                name=f"CI ({ci_qual})",
                line={"color": CI_QUALITY_COLORS.get(ci_qual, "#2ECC71"), "width": 4},
                opacity=0.5,
                showlegend=False,
            ))

    title = "Price Forecast"
    if archetype_name:
        title = f"{archetype_name} — {title}"
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Price (gold)",
        hovermode="x unified",
    )
    return fig
