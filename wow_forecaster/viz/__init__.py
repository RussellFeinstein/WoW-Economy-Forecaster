"""
Visualization layer for the WoW Economy Forecaster.

Provides publication-quality static charts (matplotlib/seaborn) and
interactive charts (Plotly) for portfolio presentation, Streamlit
dashboard, and Jupyter notebooks.

Optional dependency group — install via ``pip install -e ".[viz]"``.
"""

from __future__ import annotations

from wow_forecaster.viz.theme import (
    ACTION_COLORS,
    CATEGORY_COLORS,
    HORIZON_COLORS,
    WOW_PALETTE,
    apply_wow_theme,
    get_plotly_template,
)
from wow_forecaster.viz.utils import (
    add_watermark,
    format_gold,
    format_pct,
    save_chart,
)

__all__ = [
    "ACTION_COLORS",
    "CATEGORY_COLORS",
    "HORIZON_COLORS",
    "WOW_PALETTE",
    "add_watermark",
    "apply_wow_theme",
    "format_gold",
    "format_pct",
    "get_plotly_template",
    "save_chart",
]
