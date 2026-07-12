"""
Chart utility functions — save, format, watermark.

These are shared across all chart modules and the Streamlit dashboard.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def save_chart(
    fig: Any,
    path: str | Path,
    formats: tuple[str, ...] = ("png",),
    dpi: int = 300,
) -> list[Path]:
    """Save a matplotlib Figure to one or more file formats.

    Args:
        fig:     A matplotlib ``Figure`` instance.
        path:    Base file path (extension will be replaced per format).
        formats: Tuple of format strings, e.g. ``("png", "svg")``.
        dpi:     Resolution for raster formats.

    Returns:
        List of Path objects for each written file.
    """
    base = Path(path)
    base.parent.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for fmt in formats:
        out = base.with_suffix(f".{fmt}")
        fig.savefig(
            str(out),
            format=fmt,
            dpi=dpi,
            bbox_inches="tight",
            pad_inches=0.2,
        )
        written.append(out)
    return written


def format_gold(value: float | int | None) -> str:
    """Format a gold value for display.

    Examples::

        >>> format_gold(1234.56)
        '1,235g'
        >>> format_gold(0.5)
        '1g'
        >>> format_gold(None)
        'N/A'
    """
    if value is None:
        return "N/A"
    return f"{round(value):,}g"


def format_pct(value: float | None, decimals: int = 1) -> str:
    """Format a percentage value with sign.

    Args:
        value:    Fraction (0.1 = 10%) or percentage depending on context.
        decimals: Number of decimal places.

    Examples::

        >>> format_pct(0.123)
        '+12.3%'
        >>> format_pct(-0.05)
        '-5.0%'
        >>> format_pct(0.0)
        '0.0%'
        >>> format_pct(None)
        'N/A'
    """
    if value is None:
        return "N/A"
    pct = value * 100
    sign = "+" if pct > 0 else ""
    return f"{sign}{pct:.{decimals}f}%"


def add_watermark(fig: Any, text: str = "WoW Economy Forecaster") -> None:
    """Add a subtle watermark to the bottom-right of a matplotlib Figure.

    Args:
        fig:  A matplotlib ``Figure`` instance.
        text: Watermark text.
    """
    from wow_forecaster.viz.theme import WOW_PALETTE

    fig.text(
        0.98, 0.02, text,
        fontsize=7,
        color=WOW_PALETTE["text_muted"],
        alpha=0.4,
        ha="right",
        va="bottom",
        style="italic",
    )


def truncate_label(label: str, max_len: int = 25) -> str:
    """Truncate a label for axis display.

    Args:
        label:   The full label text.
        max_len: Maximum characters before truncation.

    Returns:
        Truncated label with ellipsis if needed.
    """
    if len(label) <= max_len:
        return label
    return label[: max_len - 3] + "..."
