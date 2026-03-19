"""Tests for wow_forecaster.viz.utils — save, format, watermark."""

from __future__ import annotations

from pathlib import Path

from wow_forecaster.viz.utils import (
    add_watermark,
    format_gold,
    format_pct,
    save_chart,
    truncate_label,
)

# ── format_gold ───────────────────────────────────────────────────────────────


class TestFormatGold:
    def test_positive_integer(self):
        assert format_gold(1234) == "1,234g"

    def test_positive_float_rounds(self):
        assert format_gold(1234.56) == "1,235g"

    def test_zero(self):
        assert format_gold(0) == "0g"

    def test_small_fraction_rounds_up(self):
        assert format_gold(0.5) == "0g"

    def test_large_value(self):
        assert format_gold(1_000_000) == "1,000,000g"

    def test_none(self):
        assert format_gold(None) == "N/A"

    def test_negative(self):
        assert format_gold(-100) == "-100g"


# ── format_pct ────────────────────────────────────────────────────────────────


class TestFormatPct:
    def test_positive(self):
        assert format_pct(0.123) == "+12.3%"

    def test_negative(self):
        assert format_pct(-0.05) == "-5.0%"

    def test_zero(self):
        assert format_pct(0.0) == "0.0%"

    def test_none(self):
        assert format_pct(None) == "N/A"

    def test_custom_decimals(self):
        assert format_pct(0.12345, decimals=2) == "+12.35%"

    def test_large_positive(self):
        assert format_pct(1.0) == "+100.0%"


# ── truncate_label ────────────────────────────────────────────────────────────


class TestTruncateLabel:
    def test_short_label_unchanged(self):
        assert truncate_label("hello") == "hello"

    def test_exact_max_len(self):
        label = "a" * 25
        assert truncate_label(label) == label

    def test_over_max_len(self):
        label = "a" * 30
        result = truncate_label(label, max_len=25)
        assert len(result) == 25
        assert result.endswith("...")

    def test_empty_string(self):
        assert truncate_label("") == ""

    def test_custom_max_len(self):
        assert truncate_label("abcdefgh", max_len=5) == "ab..."


# ── save_chart ────────────────────────────────────────────────────────────────


class TestSaveChart:
    def test_saves_png(self, tmp_path: Path):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 2, 3])
        result = save_chart(fig, tmp_path / "test_chart", formats=("png",))
        plt.close(fig)

        assert len(result) == 1
        assert result[0].suffix == ".png"
        assert result[0].exists()
        assert result[0].stat().st_size > 0

    def test_saves_multiple_formats(self, tmp_path: Path):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 2, 3])
        result = save_chart(fig, tmp_path / "test_chart", formats=("png", "svg"))
        plt.close(fig)

        assert len(result) == 2
        assert any(p.suffix == ".png" for p in result)
        assert any(p.suffix == ".svg" for p in result)
        assert all(p.exists() for p in result)

    def test_creates_parent_dirs(self, tmp_path: Path):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 2, 3])
        nested = tmp_path / "sub" / "dir" / "chart"
        result = save_chart(fig, nested, formats=("png",))
        plt.close(fig)

        assert result[0].exists()

    def test_custom_dpi(self, tmp_path: Path):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 2, 3])
        save_chart(fig, tmp_path / "test_chart", formats=("png",), dpi=72)
        plt.close(fig)
        # Just verify it doesn't crash at different DPI


# ── add_watermark ─────────────────────────────────────────────────────────────


class TestAddWatermark:
    def test_does_not_raise(self):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        add_watermark(fig)
        plt.close(fig)

    def test_custom_text(self):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        add_watermark(fig, text="Custom Watermark")
        plt.close(fig)

    def test_adds_text_artist(self):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        n_before = len(fig.texts)
        add_watermark(fig)
        assert len(fig.texts) == n_before + 1
        plt.close(fig)
