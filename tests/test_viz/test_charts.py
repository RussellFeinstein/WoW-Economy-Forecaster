"""Tests for all chart modules in wow_forecaster.viz.charts.

Each test verifies:
  1. Function returns a matplotlib Figure.
  2. Works with empty DataFrames (graceful no-data handling).
  3. Works with synthetic data fixtures.
  4. save_chart produces files from the returned Figure.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from wow_forecaster.viz.utils import save_chart

# ── Shared fixtures ───────────────────────────────────────────────────────────


@pytest.fixture
def df_hist() -> pd.DataFrame:
    """Synthetic historical price data."""
    dates = pd.date_range("2026-01-01", periods=60, freq="D")
    return pd.DataFrame({
        "obs_date": dates.strftime("%Y-%m-%d"),
        "avg_price_gold": np.random.uniform(400, 600, 60),
        "min_price_gold": np.random.uniform(350, 450, 60),
        "max_price_gold": np.random.uniform(550, 700, 60),
        "obs_count": np.random.randint(5, 50, 60),
    })


@pytest.fixture
def df_forecast() -> pd.DataFrame:
    """Synthetic forecast data."""
    return pd.DataFrame({
        "archetype_id": [1, 1, 1],
        "forecast_horizon": ["1d", "7d", "28d"],
        "target_date": ["2026-03-02", "2026-03-08", "2026-03-29"],
        "predicted_price_gold": [550.0, 580.0, 620.0],
        "confidence_lower": [480.0, 420.0, 380.0],
        "confidence_upper": [620.0, 740.0, 860.0],
        "ci_quality": ["good", "wide", "unreliable"],
        "model_slug": ["lgbm_1d", "lgbm_7d", "lgbm_28d"],
    })


@pytest.fixture
def df_backtest() -> pd.DataFrame:
    """Synthetic backtest per-prediction data."""
    n = 100
    actual = np.random.uniform(100, 1000, n)
    predicted = actual + np.random.normal(0, 50, n)
    return pd.DataFrame({
        "fold_index": np.random.choice([0, 1, 2], n),
        "archetype_id": np.random.choice([1, 2, 3, 4], n),
        "category_tag": np.random.choice(["consumable", "mat", "gear", "enchant"], n),
        "model_name": "naive_mean",
        "horizon_days": np.random.choice([1, 3], n),
        "actual_price": actual,
        "predicted_price": predicted,
        "direction_correct": np.random.choice([0, 1], n),
    })


@pytest.fixture
def df_feature_importance() -> pd.DataFrame:
    """Synthetic feature importance data."""
    features = [
        "price_mean", "price_lag_1d", "price_lag_7d", "price_roll_mean_7d",
        "price_pct_change_7d", "event_active", "day_of_week",
        "archetype_category", "quantity_sum", "is_cold_start",
        "transfer_confidence", "event_severity_max", "obs_count",
        "price_min", "price_max",
    ]
    rows = []
    for h in ["1d", "7d", "28d"]:
        gains = np.random.uniform(0, 500, len(features))
        total_gain = max(gains.sum(), 1.0)
        splits = np.random.randint(0, 100, len(features))
        total_split = max(splits.sum(), 1)
        for i, f in enumerate(features):
            rows.append({
                "feature": f,
                "gain": gains[i],
                "gain_pct": gains[i] / total_gain * 100,
                "split": splits[i],
                "split_pct": splits[i] / total_split * 100,
                "horizon": h,
            })
    return pd.DataFrame(rows)


@pytest.fixture
def df_recs() -> pd.DataFrame:
    """Synthetic recommendation data."""
    return pd.DataFrame({
        "category": ["consumable", "mat", "gear", "enchant", "gem",
                      "consumable", "mat", "gear", "enchant", "gem"],
        "rank": list(range(1, 11)),
        "archetype_id": list(range(1, 11)),
        "horizon": ["1d", "7d", "28d", "1d", "7d"] * 2,
        "action": ["buy", "buy", "sell", "hold", "avoid",
                    "buy", "sell", "hold", "buy", "avoid"],
        "score": np.random.uniform(20, 90, 10),
        "sc_opportunity": np.random.uniform(0, 100, 10),
        "sc_liquidity": np.random.uniform(0, 100, 10),
        "sc_volatility": np.random.uniform(0, 100, 10),
        "sc_event_boost": np.random.uniform(-50, 50, 10),
        "sc_uncertainty": np.random.uniform(0, 100, 10),
        "roi_pct": np.random.uniform(-0.2, 0.3, 10),
        "risk_level": ["low", "low", "medium", "high", "critical",
                        "low", "medium", "low", "low", "high"],
    })


@pytest.fixture
def df_drift() -> pd.DataFrame:
    """Synthetic drift history data."""
    return pd.DataFrame({
        "checked_at": pd.date_range("2026-02-01", periods=10, freq="3D").strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),
        "overall_drift_level": ["none", "none", "low", "low", "medium",
                                 "medium", "high", "medium", "low", "none"],
        "uncertainty_multiplier": [1.0, 1.0, 1.25, 1.25, 1.5,
                                    1.5, 2.0, 1.5, 1.25, 1.0],
        "retrain_recommended": [False, False, False, False, True,
                                 True, True, True, False, False],
    })


# ── Forecast chart tests ─────────────────────────────────────────────────────


class TestForecastChart:
    def test_returns_figure(self, df_hist, df_forecast):
        from wow_forecaster.viz.charts.forecast_chart import plot_forecast_timeline
        fig = plot_forecast_timeline(df_hist, df_forecast, "Stat Flasks")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_empty_data(self):
        from wow_forecaster.viz.charts.forecast_chart import plot_forecast_timeline
        fig = plot_forecast_timeline(pd.DataFrame(), pd.DataFrame())
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_single_horizon_filter(self, df_hist, df_forecast):
        from wow_forecaster.viz.charts.forecast_chart import plot_forecast_timeline
        fig = plot_forecast_timeline(df_hist, df_forecast, horizon="7d")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_multi_horizon(self, df_hist, df_forecast):
        from wow_forecaster.viz.charts.forecast_chart import plot_forecast_multi_horizon
        fig = plot_forecast_multi_horizon(df_hist, df_forecast, "Stat Flasks")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_saves_to_file(self, df_hist, df_forecast, tmp_path: Path):
        from wow_forecaster.viz.charts.forecast_chart import plot_forecast_timeline
        fig = plot_forecast_timeline(df_hist, df_forecast)
        paths = save_chart(fig, tmp_path / "forecast_test", formats=("png",))
        plt.close(fig)
        assert paths[0].exists()

    def test_plotly_interactive(self, df_hist, df_forecast):
        from wow_forecaster.viz.charts.forecast_chart import plot_forecast_timeline_interactive
        fig = plot_forecast_timeline_interactive(df_hist, df_forecast, "Stat Flasks")
        assert fig is not None  # Plotly figure

    def test_plotly_empty(self):
        from wow_forecaster.viz.charts.forecast_chart import plot_forecast_timeline_interactive
        fig = plot_forecast_timeline_interactive(pd.DataFrame(), pd.DataFrame())
        assert fig is not None


# ── Backtest chart tests ──────────────────────────────────────────────────────


class TestBacktestChart:
    def test_scatter_returns_figure(self, df_backtest):
        from wow_forecaster.viz.charts.backtest_chart import plot_actual_vs_predicted_scatter
        fig = plot_actual_vs_predicted_scatter(df_backtest)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_scatter_empty(self):
        from wow_forecaster.viz.charts.backtest_chart import plot_actual_vs_predicted_scatter
        fig = plot_actual_vs_predicted_scatter(pd.DataFrame())
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_residuals_returns_figure(self, df_backtest):
        from wow_forecaster.viz.charts.backtest_chart import plot_residual_distribution
        fig = plot_residual_distribution(df_backtest)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_residuals_empty(self):
        from wow_forecaster.viz.charts.backtest_chart import plot_residual_distribution
        fig = plot_residual_distribution(pd.DataFrame())
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_heatmap_returns_figure(self, df_backtest):
        from wow_forecaster.viz.charts.backtest_chart import plot_directional_accuracy_heatmap
        fig = plot_directional_accuracy_heatmap(df_backtest)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_heatmap_empty(self):
        from wow_forecaster.viz.charts.backtest_chart import plot_directional_accuracy_heatmap
        fig = plot_directional_accuracy_heatmap(pd.DataFrame())
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_walk_forward_returns_figure(self, df_backtest):
        from wow_forecaster.viz.charts.backtest_chart import plot_walk_forward_mae
        fig = plot_walk_forward_mae(df_backtest)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_walk_forward_empty(self):
        from wow_forecaster.viz.charts.backtest_chart import plot_walk_forward_mae
        fig = plot_walk_forward_mae(pd.DataFrame())
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_metrics_by_category(self, df_backtest):
        from wow_forecaster.viz.charts.backtest_chart import plot_metrics_by_category

        def _mae(x):
            return (x - df_backtest.loc[x.index, "predicted_price"]).abs().mean()

        def _mape(x):
            diff = (x - df_backtest.loc[x.index, "predicted_price"]).abs()
            return (diff / x).mean()

        summary = df_backtest.groupby("category_tag").agg(
            mae=("actual_price", _mae),
            mape=("actual_price", _mape),
        ).reset_index()
        fig = plot_metrics_by_category(summary)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_saves_to_file(self, df_backtest, tmp_path: Path):
        from wow_forecaster.viz.charts.backtest_chart import plot_actual_vs_predicted_scatter
        fig = plot_actual_vs_predicted_scatter(df_backtest)
        paths = save_chart(fig, tmp_path / "scatter", formats=("png",))
        plt.close(fig)
        assert paths[0].exists()


# ── Feature chart tests ───────────────────────────────────────────────────────


class TestFeatureChart:
    def test_importance_returns_figure(self, df_feature_importance):
        from wow_forecaster.viz.charts.feature_chart import plot_feature_importance
        fig = plot_feature_importance(df_feature_importance)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_importance_empty(self):
        from wow_forecaster.viz.charts.feature_chart import plot_feature_importance
        fig = plot_feature_importance(pd.DataFrame())
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_by_horizon_returns_figure(self, df_feature_importance):
        from wow_forecaster.viz.charts.feature_chart import plot_importance_by_horizon
        fig = plot_importance_by_horizon(df_feature_importance)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_correlation_returns_figure(self):
        from wow_forecaster.viz.charts.feature_chart import plot_feature_correlation
        df = pd.DataFrame(np.random.randn(50, 10),
                          columns=[f"feat_{i}" for i in range(10)])
        fig = plot_feature_correlation(df, top_n=10)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_correlation_empty(self):
        from wow_forecaster.viz.charts.feature_chart import plot_feature_correlation
        fig = plot_feature_correlation(pd.DataFrame())
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_saves_to_file(self, df_feature_importance, tmp_path: Path):
        from wow_forecaster.viz.charts.feature_chart import plot_feature_importance
        fig = plot_feature_importance(df_feature_importance)
        paths = save_chart(fig, tmp_path / "fi", formats=("png",))
        plt.close(fig)
        assert paths[0].exists()


# ── Recommendation chart tests ────────────────────────────────────────────────


class TestRecommendationChart:
    def test_tornado_returns_figure(self, df_recs):
        from wow_forecaster.viz.charts.recommendation_chart import plot_score_tornado
        fig = plot_score_tornado(df_recs)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_tornado_empty(self):
        from wow_forecaster.viz.charts.recommendation_chart import plot_score_tornado
        fig = plot_score_tornado(pd.DataFrame())
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_action_distribution_returns_figure(self, df_recs):
        from wow_forecaster.viz.charts.recommendation_chart import plot_action_distribution
        fig = plot_action_distribution(df_recs)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_action_distribution_empty(self):
        from wow_forecaster.viz.charts.recommendation_chart import plot_action_distribution
        fig = plot_action_distribution(pd.DataFrame())
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_roi_vs_uncertainty_returns_figure(self, df_recs):
        from wow_forecaster.viz.charts.recommendation_chart import plot_roi_vs_uncertainty
        fig = plot_roi_vs_uncertainty(df_recs)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_roi_vs_uncertainty_empty(self):
        from wow_forecaster.viz.charts.recommendation_chart import plot_roi_vs_uncertainty
        fig = plot_roi_vs_uncertainty(pd.DataFrame())
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_saves_to_file(self, df_recs, tmp_path: Path):
        from wow_forecaster.viz.charts.recommendation_chart import plot_score_tornado
        fig = plot_score_tornado(df_recs)
        paths = save_chart(fig, tmp_path / "tornado", formats=("png",))
        plt.close(fig)
        assert paths[0].exists()


# ── Drift chart tests ─────────────────────────────────────────────────────────


class TestDriftChart:
    def test_timeline_returns_figure(self, df_drift):
        from wow_forecaster.viz.charts.drift_chart import plot_drift_timeline
        fig = plot_drift_timeline(df_drift)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_timeline_empty(self):
        from wow_forecaster.viz.charts.drift_chart import plot_drift_timeline
        fig = plot_drift_timeline(pd.DataFrame())
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_gauge_returns_figure(self):
        from wow_forecaster.viz.charts.drift_chart import plot_mae_ratio_gauge
        health = {
            "1d": {"mae_ratio": 1.1},
            "7d": {"mae_ratio": 1.8},
            "28d": {"mae_ratio": 3.5},
        }
        fig = plot_mae_ratio_gauge(health)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_gauge_empty(self):
        from wow_forecaster.viz.charts.drift_chart import plot_mae_ratio_gauge
        fig = plot_mae_ratio_gauge({})
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_saves_to_file(self, df_drift, tmp_path: Path):
        from wow_forecaster.viz.charts.drift_chart import plot_drift_timeline
        fig = plot_drift_timeline(df_drift)
        paths = save_chart(fig, tmp_path / "drift", formats=("png",))
        plt.close(fig)
        assert paths[0].exists()


# ── Transfer chart tests ──────────────────────────────────────────────────────


class TestTransferChart:
    def test_ci_width_returns_figure(self, df_forecast):
        from wow_forecaster.viz.charts.transfer_chart import plot_ci_width_by_category
        fig = plot_ci_width_by_category(df_forecast)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_ci_width_empty(self):
        from wow_forecaster.viz.charts.transfer_chart import plot_ci_width_by_category
        fig = plot_ci_width_by_category(pd.DataFrame())
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_blend_diagram(self):
        from wow_forecaster.viz.charts.transfer_chart import plot_cold_start_blend_diagram
        fig = plot_cold_start_blend_diagram()
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_blend_diagram_saves(self, tmp_path: Path):
        from wow_forecaster.viz.charts.transfer_chart import plot_cold_start_blend_diagram
        fig = plot_cold_start_blend_diagram()
        paths = save_chart(fig, tmp_path / "blend", formats=("png",))
        plt.close(fig)
        assert paths[0].exists()
