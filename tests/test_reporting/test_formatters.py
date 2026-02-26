"""Tests for wow_forecaster.reporting.formatters."""

from __future__ import annotations

import pytest

from wow_forecaster.reporting.formatters import (
    format_drift_health_summary,
    format_forecast_summary,
    format_freshness_banner,
    format_status_summary,
    format_top_items_table,
    format_volatility_watchlist,
)


# ── format_freshness_banner ───────────────────────────────────────────────────


def test_freshness_banner_fresh() -> None:
    """Fresh report shows [FRESH] tag."""
    banner = format_freshness_banner(is_fresh=True, age_hours=1.5)
    assert "[FRESH]" in banner
    assert "1.5h" in banner


def test_freshness_banner_stale() -> None:
    """Stale report shows [STALE] tag with warning."""
    banner = format_freshness_banner(is_fresh=False, age_hours=26.0)
    assert "[STALE]" in banner
    assert "26.0h" in banner


def test_freshness_banner_unknown_age() -> None:
    """Unknown age shows [AGE UNKNOWN] tag."""
    banner = format_freshness_banner(is_fresh=False, age_hours=None)
    assert "[AGE UNKNOWN]" in banner


def test_freshness_banner_with_source_file() -> None:
    """Optional source file appears on a second line."""
    banner = format_freshness_banner(
        is_fresh=True, age_hours=0.5, source_file="drift_status_*.json"
    )
    assert "drift_status_*.json" in banner
    lines = banner.strip().split("\n")
    assert len(lines) == 2


def test_freshness_banner_no_source_file() -> None:
    """When source_file is empty the banner is a single line."""
    banner = format_freshness_banner(is_fresh=True, age_hours=2.0, source_file="")
    assert "\n" not in banner.strip()


# ── format_top_items_table ────────────────────────────────────────────────────


_SAMPLE_CATEGORIES = {
    "mat.ore.common": [
        {
            "rank": 1,
            "archetype_id": "mat.ore.common",
            "horizon": "1d",
            "current_price": 50.0,
            "predicted_price": 60.0,
            "roi_pct": 0.20,
            "score": 72.5,
            "action": "buy",
        }
    ],
    "consumable.flask.stat": [
        {
            "rank": 1,
            "archetype_id": "consumable.flask.stat",
            "horizon": "7d",
            "current_price": 200.0,
            "predicted_price": 190.0,
            "roi_pct": -0.05,
            "score": 45.0,
            "action": "hold",
        }
    ],
}


def test_format_top_items_table_header() -> None:
    """Output includes the header and realm."""
    out = format_top_items_table(
        _SAMPLE_CATEGORIES, "area-52", "2025-01-15", True, 1.0
    )
    assert "Top Recommendations by Category" in out
    assert "area-52" in out


def test_format_top_items_table_categories_shown() -> None:
    """Each category appears in its own block."""
    out = format_top_items_table(
        _SAMPLE_CATEGORIES, "area-52", "2025-01-15", True, 1.0
    )
    assert "MAT.ORE.COMMON" in out
    assert "CONSUMABLE.FLASK.STAT" in out


def test_format_top_items_table_item_data() -> None:
    """Price and action values are rendered."""
    out = format_top_items_table(
        _SAMPLE_CATEGORIES, "area-52", "2025-01-15", True, 1.0
    )
    assert "60.0g" in out   # predicted price
    assert "buy" in out
    assert "hold" in out


def test_format_top_items_table_stale_banner() -> None:
    """[STALE] banner appears when is_fresh=False."""
    out = format_top_items_table(
        _SAMPLE_CATEGORIES, "area-52", "2025-01-15", False, 10.0
    )
    assert "[STALE]" in out


def test_format_top_items_table_empty_categories() -> None:
    """Graceful message when no recommendations are present."""
    out = format_top_items_table({}, "area-52", "2025-01-15", True, 0.5)
    assert "no recommendations available" in out.lower()


def test_format_top_items_table_fresh_banner() -> None:
    """[FRESH] banner appears when is_fresh=True."""
    out = format_top_items_table(
        _SAMPLE_CATEGORIES, "area-52", "2025-01-15", True, 1.2
    )
    assert "[FRESH]" in out


# ── format_forecast_summary ───────────────────────────────────────────────────


_SAMPLE_RECORDS = [
    {
        "archetype_id": "mat.ore.common",
        "realm_slug":   "area-52",
        "horizon":      "1d",
        "current_price":  "50.0",
        "predicted_price": "62.5",
        "ci_lower":      "55.0",
        "ci_upper":      "70.0",
        "roi_pct":       "+0.25",
        "score":         "72.5",
        "action":        "buy",
        "model_slug":    "lgbm_v1",
    },
    {
        "archetype_id": "consumable.flask.stat",
        "realm_slug":   "area-52",
        "horizon":      "7d",
        "current_price":  "200.0",
        "predicted_price": "195.0",
        "ci_lower":      "180.0",
        "ci_upper":      "210.0",
        "roi_pct":       "-0.025",
        "score":         "45.0",
        "action":        "hold",
        "model_slug":    "lgbm_v1",
    },
]


def test_format_forecast_summary_shows_data() -> None:
    """Archetype IDs and scores appear in the output."""
    out = format_forecast_summary(_SAMPLE_RECORDS, "area-52", is_fresh=True, age_hours=1.0)
    assert "mat.ore.common" in out
    assert "consumable.flask.stat" in out
    assert "72.5" in out


def test_format_forecast_summary_ci_width() -> None:
    """CI width is computed and shown (70 - 55 = 15g)."""
    out = format_forecast_summary(_SAMPLE_RECORDS, "area-52", is_fresh=True, age_hours=1.0)
    assert "15.0g" in out


def test_format_forecast_summary_horizon_filter() -> None:
    """Only rows matching horizon_filter are shown."""
    out = format_forecast_summary(
        _SAMPLE_RECORDS, "area-52", horizon_filter="1d", is_fresh=True, age_hours=0.5
    )
    assert "mat.ore.common" in out
    assert "consumable.flask.stat" not in out


def test_format_forecast_summary_empty() -> None:
    """Graceful message when no records are available."""
    out = format_forecast_summary([], "area-52", is_fresh=True, age_hours=0.5)
    assert "no forecast data available" in out.lower()


def test_format_forecast_summary_truncation_message() -> None:
    """Truncation message appears when top_n < total records."""
    out = format_forecast_summary(
        _SAMPLE_RECORDS, "area-52", top_n=1, is_fresh=True, age_hours=0.5
    )
    assert "showing 1 of 2" in out


def test_format_forecast_summary_stale_banner() -> None:
    """[STALE] banner appears for stale reports."""
    out = format_forecast_summary(_SAMPLE_RECORDS, "area-52", is_fresh=False, age_hours=8.0)
    assert "[STALE]" in out


# ── format_volatility_watchlist ───────────────────────────────────────────────


def test_format_volatility_watchlist_sorted_by_ci() -> None:
    """Items are shown with widest CI first."""
    records = [
        {
            "archetype_id": "narrow",
            "horizon": "1d",
            "predicted_price": "100.0",
            "ci_lower": "98.0",
            "ci_upper": "102.0",   # width = 4g
            "score": "60.0",
            "action": "hold",
        },
        {
            "archetype_id": "wide",
            "horizon": "1d",
            "predicted_price": "100.0",
            "ci_lower": "60.0",
            "ci_upper": "140.0",   # width = 80g
            "score": "40.0",
            "action": "avoid",
        },
    ]
    out = format_volatility_watchlist(records, "area-52", is_fresh=True, age_hours=1.0)
    # 'wide' should appear before 'narrow' in the output.
    idx_wide   = out.find("wide")
    idx_narrow = out.find("narrow")
    assert idx_wide < idx_narrow, "Wide CI item should appear first"


def test_format_volatility_watchlist_ci_pct() -> None:
    """Relative CI% column is shown."""
    records = [
        {
            "archetype_id": "mat.ore.common",
            "horizon": "1d",
            "predicted_price": "100.0",
            "ci_lower": "70.0",
            "ci_upper": "130.0",   # 60% of price
            "score": "50.0",
            "action": "avoid",
        }
    ]
    out = format_volatility_watchlist(records, "area-52", is_fresh=True, age_hours=0.5)
    assert "60%" in out


def test_format_volatility_watchlist_empty() -> None:
    """Graceful message when no records are available."""
    out = format_volatility_watchlist([], "area-52", is_fresh=True, age_hours=0.0)
    assert "no forecast data available" in out.lower()


def test_format_volatility_watchlist_truncation() -> None:
    """Truncation message appears when top_n < total records."""
    records = [
        {
            "archetype_id": f"item_{i}",
            "horizon": "1d",
            "predicted_price": "100.0",
            "ci_lower": str(40 - i),
            "ci_upper": str(160 + i),
            "score": "50.0",
            "action": "avoid",
        }
        for i in range(5)
    ]
    out = format_volatility_watchlist(records, "area-52", top_n=2, is_fresh=True, age_hours=0.0)
    assert "showing 2 of 5" in out


def test_format_volatility_watchlist_skips_invalid_rows() -> None:
    """Rows with unparseable CI values are silently skipped."""
    records = [
        {"archetype_id": "bad",   "horizon": "1d", "predicted_price": "bad", "ci_lower": "x", "ci_upper": "y", "score": "50", "action": "hold"},
        {"archetype_id": "good",  "horizon": "1d", "predicted_price": "100.0", "ci_lower": "80.0", "ci_upper": "120.0", "score": "60", "action": "buy"},
    ]
    out = format_volatility_watchlist(records, "area-52", is_fresh=True, age_hours=0.5)
    assert "good" in out


# ── format_drift_health_summary ──────────────────────────────────────────────


_DRIFT = {
    "overall_drift_level": "medium",
    "uncertainty_multiplier": 1.5,
    "retrain_recommended": True,
    "checked_at": "2025-01-15T12:00:00+00:00",
    "data_drift":  {"drift_level": "medium", "n_series_drifted": 3, "n_series_checked": 10},
    "error_drift": {"drift_level": "low",    "n_evaluated": 20, "mae_ratio": 1.3},
    "event_shock": {"shock_active": False, "active_count": 0, "upcoming_count": 1, "active_events": []},
}

_HEALTH = {
    "realm_slug": "area-52",
    "checked_at": "2025-01-15T12:00:00+00:00",
    "horizons": [
        {"horizon_days": 1,  "health_status": "ok",       "n_evaluated": 50,
         "live_mae": 12.5, "baseline_mae": 11.0, "mae_ratio": 1.14, "live_dir_acc": 0.62},
        {"horizon_days": 7,  "health_status": "degraded", "n_evaluated": 30,
         "live_mae": 25.0, "baseline_mae": 15.0, "mae_ratio": 1.67, "live_dir_acc": 0.55},
    ],
}


def test_format_drift_health_both_present() -> None:
    """Both drift and health sections are present in the output."""
    out = format_drift_health_summary(
        _DRIFT, _HEALTH, "area-52",
        is_fresh_drift=True, age_hours_drift=1.0,
        is_fresh_health=True, age_hours_health=1.0,
    )
    assert "Drift Check" in out
    assert "Model Health" in out
    assert "MEDIUM" in out                  # overall drift level uppercased
    assert "x1.50" in out                   # uncertainty multiplier
    assert "ACTION REQUIRED" in out         # retrain message
    assert "ok" in out                      # 1d health status
    assert "degraded" in out               # 7d health status


def test_format_drift_health_no_drift() -> None:
    """Graceful message when drift is None."""
    out = format_drift_health_summary(
        None, _HEALTH, "area-52",
        is_fresh_drift=False, age_hours_drift=None,
        is_fresh_health=True, age_hours_health=1.0,
    )
    assert "no drift report available" in out.lower()
    assert "Model Health" in out


def test_format_drift_health_no_health() -> None:
    """Graceful message when health is None."""
    out = format_drift_health_summary(
        _DRIFT, None, "area-52",
        is_fresh_drift=True, age_hours_drift=2.0,
        is_fresh_health=False, age_hours_health=None,
    )
    assert "no health report available" in out.lower()
    assert "Drift Check" in out


def test_format_drift_health_both_none() -> None:
    """Both sections show 'no data' messages when both are None."""
    out = format_drift_health_summary(
        None, None, "area-52",
        is_fresh_drift=False, age_hours_drift=None,
        is_fresh_health=False, age_hours_health=None,
    )
    assert "no drift report available" in out.lower()
    assert "no health report available" in out.lower()


def test_format_drift_health_no_retrain() -> None:
    """No ACTION REQUIRED when retrain_recommended is False."""
    drift_no_retrain = {**_DRIFT, "retrain_recommended": False}
    out = format_drift_health_summary(
        drift_no_retrain, _HEALTH, "area-52",
        is_fresh_drift=True, age_hours_drift=1.0,
        is_fresh_health=True, age_hours_health=1.0,
    )
    assert "ACTION REQUIRED" not in out


# ── format_status_summary ─────────────────────────────────────────────────────


_PROVENANCE = {
    "realm_slug":     "area-52",
    "checked_at":     "2025-01-15T12:00:00+00:00",
    "freshness_hours": 2.5,
    "is_fresh":        True,
    "sources": [
        {
            "source":             "undermine",
            "last_snapshot_at":   "2025-01-15T11:00:00+00:00",
            "snapshot_count_24h": 24,
            "total_records_24h":  15000,
            "success_rate_24h":   0.96,
            "is_stale":           False,
        },
        {
            "source":             "blizzard_api",
            "last_snapshot_at":   None,
            "snapshot_count_24h": 0,
            "total_records_24h":  0,
            "success_rate_24h":   0.0,
            "is_stale":           True,
        },
    ],
}


def test_format_status_summary_shows_sources() -> None:
    """Source names and status flags appear in the output."""
    out = format_status_summary(_PROVENANCE, "area-52", is_fresh_prov=True, age_hours_prov=0.5)
    assert "undermine" in out
    assert "blizzard_api" in out
    assert "[OK]" in out
    assert "[STALE]" in out


def test_format_status_summary_freshness_info() -> None:
    """Data freshness hours and checked_at appear."""
    out = format_status_summary(_PROVENANCE, "area-52", is_fresh_prov=True, age_hours_prov=0.5)
    assert "2.5h" in out
    assert "FRESH" in out


def test_format_status_summary_no_provenance() -> None:
    """Graceful message when provenance is None."""
    out = format_status_summary(None, "area-52", is_fresh_prov=False, age_hours_prov=None)
    assert "no provenance report available" in out.lower()


def test_format_status_summary_distinguishes_report_from_data_freshness() -> None:
    """Note about report age vs data freshness appears."""
    out = format_status_summary(_PROVENANCE, "area-52", is_fresh_prov=True, age_hours_prov=0.5)
    assert "report age != data freshness" in out.lower()


def test_format_status_summary_stale_data() -> None:
    """STALE tag shown when is_fresh=False in provenance."""
    stale_prov = {**_PROVENANCE, "is_fresh": False, "freshness_hours": 36.0}
    out = format_status_summary(
        stale_prov, "area-52", is_fresh_prov=True, age_hours_prov=0.5
    )
    assert "STALE" in out


def test_format_status_summary_never_seen_snapshot() -> None:
    """Source with last_snapshot_at=None renders 'never' gracefully."""
    out = format_status_summary(_PROVENANCE, "area-52", is_fresh_prov=True, age_hours_prov=0.5)
    assert "never" in out
