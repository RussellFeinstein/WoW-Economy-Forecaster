"""Tests for wow_forecaster.reporting.export."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from wow_forecaster.reporting.export import (
    export_to_csv,
    export_to_json,
    flatten_forecast_records_for_export,
    flatten_recommendations_for_export,
)


# ── export_to_csv ─────────────────────────────────────────────────────────────


def test_export_to_csv_basic(tmp_path: Path) -> None:
    """Writes a valid CSV with correct headers and values."""
    records = [
        {"archetype_id": "mat.ore.common", "score": 72.5, "action": "buy"},
        {"archetype_id": "consumable.flask.stat", "score": 45.0, "action": "hold"},
    ]
    out = tmp_path / "test.csv"
    result = export_to_csv(records, out)

    assert result == out
    assert out.exists()

    with out.open(encoding="utf-8") as f:
        reader = list(csv.DictReader(f))
    assert len(reader) == 2
    assert reader[0]["archetype_id"] == "mat.ore.common"
    assert reader[1]["action"] == "hold"


def test_export_to_csv_custom_fieldnames(tmp_path: Path) -> None:
    """Custom fieldnames control column order."""
    records = [{"a": 1, "b": 2, "c": 3}]
    out = tmp_path / "cols.csv"
    export_to_csv(records, out, fieldnames=["c", "a"])

    with out.open(encoding="utf-8") as f:
        header = f.readline().strip()
    assert header == "c,a"


def test_export_to_csv_extra_keys_ignored(tmp_path: Path) -> None:
    """Keys not in fieldnames are silently ignored (extrasaction='ignore')."""
    records = [{"name": "foo", "extra": "bar"}]
    out = tmp_path / "e.csv"
    export_to_csv(records, out, fieldnames=["name"])

    with out.open(encoding="utf-8") as f:
        reader = list(csv.DictReader(f))
    assert "extra" not in reader[0]
    assert reader[0]["name"] == "foo"


def test_export_to_csv_empty_records(tmp_path: Path) -> None:
    """Empty records list writes an empty file without raising."""
    out = tmp_path / "empty.csv"
    result = export_to_csv([], out)
    assert result == out
    assert out.exists()
    assert out.read_text(encoding="utf-8") == ""


def test_export_to_csv_creates_parent_dirs(tmp_path: Path) -> None:
    """Parent directories are created if they don't exist."""
    out = tmp_path / "nested" / "dir" / "output.csv"
    export_to_csv([{"x": 1}], out)
    assert out.exists()


def test_export_to_csv_returns_path(tmp_path: Path) -> None:
    """Return value is the same Path that was passed in."""
    out = tmp_path / "ret.csv"
    result = export_to_csv([{"k": "v"}], out)
    assert result == out


# ── export_to_json ────────────────────────────────────────────────────────────


def test_export_to_json_dict(tmp_path: Path) -> None:
    """Writes a pretty-printed JSON dict."""
    data = {"realm": "area-52", "count": 42}
    out  = tmp_path / "test.json"
    result = export_to_json(data, out)
    assert result == out
    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert loaded["realm"] == "area-52"
    assert loaded["count"] == 42


def test_export_to_json_list(tmp_path: Path) -> None:
    """Handles list payloads correctly."""
    data = [{"a": 1}, {"b": 2}]
    out  = tmp_path / "list.json"
    export_to_json(data, out)
    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert isinstance(loaded, list)
    assert len(loaded) == 2


def test_export_to_json_creates_parent_dirs(tmp_path: Path) -> None:
    """Parent directories are created if they don't exist."""
    out = tmp_path / "a" / "b" / "c.json"
    export_to_json({"x": 1}, out)
    assert out.exists()


def test_export_to_json_non_serialisable_uses_default_str(tmp_path: Path) -> None:
    """Non-serialisable objects are converted to strings (via ``default=str``)."""
    from datetime import date
    data = {"date": date(2025, 1, 15)}
    out  = tmp_path / "date.json"
    export_to_json(data, out)
    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert loaded["date"] == "2025-01-15"


# ── flatten_recommendations_for_export ────────────────────────────────────────


_RECS_JSON = {
    "schema_version": "v0.5.0",
    "realm_slug":     "area-52",
    "generated_at":   "2025-01-15",
    "run_slug":       "run-abc123",
    "categories": {
        "mat.ore.common": [
            {
                "rank":            1,
                "archetype_id":    "mat.ore.common",
                "horizon":         "1d",
                "target_date":     "2025-01-16",
                "current_price":   50.0,
                "predicted_price": 62.5,
                "ci_lower":        55.0,
                "ci_upper":        70.0,
                "roi_pct":         0.25,
                "score":           72.5,
                "action":          "buy",
                "reasoning":       "High opportunity score.",
                "score_components": {
                    "opportunity":  0.80,
                    "liquidity":    0.60,
                    "volatility":   0.10,
                    "event_boost":  0.05,
                    "uncertainty":  0.15,
                },
                "model_slug": "lgbm_v1",
            }
        ],
        "consumable.flask.stat": [
            {
                "rank":            1,
                "archetype_id":    "consumable.flask.stat",
                "horizon":         "7d",
                "target_date":     "2025-01-22",
                "current_price":   200.0,
                "predicted_price": 195.0,
                "ci_lower":        180.0,
                "ci_upper":        210.0,
                "roi_pct":         -0.025,
                "score":           45.0,
                "action":          "hold",
                "reasoning":       "Stable market.",
                "score_components": {
                    "opportunity":  0.30,
                    "liquidity":    0.70,
                    "volatility":   0.20,
                    "event_boost":  0.00,
                    "uncertainty":  0.10,
                },
                "model_slug": "lgbm_v1",
            }
        ],
    },
}


def test_flatten_recommendations_row_count() -> None:
    """One row per recommendation item (2 categories × 1 item each = 2 rows)."""
    rows = flatten_recommendations_for_export(_RECS_JSON)
    assert len(rows) == 2


def test_flatten_recommendations_metadata_columns() -> None:
    """realm_slug, generated_at, run_slug are present on every row."""
    rows = flatten_recommendations_for_export(_RECS_JSON)
    for row in rows:
        assert row["realm_slug"]   == "area-52"
        assert row["generated_at"] == "2025-01-15"
        assert row["run_slug"]     == "run-abc123"


def test_flatten_recommendations_score_components() -> None:
    """Score component columns are expanded as sc_* keys."""
    rows = flatten_recommendations_for_export(_RECS_JSON)
    ore_row = next(r for r in rows if r["archetype_id"] == "mat.ore.common")
    assert ore_row["sc_opportunity"] == 0.80
    assert ore_row["sc_liquidity"]   == 0.60
    assert ore_row["sc_volatility"]  == 0.10
    assert ore_row["sc_event_boost"] == 0.05
    assert ore_row["sc_uncertainty"] == 0.15


def test_flatten_recommendations_all_expected_columns() -> None:
    """All expected columns exist in each row."""
    expected = {
        "realm_slug", "generated_at", "run_slug",
        "category", "rank", "archetype_id", "horizon", "target_date",
        "current_price", "predicted_price", "ci_lower", "ci_upper",
        "roi_pct", "score", "action", "reasoning",
        "sc_opportunity", "sc_liquidity", "sc_volatility",
        "sc_event_boost", "sc_uncertainty", "model_slug",
    }
    rows = flatten_recommendations_for_export(_RECS_JSON)
    for row in rows:
        assert expected == set(row.keys())


def test_flatten_recommendations_empty_categories() -> None:
    """Empty categories dict produces an empty list."""
    empty = {"realm_slug": "area-52", "generated_at": "2025-01-15", "run_slug": "", "categories": {}}
    assert flatten_recommendations_for_export(empty) == []


def test_flatten_recommendations_category_key() -> None:
    """Category name is preserved in each row's 'category' field."""
    rows = flatten_recommendations_for_export(_RECS_JSON)
    cats = {r["category"] for r in rows}
    assert cats == {"mat.ore.common", "consumable.flask.stat"}


# ── flatten_forecast_records_for_export ──────────────────────────────────────


def test_flatten_forecast_records_adds_ci_width() -> None:
    """ci_width_gold and ci_pct_of_price are added to each row."""
    records = [
        {
            "archetype_id":   "mat.ore.common",
            "ci_lower":       "80.0",
            "ci_upper":       "120.0",
            "predicted_price": "100.0",
            "score":           "65.0",
        }
    ]
    result = flatten_forecast_records_for_export(records)
    assert len(result) == 1
    assert result[0]["ci_width_gold"]     == 40.0
    assert result[0]["ci_pct_of_price"]   == 0.4


def test_flatten_forecast_records_zero_predicted() -> None:
    """ci_pct_of_price is None when predicted_price is 0."""
    records = [{"archetype_id": "x", "ci_lower": "0", "ci_upper": "10", "predicted_price": "0"}]
    result = flatten_forecast_records_for_export(records)
    assert result[0]["ci_pct_of_price"] is None


def test_flatten_forecast_records_invalid_values() -> None:
    """Rows with unparseable numbers get None for derived columns."""
    records = [{"archetype_id": "bad", "ci_lower": "N/A", "ci_upper": "N/A", "predicted_price": "N/A"}]
    result = flatten_forecast_records_for_export(records)
    assert result[0]["ci_width_gold"]   is None
    assert result[0]["ci_pct_of_price"] is None


def test_flatten_forecast_records_preserves_original_keys() -> None:
    """Original keys are preserved alongside new derived columns."""
    records = [
        {
            "archetype_id": "ore",
            "score": "70",
            "ci_lower": "90",
            "ci_upper": "110",
            "predicted_price": "100",
        }
    ]
    result = flatten_forecast_records_for_export(records)
    assert result[0]["archetype_id"] == "ore"
    assert result[0]["score"] == "70"
    assert "ci_width_gold" in result[0]
