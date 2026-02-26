"""Tests for wow_forecaster.reporting.reader."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pytest

from wow_forecaster.reporting.reader import (
    check_freshness,
    find_latest_file,
    load_drift_report,
    load_forecast_records,
    load_health_report,
    load_provenance_report,
    load_recommendations_report,
)


# ── find_latest_file ──────────────────────────────────────────────────────────


def test_find_latest_file_returns_most_recent(tmp_path: Path) -> None:
    """The most-recently-modified file is returned when multiple files match."""
    older = tmp_path / "recs_area-52_2025-01-01.json"
    newer = tmp_path / "recs_area-52_2025-01-10.json"
    older.write_text("{}", encoding="utf-8")
    # Ensure different mtime.
    time.sleep(0.01)
    newer.write_text("{}", encoding="utf-8")

    result = find_latest_file(tmp_path, "recs_area-52_*.json")
    assert result == newer


def test_find_latest_file_single_match(tmp_path: Path) -> None:
    """Returns the single matching file."""
    p = tmp_path / "drift_status_area-52_2025-01-15.json"
    p.write_text("{}")
    assert find_latest_file(tmp_path, "drift_status_area-52_*.json") == p


def test_find_latest_file_no_directory(tmp_path: Path) -> None:
    """Returns None when the directory does not exist."""
    missing = tmp_path / "nonexistent"
    assert find_latest_file(missing, "*.json") is None


def test_find_latest_file_no_match(tmp_path: Path) -> None:
    """Returns None when no files match the pattern."""
    (tmp_path / "other_file.txt").write_text("x")
    assert find_latest_file(tmp_path, "*.json") is None


def test_find_latest_file_empty_directory(tmp_path: Path) -> None:
    """Returns None for an empty directory."""
    assert find_latest_file(tmp_path, "*.json") is None


# ── check_freshness ───────────────────────────────────────────────────────────


def test_check_freshness_fresh_datetime() -> None:
    """A datetime 1 h ago is within the 4 h default window."""
    now = datetime.now(tz=timezone.utc)
    one_h_ago = now.replace(hour=now.hour - 1) if now.hour >= 1 else now
    ts = one_h_ago.isoformat()
    is_fresh, age = check_freshness(ts, max_hours=4.0)
    assert is_fresh is True
    assert age is not None
    assert 0.0 <= age <= 2.0  # generous window for CI timing


def test_check_freshness_stale_datetime() -> None:
    """A datetime 24 h ago is outside the 4 h window."""
    from datetime import timedelta
    stale = datetime.now(tz=timezone.utc) - timedelta(hours=24)
    is_fresh, age = check_freshness(stale.isoformat(), max_hours=4.0)
    assert is_fresh is False
    assert age is not None
    assert age >= 20.0  # at least 20 h old


def test_check_freshness_date_only_today() -> None:
    """A date-only string for today is treated as midnight UTC today."""
    today = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
    is_fresh, age = check_freshness(today, max_hours=25.0)
    # today midnight could be up to 24h in the past — use generous threshold
    assert age is not None
    assert 0.0 <= age <= 25.0


def test_check_freshness_none_input() -> None:
    """None input returns (False, None)."""
    is_fresh, age = check_freshness(None)
    assert is_fresh is False
    assert age is None


def test_check_freshness_empty_string() -> None:
    """Empty string returns (False, None)."""
    is_fresh, age = check_freshness("")
    assert is_fresh is False
    assert age is None


def test_check_freshness_invalid_string() -> None:
    """Unparseable string returns (False, None)."""
    is_fresh, age = check_freshness("not-a-date")
    assert is_fresh is False
    assert age is None


def test_check_freshness_custom_max_hours() -> None:
    """Custom max_hours threshold is respected."""
    from datetime import timedelta
    slightly_old = datetime.now(tz=timezone.utc) - timedelta(hours=2)
    ts = slightly_old.isoformat()
    # Strict: 1 h threshold → stale
    is_fresh_strict, _ = check_freshness(ts, max_hours=1.0)
    # Lenient: 6 h threshold → fresh
    is_fresh_lenient, _ = check_freshness(ts, max_hours=6.0)
    assert is_fresh_strict is False
    assert is_fresh_lenient is True


# ── load_recommendations_report ──────────────────────────────────────────────


def test_load_recommendations_report_success(tmp_path: Path) -> None:
    """Returns parsed dict when a valid file exists."""
    payload = {"realm_slug": "area-52", "categories": {"mat.ore.common": []}}
    (tmp_path / "recommendations_area-52_2025-01-15.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )
    result = load_recommendations_report("area-52", tmp_path)
    assert result is not None
    assert result["realm_slug"] == "area-52"


def test_load_recommendations_report_not_found(tmp_path: Path) -> None:
    """Returns None when no matching file exists."""
    assert load_recommendations_report("area-52", tmp_path) is None


def test_load_recommendations_report_bad_json(tmp_path: Path) -> None:
    """Returns None when the file contains invalid JSON."""
    (tmp_path / "recommendations_area-52_2025-01-15.json").write_text(
        "{ bad json }", encoding="utf-8"
    )
    assert load_recommendations_report("area-52", tmp_path) is None


def test_load_recommendations_report_returns_latest(tmp_path: Path) -> None:
    """Returns the most recently modified file when several exist."""
    old = tmp_path / "recommendations_area-52_2025-01-01.json"
    new = tmp_path / "recommendations_area-52_2025-01-20.json"
    old.write_text(json.dumps({"realm_slug": "old"}), encoding="utf-8")
    time.sleep(0.01)
    new.write_text(json.dumps({"realm_slug": "new"}), encoding="utf-8")
    result = load_recommendations_report("area-52", tmp_path)
    assert result is not None
    assert result["realm_slug"] == "new"


# ── load_forecast_records ────────────────────────────────────────────────────


def test_load_forecast_records_success(tmp_path: Path) -> None:
    """Returns list of row dicts from a valid CSV."""
    csv_content = (
        "archetype_id,realm_slug,horizon,score,action\n"
        "mat.ore.common,area-52,1d,72.5,buy\n"
        "consumable.flask.stat,area-52,7d,60.1,hold\n"
    )
    (tmp_path / "forecast_area-52_2025-01-15.csv").write_text(csv_content, encoding="utf-8")
    records = load_forecast_records("area-52", tmp_path)
    assert records is not None
    assert len(records) == 2
    assert records[0]["archetype_id"] == "mat.ore.common"
    assert records[0]["action"] == "buy"


def test_load_forecast_records_not_found(tmp_path: Path) -> None:
    """Returns None when no matching file exists."""
    assert load_forecast_records("area-52", tmp_path) is None


def test_load_forecast_records_empty_csv(tmp_path: Path) -> None:
    """Returns an empty list for a header-only CSV."""
    (tmp_path / "forecast_area-52_2025-01-15.csv").write_text(
        "archetype_id,realm_slug,horizon\n", encoding="utf-8"
    )
    records = load_forecast_records("area-52", tmp_path)
    assert records == []


def test_load_forecast_records_wrong_realm(tmp_path: Path) -> None:
    """Does not match files for a different realm."""
    (tmp_path / "forecast_illidan_2025-01-15.csv").write_text(
        "archetype_id,score\nfoo,5\n", encoding="utf-8"
    )
    assert load_forecast_records("area-52", tmp_path) is None


# ── load_drift_report ─────────────────────────────────────────────────────────


def test_load_drift_report_success(tmp_path: Path) -> None:
    """Returns parsed dict for a valid drift status file."""
    payload = {"realm_slug": "area-52", "overall_drift_level": "none"}
    (tmp_path / "drift_status_area-52_2025-01-15.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )
    result = load_drift_report("area-52", tmp_path)
    assert result is not None
    assert result["overall_drift_level"] == "none"


def test_load_drift_report_not_found(tmp_path: Path) -> None:
    assert load_drift_report("area-52", tmp_path) is None


# ── load_health_report ────────────────────────────────────────────────────────


def test_load_health_report_success(tmp_path: Path) -> None:
    """Returns parsed dict for a valid model health file."""
    payload = {"realm_slug": "area-52", "horizons": [{"horizon_days": 1, "health_status": "ok"}]}
    (tmp_path / "model_health_area-52_2025-01-15.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )
    result = load_health_report("area-52", tmp_path)
    assert result is not None
    assert result["horizons"][0]["health_status"] == "ok"


def test_load_health_report_not_found(tmp_path: Path) -> None:
    assert load_health_report("area-52", tmp_path) is None


# ── load_provenance_report ────────────────────────────────────────────────────


def test_load_provenance_report_success(tmp_path: Path) -> None:
    """Returns parsed dict for a valid provenance file."""
    payload = {
        "realm_slug": "area-52",
        "freshness_hours": 1.5,
        "is_fresh": True,
        "sources": [],
    }
    (tmp_path / "provenance_area-52_2025-01-15.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )
    result = load_provenance_report("area-52", tmp_path)
    assert result is not None
    assert result["is_fresh"] is True


def test_load_provenance_report_not_found(tmp_path: Path) -> None:
    assert load_provenance_report("area-52", tmp_path) is None
