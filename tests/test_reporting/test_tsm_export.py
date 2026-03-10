"""
Tests for wow_forecaster/reporting/tsm_export.py.

What we test
------------
fetch_tsm_export_items():
  - Returns empty list when no item-level forecasts exist.
  - Returns empty list when no recent price observations exist.
  - Includes items with ROI >= min_roi_pct.
  - Excludes items with ROI < min_roi_pct.
  - Uses the most recent forecast when multiple exist for the same item.
  - Excludes forecasts with ci_quality != 'good'.
  - Realm isolation: different realm returns nothing.
  - Horizon isolation: different horizon returns nothing.
  - Results are sorted by roi_pct descending.

build_tsm_import_string():
  - Returns empty string for empty list.
  - Returns correct i:XXXXX format for a single item.
  - Returns comma-separated i:XXXXX,i:YYYYY format for multiple items.

write_tsm_export():
  - Writes the TSM string to the specified path.
  - Creates parent directories if they do not exist.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from wow_forecaster.reporting.tsm_export import (
    TsmExportRow,
    build_tsm_import_string,
    fetch_tsm_export_items,
    write_tsm_export,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def conn() -> sqlite3.Connection:
    """In-memory DB with minimal schema for TSM export tests."""
    db = sqlite3.connect(":memory:")
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA foreign_keys = OFF;")
    db.executescript("""
        CREATE TABLE items (
            item_id        INTEGER PRIMARY KEY,
            name           TEXT    NOT NULL,
            archetype_id   INTEGER,
            category_id    INTEGER NOT NULL DEFAULT 1,
            expansion_slug TEXT    NOT NULL DEFAULT 'midnight',
            quality        TEXT    NOT NULL DEFAULT 'common',
            is_crafted     INTEGER NOT NULL DEFAULT 0,
            is_boe         INTEGER NOT NULL DEFAULT 0
        );
        CREATE TABLE market_observations_normalized (
            norm_id       INTEGER PRIMARY KEY AUTOINCREMENT,
            obs_id        INTEGER NOT NULL DEFAULT 1,
            item_id       INTEGER NOT NULL,
            archetype_id  INTEGER,
            realm_slug    TEXT    NOT NULL,
            faction       TEXT    NOT NULL DEFAULT 'neutral',
            observed_at   TEXT    NOT NULL,
            price_gold    REAL    NOT NULL,
            quantity_listed INTEGER,
            num_auctions  INTEGER,
            z_score       REAL,
            is_outlier    INTEGER NOT NULL DEFAULT 0,
            normalized_at TEXT    NOT NULL DEFAULT (datetime('now'))
        );
        CREATE TABLE forecast_outputs (
            forecast_id          INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id               INTEGER NOT NULL DEFAULT 1,
            archetype_id         INTEGER,
            item_id              INTEGER,
            realm_slug           TEXT    NOT NULL,
            forecast_horizon     TEXT    NOT NULL,
            target_date          TEXT    NOT NULL DEFAULT '2026-03-11',
            predicted_price_gold REAL    NOT NULL,
            confidence_lower     REAL    NOT NULL DEFAULT 0.0,
            confidence_upper     REAL    NOT NULL DEFAULT 0.0,
            confidence_pct       REAL    NOT NULL DEFAULT 0.80,
            model_slug           TEXT    NOT NULL DEFAULT 'lgbm',
            features_hash        TEXT,
            ci_quality           TEXT    NOT NULL DEFAULT 'good',
            created_at           TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now'))
        );
    """)
    db.commit()
    return db


# ── Helpers ───────────────────────────────────────────────────────────────────

def _insert_item(conn: sqlite3.Connection, item_id: int, name: str) -> None:
    conn.execute(
        "INSERT INTO items (item_id, name) VALUES (?, ?)",
        (item_id, name),
    )
    conn.commit()


def _insert_obs(
    conn: sqlite3.Connection,
    item_id: int,
    realm_slug: str,
    price_gold: float,
    days_ago: float = 0.5,
    is_outlier: int = 0,
) -> None:
    observed_at = (
        datetime.now(tz=timezone.utc) - timedelta(days=days_ago)
    ).strftime("%Y-%m-%dT%H:%M:%SZ")
    conn.execute(
        "INSERT INTO market_observations_normalized "
        "(item_id, realm_slug, price_gold, is_outlier, observed_at) "
        "VALUES (?, ?, ?, ?, ?)",
        (item_id, realm_slug, price_gold, is_outlier, observed_at),
    )
    conn.commit()


def _insert_forecast(
    conn: sqlite3.Connection,
    item_id: int,
    realm_slug: str,
    horizon: str,
    predicted_price: float,
    ci_quality: str = "good",
    ts_offset_seconds: int = 0,
) -> None:
    created_at = (
        datetime.now(tz=timezone.utc) + timedelta(seconds=ts_offset_seconds)
    ).strftime("%Y-%m-%dT%H:%M:%SZ")
    conn.execute(
        "INSERT INTO forecast_outputs "
        "(item_id, archetype_id, realm_slug, forecast_horizon, "
        "predicted_price_gold, ci_quality, created_at) "
        "VALUES (?, NULL, ?, ?, ?, ?, ?)",
        (item_id, realm_slug, horizon, predicted_price, ci_quality, created_at),
    )
    conn.commit()


# ── Tests: fetch_tsm_export_items ─────────────────────────────────────────────

class TestFetchTsmExportItems:
    def test_empty_when_no_forecasts(self, conn):
        _insert_item(conn, 100, "Herb A")
        _insert_obs(conn, 100, "us", 10.0)
        result = fetch_tsm_export_items(conn, "us", horizon="1d", min_roi_pct=0.10)
        assert result == []

    def test_empty_when_no_observations(self, conn):
        _insert_item(conn, 100, "Herb A")
        _insert_forecast(conn, 100, "us", "1d", 15.0)
        result = fetch_tsm_export_items(conn, "us", horizon="1d", min_roi_pct=0.10)
        assert result == []

    def test_includes_item_above_min_roi(self, conn):
        _insert_item(conn, 100, "Herb A")
        _insert_obs(conn, 100, "us", 100.0)
        _insert_forecast(conn, 100, "us", "1d", 120.0)  # 20% ROI
        result = fetch_tsm_export_items(conn, "us", horizon="1d", min_roi_pct=0.10)
        assert len(result) == 1
        assert result[0].item_id == 100
        assert abs(result[0].roi_pct - 0.20) < 1e-6

    def test_excludes_item_below_min_roi(self, conn):
        _insert_item(conn, 100, "Herb A")
        _insert_obs(conn, 100, "us", 100.0)
        _insert_forecast(conn, 100, "us", "1d", 105.0)  # 5% ROI
        result = fetch_tsm_export_items(conn, "us", horizon="1d", min_roi_pct=0.10)
        assert result == []

    def test_excludes_item_at_exact_threshold(self, conn):
        """Items with ROI exactly at the threshold are included (>=)."""
        _insert_item(conn, 100, "Herb A")
        _insert_obs(conn, 100, "us", 100.0)
        _insert_forecast(conn, 100, "us", "1d", 110.0)  # exactly 10% ROI
        result = fetch_tsm_export_items(conn, "us", horizon="1d", min_roi_pct=0.10)
        assert len(result) == 1

    def test_uses_most_recent_forecast(self, conn):
        """When two forecasts exist for the same item, uses MAX(created_at)."""
        _insert_item(conn, 100, "Herb A")
        _insert_obs(conn, 100, "us", 100.0)
        # Older forecast: 5% ROI (below threshold)
        _insert_forecast(conn, 100, "us", "1d", 105.0, ts_offset_seconds=-60)
        # Newer forecast: 20% ROI (above threshold)
        _insert_forecast(conn, 100, "us", "1d", 120.0, ts_offset_seconds=0)
        result = fetch_tsm_export_items(conn, "us", horizon="1d", min_roi_pct=0.10)
        assert len(result) == 1
        assert abs(result[0].roi_pct - 0.20) < 1e-6

    def test_excludes_forecast_with_bad_ci_quality(self, conn):
        """Forecasts with ci_quality != 'good' are excluded."""
        _insert_item(conn, 100, "Herb A")
        _insert_obs(conn, 100, "us", 100.0)
        _insert_forecast(conn, 100, "us", "1d", 150.0, ci_quality="clamp")
        result = fetch_tsm_export_items(conn, "us", horizon="1d", min_roi_pct=0.10)
        assert result == []

    def test_realm_isolation(self, conn):
        """Items with forecasts in a different realm are excluded."""
        _insert_item(conn, 100, "Herb A")
        _insert_obs(conn, 100, "eu", 100.0)
        _insert_forecast(conn, 100, "eu", "1d", 150.0)
        result = fetch_tsm_export_items(conn, "us", horizon="1d", min_roi_pct=0.10)
        assert result == []

    def test_horizon_isolation(self, conn):
        """Items with forecasts in a different horizon are excluded."""
        _insert_item(conn, 100, "Herb A")
        _insert_obs(conn, 100, "us", 100.0)
        _insert_forecast(conn, 100, "us", "7d", 150.0)
        result = fetch_tsm_export_items(conn, "us", horizon="1d", min_roi_pct=0.10)
        assert result == []

    def test_sorted_by_roi_descending(self, conn):
        """Results are sorted highest ROI first."""
        for item_id, price in [(1, 130.0), (2, 120.0), (3, 150.0)]:
            _insert_item(conn, item_id, f"Item {item_id}")
            _insert_obs(conn, item_id, "us", 100.0)
            _insert_forecast(conn, item_id, "us", "1d", price)
        result = fetch_tsm_export_items(conn, "us", horizon="1d", min_roi_pct=0.10)
        assert len(result) == 3
        assert result[0].item_id == 3  # 50% ROI
        assert result[1].item_id == 1  # 30% ROI
        assert result[2].item_id == 2  # 20% ROI

    def test_multiple_items_multiple_horizons(self, conn):
        """Requesting horizon=7d only returns 7d forecasts."""
        _insert_item(conn, 100, "Item A")
        _insert_obs(conn, 100, "us", 100.0)
        _insert_forecast(conn, 100, "us", "1d", 115.0)
        _insert_forecast(conn, 100, "us", "7d", 125.0)
        result_1d = fetch_tsm_export_items(conn, "us", horizon="1d", min_roi_pct=0.10)
        result_7d = fetch_tsm_export_items(conn, "us", horizon="7d", min_roi_pct=0.10)
        assert result_1d[0].horizon == "1d"
        assert result_7d[0].horizon == "7d"


# ── Tests: build_tsm_import_string ────────────────────────────────────────────

class TestBuildTsmImportString:
    def _row(self, item_id: int, roi: float = 0.20) -> TsmExportRow:
        return TsmExportRow(
            item_id=item_id,
            name=f"Item {item_id}",
            current_price=100.0,
            forecast_price=100.0 * (1 + roi),
            roi_pct=roi,
            horizon="1d",
        )

    def test_empty_list_returns_empty_string(self):
        assert build_tsm_import_string([]) == ""

    def test_single_item(self):
        result = build_tsm_import_string([self._row(12345)])
        assert result == "i:12345"

    def test_multiple_items_comma_separated(self):
        items = [self._row(100), self._row(200), self._row(300)]
        result = build_tsm_import_string(items)
        assert result == "i:100,i:200,i:300"

    def test_format_uses_integer_item_id(self):
        result = build_tsm_import_string([self._row(99999)])
        assert result.startswith("i:")
        assert "." not in result  # no decimal point


# ── Tests: write_tsm_export ───────────────────────────────────────────────────

class TestWriteTsmExport:
    def _row(self, item_id: int) -> TsmExportRow:
        return TsmExportRow(
            item_id=item_id,
            name=f"Item {item_id}",
            current_price=100.0,
            forecast_price=120.0,
            roi_pct=0.20,
            horizon="1d",
        )

    def test_writes_tsm_string_to_file(self, tmp_path: Path):
        items = [self._row(100), self._row(200)]
        out = tmp_path / "tsm_export.txt"
        result = write_tsm_export(items, out)
        assert result == out
        content = out.read_text(encoding="utf-8")
        assert "i:100,i:200" in content

    def test_creates_parent_directories(self, tmp_path: Path):
        items = [self._row(100)]
        out = tmp_path / "subdir" / "nested" / "tsm_export.txt"
        write_tsm_export(items, out)
        assert out.exists()

    def test_empty_list_writes_empty_file(self, tmp_path: Path):
        out = tmp_path / "tsm_empty.txt"
        write_tsm_export([], out)
        content = out.read_text(encoding="utf-8").strip()
        assert content == ""
