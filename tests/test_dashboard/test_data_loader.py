"""Tests for dashboard.data_loader.load_ingest_age_hours (issue #12).

Importing dashboard.data_loader here also exercises the no-streamlit
_CACHE fallback: neither the local dev venv nor CI installs the optional
[dashboard] dependency group, so a fallback that rejects @_CACHE(ttl=N)
makes the module unimportable (this was a latent bug until issue #12).
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta

import pytest

from dashboard.data_loader import load_ingest_age_hours
from wow_forecaster.db.schema import apply_schema

NOW = datetime(2026, 4, 1, 12, 0, 0, tzinfo=UTC)


def _make_db(tmp_path) -> str:
    db_file = str(tmp_path / "test.db")
    conn = sqlite3.connect(db_file)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    apply_schema(conn)
    conn.commit()
    conn.close()
    return db_file


def _insert_observation(
    db_file: str,
    observed_at: datetime,
    is_outlier: int = 0,
    item_id: int = 100,
) -> None:
    ts = observed_at.strftime("%Y-%m-%dT%H:%M:%SZ")
    conn = sqlite3.connect(db_file)
    try:
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute(
            "INSERT OR IGNORE INTO item_categories (slug, display_name, archetype_tag) "
            "VALUES ('test.cat', 'Test', 'test');"
        )
        cat_id = conn.execute(
            "SELECT category_id FROM item_categories WHERE slug='test.cat';"
        ).fetchone()[0]
        conn.execute(
            "INSERT OR IGNORE INTO items "
            "(item_id, name, category_id, expansion_slug, quality) "
            "VALUES (?, 'Item', ?, 'midnight', 'common');",
            (item_id, cat_id),
        )
        obs_id = conn.execute(
            "INSERT INTO market_observations_raw "
            "(item_id, realm_slug, faction, observed_at, source, is_processed) "
            "VALUES (?, 'us', 'neutral', ?, 'test', 1) RETURNING obs_id;",
            (item_id, ts),
        ).fetchone()[0]
        conn.execute(
            "INSERT INTO market_observations_normalized "
            "(obs_id, item_id, realm_slug, observed_at, price_gold, "
            " quantity_listed, is_outlier) "
            "VALUES (?, ?, 'us', ?, 10.0, 100, ?);",
            (obs_id, item_id, ts, is_outlier),
        )
        conn.commit()
    finally:
        conn.close()


class TestLoadIngestAgeHours:
    def test_returns_age_of_newest_observation(self, tmp_path):
        db_file = _make_db(tmp_path)
        _insert_observation(db_file, NOW - timedelta(hours=30), item_id=100)
        _insert_observation(db_file, NOW - timedelta(hours=3), item_id=101)
        assert load_ingest_age_hours(db_file, now=NOW) == pytest.approx(3.0)

    def test_outlier_rows_are_ignored(self, tmp_path):
        db_file = _make_db(tmp_path)
        _insert_observation(db_file, NOW - timedelta(hours=30), item_id=100)
        _insert_observation(
            db_file, NOW - timedelta(hours=1), is_outlier=1, item_id=101
        )
        assert load_ingest_age_hours(db_file, now=NOW) == pytest.approx(30.0)

    def test_none_when_no_observations(self, tmp_path):
        db_file = _make_db(tmp_path)
        assert load_ingest_age_hours(db_file, now=NOW) is None

    def test_none_when_db_missing(self, tmp_path):
        missing = str(tmp_path / "nope" / "missing.db")
        assert load_ingest_age_hours(missing, now=NOW) is None
