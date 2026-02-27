"""
Tests for wow_forecaster/pipeline/normalize.py.

What we test
------------
_fetch_rolling_stats():
  - Returns empty dict for empty item_ids set.
  - Returns mean/std for items with sufficient history (>= _MIN_ROLLING_OBS).
  - Excludes outlier rows from rolling stats.
  - Excludes rows outside the rolling window.
  - Excludes realms not in the batch's realm_slug set.
  - Returns empty dict when no history exists.

_normalize_batch():
  - Uses rolling stats when available (not batch stats).
  - Falls back to batch stats when rolling_stats is None.
  - Falls back to batch stats when item not in rolling_stats.
  - z_score is None when std_p is 0 (single price, no variance).
  - Outlier flag set when |z_score| > threshold.
  - price_gold = 0.0 when min_buyout_raw is NULL.
  - market/historical gold conversion.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

import pytest

from wow_forecaster.db.schema import apply_schema
from wow_forecaster.pipeline.normalize import _MIN_ROLLING_OBS, _fetch_rolling_stats, _normalize_batch


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def norm_db() -> sqlite3.Connection:
    """In-memory DB with schema; foreign keys OFF for easy raw-obs insertion."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = OFF;")
    apply_schema(conn)
    return conn


_obs_counter: list[int] = [0]


def _insert_normalized(
    conn: sqlite3.Connection,
    item_id: int,
    realm_slug: str,
    price_gold: float,
    is_outlier: bool = False,
    days_ago: int = 1,
) -> None:
    """Insert a row into market_observations_normalized for testing."""
    _obs_counter[0] += 1
    obs_id = _obs_counter[0]
    conn.execute(
        f"""
        INSERT INTO market_observations_normalized
            (obs_id, item_id, archetype_id, realm_slug, faction, observed_at,
             price_gold, z_score, is_outlier)
        VALUES (?, ?, NULL, ?, 'neutral',
                datetime('now', '-{days_ago} days'),
                ?, NULL, ?);
        """,
        (obs_id, item_id, realm_slug, price_gold, 1 if is_outlier else 0),
    )
    conn.commit()


def _make_raw_row(
    obs_id: int,
    item_id: int,
    realm_slug: str = "us",
    min_buyout_raw: int | None = 1_000_000,  # 100 gold
) -> sqlite3.Row:
    """Build a sqlite3.Row-like dict masquerading as a raw observation row."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute(
        "CREATE TABLE t (obs_id, item_id, realm_slug, faction, observed_at, "
        "source, min_buyout_raw, market_value_raw, historical_value_raw, "
        "quantity_listed, num_auctions);"
    )
    conn.execute(
        "INSERT INTO t VALUES (?, ?, ?, 'neutral', '2024-09-15T12:00:00', "
        "'blizzard_api', ?, NULL, NULL, NULL, NULL);",
        (obs_id, item_id, realm_slug, min_buyout_raw),
    )
    return conn.execute("SELECT * FROM t WHERE obs_id = ?;", (obs_id,)).fetchone()


# ── _fetch_rolling_stats ───────────────────────────────────────────────────────

class TestFetchRollingStats:
    def test_empty_item_ids_returns_empty(self, norm_db):
        result = _fetch_rolling_stats(norm_db, set(), {"us"}, 30)
        assert result == {}

    def test_no_history_returns_empty(self, norm_db):
        result = _fetch_rolling_stats(norm_db, {1001}, {"us"}, 30)
        assert result == {}

    def test_insufficient_history_excluded(self, norm_db):
        # Only 1 row — below _MIN_ROLLING_OBS (2)
        _insert_normalized(norm_db, 1001, "us", 100.0)
        result = _fetch_rolling_stats(norm_db, {1001}, {"us"}, 30)
        assert result == {}

    def test_sufficient_history_returns_stats(self, norm_db):
        _insert_normalized(norm_db, 1001, "us", 100.0, days_ago=2)
        _insert_normalized(norm_db, 1001, "us", 200.0, days_ago=3)
        result = _fetch_rolling_stats(norm_db, {1001}, {"us"}, 30)
        assert (1001, "us") in result
        mean_p, std_p = result[(1001, "us")]
        assert mean_p == pytest.approx(150.0)
        assert std_p > 0.0

    def test_outlier_rows_excluded_from_stats(self, norm_db):
        _insert_normalized(norm_db, 1001, "us", 100.0, is_outlier=False, days_ago=2)
        _insert_normalized(norm_db, 1001, "us", 100.0, is_outlier=False, days_ago=3)
        # This outlier spike should NOT affect mean/std
        _insert_normalized(norm_db, 1001, "us", 999_999.0, is_outlier=True, days_ago=4)
        result = _fetch_rolling_stats(norm_db, {1001}, {"us"}, 30)
        mean_p, _ = result[(1001, "us")]
        assert mean_p == pytest.approx(100.0)

    def test_rows_outside_window_excluded(self, norm_db):
        _insert_normalized(norm_db, 1001, "us", 100.0, days_ago=2)
        _insert_normalized(norm_db, 1001, "us", 200.0, days_ago=3)
        # 60 days old — outside a 30-day window
        _insert_normalized(norm_db, 1001, "us", 999_999.0, days_ago=60)
        result = _fetch_rolling_stats(norm_db, {1001}, {"us"}, 30)
        mean_p, _ = result[(1001, "us")]
        assert mean_p == pytest.approx(150.0)

    def test_realm_slug_filter(self, norm_db):
        _insert_normalized(norm_db, 1001, "eu", 100.0, days_ago=2)
        _insert_normalized(norm_db, 1001, "eu", 200.0, days_ago=3)
        # realm_slugs param only contains "us" — "eu" rows should be filtered
        result = _fetch_rolling_stats(norm_db, {1001}, {"us"}, 30)
        assert result == {}

    def test_std_is_nonnegative(self, norm_db):
        # Identical prices → variance should be 0 (not negative due to float)
        for i in range(3):
            _insert_normalized(norm_db, 1001, "us", 100.0, days_ago=i + 1)
        result = _fetch_rolling_stats(norm_db, {1001}, {"us"}, 30)
        _, std_p = result[(1001, "us")]
        assert std_p >= 0.0


# ── _normalize_batch ──────────────────────────────────────────────────────────

class TestNormalizeBatch:
    def test_uses_rolling_stats_when_available(self):
        row = _make_raw_row(obs_id=1, item_id=1001, realm_slug="us",
                            min_buyout_raw=1_500_000)  # 150 gold
        # Rolling history: mean=100, std=50
        rolling = {(1001, "us"): (100.0, 50.0)}
        normalized, obs_ids = _normalize_batch([row], z_threshold=3.0, rolling_stats=rolling)
        assert len(normalized) == 1
        obs = normalized[0]
        # z = (150 - 100) / 50 = 1.0
        assert obs.z_score == pytest.approx(1.0)
        assert not obs.is_outlier

    def test_falls_back_to_batch_stats_when_rolling_is_none(self):
        rows = [
            _make_raw_row(obs_id=1, item_id=1001, realm_slug="us", min_buyout_raw=1_000_000),
            _make_raw_row(obs_id=2, item_id=1001, realm_slug="us", min_buyout_raw=3_000_000),
        ]
        normalized, _ = _normalize_batch(rows, z_threshold=3.0, rolling_stats=None)
        # Batch mean=200, std=100 → z for row1 = (100-200)/100 = -1.0
        assert normalized[0].z_score == pytest.approx(-1.0)
        assert normalized[1].z_score == pytest.approx(1.0)

    def test_falls_back_to_batch_when_item_not_in_rolling(self):
        rows = [
            _make_raw_row(obs_id=1, item_id=1001, realm_slug="us", min_buyout_raw=1_000_000),
            _make_raw_row(obs_id=2, item_id=1001, realm_slug="us", min_buyout_raw=3_000_000),
        ]
        # Rolling stats exist for a different item
        rolling = {(9999, "us"): (500.0, 50.0)}
        normalized, _ = _normalize_batch(rows, z_threshold=3.0, rolling_stats=rolling)
        # Still uses batch stats for item 1001
        assert normalized[0].z_score == pytest.approx(-1.0)

    def test_outlier_flagged_when_z_exceeds_threshold(self):
        row = _make_raw_row(obs_id=1, item_id=1001, realm_slug="us",
                            min_buyout_raw=5_000_000)  # 500 gold
        # Rolling: mean=100, std=50 → z = (500-100)/50 = 8.0 > 3.0
        rolling = {(1001, "us"): (100.0, 50.0)}
        normalized, _ = _normalize_batch([row], z_threshold=3.0, rolling_stats=rolling)
        assert normalized[0].z_score == pytest.approx(8.0)
        assert normalized[0].is_outlier

    def test_z_score_none_when_std_is_zero(self):
        row = _make_raw_row(obs_id=1, item_id=1001, realm_slug="us",
                            min_buyout_raw=1_000_000)
        # Rolling: std=0 (all identical prices)
        rolling = {(1001, "us"): (100.0, 0.0)}
        normalized, _ = _normalize_batch([row], z_threshold=3.0, rolling_stats=rolling)
        assert normalized[0].z_score is None
        assert not normalized[0].is_outlier

    def test_null_min_buyout_gives_zero_price_gold(self):
        row = _make_raw_row(obs_id=1, item_id=1001, realm_slug="us",
                            min_buyout_raw=None)
        normalized, _ = _normalize_batch([row], z_threshold=3.0, rolling_stats=None)
        assert normalized[0].price_gold == pytest.approx(0.0)
        assert normalized[0].z_score is None

    def test_price_gold_copper_conversion(self):
        row = _make_raw_row(obs_id=1, item_id=1001, realm_slug="us",
                            min_buyout_raw=500_000)  # 50 gold
        normalized, _ = _normalize_batch([row], z_threshold=3.0, rolling_stats=None)
        assert normalized[0].price_gold == pytest.approx(50.0)

    def test_obs_ids_returned_correctly(self):
        rows = [
            _make_raw_row(obs_id=10, item_id=1001, realm_slug="us"),
            _make_raw_row(obs_id=20, item_id=1002, realm_slug="us"),
        ]
        _, obs_ids = _normalize_batch(rows, z_threshold=3.0, rolling_stats=None)
        assert sorted(obs_ids) == [10, 20]
