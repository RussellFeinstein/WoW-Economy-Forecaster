"""
Tests for wow_forecaster/pipeline/normalize.py.

What we test
------------
_fetch_rolling_stats() — reads the baseline from ``daily_rollup_item`` (issue #123):
  - Returns empty dict for empty item_ids set.
  - Returns empty dict when no rollup rows exist.
  - Excludes pairs whose summed obs_count is below _MIN_ROLLING_OBS.
  - Combines partial sums correctly across multiple days.
  - Excludes days outside the rolling window, against an injected clock.
  - Includes the cutoff day itself and excludes the day before it.
  - Excludes realms not in the batch's realm_slug set.
  - Clamps variance so std is never negative.
  - Matches the pre-#123 query and statistics ground truth on the same rows.
  - Still excludes outliers, now via the rollup UPSERT's own is_outlier filter.
  - Seeks the daily_rollup_item index and never reads the normalized table.

_check_rollup_freshness():
  - Silent on an empty rollup table (cold start is not a fault).
  - Warns when a batch realm has no rollup rows but other realms do.
  - Warns when a batch realm's newest rollup date is more than a day old.
  - Silent when the rollups are current.

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

import logging
import sqlite3
import statistics
from datetime import UTC, datetime, timedelta

import pytest

from wow_forecaster.db.rollup import upsert_item_rollup
from wow_forecaster.db.schema import apply_schema
from wow_forecaster.pipeline.normalize import (
    _MIN_ROLLING_OBS,
    _check_rollup_freshness,
    _fetch_archetype_map,
    _fetch_rolling_stats,
    _normalize_batch,
    _rolling_stats_sql,
)

# ── Fixtures ───────────────────────────────────────────────────────────────────

# Fixed reference clock. Every rolling-stats test injects it, so a run at any
# wall-clock time in any timezone exercises the same window.
_NOW = datetime(2026, 7, 31, 9, 0, 0, tzinfo=UTC)


def _date(days_ago: int) -> str:
    """UTC date string ``days_ago`` days before the reference clock."""
    return (_NOW - timedelta(days=days_ago)).date().isoformat()


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
    obs_date: str,
    is_outlier: bool = False,
    hour: int = 12,
) -> None:
    """Insert one row into market_observations_normalized on a given UTC date."""
    _obs_counter[0] += 1
    obs_id = _obs_counter[0]
    conn.execute(
        """
        INSERT INTO market_observations_normalized
            (obs_id, item_id, archetype_id, realm_slug, faction, observed_at,
             price_gold, z_score, is_outlier)
        VALUES (?, ?, NULL, ?, 'neutral', ?, ?, NULL, ?);
        """,
        (
            obs_id, item_id, realm_slug, f"{obs_date}T{hour:02d}:00:00",
            price_gold, 1 if is_outlier else 0,
        ),
    )
    conn.commit()


def _seed_rollup(
    conn: sqlite3.Connection,
    rows: list[tuple[int, str, float, str]],
    outliers: list[tuple[int, str, float, str]] | None = None,
) -> None:
    """Insert normalized rows, then build the rollups from them.

    Fixtures go through the real ``upsert_item_rollup()`` rather than
    hand-written daily_rollup_item rows, so a change to the rollup's own
    aggregation cannot silently drift away from what the baseline query reads.

    Args:
        conn:     Open connection with the schema applied.
        rows:     ``(item_id, realm_slug, price_gold, obs_date)`` non-outliers.
        outliers: Same shape, inserted with ``is_outlier = 1``.
    """
    affected: set[tuple[str, str]] = set()
    for item_id, realm_slug, price_gold, obs_date in rows:
        _insert_normalized(conn, item_id, realm_slug, price_gold, obs_date)
        affected.add((realm_slug, obs_date))
    for item_id, realm_slug, price_gold, obs_date in outliers or []:
        _insert_normalized(
            conn, item_id, realm_slug, price_gold, obs_date, is_outlier=True
        )
        affected.add((realm_slug, obs_date))
    for realm_slug, obs_date in sorted(affected):
        upsert_item_rollup(conn, realm_slug, obs_date)


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
        result = _fetch_rolling_stats(norm_db, set(), {"us"}, 30, now=_NOW)
        assert result == {}

    def test_no_rollup_rows_returns_empty(self, norm_db):
        result = _fetch_rolling_stats(norm_db, {1001}, {"us"}, 30, now=_NOW)
        assert result == {}

    def test_insufficient_obs_count_excluded(self, norm_db):
        # One observation on one day → summed obs_count of 1, below the minimum.
        _seed_rollup(norm_db, [(1001, "us", 100.0, _date(2))])
        result = _fetch_rolling_stats(norm_db, {1001}, {"us"}, 30, now=_NOW)
        assert result == {}
        assert _MIN_ROLLING_OBS == 2

    def test_stats_combine_across_days(self, norm_db):
        """Partial sums from separate rollup rows add up to the pooled statistic."""
        prices_by_day = {_date(2): [100.0, 120.0], _date(3): [140.0, 90.0, 110.0]}
        _seed_rollup(norm_db, [
            (1001, "us", price, day)
            for day, prices in prices_by_day.items()
            for price in prices
        ])
        result = _fetch_rolling_stats(norm_db, {1001}, {"us"}, 30, now=_NOW)
        all_prices = [p for prices in prices_by_day.values() for p in prices]
        mean_p, std_p = result[(1001, "us")]
        assert mean_p == pytest.approx(statistics.fmean(all_prices))
        assert std_p == pytest.approx(statistics.pstdev(all_prices))

    def test_days_outside_window_excluded(self, norm_db):
        _seed_rollup(norm_db, [
            (1001, "us", 100.0, _date(2)),
            (1001, "us", 200.0, _date(3)),
            (1001, "us", 999_999.0, _date(60)),  # outside a 30-day window
        ])
        result = _fetch_rolling_stats(norm_db, {1001}, {"us"}, 30, now=_NOW)
        mean_p, _ = result[(1001, "us")]
        assert mean_p == pytest.approx(150.0)

    def test_cutoff_day_included_and_prior_day_excluded(self, norm_db):
        """The window edge is a calendar day and is inclusive at the cutoff."""
        _seed_rollup(norm_db, [
            (1001, "us", 100.0, _date(30)),
            (1001, "us", 100.0, _date(30)),
            (1001, "us", 900.0, _date(31)),
            (1001, "us", 900.0, _date(31)),
        ])
        result = _fetch_rolling_stats(norm_db, {1001}, {"us"}, 30, now=_NOW)
        mean_p, _ = result[(1001, "us")]
        assert mean_p == pytest.approx(100.0)

    def test_realm_slug_filter(self, norm_db):
        _seed_rollup(norm_db, [
            (1001, "eu", 100.0, _date(2)),
            (1001, "eu", 200.0, _date(3)),
        ])
        # realm_slugs param only contains "us" — "eu" rows should be filtered
        result = _fetch_rolling_stats(norm_db, {1001}, {"us"}, 30, now=_NOW)
        assert result == {}

    def test_std_is_nonnegative(self, norm_db):
        # Identical prices → variance should be 0 (not negative due to float)
        _seed_rollup(norm_db, [(1001, "us", 100.0, _date(i + 1)) for i in range(3)])
        result = _fetch_rolling_stats(norm_db, {1001}, {"us"}, 30, now=_NOW)
        _, std_p = result[(1001, "us")]
        assert std_p >= 0.0

    def test_outlier_rows_excluded_by_the_rollup(self, norm_db):
        """The is_outlier filter still applies, one layer earlier than before.

        Nothing in _fetch_rolling_stats mentions is_outlier now. The exclusion
        happens in _UPSERT_ITEM_SQL, so this test is what proves the semantic
        survived the move.
        """
        _seed_rollup(
            norm_db,
            rows=[(1001, "us", 100.0, _date(2)), (1001, "us", 100.0, _date(3))],
            outliers=[(1001, "us", 999_999.0, _date(4))],
        )
        result = _fetch_rolling_stats(norm_db, {1001}, {"us"}, 30, now=_NOW)
        mean_p, _ = result[(1001, "us")]
        assert mean_p == pytest.approx(100.0)


# ── Parity with the pre-#123 implementation ───────────────────────────────────

# The query _fetch_rolling_stats used before issue #123, kept here only as a
# parity reference. The literal cutoff replaces datetime('now', '-N days') so
# both paths cover exactly the same rows and the comparison is about the
# aggregation rather than the documented window-edge shift.
_LEGACY_ROLLING_STATS_SQL = """
    SELECT item_id, realm_slug,
           AVG(price_gold)                                          AS mean_p,
           AVG(price_gold * price_gold) - AVG(price_gold) * AVG(price_gold)
                                                                    AS variance,
           COUNT(*)                                                 AS n
    FROM market_observations_normalized
    WHERE item_id IN (?, ?)
      AND is_outlier = 0
      AND observed_at >= ?
    GROUP BY item_id, realm_slug
    HAVING COUNT(*) >= 2;
"""


class TestRollingStatsParity:
    def test_matches_legacy_query_and_ground_truth(self, norm_db):
        prices = {
            (1001, _date(2)): [100.0, 120.0, 118.5],
            (1001, _date(3)): [140.0, 90.0],
            (1001, _date(9)): [101.25],
            (1002, _date(2)): [7.5, 8.25],
            (1002, _date(5)): [9.0, 8.75, 8.5, 9.25],
        }
        _seed_rollup(
            norm_db,
            rows=[
                (item_id, "us", price, day)
                for (item_id, day), day_prices in prices.items()
                for price in day_prices
            ],
            outliers=[(1001, "us", 999_999.0, _date(4))],
        )

        new = _fetch_rolling_stats(norm_db, {1001, 1002}, {"us"}, 30, now=_NOW)

        legacy_rows = norm_db.execute(
            _LEGACY_ROLLING_STATS_SQL, (1001, 1002, _date(30))
        ).fetchall()
        legacy = {
            (row["item_id"], row["realm_slug"]): (
                float(row["mean_p"]), float(max(row["variance"] or 0.0, 0.0) ** 0.5)
            )
            for row in legacy_rows
        }

        assert set(new) == set(legacy) == {(1001, "us"), (1002, "us")}
        for key, (mean_p, std_p) in new.items():
            assert mean_p == pytest.approx(legacy[key][0])
            assert std_p == pytest.approx(legacy[key][1])

        for item_id in (1001, 1002):
            truth = [
                price
                for (seeded_item, _day), day_prices in prices.items()
                if seeded_item == item_id
                for price in day_prices
            ]
            mean_p, std_p = new[(item_id, "us")]
            assert mean_p == pytest.approx(statistics.fmean(truth))
            assert std_p == pytest.approx(statistics.pstdev(truth))


# ── Query plan ────────────────────────────────────────────────────────────────

class TestRollingStatsQueryPlan:
    """The whole point of #123 is which table the prefetch reads.

    A correctness test cannot tell the two implementations apart once both
    return the same numbers, so the plan is pinned the same way migration 0009
    pins the health-check query shapes.
    """

    @staticmethod
    def _plan(conn: sqlite3.Connection) -> str:
        sql = _rolling_stats_sql(2)
        rows = conn.execute(
            f"EXPLAIN QUERY PLAN {sql}", (1001, 1002, _date(30))
        ).fetchall()
        return " | ".join(str(row[-1]) for row in rows)

    def test_seeks_the_rollup_index(self, norm_db):
        _seed_rollup(norm_db, [
            (1001, "us", 100.0, _date(2)),
            (1001, "us", 120.0, _date(3)),
            (1002, "us", 8.0, _date(2)),
        ])
        plan = self._plan(norm_db)
        assert "daily_rollup_item" in plan
        assert "SEARCH" in plan, f"Expected an index seek, got: {plan}"
        assert "SCAN daily_rollup_item" not in plan, plan

    def test_never_reads_the_normalized_table(self, norm_db):
        _seed_rollup(norm_db, [
            (1001, "us", 100.0, _date(2)),
            (1001, "us", 120.0, _date(3)),
        ])
        assert "market_observations_normalized" not in self._plan(norm_db)


# ── _check_rollup_freshness ───────────────────────────────────────────────────

class TestCheckRollupFreshness:
    def test_empty_rollup_table_does_not_warn(self, norm_db, caplog):
        """A cold start is not a fault, so it must not warn on every run."""
        with caplog.at_level(logging.WARNING):
            _check_rollup_freshness(norm_db, {"us"}, now=_NOW)
        assert caplog.records == []

    def test_current_rollups_do_not_warn(self, norm_db, caplog):
        _seed_rollup(norm_db, [
            (1001, "us", 100.0, _date(1)),
            (1001, "us", 120.0, _date(0)),
        ])
        with caplog.at_level(logging.WARNING):
            _check_rollup_freshness(norm_db, {"us"}, now=_NOW)
        assert caplog.records == []

    def test_yesterday_is_still_fresh(self, norm_db, caplog):
        """Rollups upsert the previous and current UTC dates, so a newest date
        of yesterday is the normal state right after UTC midnight."""
        _seed_rollup(norm_db, [
            (1001, "us", 100.0, _date(2)),
            (1001, "us", 120.0, _date(1)),
        ])
        with caplog.at_level(logging.WARNING):
            _check_rollup_freshness(norm_db, {"us"}, now=_NOW)
        assert caplog.records == []

    def test_stale_realm_warns(self, norm_db, caplog):
        _seed_rollup(norm_db, [
            (1001, "us", 100.0, _date(6)),
            (1001, "us", 120.0, _date(5)),
        ])
        with caplog.at_level(logging.WARNING):
            _check_rollup_freshness(norm_db, {"us"}, now=_NOW)
        assert len(caplog.records) == 1
        assert "us" in caplog.records[0].getMessage()

    def test_realm_missing_from_populated_table_warns(self, norm_db, caplog):
        _seed_rollup(norm_db, [
            (1001, "eu", 100.0, _date(1)),
            (1001, "eu", 120.0, _date(0)),
        ])
        with caplog.at_level(logging.WARNING):
            _check_rollup_freshness(norm_db, {"us"}, now=_NOW)
        assert len(caplog.records) == 1
        assert "us" in caplog.records[0].getMessage()

    def test_no_realms_is_a_no_op(self, norm_db, caplog):
        with caplog.at_level(logging.WARNING):
            _check_rollup_freshness(norm_db, set(), now=_NOW)
        assert caplog.records == []


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

    def test_archetype_id_populated_from_map(self):
        """archetype_id is set from archetype_map when provided."""
        row = _make_raw_row(obs_id=1, item_id=1001, realm_slug="us")
        archetype_map = {1001: 42}
        normalized, _ = _normalize_batch([row], z_threshold=3.0,
                                          rolling_stats=None, archetype_map=archetype_map)
        assert normalized[0].archetype_id == 42

    def test_archetype_id_none_for_unmapped_item(self):
        """Items absent from archetype_map keep archetype_id = None."""
        row = _make_raw_row(obs_id=1, item_id=9999, realm_slug="us")
        archetype_map = {1001: 42}  # 9999 not in map
        normalized, _ = _normalize_batch([row], z_threshold=3.0,
                                          rolling_stats=None, archetype_map=archetype_map)
        assert normalized[0].archetype_id is None

    def test_archetype_id_none_when_map_not_provided(self):
        """Omitting archetype_map preserves pre-v1.3.4 NULL behaviour."""
        row = _make_raw_row(obs_id=1, item_id=1001, realm_slug="us")
        normalized, _ = _normalize_batch([row], z_threshold=3.0, rolling_stats=None)
        assert normalized[0].archetype_id is None

    def test_single_obs_cold_start_z_score_none(self):
        """Single observation with no rolling history → z_score is None, not an outlier."""
        row = _make_raw_row(obs_id=1, item_id=1001, realm_slug="us",
                            min_buyout_raw=1_000_000)
        # Empty rolling_stats (not None): triggers the cold-start path,
        # item has 1 valid price so std_p = 0.0 → z_score = None.
        normalized, _ = _normalize_batch([row], z_threshold=3.0, rolling_stats={})
        assert normalized[0].z_score is None
        assert not normalized[0].is_outlier


# ── _fetch_archetype_map ───────────────────────────────────────────────────────

class TestFetchArchetypeMap:
    def test_empty_item_ids_returns_empty(self, norm_db):
        result = _fetch_archetype_map(norm_db, set())
        assert result == {}

    def test_returns_archetype_id_for_known_item(self, norm_db):
        norm_db.execute(
            "INSERT INTO item_categories (slug, display_name, archetype_tag) "
            "VALUES ('test.cat', 'Test', 'test.tag');"
        )
        norm_db.execute(
            "INSERT INTO economic_archetypes "
            "(archetype_id, slug, display_name, category_tag, sub_tag, "
            "is_transferable, transfer_confidence) "
            "VALUES (7, 'test.arch', 'Test Arch', 'test', 'test.tag', 1, 0.9);"
        )
        norm_db.execute(
            "INSERT INTO items (item_id, name, category_id, archetype_id, "
            "expansion_slug, quality, is_crafted, is_boe) "
            "VALUES (1001, 'Test Item', 1, 7, 'tww', 'common', 0, 0);"
        )
        norm_db.commit()
        result = _fetch_archetype_map(norm_db, {1001})
        assert result == {1001: 7}

    def test_returns_none_for_item_without_archetype(self, norm_db):
        norm_db.execute(
            "INSERT INTO item_categories (slug, display_name, archetype_tag) "
            "VALUES ('test.cat', 'Test', 'test.tag');"
        )
        norm_db.execute(
            "INSERT INTO items (item_id, name, category_id, archetype_id, "
            "expansion_slug, quality, is_crafted, is_boe) "
            "VALUES (2002, 'No Arch Item', 1, NULL, 'tww', 'common', 0, 0);"
        )
        norm_db.commit()
        result = _fetch_archetype_map(norm_db, {2002})
        assert result == {2002: None}

    def test_unknown_item_absent_from_result(self, norm_db):
        """Items not in the items table are simply absent (not mapped to None)."""
        result = _fetch_archetype_map(norm_db, {9999})
        assert 9999 not in result


# ── NormalizeStage wiring ─────────────────────────────────────────────────────

class TestNormalizeStageUsesRollupBaseline:
    """End to end proof that _execute() reaches the rollup baseline.

    Runs against a real file DB with foreign keys on, which is what the stage
    itself opens. A single pending raw row cannot produce a z-score from batch
    statistics (one price gives std 0, so the z-score is None), so a computed
    z-score can only have come from the rollup.
    """

    def test_z_score_comes_from_the_rollup(self, tmp_path):
        from wow_forecaster.config import AppConfig, DatabaseConfig
        from wow_forecaster.models.meta import RunMetadata
        from wow_forecaster.pipeline.normalize import NormalizeStage
        from wow_forecaster.utils.time_utils import utcnow

        db_file = str(tmp_path / "wiring.db")
        now = utcnow()
        history_days = [
            (now - timedelta(days=2)).date().isoformat(),
            (now - timedelta(days=1)).date().isoformat(),
        ]

        conn = sqlite3.connect(db_file)
        conn.row_factory = sqlite3.Row
        apply_schema(conn)
        conn.execute(
            "INSERT INTO item_categories (slug, display_name, archetype_tag) "
            "VALUES ('test.cat', 'Test', 'test.tag');"
        )
        conn.execute(
            "INSERT INTO items (item_id, name, category_id, archetype_id, "
            "expansion_slug, quality, is_crafted, is_boe) "
            "VALUES (1001, 'Test Item', 1, NULL, 'tww', 'common', 0, 0);"
        )
        # History: prices 50 and 150 → mean 100, population std 50.
        history = zip([50.0, 150.0], history_days, strict=True)
        for obs_id, (price, day) in enumerate(history, start=1):
            conn.execute(
                "INSERT INTO market_observations_raw "
                "(obs_id, item_id, realm_slug, observed_at, source, "
                " min_buyout_raw, is_processed) "
                "VALUES (?, 1001, 'us', ?, 'blizzard_api', ?, 1);",
                (obs_id, f"{day}T12:00:00", int(price * 10_000)),
            )
            conn.execute(
                "INSERT INTO market_observations_normalized "
                "(obs_id, item_id, realm_slug, observed_at, price_gold, is_outlier) "
                "VALUES (?, 1001, 'us', ?, ?, 0);",
                (obs_id, f"{day}T12:00:00", price),
            )
        for day in history_days:
            upsert_item_rollup(conn, "us", day)
        # The one pending row: 150 gold against a mean of 100 and std of 50.
        conn.execute(
            "INSERT INTO market_observations_raw "
            "(obs_id, item_id, realm_slug, observed_at, source, "
            " min_buyout_raw, is_processed) "
            "VALUES (99, 1001, 'us', ?, 'blizzard_api', 1500000, 0);",
            (now.strftime("%Y-%m-%dT%H:%M:%S"),),
        )
        conn.commit()
        conn.close()

        config = AppConfig(database=DatabaseConfig(db_path=db_file))
        stage = NormalizeStage(config=config, db_path=db_file)
        written = stage._execute(run=RunMetadata(
            run_slug="test-rollup-baseline",
            pipeline_stage="normalize",
            config_snapshot={},
            started_at=utcnow(),
        ))

        assert written == 1
        verify = sqlite3.connect(db_file)
        verify.row_factory = sqlite3.Row
        row = verify.execute(
            "SELECT z_score FROM market_observations_normalized WHERE obs_id = 99;"
        ).fetchone()
        verify.close()
        assert row["z_score"] == pytest.approx(1.0)
