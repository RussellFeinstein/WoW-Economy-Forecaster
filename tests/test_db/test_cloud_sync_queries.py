"""Tests for the two repository queries the cloud catch-up path added (issue #43).

``get_covered_hours`` decides whether an hour is already in the database and
``get_ingested_paths_since`` decides whether an object has already been drained.
Both are the difference between an idempotent re-run and a double-count, so they
are pinned directly rather than only through the stage.

The EXPLAIN assertion checks the *seek term*, not the index name.  An index-name
assertion would stay green if the predicate were rewritten as
``DATE(observed_at) = ?``, because that form still uses the index and simply
stops seeking on the column (the trap found on issue #65).
"""

from __future__ import annotations

import sqlite3
from datetime import datetime

import pytest

from wow_forecaster.db.repositories.ingestion_repo import (
    IngestionSnapshot,
    IngestionSnapshotRepository,
)
from wow_forecaster.db.repositories.market_repo import MarketObservationRepository

WINDOW_START = datetime(2026, 7, 20, 0, 0, 0)
WINDOW_END = datetime(2026, 7, 24, 0, 0, 0)

COVERED_HOURS_SQL = (
    "SELECT DISTINCT substr(observed_at, 1, 13) AS hour_key "
    "FROM market_observations_raw "
    "WHERE observed_at >= '2026-07-20T00:00:00' "
    "  AND observed_at <  '2026-07-24T00:00:00' "
    "  AND source = 'blizzard_api'"
)


def _seed_item(conn: sqlite3.Connection, item_id: int = 190_001) -> int:
    conn.execute(
        "INSERT OR IGNORE INTO item_categories (slug, display_name, archetype_tag) "
        "VALUES ('test.cat', 'Test', 'test');"
    )
    cat_id = conn.execute(
        "SELECT category_id FROM item_categories WHERE slug='test.cat';"
    ).fetchone()[0]
    conn.execute(
        "INSERT OR IGNORE INTO items (item_id, name, category_id, expansion_slug, quality) "
        "VALUES (?, 'Item', ?, 'midnight', 'common');",
        (item_id, cat_id),
    )
    return item_id


def _insert_obs(conn, observed_at: str, source: str = "blizzard_api") -> None:
    item_id = _seed_item(conn)
    conn.execute(
        "INSERT INTO market_observations_raw "
        "(item_id, realm_slug, faction, observed_at, source) "
        "VALUES (?, 'us', 'neutral', ?, ?);",
        (item_id, observed_at, source),
    )


class TestGetCoveredHours:
    def test_empty_table_covers_nothing(self, in_memory_db):
        repo = MarketObservationRepository(in_memory_db)
        assert repo.get_covered_hours("blizzard_api", WINDOW_START, WINDOW_END) == set()

    def test_truncates_observations_to_their_hour(self, in_memory_db):
        _insert_obs(in_memory_db, "2026-07-22T14:16:33")
        repo = MarketObservationRepository(in_memory_db)
        hours = repo.get_covered_hours("blizzard_api", WINDOW_START, WINDOW_END)
        assert hours == {datetime(2026, 7, 22, 14, 0, 0)}

    def test_collapses_many_observations_in_one_hour(self, in_memory_db):
        for minute in ("05", "16", "46"):
            _insert_obs(in_memory_db, f"2026-07-22T14:{minute}:00")
        repo = MarketObservationRepository(in_memory_db)
        assert len(repo.get_covered_hours("blizzard_api", WINDOW_START, WINDOW_END)) == 1

    def test_excludes_other_sources(self, in_memory_db):
        _insert_obs(in_memory_db, "2026-07-22T14:16:00", source="tsm_export")
        repo = MarketObservationRepository(in_memory_db)
        assert repo.get_covered_hours("blizzard_api", WINDOW_START, WINDOW_END) == set()

    def test_bounds_are_half_open(self, in_memory_db):
        _insert_obs(in_memory_db, WINDOW_START.isoformat())
        _insert_obs(in_memory_db, WINDOW_END.isoformat())
        repo = MarketObservationRepository(in_memory_db)
        hours = repo.get_covered_hours("blizzard_api", WINDOW_START, WINDOW_END)
        assert hours == {WINDOW_START}

    def test_handles_microsecond_timestamps(self, in_memory_db):
        _insert_obs(in_memory_db, "2026-07-22T14:16:33.123456")
        repo = MarketObservationRepository(in_memory_db)
        hours = repo.get_covered_hours("blizzard_api", WINDOW_START, WINDOW_END)
        assert hours == {datetime(2026, 7, 22, 14, 0, 0)}

    def test_query_seeks_on_observed_at(self, in_memory_db):
        """Pin the seek term: DATE(observed_at) in the predicate would lose it."""
        plan = " ".join(
            row[3]
            for row in in_memory_db.execute(
                f"EXPLAIN QUERY PLAN {COVERED_HOURS_SQL}"
            ).fetchall()
        )
        assert "observed_at>" in plan


class TestGetIngestedPathsSince:
    def _snapshot(self, path: str, fetched_at: datetime, success: bool = True):
        return IngestionSnapshot(
            snapshot_id=None,
            run_id=None,
            source="blizzard_api",
            endpoint="r2://bucket/key",
            snapshot_path=path,
            content_hash="abc",
            record_count=1,
            success=success,
            error_message=None,
            fetched_at=fetched_at,
        )

    def test_empty_table_returns_empty_set(self, in_memory_db):
        repo = IngestionSnapshotRepository(in_memory_db)
        assert repo.get_ingested_paths_since("blizzard_api", WINDOW_START) == set()

    def test_returns_paths_at_or_after_since(self, in_memory_db):
        repo = IngestionSnapshotRepository(in_memory_db)
        repo.insert(self._snapshot("/new.json", datetime(2026, 7, 22, 1, 0)))
        repo.insert(self._snapshot("/old.json", datetime(2026, 7, 1, 1, 0)))
        assert repo.get_ingested_paths_since("blizzard_api", WINDOW_START) == {"/new.json"}

    def test_excludes_failed_snapshots(self, in_memory_db):
        """A failed object must be retried, not treated as already drained."""
        repo = IngestionSnapshotRepository(in_memory_db)
        repo.insert(
            self._snapshot("/failed.json", datetime(2026, 7, 22, 1, 0), success=False)
        )
        assert repo.get_ingested_paths_since("blizzard_api", WINDOW_START) == set()

    def test_excludes_empty_paths(self, in_memory_db):
        repo = IngestionSnapshotRepository(in_memory_db)
        repo.insert(self._snapshot("", datetime(2026, 7, 22, 1, 0)))
        assert repo.get_ingested_paths_since("blizzard_api", WINDOW_START) == set()

    def test_excludes_other_sources(self, in_memory_db):
        repo = IngestionSnapshotRepository(in_memory_db)
        snap = self._snapshot("/news.json", datetime(2026, 7, 22, 1, 0))
        snap.source = "blizzard_news"
        repo.insert(snap)
        assert repo.get_ingested_paths_since("blizzard_api", WINDOW_START) == set()


@pytest.mark.parametrize("bad_key", ["", "garbage"])
def test_unparseable_hour_keys_are_skipped_not_fatal(in_memory_db, bad_key, caplog):
    """A corrupt observed_at must not take down the whole catch-up run."""
    item_id = _seed_item(in_memory_db)
    in_memory_db.execute(
        "INSERT INTO market_observations_raw "
        "(item_id, realm_slug, faction, observed_at, source) "
        "VALUES (?, 'us', 'neutral', ?, 'blizzard_api');",
        (item_id, bad_key),
    )
    repo = MarketObservationRepository(in_memory_db)
    assert repo.get_covered_hours("blizzard_api", datetime(1970, 1, 1), WINDOW_END) == set()
