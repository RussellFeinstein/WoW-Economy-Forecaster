"""Tests for the cloud snapshot catch-up path (issue #43).

Covers the pure selection rules in ``wow_forecaster.ingestion.cloud_sync``, the
S3 access helpers, the write lock, and an end-to-end drain through
``SyncSnapshotsStage`` into rollup rows.

Date anchors are pinned at a fixed midday UTC value rather than derived from the
wall clock: half-day offsets from a wall-clock anchor land on the previous UTC
date whenever the suite runs before noon, which has produced phantom-gap flakes
in this repo before.
"""

from __future__ import annotations

import gzip
import json
import sqlite3
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from wow_forecaster.config import (
    AppConfig,
    CloudSyncConfig,
    DatabaseConfig,
    DataConfig,
    PipelineConfig,
)
from wow_forecaster.db.schema import apply_schema
from wow_forecaster.ingestion import cloud_fetch, cloud_sync
from wow_forecaster.ingestion.cloud_sync import (
    hourly_lock,
    local_path_for_key,
    select_objects_to_ingest,
)

# Midday anchor: hour arithmetic below never crosses a UTC date boundary by accident.
NOW = datetime(2026, 7, 23, 12, 0, 0)
RAW_DIR = "data/raw"
RETENTION_CUTOFF = NOW - timedelta(days=30)


def _key(moment: datetime, region: str = "us") -> str:
    """Build a bucket key the way cloud_fetch does, from a naive UTC moment."""
    return cloud_fetch.build_object_key(region, moment.replace(tzinfo=UTC))


def _at(hours_ago: float, minute: int = 16) -> datetime:
    """A capture timestamp N hours before NOW, at a given minute past the hour."""
    base = (NOW - timedelta(hours=hours_ago)).replace(minute=minute, second=0, microsecond=0)
    return base


def _select(keys, **overrides):
    kwargs = {
        "raw_dir": RAW_DIR,
        "already_ingested": set(),
        "hours_covered": set(),
        "retention_cutoff": RETENTION_CUTOFF,
        "max_objects": None,
    }
    kwargs.update(overrides)
    return select_objects_to_ingest(keys, **kwargs)


# ── Key and path mapping ──────────────────────────────────────────────────────


class TestLocalPathForKey:
    def test_inverts_build_object_key(self):
        moment = datetime(2026, 7, 23, 4, 16, 5)
        key = _key(moment)
        expected = Path(RAW_DIR) / "snapshots" / "blizzard_api" / "2026" / "07" / "23" / (
            "commodities_us_20260723T041605Z.json"
        )
        assert local_path_for_key(RAW_DIR, key) == expected

    def test_matches_build_snapshot_path_exactly(self):
        """The catch-up path must write where the live path would have written."""
        from wow_forecaster.ingestion.snapshot import build_snapshot_path

        moment = datetime(2026, 7, 23, 4, 16, 5)
        live = build_snapshot_path(
            RAW_DIR, "blizzard_api", "commodities_us", moment.replace(tzinfo=UTC)
        )
        assert local_path_for_key(RAW_DIR, _key(moment)) == live

    def test_key_without_gz_suffix_is_handled(self):
        key = "blizzard_api/2026/07/23/commodities_us_20260723T041605Z.json"
        assert local_path_for_key(RAW_DIR, key).name.endswith(".json")


# ── Selection rules ───────────────────────────────────────────────────────────


class TestSelectObjectsToIngest:
    def test_empty_input_selects_nothing(self):
        selected, skips = _select([])
        assert selected == []
        assert skips.total() == 0

    def test_returns_oldest_first(self):
        keys = [_key(_at(1)), _key(_at(5)), _key(_at(3))]
        selected, _ = _select(keys)
        assert [o.captured_at for o in selected] == [
            _at(5), _at(3), _at(1),
        ]

    def test_unparseable_keys_are_ignored(self):
        keys = [_key(_at(1)), "blizzard_api/2026/07/23/not-a-snapshot.txt"]
        selected, skips = _select(keys)
        assert len(selected) == 1
        assert skips.unparseable_key == 1

    def test_drops_captures_beyond_the_retention_cutoff(self):
        old = NOW - timedelta(days=31)
        keys = [_key(old), _key(_at(1))]
        selected, skips = _select(keys)
        assert [o.captured_at for o in selected] == [_at(1)]
        assert skips.beyond_retention == 1

    def test_skips_objects_already_ingested(self):
        fresh, done = _at(1), _at(2)
        keys = [_key(fresh), _key(done)]
        already = {str(local_path_for_key(RAW_DIR, _key(done)))}
        selected, skips = _select(keys, already_ingested=already)
        assert [o.captured_at for o in selected] == [fresh]
        assert skips.already_ingested == 1

    def test_skips_hours_already_covered_locally(self):
        """The desktop and the cloud fetch the same AH snapshot; one per hour."""
        covered_hour = _at(2).replace(minute=0)
        keys = [_key(_at(1)), _key(_at(2))]
        selected, skips = _select(keys, hours_covered={covered_hour})
        assert [o.captured_at for o in selected] == [_at(1)]
        assert skips.hour_covered == 1

    def test_keeps_only_the_earliest_object_per_hour(self):
        """The bucket holds :06, :16 and :46 captures; one hour means one row."""
        keys = [
            _key(_at(1, minute=46)),
            _key(_at(1, minute=6)),
            _key(_at(1, minute=16)),
        ]
        selected, skips = _select(keys)
        assert len(selected) == 1
        assert selected[0].captured_at.minute == 6
        assert skips.duplicate_hour == 2

    def test_duplicate_collapse_is_order_independent(self):
        forward = [_key(_at(1, minute=6)), _key(_at(1, minute=46))]
        reverse = list(reversed(forward))
        assert _select(forward)[0][0].key == _select(reverse)[0][0].key

    def test_truncates_to_max_objects_and_reports_the_remainder(self):
        keys = [_key(_at(h)) for h in range(1, 6)]
        selected, skips = _select(keys, max_objects=2)
        assert len(selected) == 2
        assert skips.over_limit == 3
        # Oldest first, so the cap takes the oldest two and leaves the rest for
        # the next run rather than dropping the tail of the backlog.
        assert [o.captured_at for o in selected] == [_at(5), _at(4)]

    def test_max_objects_none_means_no_cap(self):
        keys = [_key(_at(h)) for h in range(1, 6)]
        selected, skips = _select(keys, max_objects=None)
        assert len(selected) == 5
        assert skips.over_limit == 0

    def test_filters_compose(self):
        old = NOW - timedelta(days=31)
        done = _at(2)
        covered = _at(3)
        keys = [
            _key(old),
            _key(done),
            _key(covered),
            _key(_at(1, minute=16)),
            _key(_at(1, minute=46)),
        ]
        selected, skips = _select(
            keys,
            already_ingested={str(local_path_for_key(RAW_DIR, _key(done)))},
            hours_covered={covered.replace(minute=0)},
        )
        assert [o.captured_at for o in selected] == [_at(1, minute=16)]
        assert (skips.beyond_retention, skips.already_ingested) == (1, 1)
        assert (skips.hour_covered, skips.duplicate_hour) == (1, 1)

    def test_captured_at_is_naive_for_database_comparison(self):
        selected, _ = _select([_key(_at(1))])
        assert selected[0].captured_at.tzinfo is None


class TestSkipCounts:
    def test_summary_omits_zero_reasons(self):
        _, skips = _select([_key(_at(1))])
        assert skips.summary() == "none"

    def test_summary_lists_only_nonzero_reasons(self):
        _, skips = _select([_key(NOW - timedelta(days=31))])
        assert skips.summary() == "beyond_retention=1"


# ── S3 helpers ────────────────────────────────────────────────────────────────


class StubS3:
    """Serves canned list and get results; records what was asked for."""

    def __init__(self, objects: dict[str, bytes] | None = None, pages: int = 1) -> None:
        self.objects = dict(objects or {})
        self.pages = pages
        self.list_prefixes: list[str] = []
        self.get_keys: list[str] = []
        self._page_state: dict[str, int] = {}

    def list_objects_v2(self, **kwargs) -> dict:
        prefix = kwargs["Prefix"]
        self.list_prefixes.append(prefix)
        matches = sorted(k for k in self.objects if k.startswith(prefix))
        if self.pages == 1:
            return {"Contents": [{"Key": k} for k in matches]} if matches else {}
        # Split matches across `pages` responses to exercise pagination.
        page = self._page_state.get(prefix, 0)
        chunk = matches[page::self.pages]
        self._page_state[prefix] = page + 1
        truncated = page + 1 < self.pages
        resp: dict = {"Contents": [{"Key": k} for k in chunk]}
        if truncated:
            resp["IsTruncated"] = True
            resp["NextContinuationToken"] = f"token-{page}"
        return resp

    def get_object(self, **kwargs) -> dict:
        key = kwargs["Key"]
        self.get_keys.append(key)
        if key not in self.objects:
            raise KeyError(key)
        return {"Body": _Body(self.objects[key])}


class _Body:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def read(self) -> bytes:
        return self._payload


def _envelope_bytes(records: list[dict], *, gzipped: bool = True) -> bytes:
    envelope = {
        "_meta": {"source": "blizzard_api", "fetcher": "cloud", "written_at": "x"},
        "data": records,
    }
    raw = json.dumps(envelope, indent=2).encode("utf-8")
    return gzip.compress(raw) if gzipped else raw


def _record(item_id: int, unit_price: int = 1_000) -> dict:
    return {
        "item_id": item_id,
        "realm_slug": "us",
        "buyout": 0,
        "bid": 0,
        "unit_price": unit_price,
        "quantity": 5,
        "time_left": "VERY_LONG",
    }


class TestListObjectsSince:
    def test_walks_one_prefix_per_day_inclusive(self):
        stub = StubS3()
        cloud_sync.list_objects_since(
            stub, "bucket", NOW - timedelta(days=2), NOW
        )
        assert stub.list_prefixes == [
            "blizzard_api/2026/07/21/",
            "blizzard_api/2026/07/22/",
            "blizzard_api/2026/07/23/",
        ]

    def test_spans_a_month_boundary(self):
        stub = StubS3()
        start = datetime(2026, 6, 29, 12, 0)
        end = datetime(2026, 7, 1, 12, 0)
        cloud_sync.list_objects_since(stub, "bucket", start, end)
        assert stub.list_prefixes == [
            "blizzard_api/2026/06/29/",
            "blizzard_api/2026/06/30/",
            "blizzard_api/2026/07/01/",
        ]

    def test_returns_every_key_under_the_prefixes(self):
        keys = [_key(_at(1)), _key(_at(26))]
        stub = StubS3({k: b"" for k in keys})
        found = cloud_sync.list_objects_since(stub, "bucket", NOW - timedelta(days=2), NOW)
        assert sorted(found) == sorted(keys)

    def test_follows_pagination_to_the_end(self):
        """A truncated listing would read exactly like a quiet day."""
        keys = [_key(_at(h)) for h in range(1, 7)]
        stub = StubS3({k: b"" for k in keys}, pages=3)
        found = cloud_sync.list_objects_since(stub, "bucket", NOW, NOW)
        assert sorted(found) == sorted(keys)


class TestDownloadSnapshot:
    def test_returns_envelope_and_raw_bytes(self):
        key = _key(_at(1))
        raw = _envelope_bytes([_record(1)], gzipped=False)
        stub = StubS3({key: gzip.compress(raw)})
        envelope, returned = cloud_sync.download_snapshot(stub, "bucket", key)
        assert envelope["data"] == [_record(1)]
        assert returned == raw

    def test_rejects_bad_gzip(self):
        key = _key(_at(1))
        stub = StubS3({key: b"not gzip at all"})
        with pytest.raises(ValueError, match="not valid gzip"):
            cloud_sync.download_snapshot(stub, "bucket", key)

    def test_rejects_bad_json(self):
        key = _key(_at(1))
        stub = StubS3({key: gzip.compress(b"{not json")})
        with pytest.raises(ValueError, match="not valid JSON"):
            cloud_sync.download_snapshot(stub, "bucket", key)

    def test_rejects_envelope_without_data_section(self):
        key = _key(_at(1))
        stub = StubS3({key: gzip.compress(json.dumps({"_meta": {}}).encode())})
        with pytest.raises(ValueError, match="missing the 'data' envelope"):
            cloud_sync.download_snapshot(stub, "bucket", key)

    def test_rejects_data_that_is_not_a_list(self):
        key = _key(_at(1))
        payload = json.dumps({"_meta": {}, "data": {"nope": 1}}).encode()
        stub = StubS3({key: gzip.compress(payload)})
        with pytest.raises(ValueError, match="'data' is not a list"):
            cloud_sync.download_snapshot(stub, "bucket", key)


class TestResolveS3Env:
    def test_names_every_missing_variable(self, monkeypatch):
        for name in cloud_sync.REQUIRED_ENV:
            monkeypatch.delenv(name, raising=False)
        with pytest.raises(RuntimeError) as exc:
            cloud_sync.resolve_s3_env()
        for name in cloud_sync.REQUIRED_ENV:
            assert name in str(exc.value)

    def test_returns_values_with_default_region(self, monkeypatch):
        for name in cloud_sync.REQUIRED_ENV:
            monkeypatch.setenv(name, f"value-for-{name}")
        monkeypatch.delenv("SNAPSHOT_S3_REGION", raising=False)
        env = cloud_sync.resolve_s3_env()
        assert env["region"] == "auto"
        assert env["bucket"] == "value-for-SNAPSHOT_S3_BUCKET"


# ── Write lock ────────────────────────────────────────────────────────────────


class TestHourlyLock:
    def test_creates_and_removes_the_lock(self, tmp_path):
        lock = tmp_path / ".hourly.lock"
        with hourly_lock(lock):
            assert lock.exists()
        assert not lock.exists()

    def test_creates_the_parent_directory(self, tmp_path):
        lock = tmp_path / "db" / ".hourly.lock"
        with hourly_lock(lock):
            assert lock.exists()

    def test_takes_over_a_stale_lock(self, tmp_path):
        lock = tmp_path / ".hourly.lock"
        lock.write_text("leaked by a crashed run")
        old = time.time() - (200 * 60)
        import os

        os.utime(lock, (old, old))
        with hourly_lock(lock, stale_minutes=180, wait_seconds=0):
            assert lock.exists()
        assert not lock.exists()

    def test_fails_loudly_when_held_by_a_live_run(self, tmp_path):
        """A quiet skip would leave a whole night unrecovered."""
        lock = tmp_path / ".hourly.lock"
        lock.write_text("hourly run in progress")
        with pytest.raises(TimeoutError, match="held by another run"):
            with hourly_lock(lock, wait_seconds=0, poll_seconds=0):
                pass
        # The other run's lock survives: we never took it, so we never drop it.
        assert lock.exists()

    def test_releases_the_lock_when_the_body_raises(self, tmp_path):
        lock = tmp_path / ".hourly.lock"
        with pytest.raises(RuntimeError):
            with hourly_lock(lock):
                raise RuntimeError("boom")
        assert not lock.exists()


# ── End-to-end drain ──────────────────────────────────────────────────────────


def _make_db(tmp_path: Path, item_ids: list[int]) -> str:
    db_file = str(tmp_path / "sync.db")
    conn = sqlite3.connect(db_file)
    try:
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        apply_schema(conn)
        conn.execute(
            "INSERT OR IGNORE INTO item_categories (slug, display_name, archetype_tag) "
            "VALUES ('test.cat', 'Test', 'test');"
        )
        cat_id = conn.execute(
            "SELECT category_id FROM item_categories WHERE slug='test.cat';"
        ).fetchone()[0]
        for item_id in item_ids:
            conn.execute(
                "INSERT OR IGNORE INTO items "
                "(item_id, name, category_id, expansion_slug, quality) "
                "VALUES (?, ?, ?, 'midnight', 'common');",
                (item_id, f"Item {item_id}", cat_id),
            )
        conn.commit()
    finally:
        conn.close()
    return db_file


def _make_config(tmp_path: Path, db_file: str, **cloud_overrides) -> AppConfig:
    return AppConfig(
        database=DatabaseConfig(db_path=db_file),
        data=DataConfig(
            raw_dir=str(tmp_path / "raw"),
            processed_dir=str(tmp_path / "processed"),
        ),
        pipeline=PipelineConfig(normalize_batch_size=100),
        cloud_sync=CloudSyncConfig(**cloud_overrides),
    )


ENV = {
    "endpoint": "https://example.r2.cloudflarestorage.com",
    "bucket": "test-bucket",
    "access_key": "unused-in-tests",
    "secret_key": "unused-in-tests",
    "region": "auto",
}


def _run_sync(config, stub, db_file, **kwargs):
    from wow_forecaster.pipeline.sync_stage import sync_snapshots

    return sync_snapshots(
        config, db_path=db_file, now=NOW, s3=stub, env=ENV, **kwargs
    )


class TestSyncSnapshotsStage:
    @pytest.fixture
    def two_objects(self, tmp_path):
        db_file = _make_db(tmp_path, [190_001, 190_002])
        config = _make_config(tmp_path, db_file)
        first, second = _at(3), _at(2)
        stub = StubS3({
            _key(first): _envelope_bytes([_record(190_001), _record(190_002, 2_000)]),
            _key(second): _envelope_bytes([_record(190_001, 1_500)]),
        })
        return config, stub, db_file, (first, second)

    def test_ingests_both_objects_through_to_rollup_rows(self, two_objects):
        config, stub, db_file, (first, second) = two_objects
        result = _run_sync(config, stub, db_file)

        assert result.ingested == 2
        assert result.observations_inserted == 3
        assert result.failures == []
        assert result.normalized_rows == 3

        conn = sqlite3.connect(db_file)
        try:
            conn.row_factory = sqlite3.Row
            raw = conn.execute(
                "SELECT COUNT(*) AS n FROM market_observations_raw;"
            ).fetchone()["n"]
            norm = conn.execute(
                "SELECT COUNT(*) AS n FROM market_observations_normalized;"
            ).fetchone()["n"]
            rollup = conn.execute(
                "SELECT COUNT(*) AS n FROM daily_rollup_item;"
            ).fetchone()["n"]
        finally:
            conn.close()

        assert (raw, norm) == (3, 3)
        assert rollup > 0
        assert result.dates_touched == [first.date().isoformat()]

    def test_writes_snapshots_where_the_live_path_would_have(self, two_objects):
        config, stub, db_file, (first, _) = two_objects
        _run_sync(config, stub, db_file)
        expected = local_path_for_key(config.data.raw_dir, _key(first))
        assert expected.exists()
        # Written verbatim: the cloud _meta block survives, so provenance shows
        # the snapshot came from the cloud fetcher rather than a live fetch.
        envelope = json.loads(expected.read_text(encoding="utf-8"))
        assert envelope["_meta"]["fetcher"] == "cloud"

    def test_records_the_bucket_key_as_the_endpoint(self, two_objects):
        config, stub, db_file, (first, _) = two_objects
        _run_sync(config, stub, db_file)
        conn = sqlite3.connect(db_file)
        try:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT endpoint, snapshot_path, record_count FROM ingestion_snapshots "
                "ORDER BY fetched_at LIMIT 1;"
            ).fetchone()
        finally:
            conn.close()
        assert row["endpoint"] == f"r2://test-bucket/{_key(first)}"
        assert row["snapshot_path"].endswith(".json")
        assert row["record_count"] == 2

    def test_second_run_is_a_no_op(self, two_objects):
        config, stub, db_file, _ = two_objects
        _run_sync(config, stub, db_file)
        again = _run_sync(config, stub, db_file)

        assert again.ingested == 0
        assert again.observations_inserted == 0
        assert again.skips.total() == 2

        conn = sqlite3.connect(db_file)
        try:
            n = conn.execute(
                "SELECT COUNT(*) AS n FROM market_observations_raw;"
            ).fetchone()[0]
        finally:
            conn.close()
        assert n == 3

    def test_dry_run_writes_nothing(self, two_objects):
        config, stub, db_file, _ = two_objects
        result = _run_sync(config, stub, db_file, dry_run=True)

        assert result.dry_run is True
        assert result.selected == 2
        assert result.ingested == 0
        assert stub.get_keys == []

        conn = sqlite3.connect(db_file)
        try:
            n = conn.execute(
                "SELECT COUNT(*) AS n FROM market_observations_raw;"
            ).fetchone()[0]
        finally:
            conn.close()
        assert n == 0

    def test_one_corrupt_object_does_not_abandon_the_rest(self, tmp_path):
        db_file = _make_db(tmp_path, [190_001])
        config = _make_config(tmp_path, db_file)
        good, bad = _at(3), _at(2)
        stub = StubS3({
            _key(good): _envelope_bytes([_record(190_001)]),
            _key(bad): b"corrupt, not gzip",
        })
        result = _run_sync(config, stub, db_file)

        assert result.ingested == 1
        assert result.observations_inserted == 1
        assert len(result.failures) == 1
        assert result.failures[0][0] == _key(bad)

    def test_a_failed_object_is_retried_on_the_next_run(self, tmp_path):
        db_file = _make_db(tmp_path, [190_001])
        config = _make_config(tmp_path, db_file)
        broken = _at(2)
        stub = StubS3({_key(broken): b"corrupt, not gzip"})
        assert _run_sync(config, stub, db_file).failures

        # Repaired in the bucket; the next run must pick it up rather than treat
        # it as already ingested.
        stub.objects[_key(broken)] = _envelope_bytes([_record(190_001)])
        result = _run_sync(config, stub, db_file)
        assert result.ingested == 1
        assert result.observations_inserted == 1

    def test_skips_an_hour_the_local_pipeline_already_covered(self, tmp_path):
        db_file = _make_db(tmp_path, [190_001])
        config = _make_config(tmp_path, db_file)
        covered = _at(2)
        conn = sqlite3.connect(db_file)
        try:
            conn.execute(
                "INSERT INTO market_observations_raw "
                "(item_id, realm_slug, faction, observed_at, source) "
                "VALUES (190001, 'us', 'neutral', ?, 'blizzard_api');",
                (covered.replace(minute=16, second=30).isoformat(),),
            )
            conn.commit()
        finally:
            conn.close()

        stub = StubS3({_key(covered): _envelope_bytes([_record(190_001)])})
        result = _run_sync(config, stub, db_file)

        assert result.ingested == 0
        assert result.skips.hour_covered == 1

    def test_respects_the_configured_object_cap(self, tmp_path):
        db_file = _make_db(tmp_path, [190_001])
        config = _make_config(tmp_path, db_file, max_objects_per_run=1)
        stub = StubS3({
            _key(_at(3)): _envelope_bytes([_record(190_001)]),
            _key(_at(2)): _envelope_bytes([_record(190_001)]),
        })
        result = _run_sync(config, stub, db_file)

        assert result.ingested == 1
        assert result.truncated is True
        assert result.skips.over_limit == 1

    def test_limit_zero_overrides_the_cap(self, tmp_path):
        db_file = _make_db(tmp_path, [190_001])
        config = _make_config(tmp_path, db_file, max_objects_per_run=1)
        stub = StubS3({
            _key(_at(3)): _envelope_bytes([_record(190_001)]),
            _key(_at(2)): _envelope_bytes([_record(190_001)]),
        })
        result = _run_sync(config, stub, db_file, limit=0)

        assert result.ingested == 2
        assert result.truncated is False

    def test_since_is_clamped_to_the_retention_window(self, tmp_path):
        """Objects the pruner would delete on the next run are never ingested."""
        db_file = _make_db(tmp_path, [190_001])
        config = _make_config(tmp_path, db_file)
        ancient = NOW - timedelta(days=40)
        stub = StubS3({_key(ancient): _envelope_bytes([_record(190_001)])})
        result = _run_sync(config, stub, db_file, since=NOW - timedelta(days=45))

        assert result.ingested == 0
        # Clamped, so the ancient day prefix is never even listed.
        assert f"{ancient.strftime('%Y/%m/%d')}" not in " ".join(stub.list_prefixes)

    def test_unknown_items_are_counted_not_silently_dropped(self, tmp_path):
        db_file = _make_db(tmp_path, [190_001])
        config = _make_config(tmp_path, db_file)
        stub = StubS3({
            _key(_at(2)): _envelope_bytes([_record(190_001), _record(999_999)]),
        })
        result = _run_sync(config, stub, db_file)

        assert result.observations_inserted == 1
        assert result.items_skipped_fk == 1

    def test_records_a_run_metadata_row(self, two_objects):
        config, stub, db_file, _ = two_objects
        _run_sync(config, stub, db_file)
        conn = sqlite3.connect(db_file)
        try:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT pipeline_stage, status FROM run_metadata "
                "WHERE pipeline_stage = 'sync_snapshots';"
            ).fetchone()
        finally:
            conn.close()
        assert row is not None
        assert row["status"] == "success"

    def test_empty_bucket_is_a_clean_no_op(self, tmp_path):
        db_file = _make_db(tmp_path, [190_001])
        config = _make_config(tmp_path, db_file)
        result = _run_sync(config, StubS3(), db_file)

        assert (result.listed, result.selected, result.ingested) == (0, 0, 0)
        assert result.failures == []
