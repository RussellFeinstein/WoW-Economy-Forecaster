"""Tests for the cloud snapshot fetcher (wow_forecaster.ingestion.cloud_fetch)."""

from __future__ import annotations

import gzip
import json
import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from wow_forecaster.ingestion import cloud_fetch
from wow_forecaster.ingestion.blizzard_client import (
    BlizzardAuctionRecord,
    BlizzardAuctionResponse,
)
from wow_forecaster.ingestion.snapshot import build_snapshot_path

FETCHED_AT = datetime(2026, 7, 12, 13, 16, 5, tzinfo=UTC)

# Placeholder values only — the fetcher never sees real credentials in tests.
REQUIRED_ENV_VALUES = {
    "BLIZZARD_CLIENT_ID": "test-client-id",
    "BLIZZARD_CLIENT_SECRET": "test-client-secret",
    "SNAPSHOT_S3_ENDPOINT": "https://example.r2.cloudflarestorage.com",
    "SNAPSHOT_S3_BUCKET": "test-bucket",
    "AWS_ACCESS_KEY_ID": "test-access-key",
    "AWS_SECRET_ACCESS_KEY": "test-secret-key",
}


def _record(item_id: int) -> BlizzardAuctionRecord:
    return BlizzardAuctionRecord(
        item_id=item_id,
        realm_id=0,
        realm_slug="us",
        buyout=0,
        bid=0,
        unit_price=1_000,
        quantity=25,
        time_left="VERY_LONG",
    )


def _response(n: int = 3, is_fixture: bool = False) -> BlizzardAuctionResponse:
    return BlizzardAuctionResponse(
        region="us",
        realm_id=0,
        realm_slug="us",
        endpoint="data/wow/auctions/commodities",
        fetched_at=FETCHED_AT,
        records=[_record(190_000 + i) for i in range(n)],
        is_fixture=is_fixture,
    )


def _set_env(monkeypatch: pytest.MonkeyPatch, **overrides: str) -> None:
    for name, value in {**REQUIRED_ENV_VALUES, **overrides}.items():
        monkeypatch.setenv(name, value)


def _stub_client_returning(response: BlizzardAuctionResponse) -> type:
    class StubBlizzardClient:
        def __init__(self, client_id=None, client_secret=None, region="us") -> None:
            self.region = region

        def fetch_commodities(self) -> BlizzardAuctionResponse:
            return response

    return StubBlizzardClient


class StubS3:
    """Records put_object calls and serves canned list_objects_v2 results."""

    def __init__(self, keys: list[str] | None = None, fail_put: bool = False) -> None:
        self.keys = list(keys or [])
        self.fail_put = fail_put
        self.puts: list[dict] = []
        self.list_prefixes: list[str] = []

    def put_object(self, **kwargs) -> None:
        if self.fail_put:
            raise RuntimeError("upload refused")
        self.puts.append(kwargs)

    def list_objects_v2(self, **kwargs) -> dict:
        prefix = kwargs["Prefix"]
        self.list_prefixes.append(prefix)
        matches = [k for k in self.keys if k.startswith(prefix)]
        if not matches:
            return {}
        return {"Contents": [{"Key": k} for k in matches]}


# ── Pure helpers ───────────────────────────────────────────────────────────────


def test_build_object_key_format():
    key = cloud_fetch.build_object_key("us", FETCHED_AT)
    assert key == "blizzard_api/2026/07/12/commodities_us_20260712T131605Z.json.gz"


def test_build_object_key_mirrors_local_snapshot_layout():
    local = build_snapshot_path("data/raw", "blizzard_api", "commodities_us", FETCHED_AT)
    expected = local.relative_to(Path("data/raw/snapshots")).as_posix() + ".gz"
    assert cloud_fetch.build_object_key("us", FETCHED_AT) == expected


def test_parse_key_timestamp_round_trip():
    key = cloud_fetch.build_object_key("us", FETCHED_AT)
    assert cloud_fetch.parse_key_timestamp(key) == FETCHED_AT


def test_parse_key_timestamp_rejects_non_snapshot_keys():
    assert cloud_fetch.parse_key_timestamp("blizzard_api/2026/07/12/notes.txt") is None
    assert cloud_fetch.parse_key_timestamp("commodities_us_20260712.json.gz") is None


def test_records_to_dicts_matches_local_ingest_shape():
    record = _record(190_396)
    (row,) = cloud_fetch.records_to_dicts([record])
    assert list(row) == [
        "item_id", "realm_slug", "buyout", "bid", "unit_price", "quantity", "time_left",
    ]
    assert row == {
        "item_id": 190_396,
        "realm_slug": "us",
        "buyout": 0,
        "bid": 0,
        "unit_price": 1_000,
        "quantity": 25,
        "time_left": "VERY_LONG",
    }


# ── Gap guard ──────────────────────────────────────────────────────────────────

GUARD_NOW = datetime(2026, 7, 12, 15, 0, 0, tzinfo=UTC)


def _keys_hours_ago(hours: list[float]) -> list[str]:
    return [
        cloud_fetch.build_object_key("us", GUARD_NOW - timedelta(hours=h)) for h in hours
    ]


def test_gap_guard_passes_on_healthy_day():
    keys = _keys_hours_ago([h + 0.5 for h in range(23)] + [30.0, 31.0])
    ok, detail = cloud_fetch.evaluate_gap_guard(keys, GUARD_NOW)
    assert ok
    assert "23 objects" in detail


def test_gap_guard_trips_when_hours_are_missing():
    keys = _keys_hours_ago([1.0, 2.0, 3.0] + [25.0 + h for h in range(10)])
    ok, detail = cloud_fetch.evaluate_gap_guard(keys, GUARD_NOW)
    assert not ok
    assert "only 3 objects" in detail


def test_gap_guard_passes_on_bootstrap():
    ok, detail = cloud_fetch.evaluate_gap_guard(_keys_hours_ago([1.0, 2.0]), GUARD_NOW)
    assert ok
    assert "bootstrap" in detail


def test_gap_guard_ignores_unparseable_keys():
    keys = _keys_hours_ago([25.0]) + ["blizzard_api/2026/07/12/garbage.txt"]
    ok, _ = cloud_fetch.evaluate_gap_guard(keys, GUARD_NOW)
    assert not ok  # one old object, zero recent, garbage not counted


def test_list_recent_keys_queries_today_and_yesterday():
    now = datetime.now(UTC)
    stub = StubS3()
    cloud_fetch.list_recent_keys(stub, "test-bucket", now)
    expected = [
        f"blizzard_api/{(now - timedelta(days=1)).strftime('%Y/%m/%d')}/",
        f"blizzard_api/{now.strftime('%Y/%m/%d')}/",
    ]
    assert stub.list_prefixes == expected


# ── Retry helper ───────────────────────────────────────────────────────────────


def test_retry_recovers_after_transient_failures(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(cloud_fetch.time, "sleep", lambda _s: None)
    calls = {"n": 0}

    def flaky() -> str:
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("transient")
        return "ok"

    assert cloud_fetch._retry(flaky, label="flaky") == "ok"
    assert calls["n"] == 3


def test_retry_reraises_after_exhausting_attempts(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(cloud_fetch.time, "sleep", lambda _s: None)

    def always_fails() -> None:
        raise RuntimeError("permanent")

    with pytest.raises(RuntimeError, match="permanent"):
        cloud_fetch._retry(always_fails, label="always", attempts=2)


# ── main() ─────────────────────────────────────────────────────────────────────


def test_main_missing_env_returns_2(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
):
    for name in cloud_fetch.REQUIRED_ENV:
        monkeypatch.delenv(name, raising=False)
    with caplog.at_level(logging.ERROR):
        assert cloud_fetch.main() == 2
    assert "SNAPSHOT_S3_BUCKET" in caplog.text
    for value in REQUIRED_ENV_VALUES.values():
        assert value not in caplog.text


def test_main_refuses_fixture_data(monkeypatch: pytest.MonkeyPatch):
    _set_env(monkeypatch)
    monkeypatch.setattr(
        cloud_fetch, "BlizzardClient", _stub_client_returning(_response(is_fixture=True))
    )
    monkeypatch.setattr(
        cloud_fetch, "make_s3_client", lambda *a, **k: pytest.fail("must not reach S3")
    )
    assert cloud_fetch.main() == 1


def test_main_refuses_implausibly_small_snapshot(monkeypatch: pytest.MonkeyPatch):
    _set_env(monkeypatch, CLOUD_FETCH_MIN_RECORDS="10")
    monkeypatch.setattr(cloud_fetch, "BlizzardClient", _stub_client_returning(_response(n=3)))
    monkeypatch.setattr(
        cloud_fetch, "make_s3_client", lambda *a, **k: pytest.fail("must not reach S3")
    )
    assert cloud_fetch.main() == 1


def test_main_returns_1_when_fetch_fails(monkeypatch: pytest.MonkeyPatch):
    _set_env(monkeypatch)
    monkeypatch.setattr(cloud_fetch.time, "sleep", lambda _s: None)

    class BrokenClient:
        def __init__(self, **kwargs) -> None:
            pass

        def fetch_commodities(self) -> None:
            raise RuntimeError("api down")

    monkeypatch.setattr(cloud_fetch, "BlizzardClient", BrokenClient)
    assert cloud_fetch.main() == 1


def test_main_returns_1_when_upload_fails(monkeypatch: pytest.MonkeyPatch):
    _set_env(monkeypatch, CLOUD_FETCH_MIN_RECORDS="1")
    monkeypatch.setattr(cloud_fetch.time, "sleep", lambda _s: None)
    monkeypatch.setattr(cloud_fetch, "BlizzardClient", _stub_client_returning(_response(n=5)))
    monkeypatch.setattr(cloud_fetch, "make_s3_client", lambda *a, **k: StubS3(fail_put=True))
    assert cloud_fetch.main() == 1


def test_main_happy_path_uploads_expected_object(monkeypatch: pytest.MonkeyPatch):
    _set_env(monkeypatch, CLOUD_FETCH_MIN_RECORDS="1")
    response = _response(n=5)
    now = datetime.now(UTC)
    recent_keys = [
        cloud_fetch.build_object_key("us", now - timedelta(hours=h)) for h in range(1, 22)
    ]
    stub = StubS3(keys=recent_keys)
    monkeypatch.setattr(cloud_fetch, "BlizzardClient", _stub_client_returning(response))
    monkeypatch.setattr(cloud_fetch, "make_s3_client", lambda *a, **k: stub)

    assert cloud_fetch.main() == 0

    assert len(stub.puts) == 1
    put = stub.puts[0]
    assert put["Bucket"] == "test-bucket"
    assert put["Key"] == "blizzard_api/2026/07/12/commodities_us_20260712T131605Z.json.gz"
    assert put["ContentType"] == "application/gzip"

    envelope = json.loads(gzip.decompress(put["Body"]))
    assert set(envelope["_meta"]) == {
        "source", "type", "region", "is_fixture", "run_slug", "fetcher", "written_at",
    }
    assert envelope["_meta"]["source"] == "blizzard_api"
    assert envelope["_meta"]["type"] == "commodities"
    assert envelope["_meta"]["region"] == "us"
    assert envelope["_meta"]["is_fixture"] is False
    assert envelope["_meta"]["fetcher"] == "cloud"
    assert envelope["_meta"]["run_slug"].startswith("gha_")
    assert envelope["data"] == cloud_fetch.records_to_dicts(response.records)


def test_main_returns_3_when_gap_guard_trips(monkeypatch: pytest.MonkeyPatch):
    _set_env(monkeypatch, CLOUD_FETCH_MIN_RECORDS="1")
    now = datetime.now(UTC)
    stale_keys = [cloud_fetch.build_object_key("us", now - timedelta(hours=26))]
    monkeypatch.setattr(cloud_fetch, "BlizzardClient", _stub_client_returning(_response(n=5)))
    monkeypatch.setattr(cloud_fetch, "make_s3_client", lambda *a, **k: StubS3())
    monkeypatch.setattr(cloud_fetch, "list_recent_keys", lambda *a, **k: stale_keys)
    assert cloud_fetch.main() == 3


def test_main_returns_3_when_guard_listing_fails(monkeypatch: pytest.MonkeyPatch):
    _set_env(monkeypatch, CLOUD_FETCH_MIN_RECORDS="1")

    def broken_listing(*args, **kwargs) -> None:
        raise RuntimeError("list refused")

    monkeypatch.setattr(cloud_fetch, "BlizzardClient", _stub_client_returning(_response(n=5)))
    monkeypatch.setattr(cloud_fetch, "make_s3_client", lambda *a, **k: StubS3())
    monkeypatch.setattr(cloud_fetch, "list_recent_keys", broken_listing)
    assert cloud_fetch.main() == 3
