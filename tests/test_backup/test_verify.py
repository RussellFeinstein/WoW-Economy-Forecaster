"""Tests for off-box backup verification (wow_forecaster/backup/verify.py, issue #104)."""

from __future__ import annotations

import gzip
import io
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from wow_forecaster.backup import verify as verify_mod
from wow_forecaster.backup.durable_backup import build_durable_db
from wow_forecaster.backup.verify import (
    APPEND_ONLY_TABLES,
    DEFAULT_MIN_ROW_TABLES,
    main,
    parse_backup_key_timestamp,
    read_table_counts,
    select_verify_keys,
    verify_backup_db,
)
from wow_forecaster.db.migrations import run_migrations
from wow_forecaster.db.schema import apply_schema

REQUIRED_ENV = (
    "BACKUP_S3_ENDPOINT",
    "BACKUP_S3_BUCKET",
    "BACKUP_S3_ACCESS_KEY_ID",
    "BACKUP_S3_SECRET_ACCESS_KEY",
)


def _make_source_db(path: Path, *, n_forecasts: int = 3) -> None:
    """Create a fully-migrated source DB with rows in every floor-checked table."""
    con = sqlite3.connect(str(path))
    con.execute("PRAGMA foreign_keys = ON;")
    apply_schema(con)
    run_migrations(con)

    con.execute(
        "INSERT INTO item_categories(category_id,slug,display_name,archetype_tag) "
        "VALUES (1,'c','C','tag')"
    )
    con.execute(
        "INSERT INTO economic_archetypes(archetype_id,slug,display_name,category_tag) "
        "VALUES (1,'a','A','consumable')"
    )
    con.execute(
        "INSERT INTO items(item_id,name,category_id,archetype_id,expansion_slug,quality) "
        "VALUES (100,'i',1,1,'tww','rare')"
    )
    con.execute(
        "INSERT INTO model_metadata(model_id,slug,display_name,model_type) "
        "VALUES (1,'m','M','stub')"
    )
    con.execute(
        "INSERT INTO run_metadata(run_id,run_slug,pipeline_stage,config_snapshot) "
        "VALUES (1,'r','recommend','{}')"
    )
    for fid in range(1, n_forecasts + 1):
        con.execute(
            "INSERT INTO forecast_outputs(forecast_id,run_id,archetype_id,realm_slug,"
            "forecast_horizon,target_date,predicted_price_gold,confidence_lower,"
            "confidence_upper,model_slug) "
            "VALUES (?,1,1,'us','7d','2026-07-30',10,8,12,'m')",
            (fid,),
        )
    con.execute(
        "INSERT INTO daily_rollup_archetype(archetype_id,realm_slug,obs_date) "
        "VALUES (1,'us','2026-07-22')"
    )
    con.execute(
        "INSERT INTO daily_rollup_item(item_id,realm_slug,obs_date) "
        "VALUES (100,'us','2026-07-22')"
    )
    con.commit()
    con.close()


def _make_backup(tmp_path: Path, name: str = "durable.db", *, n_forecasts: int = 3) -> Path:
    src = tmp_path / f"src_{name}"
    _make_source_db(src, n_forecasts=n_forecasts)
    out = tmp_path / name
    build_durable_db(src, out)
    return out


def _key_for(ts: datetime) -> str:
    u = ts.astimezone(UTC)
    return f"db_backups/{u:%Y/%m/%d}/durable_{u:%Y%m%dT%H%M%S}Z.db.gz"


# ── key parsing and selection ───────────────────────────────────────────────────


def test_parse_backup_key_timestamp_valid() -> None:
    ts = parse_backup_key_timestamp("db_backups/2026/07/23/durable_20260723T113005Z.db.gz")
    assert ts == datetime(2026, 7, 23, 11, 30, 5, tzinfo=UTC)


def test_parse_backup_key_timestamp_rejects_other_keys() -> None:
    assert parse_backup_key_timestamp("db_backups/2026/07/23/notes.txt") is None
    assert parse_backup_key_timestamp("blizzard_api/2026/07/23/commodities_us_x.json.gz") is None


def test_select_verify_keys_orders_by_timestamp_not_listing_order() -> None:
    newest = _key_for(datetime(2026, 7, 23, 11, 30, 0, tzinfo=UTC))
    middle = _key_for(datetime(2026, 7, 22, 11, 30, 0, tzinfo=UTC))
    oldest = _key_for(datetime(2026, 7, 21, 11, 30, 0, tzinfo=UTC))
    got = select_verify_keys([middle, oldest, newest, "db_backups/2026/07/23/junk.txt"])
    assert got == (newest, middle)


def test_select_verify_keys_single_and_empty() -> None:
    only = _key_for(datetime(2026, 7, 23, 11, 30, 0, tzinfo=UTC))
    assert select_verify_keys([only]) == (only, None)
    assert select_verify_keys([]) == (None, None)
    assert select_verify_keys(["db_backups/x/junk.txt"]) == (None, None)


# ── verify_backup_db ────────────────────────────────────────────────────────────


def test_clean_backup_verifies_ok(tmp_path: Path) -> None:
    db = _make_backup(tmp_path)
    result = verify_backup_db(db)
    assert result.ok is True
    assert result.integrity_errors == []
    assert result.fk_violations == []
    assert result.floor_failures == []
    assert result.regressions == []
    assert result.table_counts["forecast_outputs"] == 3
    # obs tables exist empty in the backup and must not be floor-checked
    assert result.table_counts["market_observations_raw"] == 0
    assert "market_observations_raw" not in DEFAULT_MIN_ROW_TABLES


def test_corrupted_page_fails_verification(tmp_path: Path) -> None:
    db = _make_backup(tmp_path)
    data = bytearray(db.read_bytes())
    mid = len(data) // 2
    data[mid : mid + 256] = b"\xff" * 256
    db.write_bytes(bytes(data))

    result = verify_backup_db(db)
    assert result.ok is False
    assert result.integrity_errors


def test_truncated_file_fails_without_raising(tmp_path: Path) -> None:
    db = _make_backup(tmp_path)
    data = db.read_bytes()
    db.write_bytes(data[: len(data) // 3])

    result = verify_backup_db(db)
    assert result.ok is False
    assert result.integrity_errors


def test_empty_floor_table_fails(tmp_path: Path) -> None:
    db = _make_backup(tmp_path, n_forecasts=0)
    result = verify_backup_db(db)
    assert result.ok is False
    assert any("forecast_outputs" in f for f in result.floor_failures)


def test_shrunk_append_only_table_is_a_regression(tmp_path: Path) -> None:
    db = _make_backup(tmp_path)  # 3 forecast rows
    assert "forecast_outputs" in APPEND_ONLY_TABLES
    result = verify_backup_db(db, prev_counts={"forecast_outputs": 10})
    assert result.ok is False
    assert any("forecast_outputs" in r for r in result.regressions)


def test_equal_or_grown_append_only_counts_pass(tmp_path: Path) -> None:
    db = _make_backup(tmp_path)
    result = verify_backup_db(db, prev_counts={"forecast_outputs": 3, "daily_rollup_item": 1})
    assert result.ok is True
    assert result.regressions == []


def test_read_table_counts_matches_verify_counts(tmp_path: Path) -> None:
    db = _make_backup(tmp_path)
    counts = read_table_counts(db)
    assert counts["forecast_outputs"] == 3
    assert counts["daily_rollup_item"] == 1
    assert counts == verify_backup_db(db).table_counts


# ── main: local-file mode ───────────────────────────────────────────────────────


def test_main_local_file_clean(tmp_path: Path) -> None:
    db = _make_backup(tmp_path)
    gz = tmp_path / "durable.db.gz"
    gz.write_bytes(gzip.compress(db.read_bytes()))
    assert main([str(gz)]) == 0
    assert main([str(db)]) == 0  # uncompressed also accepted


def test_main_local_file_corrupt(tmp_path: Path) -> None:
    db = _make_backup(tmp_path)
    data = bytearray(db.read_bytes())
    mid = len(data) // 2
    data[mid : mid + 256] = b"\xff" * 256
    gz = tmp_path / "durable.db.gz"
    gz.write_bytes(gzip.compress(bytes(data)))
    assert main([str(gz)]) == 1


# ── main: bucket mode ───────────────────────────────────────────────────────────


class _StubS3:
    """Stub with paginated list_objects_v2 and get_object over an in-memory bucket."""

    def __init__(self, objects: dict[str, bytes]) -> None:
        self.objects = objects

    def list_objects_v2(self, Bucket: str, Prefix: str, **kwargs) -> dict:  # noqa: N803
        keys = sorted(k for k in self.objects if k.startswith(Prefix))
        return {"Contents": [{"Key": k} for k in keys], "IsTruncated": False}

    def get_object(self, Bucket: str, Key: str) -> dict:  # noqa: N803
        return {"Body": io.BytesIO(self.objects[Key])}


def _set_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in REQUIRED_ENV:
        monkeypatch.setenv(var, "value")


def _stub_bucket(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    newest_forecasts: int = 5,
    prev_forecasts: int = 3,
    newest_age_hours: float = 2.0,
    corrupt_newest: bool = False,
    include_prev: bool = True,
) -> _StubS3:
    now = datetime.now(tz=UTC)
    newest_db = _make_backup(tmp_path, "newest.db", n_forecasts=newest_forecasts)
    newest_bytes = bytearray(newest_db.read_bytes())
    if corrupt_newest:
        mid = len(newest_bytes) // 2
        newest_bytes[mid : mid + 256] = b"\xff" * 256
    objects = {
        _key_for(now - timedelta(hours=newest_age_hours)): gzip.compress(bytes(newest_bytes)),
    }
    if include_prev:
        prev_db = _make_backup(tmp_path, "prev.db", n_forecasts=prev_forecasts)
        objects[_key_for(now - timedelta(hours=newest_age_hours + 24))] = gzip.compress(
            prev_db.read_bytes()
        )
    stub = _StubS3(objects)
    monkeypatch.setattr(verify_mod, "_make_s3_client", lambda *a, **k: stub)
    return stub


def test_main_missing_env_exits_2(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in REQUIRED_ENV:
        monkeypatch.delenv(var, raising=False)
    assert main([]) == 2


def test_main_bucket_clean_exits_0(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _set_env(monkeypatch)
    _stub_bucket(tmp_path, monkeypatch)
    assert main([]) == 0


def test_main_bucket_corrupt_newest_exits_1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _set_env(monkeypatch)
    _stub_bucket(tmp_path, monkeypatch, corrupt_newest=True)
    assert main([]) == 1


def test_main_bucket_shrunk_counts_exit_1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _set_env(monkeypatch)
    _stub_bucket(tmp_path, monkeypatch, newest_forecasts=2, prev_forecasts=6)
    assert main([]) == 1


def test_main_bucket_stale_newest_exits_1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _set_env(monkeypatch)
    _stub_bucket(tmp_path, monkeypatch, newest_age_hours=72.0)
    assert main([]) == 1


def test_main_bucket_empty_is_failure_not_skip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _set_env(monkeypatch)
    stub = _StubS3({})
    monkeypatch.setattr(verify_mod, "_make_s3_client", lambda *a, **k: stub)
    assert main([]) == 1


def test_main_bucket_single_object_skips_regression_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _set_env(monkeypatch)
    _stub_bucket(tmp_path, monkeypatch, include_prev=False)
    assert main([]) == 0
