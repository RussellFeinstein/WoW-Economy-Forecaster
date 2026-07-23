"""Tests for the durable-table backup (wow_forecaster/backup/durable_backup.py)."""

from __future__ import annotations

import gzip
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest

from wow_forecaster.backup.durable_backup import (
    EXCLUDED_TABLES,
    build_durable_db,
    build_object_key,
    gzip_file,
    prune_local,
    run_backup,
    upload_backup,
)
from wow_forecaster.config import AppConfig, BackupConfig, DatabaseConfig
from wow_forecaster.db.migrations import run_migrations
from wow_forecaster.db.schema import apply_schema


def _make_source_db(path: Path, *, n_forecasts: int = 3) -> None:
    """Create a fully-migrated source DB with FK-consistent durable + obs rows."""
    con = sqlite3.connect(str(path))
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys = ON;")
    apply_schema(con)
    run_migrations(con)  # brings recommendation_outputs to its migrated shape

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
    # recommendation_outputs carrying the migration-added columns (the drift guard)
    con.execute(
        "INSERT INTO recommendation_outputs(rec_id,forecast_id,action,reasoning,priority,"
        "score,score_components,category_tag,risk_level) "
        "VALUES (1,1,'buy','r',1,0.9,'{\"opp\":1}','consumable','low')"
    )
    # observation rows that must be EXCLUDED from the copy
    con.execute(
        "INSERT INTO market_observations_raw(obs_id,item_id,realm_slug,observed_at,source) "
        "VALUES (1,100,'us','2026-07-23T00:00:00Z','blizzard_api')"
    )
    con.execute(
        "INSERT INTO market_observations_normalized(norm_id,obs_id,item_id,realm_slug,"
        "observed_at,price_gold) VALUES (1,1,100,'us','2026-07-23T00:00:00Z',10.0)"
    )
    con.commit()
    con.close()


def _app_config(db_path: Path, out_dir: Path, keep: int = 7) -> AppConfig:
    return AppConfig(
        database=DatabaseConfig(db_path=str(db_path)),
        backup=BackupConfig(output_dir=str(out_dir), keep_local=keep),
    )


# ── build_durable_db ────────────────────────────────────────────────────────────


def test_excluded_tables_present_but_empty(tmp_path: Path) -> None:
    src = tmp_path / "src.db"
    _make_source_db(src)
    out = tmp_path / "durable.db"
    result = build_durable_db(src, out)

    b = sqlite3.connect(str(out))
    try:
        for name in EXCLUDED_TABLES:
            # schema present (drop-in restore) but no rows copied
            assert b.execute(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name=?", (name,)
            ).fetchone()[0] == 1
            assert b.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0] == 0
            assert result.tables_copied[name] == 0
    finally:
        b.close()


def test_migrated_recommendation_columns_roundtrip(tmp_path: Path) -> None:
    """The core drift guard: migration-added columns must survive SELECT *."""
    src = tmp_path / "src.db"
    _make_source_db(src)
    out = tmp_path / "durable.db"
    build_durable_db(src, out)

    b = sqlite3.connect(str(out))
    b.row_factory = sqlite3.Row
    try:
        row = b.execute(
            "SELECT score, score_components, category_tag, risk_level "
            "FROM recommendation_outputs WHERE rec_id=1"
        ).fetchone()
        assert row["score"] == 0.9
        assert row["risk_level"] == "low"
        assert row["category_tag"] == "consumable"
        assert row["score_components"] == '{"opp":1}'
    finally:
        b.close()


def test_durable_tables_copied_row_for_row(tmp_path: Path) -> None:
    src = tmp_path / "src.db"
    _make_source_db(src, n_forecasts=5)
    out = tmp_path / "durable.db"
    result = build_durable_db(src, out)

    assert result.tables_copied["forecast_outputs"] == 5
    assert result.tables_copied["recommendation_outputs"] == 1
    assert result.tables_copied["items"] == 1

    b = sqlite3.connect(str(out))
    try:
        assert b.execute("SELECT COUNT(*) FROM forecast_outputs").fetchone()[0] == 5
    finally:
        b.close()


def test_backup_opens_valid_with_clean_fk_check(tmp_path: Path) -> None:
    src = tmp_path / "src.db"
    _make_source_db(src)
    out = tmp_path / "durable.db"
    build_durable_db(src, out)

    b = sqlite3.connect(str(out))
    try:
        assert b.execute("PRAGMA foreign_key_check").fetchall() == []
        assert b.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        # schema_versions is copied, so a restore knows which migrations ran
        assert b.execute("SELECT COUNT(*) FROM schema_versions").fetchone()[0] >= 1
    finally:
        b.close()


def test_build_overwrites_existing_output(tmp_path: Path) -> None:
    src = tmp_path / "src.db"
    _make_source_db(src)
    out = tmp_path / "durable.db"
    out.write_bytes(b"stale")  # pre-existing junk
    build_durable_db(src, out)
    b = sqlite3.connect(str(out))
    try:
        assert b.execute("SELECT COUNT(*) FROM forecast_outputs").fetchone()[0] == 3
    finally:
        b.close()


# ── gzip / key / prune ──────────────────────────────────────────────────────────


def test_gzip_file_roundtrips(tmp_path: Path) -> None:
    src = tmp_path / "src.db"
    _make_source_db(src)
    out = tmp_path / "durable.db"
    build_durable_db(src, out)
    original = out.read_bytes()

    gz_path, raw, gz = gzip_file(out)
    assert gz_path == out.with_suffix(".db.gz")
    assert raw == len(original)
    assert gz < raw
    assert gzip.decompress(gz_path.read_bytes()) == original


def test_build_object_key_is_utc_and_prefixed() -> None:
    key = build_object_key(datetime(2026, 7, 23, 18, 30, 5, tzinfo=UTC))
    assert key == "db_backups/2026/07/23/durable_20260723T183005Z.db.gz"


def test_build_object_key_converts_to_utc() -> None:
    from datetime import timedelta, timezone

    # 23:30 at +02:00 is 21:30 UTC the same day
    local = datetime(2026, 7, 23, 23, 30, 0, tzinfo=timezone(timedelta(hours=2)))
    assert build_object_key(local) == "db_backups/2026/07/23/durable_20260723T213000Z.db.gz"


def test_prune_local_keeps_newest_n(tmp_path: Path) -> None:
    names = [
        "durable_20260720T070000Z.db.gz",
        "durable_20260721T070000Z.db.gz",
        "durable_20260722T070000Z.db.gz",
        "durable_20260723T070000Z.db.gz",
    ]
    for n in names:
        (tmp_path / n).write_bytes(b"x")

    deleted = prune_local(tmp_path, keep=2)
    remaining = sorted(p.name for p in tmp_path.glob("durable_*.db.gz"))
    assert remaining == names[-2:]
    assert {p.name for p in deleted} == set(names[:2])


def test_prune_local_noop_when_under_limit(tmp_path: Path) -> None:
    (tmp_path / "durable_20260723T070000Z.db.gz").write_bytes(b"x")
    assert prune_local(tmp_path, keep=7) == []


# ── upload ──────────────────────────────────────────────────────────────────────


class _StubS3:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def put_object(self, **kwargs) -> dict:
        self.calls.append(kwargs)
        return {"ETag": "stub"}


def test_upload_backup_uses_client_with_expected_key_and_body(tmp_path: Path) -> None:
    gz = tmp_path / "durable.db.gz"
    gz.write_bytes(b"gzipped-bytes")
    stub = _StubS3()

    upload_backup(
        gz,
        endpoint="https://r2",
        bucket="backups",
        key="db_backups/2026/07/23/durable_x.db.gz",
        access_key="k",
        secret_key="s",
        s3_client=stub,
    )

    assert len(stub.calls) == 1
    call = stub.calls[0]
    assert call["Bucket"] == "backups"
    assert call["Key"] == "db_backups/2026/07/23/durable_x.db.gz"
    assert call["Body"] == b"gzipped-bytes"
    assert call["ContentType"] == "application/gzip"


# ── run_backup orchestration ────────────────────────────────────────────────────


def test_run_backup_no_upload_writes_gz_and_removes_db(tmp_path: Path) -> None:
    src = tmp_path / "src.db"
    _make_source_db(src)
    out_dir = tmp_path / "backups"
    cfg = _app_config(src, out_dir)

    now = datetime(2026, 7, 23, 12, 30, 0, tzinfo=UTC)
    result = run_backup(cfg, upload=False, now=now)

    assert result.uploaded is False
    assert result.gz_path is not None and result.gz_path.exists()
    assert result.gz_path.name == "durable_20260723T123000Z.db.gz"
    # intermediate uncompressed .db is removed
    assert not (out_dir / "durable_20260723T123000Z.db").exists()
    assert result.bytes_gz > 0
    assert result.object_key == "db_backups/2026/07/23/durable_20260723T123000Z.db.gz"


def test_run_backup_uploads_with_stub_and_env(tmp_path: Path, monkeypatch) -> None:
    src = tmp_path / "src.db"
    _make_source_db(src)
    cfg = _app_config(src, tmp_path / "backups")
    for var in (
        "BACKUP_S3_ENDPOINT",
        "BACKUP_S3_BUCKET",
        "BACKUP_S3_ACCESS_KEY_ID",
        "BACKUP_S3_SECRET_ACCESS_KEY",
    ):
        monkeypatch.setenv(var, "value")
    stub = _StubS3()

    result = run_backup(cfg, upload=True, now=datetime(2026, 7, 23, 7, 30, 0, tzinfo=UTC),
                        s3_client=stub)

    assert result.uploaded is True
    assert len(stub.calls) == 1
    assert stub.calls[0]["Key"] == result.object_key


def test_run_backup_missing_env_raises_naming_vars(tmp_path: Path, monkeypatch) -> None:
    src = tmp_path / "src.db"
    _make_source_db(src)
    cfg = _app_config(src, tmp_path / "backups")
    for var in (
        "BACKUP_S3_ENDPOINT",
        "BACKUP_S3_BUCKET",
        "BACKUP_S3_ACCESS_KEY_ID",
        "BACKUP_S3_SECRET_ACCESS_KEY",
    ):
        monkeypatch.delenv(var, raising=False)

    with pytest.raises(RuntimeError) as exc:
        run_backup(cfg, upload=True, now=datetime(2026, 7, 23, 7, 30, 0, tzinfo=UTC))
    msg = str(exc.value)
    assert "BACKUP_S3_ENDPOINT" in msg
    assert "BACKUP_S3_BUCKET" in msg
    # local backup still written before the upload attempt failed
    assert list((tmp_path / "backups").glob("durable_*.db.gz"))


def test_run_backup_prunes_old_local_copies(tmp_path: Path) -> None:
    src = tmp_path / "src.db"
    _make_source_db(src)
    out_dir = tmp_path / "backups"
    out_dir.mkdir()
    for old in ("durable_20260101T000000Z.db.gz", "durable_20260102T000000Z.db.gz"):
        (out_dir / old).write_bytes(b"x")
    cfg = _app_config(src, out_dir, keep=2)

    run_backup(cfg, upload=False, now=datetime(2026, 7, 23, 7, 30, 0, tzinfo=UTC))

    remaining = sorted(p.name for p in out_dir.glob("durable_*.db.gz"))
    assert len(remaining) == 2
    assert "durable_20260723T073000Z.db.gz" in remaining
    assert "durable_20260101T000000Z.db.gz" not in remaining
