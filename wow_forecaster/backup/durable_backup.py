"""
Durable-table backup — a self-contained, restorable SQLite snapshot.

The forecaster's durable state (rollups, forecasts, recommendations, backtests,
drift/health snapshots, and the reference tables) lives only in the local SQLite
file. Once a day's underlying raw ages past the 30-day retention window the
pruner deletes it everywhere, so the rollups become the sole surviving record of
that day and cannot be regenerated. This module protects that state off-machine.

What it produces
----------------
A ``.db.gz``: a fresh SQLite file whose schema is copied verbatim from the live
database's ``sqlite_master`` (so migration-added columns and any future tables
are captured), holding data for every table **except** the two per-observation
tables. Those two tables are recreated empty, so the file is a drop-in restore.

Why copy the schema from ``sqlite_master`` rather than ``apply_schema()``: the
live ``recommendation_outputs`` carries migration-added columns (``score``,
``score_components``, ``category_tag``, ``risk_level``) that ``schema.py``'s DDL
does not declare. ``apply_schema()`` would build a narrower table and
``INSERT ... SELECT *`` would fail on the column-count mismatch. ``ALTER TABLE
ADD COLUMN`` updates the stored ``CREATE TABLE`` text, so ``sqlite_master`` is
always the true current shape.

Machine-safety: the build never reads the ~9.7 GB observation tables — only
their ``CREATE`` text (instant) plus the data of the small durable tables. No
``VACUUM`` (the inserts already produce a compact file).

Usage
-----
::

    from wow_forecaster.backup.durable_backup import run_backup
    result = run_backup(config, upload=True)
    print(result.gz_path, result.object_key, result.uploaded)
"""

from __future__ import annotations

import gzip
import logging
import os
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import sqlite3

    from wow_forecaster.config import AppConfig

logger = logging.getLogger(__name__)

# Data for these two tables is deliberately NOT copied — they are the transient,
# 30-day-retention per-observation tables, and the whole point of the backup is
# the durable state that outlives them. Their *schema* is still copied, so the
# restored file is drop-in complete (the tables exist, empty).
EXCLUDED_TABLES: frozenset[str] = frozenset(
    {"market_observations_raw", "market_observations_normalized"}
)

# Object-store key prefix. Distinct from the raw snapshots' ``blizzard_api/``
# prefix so a bucket lifecycle rule can target one without expiring the other.
KEY_PREFIX = "db_backups"

REQUIRED_ENV = (
    "BACKUP_S3_ENDPOINT",
    "BACKUP_S3_BUCKET",
    "BACKUP_S3_ACCESS_KEY_ID",
    "BACKUP_S3_SECRET_ACCESS_KEY",
)


@dataclass
class BackupResult:
    """Summary of one backup run.

    Attributes:
        out_path:      Path to the uncompressed ``.db`` (removed after gzip
                       unless the build stopped early).
        gz_path:       Path to the gzipped backup on local disk.
        tables_copied: Table name -> rows copied.
        bytes_raw:     Uncompressed ``.db`` size in bytes.
        bytes_gz:      Gzipped size in bytes.
        object_key:    Bucket key the ``.db.gz`` was (or would be) uploaded to.
        uploaded:      True when the object was uploaded to the store.
    """

    out_path:      Path
    gz_path:       Path | None = None
    tables_copied: dict[str, int] = field(default_factory=dict)
    bytes_raw:     int = 0
    bytes_gz:      int = 0
    object_key:    str = ""
    uploaded:      bool = False


# ── Build ─────────────────────────────────────────────────────────────────────


def _schema_objects(conn: sqlite3.Connection) -> list[tuple[str, str, str]]:
    """Return (type, name, sql) for every user object in the attached source.

    ``sql IS NOT NULL`` skips auto-indexes and ``sqlite_sequence``; the
    ``sqlite_%`` name filter also drops ``sqlite_stat*``. Tables are ordered
    before indexes so a fresh database can execute them in sequence.
    """
    rows = conn.execute(
        """
        SELECT type, name, sql
        FROM   src.sqlite_master
        WHERE  sql IS NOT NULL
          AND  name NOT LIKE 'sqlite_%'
        ORDER  BY CASE type WHEN 'table' THEN 0 WHEN 'index' THEN 2 ELSE 1 END,
                  name
        """
    ).fetchall()
    return [(r[0], r[1], r[2]) for r in rows]


def build_durable_db(
    src_db_path: str | Path,
    out_path: str | Path,
    busy_timeout_ms: int = 30000,
) -> BackupResult:
    """Build a durable-only SQLite file at ``out_path`` from ``src_db_path``.

    Copies every user table and index schema from the source, then copies data
    for every table except :data:`EXCLUDED_TABLES`. Runs ``foreign_key_check``
    and raises if the result is inconsistent.

    The source is only ever read; the destination is a fresh file in the default
    rollback-journal mode (never WAL), so there is no ``-wal`` to fold in before
    gzip.

    Returns:
        A :class:`BackupResult` with ``out_path``, ``tables_copied``, and
        ``bytes_raw`` populated (gzip/upload happen in :func:`run_backup`).
    """
    import sqlite3

    src_path = Path(src_db_path).resolve()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()

    result = BackupResult(out_path=out)

    # Autocommit mode (isolation_level=None) so the explicit BEGIN/COMMIT below
    # wraps the whole copy in one transaction: reads from the attached source
    # then see a single consistent snapshot even if an hourly ingest is writing.
    conn = sqlite3.connect(str(out), isolation_level=None)
    try:
        conn.execute("PRAGMA foreign_keys = OFF;")
        conn.execute(f"PRAGMA busy_timeout = {busy_timeout_ms};")
        conn.execute("ATTACH DATABASE ? AS src;", (str(src_path),))

        objects = _schema_objects(conn)
        tables = [name for typ, name, _ in objects if typ == "table"]
        indexes = [(name, sql) for typ, name, sql in objects if typ == "index"]
        others = [(name, sql) for typ, name, sql in objects if typ not in ("table", "index")]

        conn.execute("BEGIN;")

        # 1. Recreate every table (including the excluded ones, so the restore
        #    is drop-in complete). Table order from _schema_objects respects
        #    creation dependencies (parents before children).
        for typ, _name, sql in objects:
            if typ == "table":
                conn.execute(sql)

        # 2. Copy data for the durable tables only. Column order matches because
        #    each table was created from the source's exact CREATE statement.
        for name in tables:
            if name in EXCLUDED_TABLES:
                result.tables_copied[name] = 0
                continue
            conn.execute(f'INSERT INTO main."{name}" SELECT * FROM src."{name}";')
            (changed,) = conn.execute("SELECT changes();").fetchone()
            result.tables_copied[name] = int(changed)

        # 3. Indexes (and any views/triggers) after the bulk load — faster, and
        #    UNIQUE indexes still validate against the freshly loaded rows.
        for _name, sql in indexes:
            conn.execute(sql)
        for _name, sql in others:
            conn.execute(sql)

        conn.execute("COMMIT;")

        problems = conn.execute("PRAGMA foreign_key_check;").fetchall()
        if problems:
            raise RuntimeError(
                f"foreign_key_check found {len(problems)} violation(s) in the "
                f"durable backup: {problems[:5]}"
            )

        conn.execute("DETACH DATABASE src;")
    except Exception:
        conn.close()
        if out.exists():
            out.unlink()
        raise
    conn.close()

    result.bytes_raw = out.stat().st_size
    logger.info(
        "Built durable backup: %s (%d bytes, %d tables)",
        out, result.bytes_raw, len(result.tables_copied),
    )
    return result


# ── Compress / key / prune ──────────────────────────────────────────────────────


def gzip_file(path: str | Path) -> tuple[Path, int, int]:
    """Gzip ``path`` at level 9 to ``path + '.gz'``; return (gz_path, raw, gz)."""
    src = Path(path)
    gz_path = src.with_suffix(src.suffix + ".gz")
    raw = src.read_bytes()
    body = gzip.compress(raw, compresslevel=9)
    gz_path.write_bytes(body)
    return gz_path, len(raw), len(body)


def build_object_key(now: datetime, prefix: str = KEY_PREFIX) -> str:
    """Build the bucket key ``<prefix>/YYYY/MM/DD/durable_<ts>Z.db.gz`` (UTC)."""
    u = now.astimezone(UTC)
    return (
        f"{prefix}/{u:%Y/%m/%d}/durable_{u:%Y%m%dT%H%M%S}Z.db.gz"
    )


def prune_local(directory: str | Path, keep: int) -> list[Path]:
    """Delete all but the ``keep`` newest ``durable_*.db.gz`` files.

    Files are timestamped in their names, so a lexical sort is chronological;
    this avoids depending on filesystem mtimes. Returns the deleted paths.
    """
    if keep < 0:
        keep = 0
    d = Path(directory)
    if not d.exists():
        return []
    backups = sorted(d.glob("durable_*.db.gz"), key=lambda p: p.name)
    to_delete = backups[:-keep] if keep else backups
    deleted: list[Path] = []
    for p in to_delete:
        try:
            p.unlink()
            deleted.append(p)
        except OSError as exc:
            logger.warning("Could not prune old backup %s: %s", p, exc)
    return deleted


# ── Upload ──────────────────────────────────────────────────────────────────────


def _make_s3_client(endpoint: str, access_key: str, secret_key: str, region: str) -> Any:
    """Create an S3-compatible client with explicit credentials (Cloudflare R2).

    boto3 is imported lazily so the module imports without it; a clear message
    points at the ``[cloud]`` extra when it is missing. Credentials are passed
    explicitly (not via ambient ``AWS_*``) so the backup bucket can use its own
    scoped token, independent of the snapshots bucket.
    """
    try:
        import boto3
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise RuntimeError(
            "boto3 is required to upload backups. Install it with:\n"
            '    pip install -e ".[cloud]"'
        ) from exc
    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        region_name=region,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
    )


def upload_backup(
    gz_path: str | Path,
    *,
    endpoint: str,
    bucket: str,
    key: str,
    access_key: str,
    secret_key: str,
    region: str = "auto",
    s3_client: Any | None = None,
) -> None:
    """Upload ``gz_path`` to ``bucket`` at ``key`` with retries.

    ``s3_client`` may be injected (tests pass a stub); otherwise a client is
    built from the given endpoint/credentials. Upload is retried via the shared
    ``_retry`` helper.
    """
    from wow_forecaster.ingestion.cloud_fetch import _retry

    body = Path(gz_path).read_bytes()
    s3 = s3_client or _make_s3_client(endpoint, access_key, secret_key, region)
    _retry(
        lambda: s3.put_object(
            Bucket=bucket, Key=key, Body=body, ContentType="application/gzip"
        ),
        label="backup upload",
    )


# ── Orchestration ───────────────────────────────────────────────────────────────


def _resolve_upload_env() -> dict[str, str]:
    """Read BACKUP_S3_* from the environment; raise naming any missing vars."""
    missing = [name for name in REQUIRED_ENV if not os.environ.get(name)]
    if missing:
        raise RuntimeError(
            "Missing required environment variables for backup upload: "
            + ", ".join(missing)
            + '. Add them to .env (see docs/db-backup.md) or pass --no-upload.'
        )
    return {
        "endpoint":   os.environ["BACKUP_S3_ENDPOINT"],
        "bucket":     os.environ["BACKUP_S3_BUCKET"],
        "access_key": os.environ["BACKUP_S3_ACCESS_KEY_ID"],
        "secret_key": os.environ["BACKUP_S3_SECRET_ACCESS_KEY"],
        "region":     os.environ.get("BACKUP_S3_REGION", "auto"),
    }


def run_backup(
    config: AppConfig,
    *,
    upload: bool = True,
    keep_local: int | None = None,
    output_dir: str | Path | None = None,
    now: datetime | None = None,
    s3_client: Any | None = None,
) -> BackupResult:
    """Build, compress, prune, and (optionally) upload a durable backup.

    Order: build the local ``.db.gz`` and prune old local copies FIRST, then
    upload LAST — so a failed upload still leaves a good local backup and the
    error propagates to a non-zero exit.

    Args:
        config:     Application config (db path, backup settings).
        upload:     Upload to R2 when True (reads ``BACKUP_S3_*`` from the env).
        keep_local: Local copies to retain (default ``config.backup.keep_local``).
        output_dir: Local output directory (default ``config.backup.output_dir``).
        now:        Reference UTC time (injectable for tests).
        s3_client:  Optional injected S3 client (tests).

    Returns:
        The :class:`BackupResult`, with ``uploaded`` reflecting the outcome.
    """
    if now is None:
        now = datetime.now(tz=UTC)
    out_dir = Path(output_dir) if output_dir is not None else Path(config.backup.output_dir)
    keep = keep_local if keep_local is not None else config.backup.keep_local

    out_dir.mkdir(parents=True, exist_ok=True)
    db_name = f"durable_{now.astimezone(UTC):%Y%m%dT%H%M%S}Z.db"
    out_path = out_dir / db_name

    result = build_durable_db(
        config.database.db_path,
        out_path,
        busy_timeout_ms=config.database.busy_timeout_ms,
    )

    gz_path, bytes_raw, bytes_gz = gzip_file(out_path)
    result.gz_path = gz_path
    result.bytes_raw = bytes_raw
    result.bytes_gz = bytes_gz
    out_path.unlink()  # keep only the compressed copy
    result.object_key = build_object_key(now)

    prune_local(out_dir, keep)

    if upload:
        env = _resolve_upload_env()
        upload_backup(
            gz_path,
            endpoint=env["endpoint"],
            bucket=env["bucket"],
            key=result.object_key,
            access_key=env["access_key"],
            secret_key=env["secret_key"],
            region=env["region"],
            s3_client=s3_client,
        )
        result.uploaded = True
        logger.info("Uploaded durable backup to %s (%d bytes)", result.object_key, bytes_gz)

    return result
