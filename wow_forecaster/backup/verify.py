"""
Off-box verification of durable backups (issue #104).

The nightly backup (durable_backup.py) is uploaded but was never read back:
a backup corrupted at build time would upload green every night and only be
discovered at restore time. This module verifies the newest backup object on
hardware whose RAM is not the suspect component: it runs as a scheduled GitHub
Actions workflow (.github/workflows/verify-backup.yml), where the check is free
(public repo) and the 31 MB download costs nothing (R2 egress is free).

Checks, in order:

1. ``PRAGMA integrity_check`` and ``PRAGMA foreign_key_check`` on the restored
   file.
2. Row-count floors: the key durable tables must be non-empty.
3. No-shrink comparison for the append-only tables against the previous backup
   object. The backup never deletes rows from these tables, so a lower count
   means data was lost between builds.
4. Staleness: the newest object's key timestamp must be recent. A missing or
   stale object is a FAILURE, not a skip; exit-0 skips are what hid the 96-day
   outage.

Also runs locally against a downloaded file (the restore runbook's pre-restore
check, docs/integrity-incidents.md)::

    python -m wow_forecaster.backup.verify path/to/durable_x.db.gz

Bucket mode is env-only, mirroring cloud_fetch (no dotenv): the BACKUP_S3_*
variables from durable_backup.REQUIRED_ENV, with a read-only token in CI.
``BACKUP_VERIFY_MAX_AGE_HOURS`` overrides the staleness threshold (default 30,
matching the health check's --backup-stale-hours).

Exit codes:
  0 -- newest backup verified clean
  1 -- verification failure (corrupt, incomplete, shrunk, stale, or missing)
  2 -- configuration error (missing environment variables, named in the log)
"""

from __future__ import annotations

import gzip
import logging
import os
import re
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from wow_forecaster.backup.durable_backup import REQUIRED_ENV

logger = logging.getLogger(__name__)

# Durable tables that are non-empty in any real backup of the production DB.
# An empty one means the build copied nothing for it, whatever the cause.
DEFAULT_MIN_ROW_TABLES: tuple[str, ...] = (
    "items",
    "economic_archetypes",
    "daily_rollup_archetype",
    "daily_rollup_item",
    "forecast_outputs",
    "schema_versions",
)

# Tables no code path ever deletes from (the pruner explicitly exempts the
# rollups; forecasts are append-only). A count lower than the previous backup's
# is therefore evidence of loss, not housekeeping. If a retention policy is
# ever added for one of these, this check should go red and be adjusted
# consciously in the same change.
APPEND_ONLY_TABLES: tuple[str, ...] = (
    "daily_rollup_archetype",
    "daily_rollup_item",
    "forecast_outputs",
)

MAX_AGE_HOURS_DEFAULT = 30.0

_KEY_TS_RE = re.compile(r"durable_(\d{8}T\d{6})Z\.db\.gz$")


@dataclass
class VerifyResult:
    """Outcome of verifying one backup file.

    ``ok`` is True only when every check passed. ``integrity_errors`` carries
    the raw ``integrity_check`` messages (or a single "unreadable" entry when
    the file cannot be opened as a database at all).
    """

    integrity_errors: list[str] = field(default_factory=list)
    fk_violations:    list[tuple] = field(default_factory=list)
    table_counts:     dict[str, int] = field(default_factory=dict)
    floor_failures:   list[str] = field(default_factory=list)
    regressions:      list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not (
            self.integrity_errors
            or self.fk_violations
            or self.floor_failures
            or self.regressions
        )

    def failure_lines(self) -> list[str]:
        """Human-readable failure descriptions, empty when ok."""
        lines: list[str] = []
        for msg in self.integrity_errors:
            lines.append(f"integrity_check: {msg}")
        for row in self.fk_violations:
            lines.append(f"foreign_key_check: {row}")
        lines.extend(self.floor_failures)
        lines.extend(self.regressions)
        return lines


# ── Key parsing and selection (pure, no S3) ───────────────────────────────────


def parse_backup_key_timestamp(key: str) -> datetime | None:
    """Extract the UTC build timestamp from a backup object key, or None."""
    match = _KEY_TS_RE.search(key)
    if match is None:
        return None
    return datetime.strptime(match.group(1), "%Y%m%dT%H%M%S").replace(tzinfo=UTC)


def select_verify_keys(keys: list[str]) -> tuple[str | None, str | None]:
    """Return (newest, previous) backup keys by embedded timestamp.

    Unparseable keys are ignored. Listing order is irrelevant: the timestamp in
    the key name is authoritative, the same rule prune_local uses locally.
    """
    stamped = sorted(
        ((ts, k) for k in keys if (ts := parse_backup_key_timestamp(k)) is not None),
    )
    if not stamped:
        return None, None
    if len(stamped) == 1:
        return stamped[-1][1], None
    return stamped[-1][1], stamped[-2][1]


# ── Verification (local file, no S3) ──────────────────────────────────────────


def _user_tables(conn: Any) -> list[str]:
    rows = conn.execute(
        "SELECT name FROM sqlite_master "
        "WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
    ).fetchall()
    return [r[0] for r in rows]


def read_table_counts(db_path: str | Path) -> dict[str, int]:
    """Row count per user table, via a read-only connection."""
    import sqlite3

    conn = sqlite3.connect(f"file:{Path(db_path).as_posix()}?mode=ro", uri=True)
    try:
        return {
            name: conn.execute(f'SELECT COUNT(*) FROM "{name}"').fetchone()[0]
            for name in _user_tables(conn)
        }
    finally:
        conn.close()


def verify_backup_db(
    db_path: str | Path,
    min_row_tables: tuple[str, ...] = DEFAULT_MIN_ROW_TABLES,
    prev_counts: dict[str, int] | None = None,
) -> VerifyResult:
    """Run every content check against an uncompressed backup file.

    Never raises for a bad file: an unreadable or corrupt database comes back
    as a failed :class:`VerifyResult`, so the caller's exit-code mapping stays
    in one place.
    """
    import sqlite3

    result = VerifyResult()
    try:
        conn = sqlite3.connect(f"file:{Path(db_path).as_posix()}?mode=ro", uri=True)
    except sqlite3.Error as exc:
        result.integrity_errors.append(f"unreadable: {exc}")
        return result
    try:
        rows = conn.execute("PRAGMA integrity_check;").fetchall()
        messages = [r[0] for r in rows]
        if messages != ["ok"]:
            result.integrity_errors.extend(messages)

        result.fk_violations = [tuple(r) for r in conn.execute("PRAGMA foreign_key_check;")]

        for name in _user_tables(conn):
            result.table_counts[name] = conn.execute(
                f'SELECT COUNT(*) FROM "{name}"'
            ).fetchone()[0]
    except sqlite3.Error as exc:
        result.integrity_errors.append(f"unreadable: {exc}")
        return result
    finally:
        conn.close()

    for name in min_row_tables:
        if result.table_counts.get(name, 0) == 0:
            result.floor_failures.append(f"floor: {name} is empty or missing")

    if prev_counts is not None:
        for name in APPEND_ONLY_TABLES:
            prev = prev_counts.get(name)
            cur = result.table_counts.get(name)
            if prev is not None and cur is not None and cur < prev:
                result.regressions.append(
                    f"shrink: {name} has {cur} rows, previous backup had {prev}"
                )

    return result


# ── Bucket access ─────────────────────────────────────────────────────────────


def _make_s3_client(endpoint: str, access_key: str, secret_key: str, region: str) -> Any:
    """Read-only S3 client with explicit credentials (module-level so tests stub it)."""
    from wow_forecaster.backup.durable_backup import _make_s3_client as make

    return make(endpoint, access_key, secret_key, region)


def _list_backup_keys(s3: Any, bucket: str) -> list[str]:
    """List every key under db_backups/ (paginated; the lifecycle rule bounds it)."""
    keys: list[str] = []
    kwargs: dict[str, Any] = {"Bucket": bucket, "Prefix": "db_backups/"}
    while True:
        resp = s3.list_objects_v2(**kwargs)
        keys.extend(obj["Key"] for obj in resp.get("Contents", []))
        if not resp.get("IsTruncated"):
            return keys
        kwargs["ContinuationToken"] = resp["NextContinuationToken"]


def _download_gunzip(s3: Any, bucket: str, key: str, dest: Path) -> Path:
    """Download ``key`` and write the gunzipped bytes to ``dest``."""
    from wow_forecaster.ingestion.cloud_fetch import _retry

    body = _retry(
        lambda: s3.get_object(Bucket=bucket, Key=key)["Body"].read(),
        label=f"download {key}",
    )
    dest.write_bytes(gzip.decompress(body))
    return dest


# ── Entry point ───────────────────────────────────────────────────────────────


def _verify_local(path: Path) -> int:
    """Verify a local .db or .db.gz file (floors only; no previous to compare)."""
    if not path.exists():
        logger.error("No such file: %s", path)
        return 1
    with tempfile.TemporaryDirectory() as tmp:
        if path.suffix == ".gz":
            db_path = Path(tmp) / "backup.db"
            db_path.write_bytes(gzip.decompress(path.read_bytes()))
        else:
            db_path = path
        result = verify_backup_db(db_path)
    if result.ok:
        logger.info("Backup verified clean: %s (%d tables)", path, len(result.table_counts))
        return 0
    for line in result.failure_lines():
        logger.error("%s", line)
    return 1


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if args:
        return _verify_local(Path(args[0]))

    missing = [name for name in REQUIRED_ENV if not os.environ.get(name)]
    if missing:
        logger.error("Missing required environment variables: %s", ", ".join(missing))
        return 2

    max_age_hours = float(
        os.environ.get("BACKUP_VERIFY_MAX_AGE_HOURS", str(MAX_AGE_HOURS_DEFAULT))
    )
    bucket = os.environ["BACKUP_S3_BUCKET"]
    s3 = _make_s3_client(
        os.environ["BACKUP_S3_ENDPOINT"],
        os.environ["BACKUP_S3_ACCESS_KEY_ID"],
        os.environ["BACKUP_S3_SECRET_ACCESS_KEY"],
        os.environ.get("BACKUP_S3_REGION", "auto"),
    )

    from wow_forecaster.ingestion.cloud_fetch import _retry

    try:
        keys = _retry(lambda: _list_backup_keys(s3, bucket), label="backup listing")
    except Exception as exc:
        logger.error("Backup listing failed after retries: %s", exc)
        return 1

    newest, previous = select_verify_keys(keys)
    if newest is None:
        logger.error(
            "No backup objects found under db_backups/ -- the backup task may "
            "have stopped, or the token points at the wrong bucket"
        )
        return 1

    stale = False
    newest_ts = parse_backup_key_timestamp(newest)
    age_hours = (datetime.now(tz=UTC) - newest_ts).total_seconds() / 3600.0
    if age_hours > max_age_hours:
        logger.error(
            "Newest backup %s is %.1fh old (limit %.0fh) -- verifying it anyway, "
            "but the run fails: the backup task may have stopped",
            newest, age_hours, max_age_hours,
        )
        stale = True

    with tempfile.TemporaryDirectory() as tmp:
        try:
            newest_path = _download_gunzip(s3, bucket, newest, Path(tmp) / "newest.db")
        except Exception as exc:
            logger.error("Could not download or gunzip %s: %s", newest, exc)
            return 1

        prev_counts: dict[str, int] | None = None
        if previous is not None:
            try:
                prev_path = _download_gunzip(s3, bucket, previous, Path(tmp) / "previous.db")
                prev_counts = read_table_counts(prev_path)
            except Exception as exc:
                # The newest object is the verification target; a bad previous
                # only disables the shrink comparison, and loudly.
                logger.warning(
                    "Previous backup %s unusable (%s); skipping the shrink check",
                    previous, exc,
                )

        result = verify_backup_db(newest_path, prev_counts=prev_counts)

    if result.ok and not stale:
        logger.info(
            "Verified %s: integrity ok, %d tables, %d rows in forecast_outputs",
            newest, len(result.table_counts), result.table_counts.get("forecast_outputs", 0),
        )
        return 0
    for line in result.failure_lines():
        logger.error("%s", line)
    return 1


if __name__ == "__main__":
    # Configured under the guard so importing callers and tests never mutate
    # global logging state.
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    sys.exit(main())
