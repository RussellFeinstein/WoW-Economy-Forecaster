"""
Cloud snapshot catch-up — list, download, and select R2 objects for local ingest.

The counterpart to :mod:`wow_forecaster.ingestion.cloud_fetch`. That module runs
in GitHub Actions and *writes* hourly commodities snapshots to a private R2
bucket; this one runs on the desktop and *reads* them back so hours captured
while the machine was asleep or powered off can still reach the database
(issue #43). Object keys, the snapshot envelope, and the on-disk layout are
identical in both directions, so a cloud-sourced snapshot is indistinguishable
from a locally-fetched one once written.

This module deliberately holds no database code. Selection is a pure function
over key names so ordering and deduplication are testable without S3 or SQLite;
the ingest half lives in :mod:`wow_forecaster.pipeline.sync_stage`.

Environment (read from ``.env`` on the desktop; never committed)::

    SNAPSHOT_S3_ENDPOINT           S3-compatible endpoint URL (R2)
    SNAPSHOT_S3_BUCKET             private snapshot bucket name
    SNAPSHOT_S3_ACCESS_KEY_ID      read-scoped access key
    SNAPSHOT_S3_SECRET_ACCESS_KEY  read-scoped secret
    SNAPSHOT_S3_REGION             optional signing region (default "auto")

The names are explicit rather than boto3's bare ``AWS_*`` because the desktop
may have unrelated AWS credentials in its environment. ``cloud_fetch`` keeps
using ``AWS_*``: that is boto3's standard resolution on the Actions runner,
where nothing else competes for it.
"""

from __future__ import annotations

import gzip
import json
import logging
import os
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path, PurePosixPath
from typing import Any

from wow_forecaster.ingestion.cloud_fetch import _retry, parse_key_timestamp

logger = logging.getLogger(__name__)

REQUIRED_ENV = (
    "SNAPSHOT_S3_ENDPOINT",
    "SNAPSHOT_S3_BUCKET",
    "SNAPSHOT_S3_ACCESS_KEY_ID",
    "SNAPSHOT_S3_SECRET_ACCESS_KEY",
)

KEY_PREFIX = "blizzard_api"

# Must match STALE_MINUTES in scripts/run_hourly.bat. A lock older than this is
# treated as leaked by a crashed run (the failure that wedged ingestion for 96
# days in 2026) and is taken over rather than waited on.
LOCK_STALE_MINUTES = 180


# ── Result types ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class CloudObject:
    """One bucket object that is a candidate for ingestion.

    Attributes:
        key:         Bucket key, e.g.
            ``blizzard_api/2026/07/23/commodities_us_20260723T231619Z.json.gz``.
        captured_at: Capture timestamp parsed from the key, naive UTC to match
            what the database stores.
        local_path:  Where the decompressed snapshot will be written on disk.
    """

    key:         str
    captured_at: datetime
    local_path:  Path


@dataclass
class SkipCounts:
    """Why candidate objects were not selected, one counter per reason."""

    unparseable_key:  int = 0
    beyond_retention: int = 0
    already_ingested: int = 0
    hour_covered:     int = 0
    duplicate_hour:   int = 0
    over_limit:       int = 0

    def total(self) -> int:
        return (
            self.unparseable_key
            + self.beyond_retention
            + self.already_ingested
            + self.hour_covered
            + self.duplicate_hour
            + self.over_limit
        )

    def summary(self) -> str:
        """One-line breakdown, omitting reasons that skipped nothing."""
        parts = [
            f"{name}={value}"
            for name, value in (
                ("unparseable", self.unparseable_key),
                ("beyond_retention", self.beyond_retention),
                ("already_ingested", self.already_ingested),
                ("hour_covered", self.hour_covered),
                ("duplicate_hour", self.duplicate_hour),
                ("over_limit", self.over_limit),
            )
            if value
        ]
        return ", ".join(parts) if parts else "none"


@dataclass
class SyncResult:
    """Outcome of one ``sync-snapshots`` run."""

    listed:                int = 0
    selected:              int = 0
    ingested:              int = 0
    observations_inserted: int = 0
    items_skipped_fk:      int = 0
    normalized_rows:       int = 0
    dry_run:               bool = False
    truncated:             bool = False
    skips:                 SkipCounts = field(default_factory=SkipCounts)
    dates_touched:         list[str] = field(default_factory=list)
    failures:              list[tuple[str, str]] = field(default_factory=list)


# ── Key and path mapping ──────────────────────────────────────────────────────

def local_path_for_key(raw_dir: str | Path, key: str) -> Path:
    """Map a bucket key back to the local snapshot path it will be written to.

    Exact inverse of :func:`cloud_fetch.build_object_key`, which takes the path
    ``build_snapshot_path()`` produces, drops everything up to and including the
    ``snapshots`` component, and appends ``.gz``.
    """
    stem = key[: -len(".gz")] if key.endswith(".gz") else key
    return Path(raw_dir).joinpath("snapshots", *PurePosixPath(stem).parts)


def _hour_of(moment: datetime) -> datetime:
    """Truncate a timestamp to the top of its UTC hour."""
    return moment.replace(minute=0, second=0, microsecond=0)


# ── Selection (pure) ──────────────────────────────────────────────────────────

def select_objects_to_ingest(
    keys: Sequence[str],
    *,
    raw_dir: str | Path,
    already_ingested: set[str],
    hours_covered: set[datetime],
    retention_cutoff: datetime,
    max_objects: int | None = None,
) -> tuple[list[CloudObject], SkipCounts]:
    """Decide which bucket objects to ingest, in the order to ingest them.

    Pure: no S3, no database, no clock. Every filter is a set membership or a
    comparison against a value the caller resolved, which is what makes the
    ordering and deduplication rules testable on their own.

    Filters apply in this order, and the order matters:

    1. Keys whose timestamp cannot be parsed are ignored (foreign objects).
    2. Captures older than ``retention_cutoff`` are dropped. ``SnapshotPruner``
       deletes rows by ``observed_at`` on the next hourly run, so ingesting them
       would write rows destined for immediate deletion (API ToS section 2.r).
    3. Objects already ingested are dropped, which is what makes re-running a
       no-op.
    4. Objects whose UTC hour already has local observations are dropped. The
       desktop's own hourly run and the cloud capture fetch the *same* AH
       snapshot (``fetched_at`` is client-side wall clock, not the snapshot's own
       modification time), so without this an overlapping hour is counted twice.
    5. At most one object survives per UTC hour, the earliest. The bucket holds
       two captures an hour (:16 and :46 from the Worker, plus the :06 GitHub
       fallback); ingesting all of them would silently triple the sample density
       of catch-up hours relative to every hour the desktop captured itself.
    6. Oldest first, so rolling normalization stats and daily rollups build
       forward chronologically.
    7. Truncated to ``max_objects``. The count dropped is reported rather than
       silently discarded, so a capped run never reads as a complete one.

    Args:
        keys:             Candidate bucket keys, any order.
        raw_dir:          Local raw data directory, for deriving snapshot paths.
        already_ingested: Local snapshot paths already recorded as ingested.
        hours_covered:    UTC hours (truncated) that already have observations.
        retention_cutoff: Captures older than this are refused.
        max_objects:      Cap on how many to return; ``None`` for no cap.

    Returns:
        Tuple of ``(selected, skips)``, selected oldest first.
    """
    skips = SkipCounts()
    ingested_paths = {str(p) for p in already_ingested}
    survivors: list[CloudObject] = []

    for key in keys:
        captured_at = parse_key_timestamp(key)
        if captured_at is None:
            skips.unparseable_key += 1
            continue
        # The database stores naive UTC; parse_key_timestamp returns aware.
        captured_at = captured_at.replace(tzinfo=None)

        if captured_at < retention_cutoff:
            skips.beyond_retention += 1
            continue

        local_path = local_path_for_key(raw_dir, key)
        if str(local_path) in ingested_paths:
            skips.already_ingested += 1
            continue

        if _hour_of(captured_at) in hours_covered:
            skips.hour_covered += 1
            continue

        survivors.append(CloudObject(key, captured_at, local_path))

    # One per UTC hour, earliest wins. Sort first so "earliest" is deterministic
    # regardless of the order the bucket listed them in.
    survivors.sort(key=lambda o: (o.captured_at, o.key))
    per_hour: dict[datetime, CloudObject] = {}
    for obj in survivors:
        hour = _hour_of(obj.captured_at)
        if hour in per_hour:
            skips.duplicate_hour += 1
            continue
        per_hour[hour] = obj

    selected = sorted(per_hour.values(), key=lambda o: o.captured_at)

    if max_objects is not None and len(selected) > max_objects:
        skips.over_limit = len(selected) - max_objects
        selected = selected[:max_objects]

    return selected, skips


# ── S3 access ─────────────────────────────────────────────────────────────────

def resolve_s3_env() -> dict[str, str]:
    """Read ``SNAPSHOT_S3_*`` from the environment; raise naming any missing.

    Values are never logged, only names.
    """
    missing = [name for name in REQUIRED_ENV if not os.environ.get(name)]
    if missing:
        raise RuntimeError(
            "Missing required environment variables for cloud snapshot sync: "
            + ", ".join(missing)
            + ". Add them to .env (see docs/cloud-capture.md)."
        )
    return {
        "endpoint":   os.environ["SNAPSHOT_S3_ENDPOINT"],
        "bucket":     os.environ["SNAPSHOT_S3_BUCKET"],
        "access_key": os.environ["SNAPSHOT_S3_ACCESS_KEY_ID"],
        "secret_key": os.environ["SNAPSHOT_S3_SECRET_ACCESS_KEY"],
        "region":     os.environ.get("SNAPSHOT_S3_REGION", "auto"),
    }


def make_s3_client(env: dict[str, str]) -> Any:
    """Create an S3-compatible client from :func:`resolve_s3_env` output.

    boto3 is imported lazily so this module imports without it; tests inject a
    stub client instead of patching boto3.
    """
    try:
        import boto3
    except ImportError as exc:  # pragma: no cover - exercised by hand
        raise RuntimeError(
            "boto3 is required to sync cloud snapshots. Install it with:\n"
            '    pip install -e ".[cloud]"'
        ) from exc

    return boto3.client(
        "s3",
        endpoint_url=env["endpoint"],
        region_name=env["region"],
        aws_access_key_id=env["access_key"],
        aws_secret_access_key=env["secret_key"],
    )


def list_objects_since(
    s3: Any,
    bucket: str,
    since: datetime,
    now: datetime,
) -> list[str]:
    """List every snapshot key under the day prefixes spanning ``since``..``now``.

    Walks one prefix per UTC date rather than listing the bucket flat, matching
    ``cloud_fetch.list_recent_keys``. Pagination is followed to the end: a day
    prefix holding more than one page would otherwise truncate silently, and a
    short listing reads exactly like a quiet day.
    """
    keys: list[str] = []
    day = since.date()
    last = now.date()
    while day <= last:
        prefix = f"{KEY_PREFIX}/{day.strftime('%Y/%m/%d')}/"
        token: str | None = None
        while True:
            kwargs: dict[str, Any] = {"Bucket": bucket, "Prefix": prefix}
            if token:
                kwargs["ContinuationToken"] = token
            resp = _retry(
                lambda kw=kwargs: s3.list_objects_v2(**kw),
                label=f"list {prefix}",
            )
            keys.extend(obj["Key"] for obj in resp.get("Contents", []))
            if not resp.get("IsTruncated"):
                break
            token = resp.get("NextContinuationToken")
            if not token:
                break
        day += timedelta(days=1)
    return keys


def download_snapshot(s3: Any, bucket: str, key: str) -> tuple[dict[str, Any], bytes]:
    """Download one gzipped snapshot object; return its envelope and raw bytes.

    The decompressed bytes are returned alongside the parsed envelope so the
    caller can write the file to disk verbatim.  Re-serializing would change
    the ``_meta`` block (``written_at``, ``run_slug``, ``fetcher``) and lose the
    record of where the snapshot actually came from.

    Raises:
        ValueError: If the object is not valid gzip, not valid JSON, or does not
            carry the ``_meta``/``data`` envelope the local pipeline writes.
    """
    resp = _retry(
        lambda: s3.get_object(Bucket=bucket, Key=key),
        label=f"download {key}",
    )
    body = resp["Body"].read()
    try:
        raw = gzip.decompress(body)
    except (OSError, EOFError, gzip.BadGzipFile) as exc:
        raise ValueError(f"{key} is not valid gzip: {exc}") from exc
    try:
        envelope = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{key} is not valid JSON: {exc}") from exc
    if not isinstance(envelope, dict) or "data" not in envelope:
        raise ValueError(f"{key} is missing the 'data' envelope section")
    if not isinstance(envelope["data"], list):
        raise ValueError(f"{key} envelope 'data' is not a list")
    return envelope, raw


# ── Write lock ────────────────────────────────────────────────────────────────

@contextmanager
def hourly_lock(
    lock_path: str | Path,
    *,
    wait_seconds: float = 900.0,
    stale_minutes: float = LOCK_STALE_MINUTES,
    poll_seconds: float = 5.0,
) -> Iterator[None]:
    """Hold ``data/db/.hourly.lock`` for the duration of the block.

    The same lock ``scripts/run_hourly.bat`` takes, so a catch-up run and an
    overrunning hourly run cannot write to the database at the same time. Bulk
    inserts here run to hundreds of thousands of rows and can exceed the 30s
    busy timeout, so relying on SQLite's own serialization is not enough.

    Semantics mirror the bat with one deliberate difference: waiting. The bat
    skips when the lock is fresh because a skipped hourly run costs one sample;
    a skipped catch-up would silently leave a whole night unrecovered, so this
    waits instead, then fails loudly rather than exiting quietly (the exit-0
    skip is precisely what hid the 96-day outage).

    A lock older than ``stale_minutes`` is taken over, matching the bat.

    Raises:
        TimeoutError: If the lock stays held by a live run past ``wait_seconds``.
    """
    path = Path(lock_path)
    deadline = time.monotonic() + wait_seconds
    acquired = False

    while True:
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            # Exclusive create: atomic, so two racing syncs cannot both win.
            with open(path, "x", encoding="utf-8") as handle:
                handle.write(f"sync-snapshots {datetime.now().isoformat()}\n")
            acquired = True
            break
        except FileExistsError:
            age_minutes = _lock_age_minutes(path)
            if age_minutes is None:
                # Vanished between the failed create and the stat: retry at once.
                continue
            if age_minutes > stale_minutes:
                logger.warning(
                    "STALE LOCK TAKEOVER: %s is %.0f minutes old (limit %.0f); "
                    "deleting and continuing",
                    path, age_minutes, stale_minutes,
                )
                try:
                    path.unlink()
                except FileNotFoundError:
                    pass
                continue
            if time.monotonic() >= deadline:
                # from None: the FileExistsError is expected control flow here,
                # not the cause worth chaining into the timeout.
                raise TimeoutError(
                    f"{path} held by another run for longer than {wait_seconds:.0f}s "
                    f"(lock age {age_minutes:.1f} min). The hourly pipeline is "
                    "still running; retry once it finishes."
                ) from None
            logger.info(
                "Waiting for %s (age %.1f min) before syncing", path, age_minutes
            )
            time.sleep(poll_seconds)

    try:
        yield
    finally:
        if acquired:
            try:
                path.unlink()
            except FileNotFoundError:
                logger.warning("Lock %s was already gone at release", path)


def _lock_age_minutes(path: Path) -> float | None:
    """Age of the lock file in minutes, or None if it vanished or cannot be read."""
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return None
    return (time.time() - mtime) / 60.0
