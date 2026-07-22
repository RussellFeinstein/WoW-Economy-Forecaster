"""
Cloud snapshot fetcher — hourly commodities capture that does not depend on the desktop.

Runs as a GitHub Actions scheduled workflow (.github/workflows/cloud-snapshot.yml)
so a slept or powered-off desktop no longer loses hours; the commodities endpoint
serves only the current snapshot, so a missed hour is unrecoverable. Design record,
sizing, and the one-time activation checklist: docs/cloud-capture.md (issue #41;
this module implements #42, and #43 ingests the backlog).

Reuses ``BlizzardClient.fetch_commodities()``, ``build_snapshot_path()``, and
``save_snapshot()`` so cloud objects carry the same envelope the local pipeline
writes, by construction. The import chain is stdlib plus lazily imported httpx
and boto3, which lets the workflow install the package with ``--no-deps``::

    pip install --no-deps .
    pip install httpx boto3
    python -m wow_forecaster.ingestion.cloud_fetch

Environment (set as GitHub Actions secrets; values are never logged):
  BLIZZARD_CLIENT_ID / BLIZZARD_CLIENT_SECRET  — Blizzard OAuth2 client credentials
  SNAPSHOT_S3_ENDPOINT                         — S3-compatible endpoint URL (R2)
  SNAPSHOT_S3_BUCKET                           — private bucket name
  AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY    — bucket credentials (boto3 standard)
  SNAPSHOT_S3_REGION                           — optional signing region (default "auto")
  BLIZZARD_REGION                              — optional WoW region (default "us")
  CLOUD_FETCH_MIN_RECORDS                      — optional sanity floor override
  CLOUD_FETCH_GUARD_MIN_HOURS                  — optional gap-guard floor override

Exit codes:
  0 — snapshot uploaded and the trailing 24 hours look healthy
  1 — fetch, sanity, or upload failure (nothing stored, or store failed)
  2 — configuration error (missing environment variables, named in the log)
  3 — snapshot uploaded, but the trailing 24 hours have gaps (gap guard tripped)
"""

from __future__ import annotations

import gzip
import logging
import os
import re
import sys
import tempfile
import time
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any

from wow_forecaster.ingestion.blizzard_client import BlizzardAuctionRecord, BlizzardClient
from wow_forecaster.ingestion.snapshot import build_snapshot_path, save_snapshot

logger = logging.getLogger(__name__)

REQUIRED_ENV = (
    "BLIZZARD_CLIENT_ID",
    "BLIZZARD_CLIENT_SECRET",
    "SNAPSHOT_S3_ENDPOINT",
    "SNAPSHOT_S3_BUCKET",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
)

# A healthy commodities snapshot has roughly 314K records (measured 2026-04-15).
# Far fewer means the API served a brownout or partial payload; refuse to store it.
MIN_RECORDS_DEFAULT = 50_000

# Distinct UTC hours covered in the trailing 24h below which the gap guard
# trips. Counting hours instead of objects keeps the guard meaning "hours are
# being missed" at any cron density: the 3x/hour schedule (#67) can leave ~72
# objects on a day that still misses a third of its hours.
GUARD_MIN_HOURS_DEFAULT = 20

_KEY_TS_RE = re.compile(r"_(\d{8}T\d{6}Z)\.json\.gz$")


def records_to_dicts(records: list[BlizzardAuctionRecord]) -> list[dict[str, Any]]:
    """Project auction records onto the seven-key dict shape local snapshots use.

    Mirrors ``IngestStage._fetch_blizzard_commodities()`` so cloud objects and
    local snapshot files carry identical data records.
    """
    return [
        {
            "item_id": r.item_id,
            "realm_slug": r.realm_slug,
            "buyout": r.buyout,
            "bid": r.bid,
            "unit_price": r.unit_price,
            "quantity": r.quantity,
            "time_left": r.time_left,
        }
        for r in records
    ]


def build_object_key(region: str, fetched_at: datetime) -> str:
    """Build the bucket key mirroring the local snapshot layout, plus ``.gz``.

    Derived from :func:`build_snapshot_path` so the key can never drift from
    the filename the local pipeline writes.
    """
    path = build_snapshot_path("", "blizzard_api", f"commodities_{region}", fetched_at)
    parts = path.parts[path.parts.index("snapshots") + 1 :]
    return "/".join(parts) + ".gz"


def parse_key_timestamp(key: str) -> datetime | None:
    """Extract the UTC capture timestamp from a bucket key, or None on no match."""
    match = _KEY_TS_RE.search(key)
    if match is None:
        return None
    return datetime.strptime(match.group(1), "%Y%m%dT%H%M%SZ").replace(tzinfo=UTC)


def evaluate_gap_guard(
    keys: list[str],
    now: datetime,
    min_hours: int = GUARD_MIN_HOURS_DEFAULT,
) -> tuple[bool, str]:
    """Decide whether the trailing 24 hours of captures look healthy.

    The metric is distinct UTC hours covered in the trailing 24 hours, so
    duplicate captures within an hour (the 3x/hour cron) never mask missed
    hours. Returns ``(ok, detail)``. Bootstrap rule: when no listed object is
    older than 24 hours, low coverage is expected (first day of operation, or
    a resume after a 48h+ outage whose failed runs already alerted) and passes.
    """
    stamps = [ts for ts in (parse_key_timestamp(k) for k in keys) if ts is not None]
    cutoff = now - timedelta(hours=24)
    hours_covered = len({
        ts.replace(minute=0, second=0, microsecond=0) for ts in stamps if ts >= cutoff
    })
    older = sum(1 for ts in stamps if ts < cutoff)
    if hours_covered >= min_hours:
        return True, (
            f"{hours_covered} distinct hours covered in the trailing 24h "
            f"(minimum {min_hours})"
        )
    if older == 0:
        return True, (
            f"bootstrap: {hours_covered} distinct hours covered in the "
            "trailing 24h, none older"
        )
    return False, (
        f"only {hours_covered} distinct hours covered in the trailing 24h "
        f"(minimum {min_hours}); hours are being missed"
    )


def make_s3_client(endpoint: str, region_name: str = "auto") -> Any:
    """Create an S3-compatible client (Cloudflare R2 by default).

    boto3 is imported lazily so the module imports without it; tests inject a
    stub client instead. Credentials come from the standard AWS_* environment
    variables via boto3's own resolution chain.
    """
    import boto3

    return boto3.client("s3", endpoint_url=endpoint, region_name=region_name)


def list_recent_keys(s3: Any, bucket: str, now: datetime) -> list[str]:
    """List object keys under the today, yesterday, and day-before-yesterday prefixes.

    Three prefixes keep the listing reaching past the guard's 24-hour cutoff
    even just after UTC midnight, when every object older than 24 hours lives
    under the day-before-yesterday prefix. With two prefixes the guard's
    ``older`` count read 0 in that window and the bootstrap rule fired on days
    that genuinely had gaps (issue #68).
    """
    keys: list[str] = []
    for day in (now - timedelta(days=2), now - timedelta(days=1), now):
        prefix = f"blizzard_api/{day.strftime('%Y/%m/%d')}/"
        resp = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
        keys.extend(obj["Key"] for obj in resp.get("Contents", []))
    return keys


def _retry(
    fn: Callable[[], Any],
    label: str,
    attempts: int = 3,
    delays: tuple[int, ...] = (10, 30),
) -> Any:
    """Run ``fn`` with retries; sleep ``delays[i]`` seconds after failure i+1."""
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except Exception as exc:
            if attempt == attempts:
                raise
            delay = delays[min(attempt - 1, len(delays) - 1)]
            logger.warning(
                "%s failed (attempt %d/%d): %s; retrying in %ds",
                label, attempt, attempts, exc, delay,
            )
            time.sleep(delay)


def main() -> int:
    missing = [name for name in REQUIRED_ENV if not os.environ.get(name)]
    if missing:
        logger.error("Missing required environment variables: %s", ", ".join(missing))
        return 2

    region = os.environ.get("BLIZZARD_REGION", "us")
    min_records = int(os.environ.get("CLOUD_FETCH_MIN_RECORDS", str(MIN_RECORDS_DEFAULT)))
    min_hours = int(
        os.environ.get("CLOUD_FETCH_GUARD_MIN_HOURS", str(GUARD_MIN_HOURS_DEFAULT))
    )
    endpoint = os.environ["SNAPSHOT_S3_ENDPOINT"]
    bucket = os.environ["SNAPSHOT_S3_BUCKET"]

    client = BlizzardClient(
        client_id=os.environ["BLIZZARD_CLIENT_ID"],
        client_secret=os.environ["BLIZZARD_CLIENT_SECRET"],
        region=region,
    )
    try:
        response = _retry(client.fetch_commodities, label="commodities fetch")
    except Exception as exc:
        logger.error("Commodities fetch failed after retries: %s", exc)
        return 1

    if response.is_fixture:
        logger.error("Refusing to upload fixture data")
        return 1
    if len(response.records) < min_records:
        logger.error(
            "Snapshot has %d records, below the sanity minimum of %d; refusing to store it",
            len(response.records), min_records,
        )
        return 1

    records_data = records_to_dicts(response.records)
    run_slug = f"gha_{os.environ.get('GITHUB_RUN_ID', 'manual')}"
    with tempfile.TemporaryDirectory() as tmp:
        path = build_snapshot_path(
            tmp, "blizzard_api", f"commodities_{region}", response.fetched_at
        )
        content_hash, record_count = save_snapshot(
            path,
            records_data,
            metadata={
                "source": "blizzard_api",
                "type": "commodities",
                "region": region,
                "is_fixture": response.is_fixture,
                "run_slug": run_slug,
                "fetcher": "cloud",
            },
        )
        raw_bytes = path.read_bytes()
    body = gzip.compress(raw_bytes, compresslevel=9)
    key = build_object_key(region, response.fetched_at)

    s3 = make_s3_client(endpoint, os.environ.get("SNAPSHOT_S3_REGION", "auto"))
    try:
        _retry(
            lambda: s3.put_object(
                Bucket=bucket, Key=key, Body=body, ContentType="application/gzip"
            ),
            label="snapshot upload",
        )
    except Exception as exc:
        logger.error("Snapshot upload failed after retries: %s", exc)
        return 1

    logger.info(
        "Uploaded %s: %d records, %d bytes raw, %d bytes gzipped, hash %s",
        key, record_count, len(raw_bytes), len(body), content_hash[:12],
    )

    now = datetime.now(UTC)
    try:
        keys = list_recent_keys(s3, bucket, now)
    except Exception as exc:
        logger.error("Snapshot uploaded, but the gap-guard listing failed: %s", exc)
        return 3
    ok, detail = evaluate_gap_guard(keys, now, min_hours)
    if not ok:
        logger.error("Snapshot uploaded, but the gap guard tripped: %s", detail)
        return 3
    logger.info("Gap guard: %s", detail)
    return 0


if __name__ == "__main__":
    # Configured here rather than in main() so importing callers and tests
    # never mutate global logging state.
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    sys.exit(main())
