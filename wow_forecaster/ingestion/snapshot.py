"""
Snapshot persistence — save raw API payloads to disk as timestamped JSON files.

File layout::

    data/raw/snapshots/
      undermine/
        2026/02/24/
          area-52_neutral_20260224T150000Z.json
          illidan_neutral_20260224T150000Z.json
      blizzard_api/
        2026/02/24/
          realm_area-52_20260224T150000Z.json
          commodities_20260224T150000Z.json
      blizzard_news/
        2026/02/24/
          news_20260224T150000Z.json

Each file contains::

    {
      "_meta": {
        "source": "undermine",
        "realm": "area-52",
        "is_fixture": true,
        "run_slug": "...",
        "written_at": "2026-02-24T15:00:00Z"
      },
      "data": [ ... ]
    }

The ``_meta`` section enables reproducibility — any snapshot can be re-played
without knowing the original run context.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def build_snapshot_path(
    raw_dir: str,
    source: str,
    name: str,
    fetched_at: datetime,
) -> Path:
    """Build the deterministic filesystem path for a raw snapshot.

    The path encodes source, date (YYYY/MM/DD), name, and UTC timestamp so
    that snapshots are naturally sorted and collision-free.

    Args:
        raw_dir: Base raw data directory (e.g. ``"data/raw"``).
        source: Provider name (``"undermine"``, ``"blizzard_api"``,
            ``"blizzard_news"``).
        name: Descriptive label (e.g. ``"area-52_neutral"`` or ``"commodities"``).
        fetched_at: UTC datetime the fetch occurred.

    Returns:
        :class:`~pathlib.Path` for the snapshot file (parent dirs not created yet).

    Example::

        build_snapshot_path("data/raw", "undermine", "area-52_neutral",
                            datetime(2026, 2, 24, 15, 0, 0, tzinfo=timezone.utc))
        # → Path("data/raw/snapshots/undermine/2026/02/24/area-52_neutral_20260224T150000Z.json")
    """
    ts = fetched_at.strftime("%Y%m%dT%H%M%SZ")
    date_part = fetched_at.strftime("%Y/%m/%d")
    filename = f"{name}_{ts}.json"
    return Path(raw_dir) / "snapshots" / source / date_part / filename


def compute_hash(payload: Any) -> str:
    """Compute a SHA-256 hash of any JSON-serializable payload.

    Serialization uses ``sort_keys=True`` for determinism — identical data
    always produces the same hash regardless of dict key ordering.

    Args:
        payload: Any JSON-serializable object.

    Returns:
        Hex-encoded SHA-256 digest string (64 chars).
    """
    serialized = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def save_snapshot(
    path: Path,
    payload: Any,
    metadata: dict[str, Any] | None = None,
) -> tuple[str, int]:
    """Write a raw payload to disk as an envelope JSON file.

    Creates parent directories automatically. Handles plain dicts, lists,
    dataclass instances, and lists of dataclass instances.

    The written file structure::

        {
          "_meta": { ...metadata, "written_at": "..." },
          "data": <payload>
        }

    Args:
        path: Destination :class:`~pathlib.Path` for the snapshot.
        payload: Data to persist. May be a list, dict, dataclass, or list of
            dataclasses.
        metadata: Optional dict merged into the ``"_meta"`` envelope section.

    Returns:
        Tuple ``(content_hash, record_count)`` where ``content_hash`` is the
        SHA-256 of the full envelope and ``record_count`` is ``len(payload)``
        for lists, or 1 for scalar payloads.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    # Normalise dataclasses → plain dicts for JSON serialisation
    if is_dataclass(payload) and not isinstance(payload, type):
        serializable: Any = asdict(payload)
    elif (
        isinstance(payload, list)
        and payload
        and is_dataclass(payload[0])
        and not isinstance(payload[0], type)
    ):
        serializable = [asdict(item) for item in payload]
    else:
        serializable = payload

    record_count = len(serializable) if isinstance(serializable, list) else 1

    meta = dict(metadata or {})
    meta["written_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    envelope = {"_meta": meta, "data": serializable}
    content_hash = compute_hash(envelope)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(envelope, f, indent=2, default=str)

    logger.debug(
        "Snapshot saved: %s | records=%d | hash=%s…",
        path.name, record_count, content_hash[:12],
    )
    return content_hash, record_count


def load_snapshot(path: Path) -> dict[str, Any]:
    """Load a snapshot file from disk.

    Args:
        path: Path to a snapshot JSON file.

    Returns:
        Dict with ``"_meta"`` and ``"data"`` keys.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        json.JSONDecodeError: If the file is not valid JSON.
    """
    with open(path, encoding="utf-8") as f:
        return json.load(f)
