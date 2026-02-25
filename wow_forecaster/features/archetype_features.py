"""
Archetype and category encoding, cold-start detection, and transfer features.

Purpose
-------
Adds static (per-archetype, per-realm) columns to each row:

- Category and sub-tag string encoding (from ``economic_archetypes`` table).
- Cold-start flag: True when an archetype series for the transfer-target
  expansion has fewer than ``cold_start_threshold`` total observations.
- Transfer mapping presence and confidence (from ``archetype_mappings`` table).
- Item count (number of distinct items contributing to this series).

Why static columns?
-------------------
These values do not change day-to-day for a given (archetype_id, realm_slug)
series.  Adding them once per row rather than joining at query time keeps the
SQL simple and keeps the feature logic explicit and testable in isolation.

Cold-start definition
---------------------
``is_cold_start = True`` when:
    1.  The expansion_slug of the items in this archetype series equals
        ``config.expansions.transfer_target`` (e.g. "midnight"), AND
    2.  The total count of non-outlier observations for this (archetype, realm)
        is below ``config.features.cold_start_threshold`` (default 30).

This correctly identifies newly-launched expansion items that lack sufficient
price history to train on directly and therefore benefit most from transfer
learning from TWW archetypes.

Items with archetype_id = NULL in the ``items`` table are excluded entirely
from the feature pipeline (not grouped under a phantom archetype).  Their
count is surfaced in the quality report.

Input → Output
--------------
Input:  list[dict] with obs_date, archetype_id, realm_slug keys
Output: same list augmented with archetype / transfer feature keys
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ArchetypeMetadata:
    """Static attributes for one economic archetype.

    Attributes:
        archetype_id:              Primary key from economic_archetypes.
        category_tag:              ArchetypeCategory slug (e.g. "consumable").
        sub_tag:                   Full ArchetypeTag slug or None.
        is_transferable:           Can be mapped to the transfer-target expansion.
        transfer_confidence:       Prior confidence score from the archetype row.
        has_transfer_mapping:      True if an archetype_mapping exists.
        transfer_mapping_confidence: Max confidence_score across all mappings, or None.
    """

    archetype_id: int
    category_tag: str
    sub_tag: str | None
    is_transferable: bool
    transfer_confidence: float
    has_transfer_mapping: bool
    transfer_mapping_confidence: float | None


# ── DB helpers ─────────────────────────────────────────────────────────────────

def load_archetype_metadata(
    conn: sqlite3.Connection,
    source_expansion: str,
    target_expansion: str,
) -> dict[int, ArchetypeMetadata]:
    """Load archetype metadata with transfer mapping info for all archetypes.

    Returns a dict keyed by ``archetype_id``.

    Args:
        conn:              Open SQLite connection with Row factory.
        source_expansion:  Usually "tww" — the expansion we have historical data for.
        target_expansion:  Usually "midnight" — the expansion we are predicting.
    """
    rows = conn.execute(
        """
        SELECT
            ea.archetype_id,
            ea.category_tag,
            ea.sub_tag,
            ea.is_transferable,
            ea.transfer_confidence,
            CASE WHEN am.max_confidence IS NOT NULL THEN 1 ELSE 0 END AS has_mapping,
            am.max_confidence
        FROM economic_archetypes ea
        LEFT JOIN (
            SELECT
                source_archetype_id,
                MAX(confidence_score) AS max_confidence
            FROM archetype_mappings
            WHERE source_expansion = ?
              AND target_expansion  = ?
            GROUP BY source_archetype_id
        ) am ON ea.archetype_id = am.source_archetype_id
        """,
        (source_expansion, target_expansion),
    ).fetchall()

    result: dict[int, ArchetypeMetadata] = {}
    for r in rows:
        result[r["archetype_id"]] = ArchetypeMetadata(
            archetype_id=r["archetype_id"],
            category_tag=r["category_tag"],
            sub_tag=r["sub_tag"],
            is_transferable=bool(r["is_transferable"]),
            transfer_confidence=float(r["transfer_confidence"]),
            has_transfer_mapping=bool(r["has_mapping"]),
            transfer_mapping_confidence=(
                float(r["max_confidence"]) if r["max_confidence"] is not None else None
            ),
        )
    return result


def count_obs_per_archetype_realm(
    conn: sqlite3.Connection,
    realm_slug: str,
    expansion_slug: str,
) -> dict[int, int]:
    """Count non-outlier observations per (archetype_id) for cold-start detection.

    Filters to items belonging to ``expansion_slug`` so that cold-start detection
    is scoped to the target expansion (e.g. "midnight" items with sparse history).

    Args:
        conn:           Open SQLite connection.
        realm_slug:     Realm to count observations for.
        expansion_slug: Expansion to filter items by (e.g. "midnight").

    Returns:
        Dict mapping ``archetype_id → observation count``.
    """
    rows = conn.execute(
        """
        SELECT i.archetype_id, COUNT(*) AS obs_count
        FROM market_observations_normalized mon
        JOIN items i ON mon.item_id = i.item_id
        WHERE i.archetype_id   IS NOT NULL
          AND mon.is_outlier   = 0
          AND mon.realm_slug   = ?
          AND i.expansion_slug = ?
        GROUP BY i.archetype_id
        """,
        (realm_slug, expansion_slug),
    ).fetchall()

    return {r["archetype_id"]: r["obs_count"] for r in rows}


def count_items_per_archetype(
    conn: sqlite3.Connection,
    realm_slug: str,
) -> dict[int, int]:
    """Count distinct items contributing to each archetype on this realm.

    Used for the ``item_count_in_archetype`` feature.

    Args:
        conn:       Open SQLite connection.
        realm_slug: Realm to scope the count.

    Returns:
        Dict mapping ``archetype_id → item count``.
    """
    rows = conn.execute(
        """
        SELECT i.archetype_id, COUNT(DISTINCT mon.item_id) AS item_count
        FROM market_observations_normalized mon
        JOIN items i ON mon.item_id = i.item_id
        WHERE i.archetype_id IS NOT NULL
          AND mon.is_outlier  = 0
          AND mon.realm_slug  = ?
        GROUP BY i.archetype_id
        """,
        (realm_slug,),
    ).fetchall()

    return {r["archetype_id"]: r["item_count"] for r in rows}


def count_items_without_archetype(conn: sqlite3.Connection) -> int:
    """Return the number of items in the ``items`` table with archetype_id = NULL."""
    row = conn.execute(
        "SELECT COUNT(*) AS cnt FROM items WHERE archetype_id IS NULL"
    ).fetchone()
    return int(row["cnt"]) if row else 0


# ── Feature computation ────────────────────────────────────────────────────────

def compute_archetype_features(
    rows: list[dict[str, Any]],
    archetype_id: int,
    realm_slug: str,
    metadata: dict[int, ArchetypeMetadata],
    cold_start_obs_counts: dict[int, int],
    item_counts: dict[int, int],
    cold_start_threshold: int,
    cold_start_expansion: str,
) -> list[dict[str, Any]]:
    """Add archetype encoding and transfer features to rows for one series.

    All archetype-derived columns are *static* for a given (archetype_id, realm)
    series — they are the same for every row.  The function stamps them onto each
    row for Parquet assembly convenience.

    Args:
        rows:                   Rows from ``compute_event_features()`` output.
        archetype_id:           The archetype being processed.
        realm_slug:             The realm being processed.
        metadata:               Pre-loaded ``ArchetypeMetadata`` dict from
                                ``load_archetype_metadata()``.
        cold_start_obs_counts:  Pre-loaded obs counts per archetype for the
                                transfer-target expansion (from
                                ``count_obs_per_archetype_realm()``).
        item_counts:            Pre-loaded item counts per archetype from
                                ``count_items_per_archetype()``.
        cold_start_threshold:   ``config.features.cold_start_threshold``.
        cold_start_expansion:   ``config.expansions.transfer_target`` (e.g. "midnight").

    Returns:
        Rows with archetype / transfer feature keys added.
    """
    meta = metadata.get(archetype_id)

    if meta is not None:
        category_tag  = meta.category_tag
        sub_tag       = meta.sub_tag
        is_transferable = meta.is_transferable
        has_mapping   = meta.has_transfer_mapping
        map_confidence = meta.transfer_mapping_confidence
    else:
        # Archetype was deleted or not loaded; use safe defaults.
        category_tag  = "unknown"
        sub_tag       = None
        is_transferable = False
        has_mapping   = False
        map_confidence = None

    # Cold-start: too few observations in the transfer-target expansion.
    target_obs = cold_start_obs_counts.get(archetype_id, 0)
    is_cold_start = target_obs < cold_start_threshold

    item_count = item_counts.get(archetype_id, 0)

    result: list[dict[str, Any]] = []
    for row in rows:
        result.append({
            **row,
            "archetype_category":      category_tag,
            "archetype_sub_tag":       sub_tag,
            "is_transferable":         is_transferable,
            "is_cold_start":           is_cold_start,
            "item_count_in_archetype": item_count,
            "has_transfer_mapping":    has_mapping,
            "transfer_confidence":     map_confidence,
        })
    return result
