"""
NormalizeStage — convert raw market observations to gold-priced, z-scored records.

Processing steps:
  1. Fetch unprocessed raw observations in batches from ``market_observations_raw``.
  2. Convert copper prices → gold (divide by 10_000).
  3. Compute a batch-level z-score per (item_id, realm_slug) group.
     Note: This is a *batch* z-score (within the current batch), not a true
     rolling window. For a proper rolling window, implement the TODO below
     to query historical gold prices from ``market_observations_normalized``
     and compute z-score against that history.
  4. Flag outliers: ``|z_score| > config.pipeline.outlier_z_threshold``.
  5. Write ``NormalizedMarketObservation`` records.
  6. Mark raw observations as processed (``is_processed = 1``).

TODO — rolling z-score:
  Replace the batch z-score with a proper rolling window:
    SELECT AVG(price_gold), STDEV(price_gold)
    FROM market_observations_normalized
    WHERE item_id = ? AND realm_slug = ?
      AND observed_at >= datetime('now', '-30 days')
  Use this rolling mean/std in step 3 instead of the batch statistics.

Archetype mapping TODO:
  The ``archetype_id`` field on normalized observations is currently NULL.
  To fill it: JOIN items → economic_archetypes via item_id during normalization.
"""

from __future__ import annotations

import logging
import sqlite3

from wow_forecaster.models.market import NormalizedMarketObservation
from wow_forecaster.models.meta import RunMetadata
from wow_forecaster.pipeline.base import PipelineStage

logger = logging.getLogger(__name__)


class NormalizeStage(PipelineStage):
    """Process unprocessed raw observations into gold-priced normalized records.

    Handles empty tables gracefully — returns 0 rows if nothing is pending.
    """

    stage_name = "normalize"

    def _execute(self, run: RunMetadata, **kwargs) -> int:
        """Normalize all pending raw market observations.

        Args:
            run: In-progress :class:`RunMetadata` (mutable, unused here).

        Returns:
            Total number of normalized rows written.
        """
        from wow_forecaster.db.connection import get_connection

        batch_size = self.config.pipeline.normalize_batch_size
        z_threshold = self.config.pipeline.outlier_z_threshold
        total_normalized = 0

        with get_connection(self.db_path) as conn:
            while True:
                batch = conn.execute(
                    """
                    SELECT obs_id, item_id, realm_slug, faction, observed_at,
                           source, min_buyout_raw, market_value_raw,
                           historical_value_raw, quantity_listed, num_auctions
                    FROM market_observations_raw
                    WHERE is_processed = 0
                    ORDER BY obs_id
                    LIMIT ?;
                    """,
                    (batch_size,),
                ).fetchall()

                if not batch:
                    break

                normalized_rows, obs_ids = _normalize_batch(batch, z_threshold)

                # Bulk-insert normalized rows
                conn.executemany(
                    """
                    INSERT INTO market_observations_normalized (
                        obs_id, item_id, archetype_id, realm_slug, faction, observed_at,
                        price_gold, market_value_gold, historical_value_gold,
                        quantity_listed, num_auctions, z_score, is_outlier
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                    """,
                    [
                        (
                            n.obs_id, n.item_id, n.archetype_id, n.realm_slug,
                            n.faction, n.observed_at.isoformat(),
                            n.price_gold, n.market_value_gold, n.historical_value_gold,
                            n.quantity_listed, n.num_auctions,
                            n.z_score, int(n.is_outlier),
                        )
                        for n in normalized_rows
                    ],
                )

                # Mark raw rows as processed
                placeholders = ",".join("?" for _ in obs_ids)
                conn.execute(
                    f"UPDATE market_observations_raw SET is_processed = 1 "
                    f"WHERE obs_id IN ({placeholders});",
                    tuple(obs_ids),
                )
                conn.commit()

                total_normalized += len(normalized_rows)
                logger.info(
                    "NormalizeStage: processed batch of %d → %d normalized rows (total: %d)",
                    len(batch), len(normalized_rows), total_normalized,
                )

                if len(batch) < batch_size:
                    break  # Last (partial) batch

        if total_normalized == 0:
            logger.info("NormalizeStage: no unprocessed raw observations found.")

        return total_normalized


# ── Processing helpers ─────────────────────────────────────────────────────────

def _normalize_batch(
    batch: list[sqlite3.Row],
    z_threshold: float,
) -> tuple[list[NormalizedMarketObservation], list[int]]:
    """Normalize a batch of raw rows and compute batch-level z-scores.

    Groups by (item_id, realm_slug) within the batch to compute z-scores.
    For single-observation groups, z_score is None (insufficient data).

    Args:
        batch: List of ``sqlite3.Row`` from ``market_observations_raw``.
        z_threshold: Outlier flag threshold (|z_score| > this → is_outlier=True).

    Returns:
        Tuple of (normalized observations list, obs_id list for mark_processed).
    """
    from collections import defaultdict
    from datetime import datetime

    # Group by (item_id, realm_slug) for z-score computation
    groups: dict[tuple[int, str], list[sqlite3.Row]] = defaultdict(list)
    for row in batch:
        groups[(row["item_id"], row["realm_slug"])].append(row)

    normalized: list[NormalizedMarketObservation] = []
    obs_ids: list[int] = []

    for (item_id, realm_slug), rows in groups.items():
        # Gather gold prices for z-score stats
        gold_prices = [
            r["min_buyout_raw"] / 10_000.0
            if r["min_buyout_raw"] is not None
            else None
            for r in rows
        ]
        valid_prices = [p for p in gold_prices if p is not None]

        # Batch-level mean / std  (TODO: replace with rolling window query)
        if len(valid_prices) >= 2:
            mean_p = sum(valid_prices) / len(valid_prices)
            variance = sum((p - mean_p) ** 2 for p in valid_prices) / len(valid_prices)
            std_p = variance ** 0.5 if variance > 0 else 0.0
        elif len(valid_prices) == 1:
            mean_p = valid_prices[0]
            std_p = 0.0
        else:
            mean_p = 0.0
            std_p = 0.0

        for row, price_gold in zip(rows, gold_prices):
            obs_ids.append(row["obs_id"])

            # Z-score: None when std is 0 (all prices identical) or price missing
            if price_gold is not None and std_p > 0:
                z_score = (price_gold - mean_p) / std_p
            else:
                z_score = None

            is_outlier = z_score is not None and abs(z_score) > z_threshold

            norm = NormalizedMarketObservation(
                obs_id=row["obs_id"],
                item_id=row["item_id"],
                archetype_id=None,   # TODO: look up via item→archetype join
                realm_slug=row["realm_slug"],
                faction=row["faction"],
                observed_at=datetime.fromisoformat(row["observed_at"]),
                price_gold=price_gold if price_gold is not None else 0.0,
                market_value_gold=(
                    row["market_value_raw"] / 10_000.0
                    if row["market_value_raw"] is not None else None
                ),
                historical_value_gold=(
                    row["historical_value_raw"] / 10_000.0
                    if row["historical_value_raw"] is not None else None
                ),
                quantity_listed=row["quantity_listed"],
                num_auctions=row["num_auctions"],
                z_score=z_score,
                is_outlier=is_outlier,
            )
            normalized.append(norm)

    return normalized, obs_ids
