"""
NormalizeStage — convert raw market observations to gold-priced, z-scored records.

Processing steps:
  1. Fetch unprocessed raw observations in batches from ``market_observations_raw``.
  2. Convert copper prices → gold (divide by 10_000).
  3. Compute a rolling z-score per (item_id, realm_slug) using historical mean/std
     from ``market_observations_normalized`` over a configurable window
     (``config.pipeline.normalize_rolling_days``, default 30 days).
     Falls back to batch-level stats for items with no prior history (cold-start).
  4. Flag outliers: ``|z_score| > config.pipeline.outlier_z_threshold``.
  5. Write ``NormalizedMarketObservation`` records.
  6. Mark raw observations as processed (``is_processed = 1``).

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

        rolling_days = self.config.pipeline.normalize_rolling_days

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

                # Pre-fetch rolling mean/std from historical normalized data
                item_ids   = {row["item_id"]   for row in batch}
                realm_slugs = {row["realm_slug"] for row in batch}
                rolling_stats = _fetch_rolling_stats(conn, item_ids, realm_slugs, rolling_days)

                normalized_rows, obs_ids = _normalize_batch(batch, z_threshold, rolling_stats)

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

# Minimum historical observations required to use rolling stats instead of
# falling back to batch-level stats (which are meaningless for single items).
_MIN_ROLLING_OBS = 2


def _fetch_rolling_stats(
    conn: sqlite3.Connection,
    item_ids: set[int],
    realm_slugs: set[str],
    window_days: int,
) -> dict[tuple[int, str], tuple[float, float]]:
    """Fetch rolling mean and std from historical normalized data for a set of items.

    Uses the identity Var(X) = E[X²] - E[X]² to compute variance in a single
    SQL pass (SQLite has no built-in STDEV).  Only non-outlier rows within the
    rolling window are included so that previously flagged spikes don't corrupt
    future baselines.

    Args:
        conn:        Open SQLite connection with ``row_factory = sqlite3.Row``.
        item_ids:    Set of item_ids to look up.
        realm_slugs: Set of realm_slugs present in the batch (used to filter rows).
        window_days: How many calendar days of history to include.

    Returns:
        Mapping of ``(item_id, realm_slug)`` → ``(mean_price, std_price)``.
        Only pairs with at least ``_MIN_ROLLING_OBS`` observations are included;
        items with insufficient history are absent (caller falls back to batch stats).
    """
    if not item_ids:
        return {}

    placeholders = ",".join("?" for _ in item_ids)
    rows = conn.execute(
        f"""
        SELECT item_id, realm_slug,
               AVG(price_gold)                                          AS mean_p,
               AVG(price_gold * price_gold) - AVG(price_gold) * AVG(price_gold)
                                                                        AS variance,
               COUNT(*)                                                 AS n
        FROM market_observations_normalized
        WHERE item_id IN ({placeholders})
          AND is_outlier = 0
          AND observed_at >= datetime('now', '-{window_days} days')
        GROUP BY item_id, realm_slug
        HAVING COUNT(*) >= {_MIN_ROLLING_OBS};
        """,
        tuple(item_ids),
    ).fetchall()

    result: dict[tuple[int, str], tuple[float, float]] = {}
    for row in rows:
        if row["realm_slug"] not in realm_slugs:
            continue
        # Guard against tiny negative floating-point variance
        variance = max(row["variance"] or 0.0, 0.0)
        result[(row["item_id"], row["realm_slug"])] = (
            float(row["mean_p"]),
            float(variance ** 0.5),
        )
    return result


def _normalize_batch(
    batch: list[sqlite3.Row],
    z_threshold: float,
    rolling_stats: dict[tuple[int, str], tuple[float, float]] | None = None,
) -> tuple[list[NormalizedMarketObservation], list[int]]:
    """Normalize a batch of raw rows and compute z-scores.

    Z-score baseline priority:
      1. Rolling historical stats from ``_fetch_rolling_stats()`` when available
         (mean/std over the configured lookback window from normalized history).
      2. Batch-level stats (mean/std within the current batch group) as a
         cold-start fallback for items with no prior history.

    For single-observation groups with no rolling history, z_score is None
    (insufficient data to compute a meaningful score).

    Args:
        batch:         List of ``sqlite3.Row`` from ``market_observations_raw``.
        z_threshold:   Outlier flag threshold (|z_score| > this → is_outlier=True).
        rolling_stats: Pre-fetched rolling baselines from ``_fetch_rolling_stats()``.
                       If None, always falls back to batch-level stats.

    Returns:
        Tuple of (normalized observations list, obs_id list for mark_processed).
    """
    from collections import defaultdict
    from datetime import datetime

    # Group by (item_id, realm_slug) so we can compute batch fallback stats
    groups: dict[tuple[int, str], list[sqlite3.Row]] = defaultdict(list)
    for row in batch:
        groups[(row["item_id"], row["realm_slug"])].append(row)

    normalized: list[NormalizedMarketObservation] = []
    obs_ids: list[int] = []

    for (item_id, realm_slug), rows in groups.items():
        # Gather gold prices for this group
        gold_prices = [
            r["min_buyout_raw"] / 10_000.0
            if r["min_buyout_raw"] is not None
            else None
            for r in rows
        ]
        valid_prices = [p for p in gold_prices if p is not None]

        # ── Z-score baseline: rolling history preferred, batch fallback ────────
        if rolling_stats is not None and (item_id, realm_slug) in rolling_stats:
            mean_p, std_p = rolling_stats[(item_id, realm_slug)]
        elif len(valid_prices) >= 2:
            # Cold-start: no history yet — use batch group stats
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
