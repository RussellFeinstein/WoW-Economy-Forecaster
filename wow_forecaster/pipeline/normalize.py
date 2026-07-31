"""
NormalizeStage — convert raw market observations to gold-priced, z-scored records.

Processing steps:
  1. Fetch unprocessed raw observations in batches from ``market_observations_raw``.
  2. Convert copper prices → gold (divide by 10_000).
  3. Compute a rolling z-score per (item_id, realm_slug) using historical mean/std
     from the pre-aggregated ``daily_rollup_item`` partial sums over a configurable
     window (``config.pipeline.normalize_rolling_days``, default 30 days).
     Falls back to batch-level stats for items with no prior history (cold-start).
     The rollup step runs after this one, so the current day's rollup row can be up
     to an hour stale here; against a 30-day window that is negligible.
  4. Flag outliers: ``|z_score| > config.pipeline.outlier_z_threshold``.
  5. Write ``NormalizedMarketObservation`` records.
  6. Mark raw observations as processed (``is_processed = 1``).

Archetype mapping:
  ``archetype_id`` is populated via a pre-batch lookup of item_id → archetype_id
  from the ``items`` table.  Items without an archetype assignment remain NULL.
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timedelta

from wow_forecaster.models.market import NormalizedMarketObservation
from wow_forecaster.models.meta import RunMetadata
from wow_forecaster.pipeline.base import PipelineStage
from wow_forecaster.utils.time_utils import utcnow

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

        batch_size  = self.config.pipeline.normalize_batch_size
        z_threshold = self.config.pipeline.outlier_z_threshold
        rolling_days = self.config.pipeline.normalize_rolling_days

        total_normalized = 0
        total_processed  = 0

        with get_connection(
            self.db_path,
            wal_mode=self.config.database.wal_mode,
            busy_timeout_ms=self.config.database.busy_timeout_ms,
        ) as conn:
            # Count pending rows upfront for X/Y progress reporting.
            total_pending = conn.execute(
                "SELECT COUNT(*) FROM market_observations_raw WHERE is_processed = 0;"
            ).fetchone()[0]

            if total_pending == 0:
                logger.info("NormalizeStage: no unprocessed raw observations found.")
                return 0

            logger.info("NormalizeStage: %d raw rows pending normalization.", total_pending)

            # Pre-fetch rolling stats and archetype map once for all pending items.
            # Previously these were re-fetched per batch (~N_batches queries).
            # The 30-day rolling window dwarfs what one ingest run adds, so a single
            # pre-fetch produces effectively identical baselines for every batch.
            pending_meta = conn.execute(
                "SELECT DISTINCT item_id, realm_slug "
                "FROM market_observations_raw WHERE is_processed = 0;"
            ).fetchall()
            pending_item_ids   = {row["item_id"]   for row in pending_meta}
            pending_realm_slugs = {row["realm_slug"] for row in pending_meta}

            rolling_stats = _fetch_rolling_stats(
                conn, pending_item_ids, pending_realm_slugs, rolling_days
            )
            _check_rollup_freshness(conn, pending_realm_slugs)
            archetype_map = _fetch_archetype_map(conn, pending_item_ids)

            logger.info(
                "NormalizeStage: rolling stats for %d item/realm pairs | "
                "archetype map: %d items.",
                len(rolling_stats), len(archetype_map),
            )

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

                normalized_rows, obs_ids = _normalize_batch(
                    batch, z_threshold, rolling_stats, archetype_map
                )

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

                total_processed  += len(batch)
                total_normalized += len(normalized_rows)
                pct = total_processed * 100 // total_pending
                logger.info(
                    "NormalizeStage: %d / %d rows (%d%%) | normalized=%d",
                    total_processed, total_pending, pct, total_normalized,
                )

                if len(batch) < batch_size:
                    break  # Last (partial) batch

        return total_normalized


# ── Processing helpers ─────────────────────────────────────────────────────────

# Minimum historical observations required to use rolling stats instead of
# falling back to batch-level stats (which are meaningless for single items).
_MIN_ROLLING_OBS = 2


# Baseline query template.  Exposed through _rolling_stats_sql() so tests can
# pin its query plan against the exact string production runs, rather than a
# copy that can drift.
_ROLLING_STATS_SQL = """
        SELECT item_id, realm_slug,
               SUM(price_sum) / SUM(obs_count)                           AS mean_p,
               SUM(price_sum_sq) / SUM(obs_count)
                 - (SUM(price_sum) / SUM(obs_count))
                   * (SUM(price_sum) / SUM(obs_count))                   AS variance,
               SUM(obs_count)                                            AS n
        FROM daily_rollup_item
        WHERE item_id IN ({placeholders})
          AND obs_date >= ?
        GROUP BY item_id, realm_slug
        HAVING SUM(obs_count) >= {min_obs};
"""


def _rolling_stats_sql(n_items: int) -> str:
    """Return the baseline query for an ``item_id IN (...)`` list of ``n_items``."""
    return _ROLLING_STATS_SQL.format(
        placeholders=",".join("?" for _ in range(n_items)),
        min_obs=_MIN_ROLLING_OBS,
    )


def _fetch_rolling_stats(
    conn: sqlite3.Connection,
    item_ids: set[int],
    realm_slugs: set[str],
    window_days: int,
    now: datetime | None = None,
) -> dict[tuple[int, str], tuple[float, float]]:
    """Fetch rolling mean and std from the daily item rollups for a set of items.

    Reads the pre-aggregated partial sums in ``daily_rollup_item`` rather than
    re-scanning every underlying observation.  COUNT, SUM and SUM of squares are
    sufficient statistics for a mean and a variance, so summing them across days
    reproduces the same numbers exactly.

    ``daily_rollup_item`` is built from ``market_observations_normalized`` under
    the same ``is_outlier = 0`` filter, so only non-outlier rows within the
    rolling window are included so that previously flagged spikes don't corrupt
    future baselines.  The swap depends on ``price_gold`` never being NULL, since
    SUM skips NULLs while COUNT(*) counts them; the schema holds that with
    ``REAL NOT NULL`` and normalize coerces a missing price to 0.0 on write.

    Uses the identity Var(X) = E[X²] - E[X]² to compute variance in a single
    SQL pass (SQLite has no built-in STDEV).

    Args:
        conn:        Open SQLite connection with ``row_factory = sqlite3.Row``.
        item_ids:    Set of item_ids to look up.
        realm_slugs: Set of realm_slugs present in the batch (used to filter rows).
        window_days: How many calendar days of history to include.
        now:         Reference clock for the window (default: current UTC time).
                     Injectable so tests are deterministic at any wall-clock time.

    Returns:
        Mapping of ``(item_id, realm_slug)`` → ``(mean_price, std_price)``.
        Only pairs with at least ``_MIN_ROLLING_OBS`` observations are included;
        items with insufficient history are absent (caller falls back to batch stats).
    """
    if not item_ids:
        return {}

    if now is None:
        now = utcnow()
    cutoff = (now - timedelta(days=window_days)).date().isoformat()

    rows = conn.execute(
        _rolling_stats_sql(len(item_ids)),
        (*item_ids, cutoff),
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


def _check_rollup_freshness(
    conn: sqlite3.Connection,
    realm_slugs: set[str],
    now: datetime | None = None,
) -> None:
    """Log a warning when the rollups backing the baseline have stopped updating.

    Items below ``_MIN_ROLLING_OBS`` already fall back to batch statistics, so a
    rollup outage degrades to cold-start behaviour rather than failing.  That is
    the right behaviour and the wrong silence: the z-scores quietly get worse and
    nothing says why.  ``backfill-rollups`` is the repair path.

    A completely empty table is a cold start, not an outage, so it logs at INFO.

    Args:
        conn:        Open SQLite connection with ``row_factory = sqlite3.Row``.
        realm_slugs: Set of realm_slugs present in the batch.
        now:         Reference clock (default: current UTC time).
    """
    if not realm_slugs:
        return
    if now is None:
        now = utcnow()

    if not conn.execute("SELECT EXISTS(SELECT 1 FROM daily_rollup_item);").fetchone()[0]:
        logger.info(
            "NormalizeStage: daily_rollup_item is empty; every pair falls back to "
            "batch stats until the first rollup runs (cold start)."
        )
        return

    placeholders = ",".join("?" for _ in realm_slugs)
    newest_by_realm = {
        row["realm_slug"]: row["newest"]
        for row in conn.execute(
            f"SELECT realm_slug, MAX(obs_date) AS newest FROM daily_rollup_item "
            f"WHERE realm_slug IN ({placeholders}) GROUP BY realm_slug;",
            tuple(realm_slugs),
        ).fetchall()
    }

    # The rollup step upserts both the previous and current UTC dates on every
    # run, so a newest date of yesterday is the normal state just after UTC
    # midnight.  Anything older means the step has stopped landing.
    stale_before = (now - timedelta(days=1)).date().isoformat()
    for realm in sorted(realm_slugs):
        newest = newest_by_realm.get(realm)
        if newest is None:
            logger.warning(
                "NormalizeStage: daily_rollup_item holds no rows for realm '%s' "
                "although other realms are present; its rolling baseline is empty "
                "and every pair falls back to batch stats.",
                realm,
            )
        elif newest < stale_before:
            logger.warning(
                "NormalizeStage: newest daily_rollup_item date for realm '%s' is %s, "
                "older than %s; the rolling baseline is running on stale rollups. "
                "Repair with backfill-rollups.",
                realm, newest, stale_before,
            )


def _fetch_archetype_map(
    conn: sqlite3.Connection,
    item_ids: set[int],
) -> dict[int, int | None]:
    """Return a mapping of item_id → archetype_id for the given item IDs.

    Items not present in the ``items`` table, or items with no archetype
    assignment, map to ``None``.

    Args:
        conn:     Open SQLite connection with ``row_factory = sqlite3.Row``.
        item_ids: Set of item_ids present in the current batch.

    Returns:
        Dict of ``{item_id: archetype_id}``; archetype_id is ``None`` for
        unregistered or unassigned items.
    """
    if not item_ids:
        return {}
    placeholders = ",".join("?" for _ in item_ids)
    rows = conn.execute(
        f"SELECT item_id, archetype_id FROM items WHERE item_id IN ({placeholders});",
        tuple(item_ids),
    ).fetchall()
    return {row["item_id"]: row["archetype_id"] for row in rows}


def _normalize_batch(
    batch: list[sqlite3.Row],
    z_threshold: float,
    rolling_stats: dict[tuple[int, str], tuple[float, float]] | None = None,
    archetype_map: dict[int, int | None] | None = None,
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
        archetype_map: Mapping of item_id → archetype_id from ``_fetch_archetype_map()``.
                       If None, archetype_id is left as NULL (pre-v1.3.4 behaviour).

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

        for row, price_gold in zip(rows, gold_prices, strict=True):
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
                archetype_id=archetype_map.get(row["item_id"]) if archetype_map else None,
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
