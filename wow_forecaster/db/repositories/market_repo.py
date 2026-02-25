"""
Repository for raw and normalized market observations.
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import datetime
from typing import Optional

from wow_forecaster.db.repositories.base import BaseRepository
from wow_forecaster.models.market import NormalizedMarketObservation, RawMarketObservation

logger = logging.getLogger(__name__)


class MarketObservationRepository(BaseRepository):
    """Read/write access to ``market_observations_raw`` and
    ``market_observations_normalized`` tables."""

    # ── Raw observations ──────────────────────────────────────────────────────

    def insert_raw(self, obs: RawMarketObservation) -> int:
        """Insert a raw market observation.

        Args:
            obs: The ``RawMarketObservation`` to persist.

        Returns:
            The newly assigned ``obs_id``.
        """
        self.execute(
            """
            INSERT INTO market_observations_raw (
                item_id, realm_slug, faction, observed_at, source,
                min_buyout_raw, market_value_raw, historical_value_raw,
                quantity_listed, num_auctions, raw_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            (
                obs.item_id,
                obs.realm_slug,
                obs.faction,
                obs.observed_at.isoformat(),
                obs.source,
                obs.min_buyout_raw,
                obs.market_value_raw,
                obs.historical_value_raw,
                obs.quantity_listed,
                obs.num_auctions,
                obs.raw_json,
            ),
        )
        return self.last_insert_rowid()

    def insert_raw_batch(self, observations: list[RawMarketObservation]) -> int:
        """Bulk-insert raw observations.

        Args:
            observations: List of ``RawMarketObservation`` instances.

        Returns:
            Number of rows inserted.
        """
        if not observations:
            return 0
        params = [
            (
                o.item_id, o.realm_slug, o.faction, o.observed_at.isoformat(),
                o.source, o.min_buyout_raw, o.market_value_raw,
                o.historical_value_raw, o.quantity_listed, o.num_auctions, o.raw_json,
            )
            for o in observations
        ]
        self.executemany(
            """
            INSERT INTO market_observations_raw (
                item_id, realm_slug, faction, observed_at, source,
                min_buyout_raw, market_value_raw, historical_value_raw,
                quantity_listed, num_auctions, raw_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            params,
        )
        return len(observations)

    def get_unprocessed_raw(self, limit: int = 1000) -> list[RawMarketObservation]:
        """Fetch raw observations not yet normalized.

        Args:
            limit: Maximum number of rows to return.

        Returns:
            List of ``RawMarketObservation`` objects.
        """
        rows = self.fetchall(
            """
            SELECT * FROM market_observations_raw
            WHERE is_processed = 0
            ORDER BY obs_id
            LIMIT ?;
            """,
            (limit,),
        )
        return [_row_to_raw(r) for r in rows]

    def mark_processed(self, obs_ids: list[int]) -> int:
        """Mark raw observations as processed (``is_processed = 1``).

        Args:
            obs_ids: List of ``obs_id`` values to mark.

        Returns:
            Number of rows updated.
        """
        if not obs_ids:
            return 0
        placeholders = ",".join("?" for _ in obs_ids)
        cursor = self.execute(
            f"UPDATE market_observations_raw SET is_processed = 1 WHERE obs_id IN ({placeholders});",
            tuple(obs_ids),
        )
        return cursor.rowcount

    # ── Normalized observations ───────────────────────────────────────────────

    def insert_normalized(self, norm: NormalizedMarketObservation) -> int:
        """Insert a normalized market observation.

        Args:
            norm: The ``NormalizedMarketObservation`` to persist.

        Returns:
            The newly assigned ``norm_id``.
        """
        self.execute(
            """
            INSERT INTO market_observations_normalized (
                obs_id, item_id, archetype_id, realm_slug, faction, observed_at,
                price_gold, market_value_gold, historical_value_gold,
                quantity_listed, num_auctions, z_score, is_outlier
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            (
                norm.obs_id,
                norm.item_id,
                norm.archetype_id,
                norm.realm_slug,
                norm.faction,
                norm.observed_at.isoformat(),
                norm.price_gold,
                norm.market_value_gold,
                norm.historical_value_gold,
                norm.quantity_listed,
                norm.num_auctions,
                norm.z_score,
                int(norm.is_outlier),
            ),
        )
        return self.last_insert_rowid()

    def get_normalized_for_item(
        self,
        item_id: int,
        realm_slug: Optional[str] = None,
        limit: int = 500,
    ) -> list[NormalizedMarketObservation]:
        """Fetch normalized observations for a specific item.

        Args:
            item_id: WoW item ID.
            realm_slug: Optional realm filter.
            limit: Maximum rows to return.

        Returns:
            List of ``NormalizedMarketObservation``, most recent first.
        """
        if realm_slug:
            rows = self.fetchall(
                """
                SELECT * FROM market_observations_normalized
                WHERE item_id = ? AND realm_slug = ? AND is_outlier = 0
                ORDER BY observed_at DESC LIMIT ?;
                """,
                (item_id, realm_slug, limit),
            )
        else:
            rows = self.fetchall(
                """
                SELECT * FROM market_observations_normalized
                WHERE item_id = ? AND is_outlier = 0
                ORDER BY observed_at DESC LIMIT ?;
                """,
                (item_id, limit),
            )
        return [_row_to_normalized(r) for r in rows]


# ── Private helpers ────────────────────────────────────────────────────────────

def _row_to_raw(row: sqlite3.Row) -> RawMarketObservation:
    return RawMarketObservation(
        item_id=row["item_id"],
        realm_slug=row["realm_slug"],
        faction=row["faction"],
        observed_at=datetime.fromisoformat(row["observed_at"]),
        source=row["source"],
        min_buyout_raw=row["min_buyout_raw"],
        market_value_raw=row["market_value_raw"],
        historical_value_raw=row["historical_value_raw"],
        quantity_listed=row["quantity_listed"],
        num_auctions=row["num_auctions"],
        raw_json=row["raw_json"],
    )


def _row_to_normalized(row: sqlite3.Row) -> NormalizedMarketObservation:
    return NormalizedMarketObservation(
        obs_id=row["obs_id"],
        item_id=row["item_id"],
        archetype_id=row["archetype_id"],
        realm_slug=row["realm_slug"],
        faction=row["faction"],
        observed_at=datetime.fromisoformat(row["observed_at"]),
        price_gold=row["price_gold"],
        market_value_gold=row["market_value_gold"],
        historical_value_gold=row["historical_value_gold"],
        quantity_listed=row["quantity_listed"],
        num_auctions=row["num_auctions"],
        z_score=row["z_score"],
        is_outlier=bool(row["is_outlier"]),
    )
