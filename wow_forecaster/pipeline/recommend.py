"""
RecommendStage — derive ranked trading recommendations from forecast outputs.

Recommendation flow
-------------------
For each realm:
  1. Load the most recent ForecastOutput rows from forecast_outputs (DB).
  2. Load the latest inference Parquet for current market state
     (current price, volume, rolling stats, event features).
  3. Score all forecasts via scorer.compute_score() (5-component formula).
  4. Determine action (buy/sell/hold/avoid) and build reasoning strings.
  5. Run top_n_per_category() → top-3 per archetype category by default.
  6. Persist RecommendationOutput rows to recommendation_outputs (DB).
  7. Write CSV + JSON report files to config.model.recommendation_output_dir.

Score formula (0–~100)
-----------------------
    total = 0.35 × opportunity  +  0.20 × liquidity
          − 0.20 × volatility   +  0.15 × event_boost
          − 0.10 × uncertainty

Returns total number of RecommendationOutput rows written.
"""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

import pyarrow.parquet as pq

from wow_forecaster.models.meta import RunMetadata
from wow_forecaster.pipeline.base import PipelineStage

logger = logging.getLogger(__name__)


class RecommendStage(PipelineStage):
    """Convert forecast outputs into ranked buy/sell/hold/avoid recommendations."""

    stage_name = "recommend"

    def _execute(
        self,
        run: RunMetadata,
        top_n_per_category: int | None = None,
        realm_slug: str | None = None,
        forecast_run_id: int | None = None,
        **kwargs,
    ) -> int:
        """Generate and persist recommendation outputs.

        Args:
            run:               In-progress RunMetadata (mutable).
            top_n_per_category: Override for max recommendations per category.
                               Defaults to config.model.top_n_per_category.
            realm_slug:        Single realm to target. If None, uses config defaults.
            forecast_run_id:   Specific forecast run_id to recommend from.
                               If None, uses the most recent run per realm.

        Returns:
            Total RecommendationOutput rows written.
        """
        from wow_forecaster.db.connection import get_connection
        from wow_forecaster.db.repositories.forecast_repo import ForecastOutputRepository
        from wow_forecaster.ml.predictor import find_latest_inference_parquet
        from wow_forecaster.recommendations.ranker import (
            build_recommendation_outputs,
            build_scored_forecasts,
            top_n_per_category as rank_top_n,
        )
        from wow_forecaster.recommendations.reporter import (
            write_forecast_csv,
            write_recommendation_csv,
            write_recommendation_json,
        )

        self._persist_run(run)

        realms         = [realm_slug] if realm_slug else list(self.config.realms.defaults)
        n              = top_n_per_category or self.config.model.top_n_per_category
        processed_dir  = Path(self.config.data.processed_dir)
        rec_output_dir = Path(self.config.model.recommendation_output_dir)
        fc_output_dir  = Path(self.config.model.forecast_output_dir)
        today          = date.today()

        total_recs = 0

        for realm in realms:
            # ── Load inference Parquet (current market state) ─────────────────
            inf_path = find_latest_inference_parquet(processed_dir, realm)
            if inf_path is None:
                logger.warning(
                    "No inference Parquet for realm=%s; skipping.", realm
                )
                continue

            inf_table = pq.read_table(str(inf_path))
            inf_rows  = inf_table.to_pylist()

            # ── Load forecast outputs from DB ─────────────────────────────────
            with get_connection(
                self.db_path,
                wal_mode=self.config.database.wal_mode,
                busy_timeout_ms=self.config.database.busy_timeout_ms,
            ) as conn:
                forecasts = _load_recent_forecasts(
                    conn=conn,
                    realm_slug=realm,
                    run_id=forecast_run_id,
                )

            if not forecasts:
                logger.warning(
                    "No forecast outputs found for realm=%s run_id=%s. "
                    "Run 'run-daily-forecast' first.",
                    realm, forecast_run_id,
                )
                continue

            logger.info(
                "Scoring %d forecasts for realm=%s  n_per_cat=%d",
                len(forecasts), realm, n,
            )

            # ── Score + rank ──────────────────────────────────────────────────
            scored       = build_scored_forecasts(forecasts, inf_rows)
            top_by_cat   = rank_top_n(scored, n=n)
            rec_outputs  = build_recommendation_outputs(top_by_cat)

            # ── Persist recommendations ───────────────────────────────────────
            with get_connection(
                self.db_path,
                wal_mode=self.config.database.wal_mode,
                busy_timeout_ms=self.config.database.busy_timeout_ms,
            ) as conn:
                repo = ForecastOutputRepository(conn)
                for rec in rec_outputs:
                    repo.insert_recommendation(rec)

            total_recs += len(rec_outputs)

            # ── Write report files ────────────────────────────────────────────
            write_forecast_csv(scored, fc_output_dir, realm, today)
            write_recommendation_csv(top_by_cat, rec_output_dir, realm, today)
            write_recommendation_json(
                top_by_cat, rec_output_dir, realm, today, run_slug=run.run_slug
            )

            logger.info(
                "realm=%s: %d recommendation(s) written.", realm, len(rec_outputs)
            )

        logger.info(
            "RecommendStage complete: %d recommendation(s) across %d realm(s).",
            total_recs, len(realms),
        )
        return total_recs


def _load_recent_forecasts(conn, realm_slug: str, run_id: int | None):
    """Load the most recent forecast outputs for a realm from the DB."""
    from wow_forecaster.db.repositories.forecast_repo import ForecastOutputRepository
    from datetime import date

    repo = ForecastOutputRepository(conn)

    if run_id is not None:
        # Specific run
        rows = conn.execute(
            """
            SELECT * FROM forecast_outputs
            WHERE run_id = ? AND realm_slug = ?
            ORDER BY created_at DESC;
            """,
            (run_id, realm_slug),
        ).fetchall()
    else:
        # Latest run for this realm (by the most recent created_at)
        rows = conn.execute(
            """
            SELECT fo.* FROM forecast_outputs fo
            INNER JOIN (
                SELECT MAX(created_at) AS max_ts
                FROM forecast_outputs
                WHERE realm_slug = ?
            ) AS latest ON fo.created_at = latest.max_ts
            WHERE fo.realm_slug = ?;
            """,
            (realm_slug, realm_slug),
        ).fetchall()
        if not rows:
            # Fallback: any forecasts for this realm ordered by created_at
            rows = conn.execute(
                """
                SELECT * FROM forecast_outputs
                WHERE realm_slug = ?
                ORDER BY created_at DESC LIMIT 500;
                """,
                (realm_slug,),
            ).fetchall()

    from wow_forecaster.models.forecast import ForecastOutput

    outputs = []
    for r in rows:
        try:
            outputs.append(
                ForecastOutput(
                    forecast_id=r["forecast_id"],
                    run_id=r["run_id"],
                    archetype_id=r["archetype_id"],
                    item_id=r["item_id"],
                    realm_slug=r["realm_slug"],
                    forecast_horizon=r["forecast_horizon"],
                    target_date=date.fromisoformat(r["target_date"]),
                    predicted_price_gold=r["predicted_price_gold"],
                    confidence_lower=r["confidence_lower"],
                    confidence_upper=r["confidence_upper"],
                    confidence_pct=r["confidence_pct"],
                    model_slug=r["model_slug"],
                    features_hash=r["features_hash"],
                )
            )
        except Exception as exc:
            logger.warning("Skipping malformed forecast row: %s", exc)

    return outputs
