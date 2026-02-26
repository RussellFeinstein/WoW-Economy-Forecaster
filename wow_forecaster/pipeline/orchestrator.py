"""
Hourly refresh orchestration for the WoW Economy Forecaster.

The ``HourlyOrchestrator`` coordinates the full hourly pipeline in a
deterministic, testable sequence:

  Step 1 — Pre-flight:    Verify DB accessible, schema current, log start.
  Step 2 — Ingest:        Call IngestStage per realm (failures isolated per-realm).
  Step 3 — Normalize:     Call NormalizeStage once for all unprocessed raw obs.
  Step 4 — Drift check:   Run DriftChecker per realm on fresh normalized data.
  Step 5 — Adaptive:      Map drift level to uncertainty multiplier + retrain flag.
            (Steps 4–5 are interleaved per-realm for fresh drift state.)
  Step 6 — Provenance:    Build source attribution summary per realm.
  Step 7 — Write outputs: Persist to DB + write monitoring JSON files.

Failure isolation
-----------------
- Per-realm ingest failure:  Recorded in realm_failures, orchestration continues.
- NormalizeStage failure:    Marked normalize_failed=True; drift check still runs
                              (on existing normalized data, which may be stale).
- Drift check failure:       Non-fatal; multiplier stays 1.0, no retrain flag.
- File write failure:        Non-fatal; DB records still written.
- All failures:              Written to run_metadata.error_message.

Provider throttling hook
------------------------
Each IngestStage.run() call is preceded by a throttle_hook() call point.
Currently this is a no-op (the IngestStage operates in fixture mode and
makes no real HTTP calls).  When real API clients are enabled, implement
rate-limiting / exponential backoff inside the hook:

    # Example future hook:
    def _throttle_hook(realm_slug: str, attempt: int = 0) -> None:
        if attempt > 0:
            time.sleep(min(2 ** attempt, 60))  # exponential backoff cap 60s

See the HOOK comment in _run_ingest() below.

Future scheduler integration
-----------------------------
This orchestrator has no scheduling logic.  Every run() call is self-contained.
To run on a schedule:

    # Windows Task Scheduler (hourly):
    #   schtasks /create /sc HOURLY /tn "WoWForecaster" \
    #            /tr "C:\\path\\to\\.venv\\Scripts\\wow-forecaster.exe run-hourly-refresh"

    # Linux cron (hourly):
    #   0 * * * * cd /srv/wow-forecaster && .venv/bin/wow-forecaster run-hourly-refresh

    # Python-APScheduler (future):
    #   from apscheduler.schedulers.blocking import BlockingScheduler
    #   scheduler = BlockingScheduler()
    #   scheduler.add_job(lambda: HourlyOrchestrator(config).run(...),
    #                     trigger='interval', hours=1)
    #   scheduler.start()
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from uuid import uuid4

from wow_forecaster.config import AppConfig

logger = logging.getLogger(__name__)


# ── Result types ──────────────────────────────────────────────────────────────

@dataclass
class RealmIngestionResult:
    """Outcome of IngestStage for a single realm.

    Attributes:
        realm_slug:   Realm that was processed.
        success:      True if IngestStage completed without exception.
        rows_written: Number of raw observation rows written (0 on failure).
        error:        Exception message if success=False.
    """

    realm_slug:   str
    success:      bool
    rows_written: int = 0
    error:        Optional[str] = None


@dataclass
class OrchestratorResult:
    """Complete result of one hourly orchestration run.

    Attributes:
        run_id:            DB run_id for the orchestrator run_metadata record.
        started_at:        UTC datetime when the run started.
        finished_at:       UTC datetime when the run finished (None if in-flight).
        realm_results:     Per-realm ingest outcomes.
        normalize_success: Whether NormalizeStage completed without error.
        normalize_rows:    Rows normalized.
        drift_results:     DriftCheckResult per realm (may be empty on failure).
        provenance:        ProvenanceSummary per realm (may be empty on failure).
        monitoring_files:  Paths of monitoring JSON files written.
        errors:            Accumulated error messages.
        status:            "success", "partial", or "failed".
    """

    run_id:             Optional[int]          = None
    started_at:         Optional[datetime]     = None
    finished_at:        Optional[datetime]     = None
    realm_results:      list[RealmIngestionResult] = field(default_factory=list)
    normalize_success:  bool                   = False
    normalize_rows:     int                    = 0
    drift_results:      dict                   = field(default_factory=dict)  # realm -> DriftCheckResult
    provenance:         dict                   = field(default_factory=dict)  # realm -> ProvenanceSummary
    monitoring_files:   list[str]              = field(default_factory=list)
    errors:             list[str]              = field(default_factory=list)
    status:             str                    = "started"


# ── Orchestrator ──────────────────────────────────────────────────────────────

class HourlyOrchestrator:
    """Coordinates the full hourly refresh pipeline.

    Args:
        config:  AppConfig for this run.
        db_path: Override DB path (defaults to config.database.db_path).
    """

    def __init__(
        self,
        config: AppConfig,
        db_path: Optional[str] = None,
    ) -> None:
        self.config  = config
        self.db_path = db_path or config.database.db_path

    def run(
        self,
        realm_slugs: Optional[list[str]] = None,
        check_drift: bool   = True,
        apply_adaptive: bool = True,
    ) -> OrchestratorResult:
        """Execute the full hourly refresh pipeline.

        Args:
            realm_slugs:    Realms to process.  Defaults to config.realms.defaults.
            check_drift:    Whether to run drift detection after normalize.
            apply_adaptive: Whether to compute adaptive policy from drift.

        Returns:
            OrchestratorResult summarising all steps.
        """
        realms  = realm_slugs or list(self.config.realms.defaults)
        result  = OrchestratorResult(started_at=datetime.now(tz=timezone.utc))
        run_slug = str(uuid4())

        # ── Step 1: Pre-flight ────────────────────────────────────────────────
        logger.info("HourlyOrchestrator | run_slug=%s | realms=%s", run_slug, realms)
        try:
            self._ensure_schema()
        except Exception as exc:
            result.status = "failed"
            result.errors.append(f"Pre-flight schema check failed: {exc}")
            result.finished_at = datetime.now(tz=timezone.utc)
            logger.error("Pre-flight failed: %s", exc)
            return result

        # Pre-persist orchestrator run record
        run_id = self._persist_run_start(run_slug, realms)
        result.run_id = run_id

        # ── Step 2: Ingest (per realm, isolated) ──────────────────────────────
        logger.info("[1/4] IngestStage per realm ...")
        for realm in realms:
            realm_result = self._run_ingest(realm, run_id)
            result.realm_results.append(realm_result)
            if not realm_result.success:
                result.errors.append(
                    f"IngestStage[{realm}]: {realm_result.error}"
                )

        # ── Step 3: Normalize ─────────────────────────────────────────────────
        logger.info("[2/4] NormalizeStage ...")
        norm_rows, norm_ok, norm_err = self._run_normalize(run_id)
        result.normalize_success = norm_ok
        result.normalize_rows    = norm_rows
        if not norm_ok and norm_err:
            result.errors.append(f"NormalizeStage: {norm_err}")

        # ── Step 4: Drift check + adaptive policy per realm ───────────────────
        if check_drift:
            logger.info("[3/4] Drift check + adaptive policy ...")
            for realm in realms:
                try:
                    drift_result, prov = self._run_drift_and_provenance(
                        realm, run_id, apply_adaptive
                    )
                except Exception as exc:
                    logger.error(
                        "Drift step raised unexpectedly for realm=%s: %s", realm, exc
                    )
                    drift_result, prov = None, None
                if drift_result is not None:
                    result.drift_results[realm] = drift_result
                if prov is not None:
                    result.provenance[realm] = prov
        else:
            logger.info("[3/4] Drift check skipped (check_drift=False).")

        # ── Step 5: Write monitoring outputs ──────────────────────────────────
        logger.info("[4/4] Writing monitoring outputs ...")
        output_dir = Path(self.config.monitoring.monitoring_output_dir)
        for realm, drift_result in result.drift_results.items():
            try:
                from wow_forecaster.monitoring.reporter import (
                    write_drift_report,
                    write_provenance_report,
                )
                p = write_drift_report(drift_result, output_dir)
                result.monitoring_files.append(str(p))
            except Exception as exc:
                logger.warning("Failed to write drift report for realm=%s: %s", realm, exc)

            prov = result.provenance.get(realm)
            if prov is not None:
                try:
                    from wow_forecaster.monitoring.reporter import write_provenance_report
                    p = write_provenance_report(prov, output_dir)
                    result.monitoring_files.append(str(p))
                except Exception as exc:
                    logger.warning("Failed to write provenance for realm=%s: %s", realm, exc)

        # ── Finalise result ───────────────────────────────────────────────────
        result.finished_at = datetime.now(tz=timezone.utc)
        n_ingest_ok = sum(1 for r in result.realm_results if r.success)

        if not result.errors:
            result.status = "success"
        elif n_ingest_ok > 0 or result.normalize_success:
            result.status = "partial"
        else:
            result.status = "failed"

        # Update run record with final status
        self._persist_run_finish(run_id, result)

        logger.info(
            "HourlyOrchestrator finished | status=%s | realms_ok=%d/%d | norm_rows=%d | errors=%d",
            result.status, n_ingest_ok, len(realms), result.normalize_rows, len(result.errors),
        )
        return result

    # ── Private helpers ───────────────────────────────────────────────────────

    def _ensure_schema(self) -> None:
        """Verify DB is accessible and schema is current (idempotent)."""
        from wow_forecaster.db.connection import get_connection
        from wow_forecaster.db.migrations import run_migrations
        from wow_forecaster.db.schema import apply_schema

        with get_connection(
            self.db_path,
            wal_mode=self.config.database.wal_mode,
            busy_timeout_ms=self.config.database.busy_timeout_ms,
        ) as conn:
            apply_schema(conn)
            run_migrations(conn)

    def _run_ingest(
        self, realm_slug: str, run_id: Optional[int]
    ) -> RealmIngestionResult:
        """Run IngestStage for one realm, isolating failures.

        HOOK — Provider rate-limiting/backoff:
            Before calling stage.run(), this is the correct place to
            implement per-realm throttle logic when real API keys are set.
            Example:
                _throttle_hook(realm_slug)  # e.g. sleep(1) between realms
            The IngestStage currently operates in fixture mode (no real HTTP
            calls), so no throttle is applied here.

        Args:
            realm_slug: Realm to ingest.
            run_id:     Orchestrator run ID (for logging).

        Returns:
            RealmIngestionResult.
        """
        # HOOK: throttle / backoff before API call
        # _throttle_hook(realm_slug)  # stub — add when real HTTP is enabled

        try:
            from wow_forecaster.pipeline.ingest import IngestStage

            stage     = IngestStage(config=self.config, db_path=self.db_path)
            stage_run = stage.run(realm_slugs=[realm_slug])
            return RealmIngestionResult(
                realm_slug=realm_slug,
                success=stage_run.status == "success",
                rows_written=stage_run.rows_processed,
            )
        except Exception as exc:
            logger.error("IngestStage[%s] failed: %s", realm_slug, exc)
            return RealmIngestionResult(
                realm_slug=realm_slug,
                success=False,
                rows_written=0,
                error=str(exc),
            )

    def _run_normalize(
        self, run_id: Optional[int]
    ) -> tuple[int, bool, Optional[str]]:
        """Run NormalizeStage for all unprocessed observations.

        Returns:
            Tuple of (rows_normalized, success, error_message).
        """
        try:
            from wow_forecaster.pipeline.normalize import NormalizeStage

            stage     = NormalizeStage(config=self.config, db_path=self.db_path)
            stage_run = stage.run()
            return stage_run.rows_processed, stage_run.status == "success", None
        except Exception as exc:
            logger.error("NormalizeStage failed: %s", exc)
            return 0, False, str(exc)

    def _run_drift_and_provenance(
        self,
        realm_slug: str,
        run_id: Optional[int],
        apply_adaptive: bool,
    ):
        """Run drift detection + provenance for one realm.

        Failure is non-fatal: returns (None, None) on exception.

        Returns:
            Tuple of (DriftCheckResult | None, ProvenanceSummary | None).
        """
        from wow_forecaster.db.connection import get_connection

        drift_result = None
        prov         = None

        try:
            with get_connection(
                self.db_path,
                wal_mode=self.config.database.wal_mode,
                busy_timeout_ms=self.config.database.busy_timeout_ms,
            ) as conn:
                from wow_forecaster.monitoring.drift import DriftChecker
                from wow_forecaster.monitoring.provenance import build_provenance_summary
                from wow_forecaster.monitoring.reporter import (
                    persist_drift_to_db,
                )

                mc = self.config.monitoring
                checker = DriftChecker(
                    conn=conn,
                    drift_window_hours=mc.drift_window_hours,
                    baseline_days=mc.drift_baseline_days,
                    z_threshold=mc.drift_z_threshold,
                    error_window_days=mc.error_drift_window_days,
                    mae_thresholds=(
                        mc.error_drift_mae_ratio_low,
                        mc.error_drift_mae_ratio_medium,
                        mc.error_drift_mae_ratio_high,
                        mc.error_drift_mae_ratio_critical,
                    ),
                    shock_window_days=mc.event_shock_window_days,
                )

                drift_result = checker.run_all(realm_slug)

                if run_id is not None:
                    persist_drift_to_db(conn, run_id, drift_result)

                prov = build_provenance_summary(
                    conn,
                    realm_slug=realm_slug,
                    lookback_hours=mc.drift_window_hours,
                    stale_threshold_hours=mc.drift_window_hours,
                )

        except Exception as exc:
            logger.error("Drift/provenance check failed for realm=%s: %s", realm_slug, exc)

        return drift_result, prov

    def _persist_run_start(
        self, run_slug: str, realms: list[str]
    ) -> Optional[int]:
        """Write the initial orchestrator run_metadata record.

        Returns the run_id, or None if persistence fails (non-fatal).
        """
        try:
            from wow_forecaster.db.connection import get_connection
            from wow_forecaster.db.repositories.forecast_repo import RunMetadataRepository
            from wow_forecaster.models.meta import RunMetadata
            from wow_forecaster.utils.time_utils import utcnow

            run = RunMetadata(
                run_slug=run_slug,
                pipeline_stage="orchestrator",
                config_snapshot={
                    "realms": realms,
                    "monitoring": self.config.monitoring.model_dump(),
                },
                started_at=utcnow(),
            )
            with get_connection(
                self.db_path,
                wal_mode=self.config.database.wal_mode,
                busy_timeout_ms=self.config.database.busy_timeout_ms,
            ) as conn:
                repo   = RunMetadataRepository(conn)
                run_id = repo.insert_run(run)
                return run_id
        except Exception as exc:
            logger.warning("Could not persist orchestrator run start: %s", exc)
            return None

    def _persist_run_finish(
        self, run_id: Optional[int], result: OrchestratorResult
    ) -> None:
        """Update the orchestrator run_metadata record with final status."""
        if run_id is None:
            return
        try:
            from wow_forecaster.db.connection import get_connection
            from wow_forecaster.utils.time_utils import utcnow

            rows = result.normalize_rows + sum(
                r.rows_written for r in result.realm_results
            )
            error_msg = "; ".join(result.errors) if result.errors else None
            finished  = utcnow()

            with get_connection(
                self.db_path,
                wal_mode=self.config.database.wal_mode,
                busy_timeout_ms=self.config.database.busy_timeout_ms,
            ) as conn:
                conn.execute(
                    """
                    UPDATE run_metadata
                    SET status = ?, rows_processed = ?, error_message = ?, finished_at = ?
                    WHERE run_id = ?;
                    """,
                    (result.status, rows, error_msg, finished.isoformat(), run_id),
                )
                conn.commit()
        except Exception as exc:
            logger.warning("Could not persist orchestrator run finish: %s", exc)
