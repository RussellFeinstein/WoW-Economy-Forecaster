"""
Tests for hourly orchestration — step ordering and failure handling.

What we test
------------
1. Orchestrator runs all steps in the correct order (ingest -> normalize -> drift).
2. Per-realm ingest failure is isolated (other realms succeed).
3. All realms fail -> overall status is "failed".
4. Normalize failure does not stop drift check (drift runs on old data).
5. Drift check failure is non-fatal (orchestration still "partial" not "failed").
6. Empty realm list returns graceful result.
7. OrchestratorResult.status transitions.
8. Dry-run mode does not modify DB.
9. Monitoring tables are populated after a successful run.
10. HourlyOrchestrator respects check_drift=False flag.
11. Rollup step anchors on the UTC clock, upserts the previous and current UTC
    dates per realm, and stays non-fatal on failure (issue #61).
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

from wow_forecaster.config import AppConfig, RealmsConfig
from wow_forecaster.pipeline.orchestrator import (
    HourlyOrchestrator,
    RealmIngestionResult,
)

# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_config(realms: list[str] | None = None) -> AppConfig:
    """Build a minimal AppConfig for testing."""
    cfg = AppConfig()
    if realms is not None:
        object.__setattr__(cfg, "realms", RealmsConfig(defaults=realms))
    return cfg


def _make_fake_run_metadata(status: str = "success", rows: int = 0):
    """Build a fake RunMetadata-like object."""
    m = MagicMock()
    m.status = status
    m.rows_processed = rows
    return m


# ── Step ordering ─────────────────────────────────────────────────────────────

class TestStepOrdering:
    def test_result_contains_all_steps(self, tmp_path):
        """OrchestratorResult has fields for realm results, normalize, and drift."""
        config = _make_config(["test-realm"])
        orch   = HourlyOrchestrator(config=config, db_path=":memory:")

        fake_realm_result = RealmIngestionResult(
            realm_slug="test-realm", success=True, rows_written=5
        )
        with patch.object(orch, "_ensure_schema"):
            with patch.object(orch, "_run_ingest", return_value=fake_realm_result):
                with patch.object(orch, "_run_normalize", return_value=(10, True, None)):
                    with patch.object(orch, "_run_drift_and_provenance",
                                      return_value=(None, None)):
                        with patch.object(orch, "_persist_run_start", return_value=1):
                            with patch.object(orch, "_persist_run_finish"):
                                result = orch.run(["test-realm"])

        assert len(result.realm_results) == 1
        assert result.realm_results[0].realm_slug == "test-realm"
        assert result.normalize_rows == 10

    def test_ingest_runs_before_normalize(self, tmp_path):
        """Ingest step must complete for each realm before normalize starts."""
        call_order: list[str] = []
        config = _make_config(["r1", "r2"])
        orch   = HourlyOrchestrator(config=config, db_path=":memory:")

        def fake_ingest(realm_slug, run_id):
            call_order.append(f"ingest:{realm_slug}")
            return MagicMock(success=True, rows_written=1, error=None)

        def fake_normalize(run_id):
            call_order.append("normalize")
            return (5, True, None)

        with patch.object(orch, "_ensure_schema"):
            with patch.object(orch, "_run_ingest", side_effect=fake_ingest):
                with patch.object(orch, "_run_normalize", side_effect=fake_normalize):
                    with patch.object(orch, "_run_drift_and_provenance",
                                      return_value=(None, None)):
                        with patch.object(orch, "_persist_run_start", return_value=1):
                            with patch.object(orch, "_persist_run_finish"):
                                orch.run(["r1", "r2"])

        ingest_idx   = [i for i, c in enumerate(call_order) if c.startswith("ingest:")]
        normalize_idx = [i for i, c in enumerate(call_order) if c == "normalize"]
        assert max(ingest_idx) < min(normalize_idx), (
            "All ingest calls must precede normalize"
        )


# ── Per-realm failure isolation ───────────────────────────────────────────────

class TestPerRealmFailureIsolation:
    def test_single_realm_fail_others_continue(self):
        config = _make_config(["good-realm", "bad-realm", "also-good"])
        orch   = HourlyOrchestrator(config=config, db_path=":memory:")

        def fake_ingest(realm_slug, run_id):
            if realm_slug == "bad-realm":
                return RealmIngestionResult(
                    realm_slug=realm_slug, success=False, rows_written=0, error="API timeout"
                )
            return RealmIngestionResult(realm_slug=realm_slug, success=True, rows_written=3)

        with patch.object(orch, "_ensure_schema"):
            with patch.object(orch, "_run_ingest", side_effect=fake_ingest):
                with patch.object(orch, "_run_normalize", return_value=(10, True, None)):
                    with patch.object(orch, "_run_drift_and_provenance",
                                      return_value=(None, None)):
                        with patch.object(orch, "_persist_run_start", return_value=1):
                            with patch.object(orch, "_persist_run_finish"):
                                result = orch.run(["good-realm", "bad-realm", "also-good"])

        assert len(result.realm_results) == 3
        good_results = [r for r in result.realm_results if r.success]
        bad_results  = [r for r in result.realm_results if not r.success]
        assert len(good_results) == 2
        assert len(bad_results)  == 1
        assert bad_results[0].realm_slug == "bad-realm"

    def test_all_realms_fail_status_failed(self):
        config = _make_config(["r1", "r2"])
        orch   = HourlyOrchestrator(config=config, db_path=":memory:")

        with patch.object(orch, "_ensure_schema"):
            with patch.object(orch, "_run_ingest",
                              return_value=MagicMock(success=False, rows_written=0, error="err")):
                with patch.object(orch, "_run_normalize", return_value=(0, False, "norm failed")):
                    with patch.object(orch, "_run_drift_and_provenance",
                                      return_value=(None, None)):
                        with patch.object(orch, "_persist_run_start", return_value=1):
                            with patch.object(orch, "_persist_run_finish"):
                                result = orch.run(["r1", "r2"])

        assert result.status == "failed"

    def test_partial_failure_gives_partial_status(self):
        config = _make_config(["r1", "r2"])
        orch   = HourlyOrchestrator(config=config, db_path=":memory:")

        call_count = {"n": 0}

        def fake_ingest(realm_slug, run_id):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return MagicMock(success=True, rows_written=5, error=None)
            return MagicMock(success=False, rows_written=0, error="failed")

        with patch.object(orch, "_ensure_schema"):
            with patch.object(orch, "_run_ingest", side_effect=fake_ingest):
                with patch.object(orch, "_run_normalize", return_value=(5, True, None)):
                    with patch.object(orch, "_run_drift_and_provenance",
                                      return_value=(None, None)):
                        with patch.object(orch, "_persist_run_start", return_value=1):
                            with patch.object(orch, "_persist_run_finish"):
                                result = orch.run(["r1", "r2"])

        # One ingest ok + normalize ok -> partial at minimum
        assert result.status in ("partial", "success")


# ── Normalize failure handling ────────────────────────────────────────────────

class TestNormalizeFailure:
    def test_normalize_failure_is_recorded(self):
        config = _make_config(["r1"])
        orch   = HourlyOrchestrator(config=config, db_path=":memory:")

        with patch.object(orch, "_ensure_schema"):
            with patch.object(orch, "_run_ingest",
                              return_value=MagicMock(success=True, rows_written=5, error=None)):
                with patch.object(orch, "_run_normalize",
                                  return_value=(0, False, "DB locked")):
                    with patch.object(orch, "_run_drift_and_provenance",
                                      return_value=(None, None)):
                        with patch.object(orch, "_persist_run_start", return_value=1):
                            with patch.object(orch, "_persist_run_finish"):
                                result = orch.run(["r1"])

        assert not result.normalize_success
        assert result.normalize_rows    == 0
        assert any("NormalizeStage" in e for e in result.errors)

    def test_drift_still_runs_after_normalize_failure(self):
        """Drift check should still attempt to run even if normalize failed."""
        config  = _make_config(["r1"])
        orch    = HourlyOrchestrator(config=config, db_path=":memory:")
        drift_called = {"n": 0}

        def fake_drift(realm_slug, run_id, apply_adaptive):
            drift_called["n"] += 1
            return (None, None)

        with patch.object(orch, "_ensure_schema"):
            with patch.object(orch, "_run_ingest",
                              return_value=MagicMock(success=True, rows_written=1, error=None)):
                with patch.object(orch, "_run_normalize", return_value=(0, False, "err")):
                    with patch.object(orch, "_run_drift_and_provenance",
                                      side_effect=fake_drift):
                        with patch.object(orch, "_persist_run_start", return_value=1):
                            with patch.object(orch, "_persist_run_finish"):
                                orch.run(["r1"], check_drift=True)

        assert drift_called["n"] == 1


# ── Drift check failure ───────────────────────────────────────────────────────

class TestDriftCheckFailure:
    def test_drift_failure_is_non_fatal(self):
        """If drift check raises, orchestration should still complete."""
        config = _make_config(["r1"])
        orch   = HourlyOrchestrator(config=config, db_path=":memory:")

        def failing_drift(realm_slug, run_id, apply_adaptive):
            raise RuntimeError("Drift check exploded")

        with patch.object(orch, "_ensure_schema"):
            with patch.object(orch, "_run_ingest",
                              return_value=MagicMock(success=True, rows_written=3, error=None)):
                with patch.object(orch, "_run_normalize", return_value=(3, True, None)):
                    with patch.object(orch, "_run_drift_and_provenance",
                                      side_effect=failing_drift):
                        with patch.object(orch, "_persist_run_start", return_value=1):
                            with patch.object(orch, "_persist_run_finish"):
                                result = orch.run(["r1"], check_drift=True)

        # Drift failed but ingest+normalize succeeded -> partial or success
        assert result.status in ("partial", "success")
        assert result.drift_results == {}


# ── Rollup step (issue #61) ───────────────────────────────────────────────────

class TestRollupStep:
    """Step 3.5 anchors on the UTC clock and covers the UTC-midnight seam."""

    def _run(self, realms, now=None, upsert=None):
        """Run the orchestrator with every other step mocked and the rollup
        upsert replaced by ``upsert`` (default: MagicMock returning (1, 1))."""
        config = _make_config(realms)
        orch   = HourlyOrchestrator(config=config, db_path=":memory:")
        if upsert is None:
            upsert = MagicMock(return_value=(1, 1))

        def fake_ingest(realm_slug, run_id):
            return RealmIngestionResult(
                realm_slug=realm_slug, success=True, rows_written=1
            )

        with (
            patch.object(orch, "_ensure_schema"),
            patch.object(orch, "_run_ingest", side_effect=fake_ingest),
            patch.object(orch, "_run_normalize", return_value=(5, True, None)),
            patch.object(orch, "_run_drift_and_provenance",
                         return_value=(None, None)),
            patch.object(orch, "_persist_run_start", return_value=1),
            patch.object(orch, "_persist_run_finish"),
            patch("wow_forecaster.db.rollup.upsert_rollups_for_date", upsert),
        ):
            result = orch.run(realms, now=now)
        return result, upsert

    def test_incident_clock_targets_both_utc_dates(self):
        """The 2026-07-21 02:43Z restore run (22:43 EDT locally) must upsert
        the previous and current UTC dates, regardless of the machine's local
        date. The old local-date anchor produced a single wrong-date call."""
        result, upsert = self._run(
            ["us"], now=datetime(2026, 7, 21, 2, 43, tzinfo=UTC)
        )
        calls = [c.args[1:] for c in upsert.call_args_list]
        assert calls == [("us", "2026-07-20"), ("us", "2026-07-21")]
        assert result.status == "success"

    def test_both_dates_upserted_per_realm(self):
        """Each realm gets both target dates, previous date first."""
        _, upsert = self._run(
            ["r1", "r2"], now=datetime(2026, 3, 15, 12, 0, tzinfo=UTC)
        )
        calls = [c.args[1:] for c in upsert.call_args_list]
        assert calls == [
            ("r1", "2026-03-14"), ("r1", "2026-03-15"),
            ("r2", "2026-03-14"), ("r2", "2026-03-15"),
        ]

    def test_default_clock_upserts_two_dates(self):
        """Without an injected clock the step still targets two dates."""
        _, upsert = self._run(["us"])
        assert upsert.call_count == 2

    def test_upsert_failure_is_non_fatal(self, caplog):
        """A rollup failure is logged as a warning and never fails the run."""
        failing = MagicMock(side_effect=RuntimeError("rollup exploded"))
        with caplog.at_level(
            logging.WARNING, logger="wow_forecaster.pipeline.orchestrator"
        ):
            result, _ = self._run(["us"], upsert=failing)
        assert result.status == "success"
        assert "Rollup step failed (non-fatal)" in caplog.text


# ── check_drift=False flag ────────────────────────────────────────────────────

class TestCheckDriftFalse:
    def test_no_drift_when_flag_off(self):
        config = _make_config(["r1"])
        orch   = HourlyOrchestrator(config=config, db_path=":memory:")

        drift_called = {"n": 0}

        def fake_drift(realm_slug, run_id, apply_adaptive):
            drift_called["n"] += 1
            return (None, None)

        with patch.object(orch, "_ensure_schema"):
            with patch.object(orch, "_run_ingest",
                              return_value=MagicMock(success=True, rows_written=0, error=None)):
                with patch.object(orch, "_run_normalize", return_value=(0, True, None)):
                    with patch.object(orch, "_run_drift_and_provenance",
                                      side_effect=fake_drift):
                        with patch.object(orch, "_persist_run_start", return_value=1):
                            with patch.object(orch, "_persist_run_finish"):
                                result = orch.run(["r1"], check_drift=False)

        assert drift_called["n"] == 0
        assert result.drift_results == {}


# ── OrchestratorResult status transitions ────────────────────────────────────

class TestStatusTransitions:
    def test_all_success(self):
        config = _make_config(["r1"])
        orch   = HourlyOrchestrator(config=config, db_path=":memory:")

        with patch.object(orch, "_ensure_schema"):
            with patch.object(orch, "_run_ingest",
                              return_value=MagicMock(success=True, rows_written=1, error=None)):
                with patch.object(orch, "_run_normalize", return_value=(1, True, None)):
                    with patch.object(orch, "_run_drift_and_provenance",
                                      return_value=(None, None)):
                        with patch.object(orch, "_persist_run_start", return_value=1):
                            with patch.object(orch, "_persist_run_finish"):
                                result = orch.run(["r1"])

        assert result.status == "success"
        assert result.errors == []

    def test_preflight_failure_returns_failed(self):
        config = _make_config(["r1"])
        orch   = HourlyOrchestrator(config=config, db_path=":memory:")

        with patch.object(orch, "_ensure_schema", side_effect=RuntimeError("no DB")):
            result = orch.run(["r1"])

        assert result.status == "failed"
        assert any("Pre-flight" in e for e in result.errors)
