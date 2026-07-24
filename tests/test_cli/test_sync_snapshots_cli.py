"""CLI tests for ``sync-snapshots`` (issue #43).

The drain itself is covered in ``tests/test_ingestion/test_cloud_sync.py``.
These tests pin the command surface: argument validation, what the operator
sees, and the exit codes downstream scheduling depends on.
"""

from __future__ import annotations

from typer.testing import CliRunner

from wow_forecaster.cli import app
from wow_forecaster.ingestion.cloud_sync import SkipCounts, SyncResult

runner = CliRunner()


def _patch_sync(monkeypatch, result: SyncResult) -> dict:
    """Replace the stage entry point; return the kwargs it was called with."""
    captured: dict = {}

    def fake_sync(config, **kwargs):
        captured.update(kwargs)
        return result

    monkeypatch.setattr(
        "wow_forecaster.pipeline.sync_stage.sync_snapshots", fake_sync
    )
    return captured


class TestArgumentHandling:
    def test_help_exits_zero(self):
        result = runner.invoke(app, ["sync-snapshots", "--help"])
        assert result.exit_code == 0, result.output
        # "Usage:" only, matching test_cli_smoke.py: rich wraps and colorizes
        # the options table, so option names are not reliably contiguous in
        # `result.output` at the CI runner's terminal width.
        assert "Usage:" in result.output

    def test_rejects_a_malformed_since_date(self):
        result = runner.invoke(app, ["sync-snapshots", "--since", "23-07-2026"])
        assert result.exit_code == 1
        assert "--since must be YYYY-MM-DD" in result.output

    def test_parses_a_valid_since_date(self, monkeypatch):
        captured = _patch_sync(monkeypatch, SyncResult(dry_run=True))
        result = runner.invoke(app, ["sync-snapshots", "--since", "2026-07-20", "--dry-run"])
        assert result.exit_code == 0
        assert captured["since"].date().isoformat() == "2026-07-20"

    def test_passes_limit_through(self, monkeypatch):
        captured = _patch_sync(monkeypatch, SyncResult(dry_run=True))
        runner.invoke(app, ["sync-snapshots", "--limit", "0", "--dry-run"])
        assert captured["limit"] == 0


class TestMissingCredentials:
    def test_exits_one_and_names_the_missing_variables(self, monkeypatch):
        from wow_forecaster.ingestion.cloud_sync import REQUIRED_ENV

        for name in REQUIRED_ENV:
            monkeypatch.delenv(name, raising=False)
        result = runner.invoke(app, ["sync-snapshots", "--dry-run"])
        assert result.exit_code == 1
        assert "SNAPSHOT_S3_ENDPOINT" in result.output


class TestOutput:
    def test_dry_run_says_nothing_was_written(self, monkeypatch):
        _patch_sync(
            monkeypatch,
            SyncResult(listed=5, selected=2, dry_run=True, skips=SkipCounts(hour_covered=3)),
        )
        result = runner.invoke(app, ["sync-snapshots", "--dry-run"])
        assert result.exit_code == 0
        assert "nothing was written" in result.output
        assert "hour_covered=3" in result.output

    def test_success_reports_counts(self, monkeypatch):
        _patch_sync(
            monkeypatch,
            SyncResult(
                listed=4,
                selected=2,
                ingested=2,
                observations_inserted=500_000,
                normalized_rows=500_000,
                dates_touched=["2026-07-23"],
            ),
        )
        result = runner.invoke(app, ["sync-snapshots"])
        assert result.exit_code == 0
        assert "500,000" in result.output
        assert "2026-07-23" in result.output
        assert "[OK] sync-snapshots complete." in result.output

    def test_reports_unknown_items_when_any_were_skipped(self, monkeypatch):
        _patch_sync(
            monkeypatch,
            SyncResult(ingested=1, observations_inserted=10, items_skipped_fk=7),
        )
        result = runner.invoke(app, ["sync-snapshots"])
        assert "Unknown items" in result.output
        assert "bootstrap-items" in result.output

    def test_reports_a_capped_run_rather_than_looking_complete(self, monkeypatch):
        _patch_sync(
            monkeypatch,
            SyncResult(
                ingested=96,
                observations_inserted=1,
                truncated=True,
                skips=SkipCounts(over_limit=12),
            ),
        )
        result = runner.invoke(app, ["sync-snapshots"])
        assert result.exit_code == 0
        assert "12 more objects waiting" in result.output


class TestFailureExitCode:
    def test_exits_one_when_any_object_failed(self, monkeypatch):
        _patch_sync(
            monkeypatch,
            SyncResult(
                ingested=1,
                observations_inserted=10,
                failures=[("blizzard_api/2026/07/23/x.json.gz", "not valid gzip")],
            ),
        )
        result = runner.invoke(app, ["sync-snapshots"])
        assert result.exit_code == 1
        assert "not valid gzip" in result.output
        assert "retried on the next run" in result.output

    def test_exits_one_when_the_stage_raises(self, monkeypatch):
        def boom(config, **kwargs):
            raise RuntimeError("bucket unreachable")

        monkeypatch.setattr(
            "wow_forecaster.pipeline.sync_stage.sync_snapshots", boom
        )
        result = runner.invoke(app, ["sync-snapshots"])
        assert result.exit_code == 1
        assert "bucket unreachable" in result.output
