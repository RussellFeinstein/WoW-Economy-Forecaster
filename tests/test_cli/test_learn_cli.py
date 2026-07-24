"""
Smoke tests for the ``wowfc learn`` command group.

``--help`` assertions check only ``"Usage:"``, matching the convention in
test_cli_smoke.py: rich colorizes and wraps the options table, so option names
are not contiguous in ``result.output`` at the CI runner's terminal width.

Every test pins ``WOWFC_LEARN_DB`` at tmp_path through CliRunner's ``env``
override, so a test run never writes ``data/learn/`` into the checkout.
"""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from wow_forecaster.cli import app
from wow_forecaster.learning.store import LEARN_DB_ENV

runner = CliRunner()

SUBCOMMANDS = ["status", "next", "module", "exam", "lab", "validate", "reset"]


@pytest.fixture
def learn_env(tmp_path):
    """Environment pinning the progress database inside tmp_path."""
    return {LEARN_DB_ENV: str(tmp_path / "progress.db")}


class TestHelp:
    def test_group_help_exits_zero(self):
        result = runner.invoke(app, ["learn", "--help"])
        assert result.exit_code == 0, result.output
        assert "Usage:" in result.output

    @pytest.mark.parametrize("command", SUBCOMMANDS)
    def test_subcommand_help_exits_zero(self, command):
        result = runner.invoke(app, ["learn", command, "--help"])
        assert result.exit_code == 0, (
            f"`learn {command} --help` exited {result.exit_code}:\n{result.output}"
        )
        assert "Usage:" in result.output

    def test_group_appears_in_root_help(self):
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "learn" in result.output

    def test_bare_group_shows_help_rather_than_erroring(self):
        result = runner.invoke(app, ["learn"])
        assert "Usage:" in result.output


class TestStatus:
    def test_reports_every_module(self, learn_env):
        result = runner.invoke(app, ["learn", "status"], env=learn_env)
        assert result.exit_code == 0, result.output
        assert "m06" in result.output
        assert "m20" in result.output

    def test_creates_the_progress_database(self, tmp_path, learn_env):
        runner.invoke(app, ["learn", "status"], env=learn_env)
        assert (tmp_path / "progress.db").is_file()

    def test_unauthored_modules_are_labelled(self, learn_env):
        result = runner.invoke(app, ["learn", "status"], env=learn_env)
        assert "not authored yet" in result.output


class TestValidate:
    def test_committed_content_passes(self, learn_env):
        result = runner.invoke(app, ["learn", "validate"], env=learn_env)
        assert result.exit_code == 0, result.output
        assert "[OK]" in result.output


class TestNext:
    def test_list_shows_the_queue_without_prompting(self, learn_env):
        result = runner.invoke(app, ["learn", "next", "--list", "-n", "3"], env=learn_env)
        assert result.exit_code == 0, result.output
        assert "m06-q01" in result.output
        assert result.output.count("m06-q") == 3

    def test_list_writes_no_review_state(self, tmp_path, learn_env):
        import sqlite3

        runner.invoke(app, ["learn", "next", "--list"], env=learn_env)
        conn = sqlite3.connect(tmp_path / "progress.db")
        assert conn.execute("SELECT COUNT(*) FROM review_state;").fetchone()[0] == 0
        conn.close()

    def test_unknown_module_exits_one(self, learn_env):
        result = runner.invoke(app, ["learn", "next", "-m", "m99"], env=learn_env)
        assert result.exit_code == 1
        assert "m99" in result.output

    def test_unauthored_module_exits_one_and_says_what_exists(self, learn_env):
        result = runner.invoke(app, ["learn", "next", "-m", "m20"], env=learn_env)
        assert result.exit_code == 1
        assert "m06" in result.output

    def test_grading_persists_and_reports_the_next_interval(self, tmp_path, learn_env):
        result = runner.invoke(
            app, ["learn", "next", "-m", "m06", "-n", "1"], input="\n3\n", env=learn_env
        )
        assert result.exit_code == 0, result.output
        assert "ANSWER" in result.output
        assert "next review in" in result.output

        import sqlite3

        conn = sqlite3.connect(tmp_path / "progress.db")
        rows = conn.execute("SELECT question_id, reps FROM review_state;").fetchall()
        conn.close()
        assert rows == [("m06-q01", 1)]

    def test_quitting_records_nothing(self, tmp_path, learn_env):
        runner.invoke(app, ["learn", "next", "-n", "1"], input="q\n", env=learn_env)
        import sqlite3

        conn = sqlite3.connect(tmp_path / "progress.db")
        assert conn.execute("SELECT COUNT(*) FROM review_state;").fetchone()[0] == 0
        conn.close()

    def test_exhausted_stdin_stops_cleanly_rather_than_crashing(self, learn_env):
        result = runner.invoke(app, ["learn", "next", "-n", "2"], input="", env=learn_env)
        assert result.exit_code == 0, result.output
        assert result.exception is None or isinstance(result.exception, SystemExit)

    def test_passed_card_is_not_reserved_the_same_day(self, learn_env):
        runner.invoke(app, ["learn", "next", "-n", "1"], input="\n3\n", env=learn_env)
        second = runner.invoke(app, ["learn", "next", "--list", "-n", "1"], env=learn_env)
        assert "m06-q01" not in second.output


class TestModule:
    def test_shows_objectives_and_reading_list(self, learn_env):
        result = runner.invoke(app, ["learn", "module", "m06"], env=learn_env)
        assert result.exit_code == 0, result.output
        assert "splits.py" in result.output
        assert "lab-01-purge-embargo" in result.output

    def test_unauthored_module_still_renders(self, learn_env):
        result = runner.invoke(app, ["learn", "module", "m18"], env=learn_env)
        assert result.exit_code == 0
        assert "not authored yet" in result.output

    def test_unknown_module_exits_one(self, learn_env):
        result = runner.invoke(app, ["learn", "module", "m99"], env=learn_env)
        assert result.exit_code == 1


class TestExam:
    def test_list_samples_without_prompting(self, learn_env):
        result = runner.invoke(
            app, ["learn", "exam", "-m", "m06", "-n", "5", "--seed", "1", "--list"], env=learn_env
        )
        assert result.exit_code == 0, result.output
        assert result.output.count("m06-q") == 5

    def test_seed_makes_the_sample_repeatable(self, learn_env):
        args = ["learn", "exam", "-n", "5", "--seed", "42", "--list"]
        assert runner.invoke(app, args, env=learn_env).output == (
            runner.invoke(app, args, env=learn_env).output
        )

    def test_records_nothing_without_the_record_flag(self, tmp_path, learn_env):
        runner.invoke(
            app,
            ["learn", "exam", "-m", "m06", "-n", "1", "--seed", "3"],
            input="\n0\n",
            env=learn_env,
        )
        import sqlite3

        db = tmp_path / "progress.db"
        if db.is_file():
            conn = sqlite3.connect(db)
            assert conn.execute("SELECT COUNT(*) FROM review_state;").fetchone()[0] == 0
            conn.close()

    def test_record_flag_writes_review_state(self, tmp_path, learn_env):
        result = runner.invoke(
            app,
            ["learn", "exam", "-m", "m06", "-n", "1", "--seed", "3", "--record"],
            input="\n1\n",
            env=learn_env,
        )
        assert result.exit_code == 0, result.output
        import sqlite3

        conn = sqlite3.connect(tmp_path / "progress.db")
        assert conn.execute("SELECT COUNT(*) FROM review_state;").fetchone()[0] == 1
        conn.close()

    def test_unauthored_module_exits_one(self, learn_env):
        result = runner.invoke(app, ["learn", "exam", "-m", "m20"], env=learn_env)
        assert result.exit_code == 1


class TestLab:
    def test_prints_the_brief(self, learn_env):
        result = runner.invoke(app, ["learn", "lab", "lab-01-purge-embargo"], env=learn_env)
        assert result.exit_code == 0, result.output
        assert "purge" in result.output.lower()
        assert "not_started" in result.output.lower()

    def test_unknown_lab_exits_one(self, learn_env):
        result = runner.invoke(app, ["learn", "lab", "lab-99-nope"], env=learn_env)
        assert result.exit_code == 1

    def test_start_then_done_records_progress(self, learn_env):
        started = runner.invoke(
            app,
            ["learn", "lab", "lab-01-purge-embargo", "--start", "--branch", "fix/1-x"],
            env=learn_env,
        )
        assert started.exit_code == 0
        assert "in_progress" in started.output

        done = runner.invoke(
            app, ["learn", "lab", "lab-01-purge-embargo", "--done"], env=learn_env
        )
        assert "done" in done.output

        shown = runner.invoke(app, ["learn", "lab", "lab-01-purge-embargo"], env=learn_env)
        assert "fix/1-x" in shown.output

    def test_start_and_done_together_exit_one(self, learn_env):
        result = runner.invoke(
            app, ["learn", "lab", "lab-01-purge-embargo", "--start", "--done"], env=learn_env
        )
        assert result.exit_code == 1


class TestReset:
    def test_yes_flag_skips_confirmation(self, learn_env):
        runner.invoke(app, ["learn", "next", "-n", "1"], input="\n3\n", env=learn_env)
        result = runner.invoke(app, ["learn", "reset", "--yes"], env=learn_env)
        assert result.exit_code == 0, result.output
        assert "cleared 1" in result.output

    def test_declining_the_prompt_keeps_progress(self, learn_env):
        runner.invoke(app, ["learn", "next", "-n", "1"], input="\n3\n", env=learn_env)
        result = runner.invoke(app, ["learn", "reset"], input="n\n", env=learn_env)
        assert "Cancelled" in result.output
        assert "m06-q01" not in runner.invoke(
            app, ["learn", "next", "--list", "-n", "1"], env=learn_env
        ).output

    def test_scoped_reset_requires_an_authored_module(self, learn_env):
        result = runner.invoke(app, ["learn", "reset", "-m", "m20", "--yes"], env=learn_env)
        assert result.exit_code == 1
