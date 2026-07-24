"""Tests for the learning progress store."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pytest

from wow_forecaster.learning.models import Grade, LabState, LabStatus, ReviewState
from wow_forecaster.learning.scheduler import new_state, review
from wow_forecaster.learning.store import LEARN_DB_ENV, ProgressStore, default_db_path


class TestSchemaAndPath:
    def test_apply_schema_is_idempotent(self, store):
        store.apply_schema()
        store.apply_schema()
        assert store.get_states() == {}

    def test_creates_parent_directories(self, tmp_path, monkeypatch):
        nested = tmp_path / "deep" / "deeper" / "progress.db"
        monkeypatch.setenv(LEARN_DB_ENV, str(nested))
        ProgressStore().apply_schema()
        assert nested.is_file()

    def test_env_override_wins(self, tmp_path, monkeypatch):
        monkeypatch.setenv(LEARN_DB_ENV, str(tmp_path / "elsewhere.db"))
        assert default_db_path() == tmp_path / "elsewhere.db"

    def test_default_path_is_under_data_learn(self, monkeypatch, repo_root):
        """Unset, the default sits in the gitignored data/learn/ directory."""
        monkeypatch.delenv(LEARN_DB_ENV, raising=False)
        assert default_db_path(repo_root) == repo_root / Path("data/learn/progress.db")


class TestReviewState:
    def test_round_trip(self, store, today):
        state = review(new_state("m06-q01"), Grade.GOOD, today)
        store.save_state(state)
        loaded = store.get_state("m06-q01")
        assert loaded == state

    def test_missing_question_returns_none(self, store):
        assert store.get_state("m06-q99") is None

    def test_upsert_replaces_rather_than_duplicating(self, store, today):
        first = review(new_state("m06-q01"), Grade.GOOD, today)
        store.save_state(first)
        second = review(first, Grade.EASY, today + timedelta(days=1))
        store.save_state(second)
        assert len(store.get_states()) == 1
        assert store.get_state("m06-q01").reps == 2

    def test_get_states_filters(self, store, today):
        for qid in ("m06-q01", "m06-q02", "m06-q03"):
            store.save_state(review(new_state(qid), Grade.GOOD, today))
        assert set(store.get_states(["m06-q01", "m06-q03"])) == {"m06-q01", "m06-q03"}
        assert len(store.get_states()) == 3

    def test_empty_filter_returns_empty_without_querying(self, store):
        """An empty IN () list is a SQLite syntax error, so it must short-circuit."""
        assert store.get_states([]) == {}

    def test_none_grade_and_dates_survive_the_round_trip(self, store):
        state = ReviewState(question_id="m06-q01")
        store.save_state(state)
        loaded = store.get_state("m06-q01")
        assert loaded.last_grade is None
        assert loaded.due_date is None
        assert loaded.is_new


class TestReviewLog:
    def test_logs_accumulate(self, store, today):
        state = review(new_state("m06-q01"), Grade.HARD, today)
        store.log_review(state, mode="drill")
        store.log_review(state, mode="exam")
        assert store.review_counts_by_grade() == {Grade.HARD: 2}

    def test_ungraded_state_is_rejected(self, store):
        with pytest.raises(ValueError, match="ungraded"):
            store.log_review(new_state("m06-q01"), mode="drill")


class TestReset:
    def test_scoped_reset_leaves_other_modules_alone(self, store, today):
        for qid in ("m06-q01", "m06-q02", "m05-q01"):
            store.save_state(review(new_state(qid), Grade.GOOD, today))
        deleted = store.reset(["m06-q01", "m06-q02"])
        assert deleted == 2
        assert set(store.get_states()) == {"m05-q01"}

    def test_full_reset_clears_labs_too(self, store, today):
        store.save_state(review(new_state("m06-q01"), Grade.GOOD, today))
        store.save_lab(LabState(lab_id="lab-01", status=LabStatus.IN_PROGRESS))
        store.reset()
        assert store.get_states() == {}
        assert store.all_labs() == {}

    def test_empty_id_list_deletes_nothing(self, store, today):
        store.save_state(review(new_state("m06-q01"), Grade.GOOD, today))
        assert store.reset([]) == 0
        assert len(store.get_states()) == 1


class TestLabs:
    def test_unrecorded_lab_defaults_to_not_started(self, store):
        state = store.get_lab("lab-01-purge-embargo")
        assert state.status is LabStatus.NOT_STARTED
        assert state.started_at is None

    def test_round_trip(self, store):
        state = LabState(
            lab_id="lab-01-purge-embargo",
            status=LabStatus.DONE,
            branch="fix/99-purge-embargo",
            started_at=date(2026, 7, 24),
            completed_at=date(2026, 7, 25),
        )
        store.save_lab(state)
        assert store.get_lab("lab-01-purge-embargo") == state

    def test_upsert_replaces(self, store):
        store.save_lab(LabState(lab_id="lab-01", status=LabStatus.IN_PROGRESS))
        store.save_lab(LabState(lab_id="lab-01", status=LabStatus.DONE))
        assert len(store.all_labs()) == 1
        assert store.get_lab("lab-01").status is LabStatus.DONE
