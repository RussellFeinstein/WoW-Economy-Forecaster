"""Tests for question selection and exam scoring."""

from __future__ import annotations

import random
from datetime import timedelta

import pytest

from tests.test_learning.conftest import make_question
from wow_forecaster.learning.models import Grade, ReviewState
from wow_forecaster.learning.scheduler import new_state, review
from wow_forecaster.learning.session import (
    ExamAnswer,
    ExamReport,
    select_drill,
    select_exam,
)


def q(n: int, module: str = "m06", kind: str = "recall"):
    return make_question(qid=f"{module}-q{n:02d}", kind=kind)


class TestSelectDrill:
    def test_new_questions_come_in_authored_order(self, today):
        questions = [q(1), q(2), q(3)]
        picked = select_drill(questions, {}, count=3, today=today)
        assert [i.question.id for i in picked] == ["m06-q01", "m06-q02", "m06-q03"]
        assert all(i.is_new for i in picked)

    def test_due_reviews_precede_new_material(self, today):
        questions = [q(1), q(2)]
        overdue = ReviewState(
            question_id="m06-q02",
            reps=3,
            interval_days=5,
            due_date=today - timedelta(days=2),
            last_reviewed_at=today - timedelta(days=7),
        )
        picked = select_drill(questions, {"m06-q02": overdue}, count=2, today=today)
        assert [i.question.id for i in picked] == ["m06-q02", "m06-q01"]
        assert picked[0].is_new is False

    def test_oldest_due_first(self, today):
        questions = [q(1), q(2), q(3)]
        states = {}
        for n, days in ((1, 1), (2, 5), (3, 3)):
            states[f"m06-q{n:02d}"] = ReviewState(
                question_id=f"m06-q{n:02d}",
                reps=2,
                interval_days=4,
                due_date=today - timedelta(days=days),
                last_reviewed_at=today - timedelta(days=days + 4),
            )
        picked = select_drill(questions, states, count=3, today=today)
        assert [i.question.id for i in picked] == ["m06-q02", "m06-q03", "m06-q01"]

    def test_count_caps_the_queue(self, today):
        picked = select_drill([q(n) for n in range(1, 11)], {}, count=3, today=today)
        assert len(picked) == 3

    def test_count_below_one_returns_nothing(self, today):
        assert select_drill([q(1)], {}, count=0, today=today) == ()

    def test_passed_card_is_not_served_again_the_same_day(self, today):
        """Re-running a drill serves no repeat of anything graded good or better."""
        graded = review(new_state("m06-q01"), Grade.GOOD, today)
        picked = select_drill([q(1)], {"m06-q01": graded}, count=5, today=today)
        assert picked == ()

    def test_failed_card_does_come_back_the_same_day(self, today):
        graded = review(new_state("m06-q01"), Grade.AGAIN, today)
        picked = select_drill([q(1)], {"m06-q01": graded}, count=5, today=today)
        assert [i.question.id for i in picked] == ["m06-q01"]

    def test_reviews_only_excludes_new_material(self, today):
        questions = [q(1), q(2)]
        overdue = ReviewState(
            question_id="m06-q01",
            reps=2,
            interval_days=3,
            due_date=today,
            last_reviewed_at=today - timedelta(days=3),
        )
        picked = select_drill(
            questions, {"m06-q01": overdue}, count=5, today=today, include_new=False
        )
        assert [i.question.id for i in picked] == ["m06-q01"]

    def test_missing_state_is_treated_as_new_not_an_error(self, today):
        picked = select_drill([q(1)], {}, count=1, today=today)
        assert picked[0].state.is_new


class TestSelectExam:
    def test_is_repeatable_with_a_seed(self):
        banks = {"m05": [q(n, "m05") for n in range(1, 6)], "m06": [q(n) for n in range(1, 6)]}
        first = select_exam(banks, 6, random.Random(42))
        second = select_exam(banks, 6, random.Random(42))
        assert [x.id for x in first] == [x.id for x in second]

    def test_rotates_across_modules_before_repeating_one(self):
        """A flat sample over uneven banks over-weights the largest module."""
        banks = {"m05": [q(n, "m05") for n in range(1, 21)], "m06": [q(1), q(2)]}
        picked = select_exam(banks, 4, random.Random(7))
        modules = [x.module_id for x in picked]
        assert modules.count("m06") == 2
        assert modules.count("m05") == 2

    def test_drains_exhausted_modules_without_stalling(self):
        banks = {"m05": [q(1, "m05")], "m06": [q(n) for n in range(1, 5)]}
        picked = select_exam(banks, 5, random.Random(1))
        assert len(picked) == 5
        assert len({x.id for x in picked}) == 5

    def test_returns_fewer_than_requested_when_banks_are_small(self):
        assert len(select_exam({"m06": [q(1), q(2)]}, 10, random.Random(1))) == 2

    def test_count_below_one_returns_nothing(self):
        assert select_exam({"m06": [q(1)]}, 0, random.Random(1)) == ()

    def test_empty_banks_return_nothing(self):
        assert select_exam({}, 5, random.Random(1)) == ()


class TestExamScoring:
    def test_selection_question_scores_on_the_correct_index(self):
        question = make_question(kind="mcq")
        assert ExamAnswer(question=question, selected_index=0).passed
        assert not ExamAnswer(question=question, selected_index=1).passed

    def test_unanswered_selection_is_a_miss(self):
        answer = ExamAnswer(question=make_question(kind="mcq"), selected_index=None)
        assert not answer.passed
        assert answer.score == 0.0

    def test_free_response_earns_partial_credit(self):
        question = make_question(
            kind="recall", rubric=("a", "b", "c", "d")
        )
        assert ExamAnswer(question=question, rubric_hits=3).score == pytest.approx(0.75)
        assert ExamAnswer(question=question, rubric_hits=3).passed
        assert not ExamAnswer(question=question, rubric_hits=1).passed

    def test_score_is_capped_at_one(self):
        question = make_question(kind="recall", rubric=("a", "b"))
        assert ExamAnswer(question=question, rubric_hits=99).score == 1.0

    def test_skipped_scores_zero_regardless_of_hits(self):
        question = make_question(kind="recall", rubric=("a", "b"))
        answer = ExamAnswer(question=question, rubric_hits=2, skipped=True)
        assert answer.score == 0.0
        assert not answer.passed


class TestExamReport:
    def _report(self):
        mcq = make_question(qid="m06-q01", kind="mcq")
        recall = make_question(qid="m05-q01", kind="recall", rubric=("a", "b"))
        return ExamReport(
            answers=(
                ExamAnswer(question=mcq, selected_index=0),
                ExamAnswer(question=recall, rubric_hits=1),
            )
        )

    def test_partial_credit_shows_in_the_total(self):
        report = self._report()
        assert report.total_score == pytest.approx(1.5)
        assert report.n_passed == 1
        assert report.percent == pytest.approx(75.0)

    def test_misses_are_listed(self):
        assert [a.question.id for a in self._report().misses()] == ["m05-q01"]

    def test_by_module_groups_and_sorts(self):
        assert self._report().by_module() == {"m05": (0, 1), "m06": (1, 1)}

    def test_empty_report_does_not_divide_by_zero(self):
        empty = ExamReport(answers=())
        assert empty.percent == 0.0
        assert empty.by_module() == {}
