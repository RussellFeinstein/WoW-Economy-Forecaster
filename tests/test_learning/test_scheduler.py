"""
Tests for SM-2 scheduling.

Every test injects ``today`` rather than relying on the wall clock, which is what
makes interval arithmetic assertable at all.
"""

from __future__ import annotations

from datetime import timedelta

import pytest

from wow_forecaster.learning.models import Grade, ReviewState
from wow_forecaster.learning.scheduler import (
    FIRST_INTERVAL_DAYS,
    MASTERY_INTERVAL_DAYS,
    MIN_EASE,
    SECOND_INTERVAL_DAYS,
    is_mastered,
    mastery_fraction,
    new_state,
    review,
)


class TestFirstReview:
    """A card's first passing review uses fixed intervals, not the ease factor."""

    @pytest.mark.parametrize(
        ("grade", "expected"),
        [
            (Grade.HARD, FIRST_INTERVAL_DAYS[Grade.HARD]),
            (Grade.GOOD, FIRST_INTERVAL_DAYS[Grade.GOOD]),
            (Grade.EASY, FIRST_INTERVAL_DAYS[Grade.EASY]),
        ],
    )
    def test_interval(self, grade, expected, today):
        result = review(new_state("m06-q01"), grade, today)
        assert result.interval_days == expected
        assert result.due_date == today + timedelta(days=expected)
        assert result.reps == 1

    def test_again_leaves_card_due_today(self, today):
        """A failed card returns in the same session; that is relearning."""
        result = review(new_state("m06-q01"), Grade.AGAIN, today)
        assert result.interval_days == 0
        assert result.due_date == today
        assert result.is_due(today)
        assert result.lapses == 1

    def test_input_state_is_not_mutated(self, today):
        original = new_state("m06-q01")
        review(original, Grade.EASY, today)
        assert original.reps == 0
        assert original.interval_days == 0


class TestSecondReview:
    def test_good_uses_second_interval(self, today):
        first = review(new_state("m06-q01"), Grade.GOOD, today)
        second = review(first, Grade.GOOD, first.due_date)
        assert second.interval_days == SECOND_INTERVAL_DAYS[Grade.GOOD]
        assert second.reps == 2


class TestEaseFactor:
    def test_again_lowers_ease_and_easy_raises_it(self, today):
        base = new_state("m06-q01")
        assert review(base, Grade.AGAIN, today).ease < base.ease
        assert review(base, Grade.EASY, today).ease > base.ease
        assert review(base, Grade.GOOD, today).ease == base.ease

    def test_ease_floors(self, today):
        """Repeated failures cannot drive ease below the floor."""
        state = new_state("m06-q01")
        day = today
        for _ in range(20):
            state = review(state, Grade.AGAIN, day)
            day += timedelta(days=1)
        assert state.ease == pytest.approx(MIN_EASE)


class TestIntervalGrowth:
    def test_passed_cards_always_grow(self, today):
        """A pass must move the card further out, even when rounding says otherwise."""
        state = ReviewState(question_id="m06-q01", interval_days=3, reps=4, ease=1.3)
        result = review(state, Grade.HARD, today)
        assert result.interval_days > state.interval_days

    def test_easy_outpaces_good(self, today):
        state = ReviewState(question_id="m06-q01", interval_days=10, reps=4)
        assert (
            review(state, Grade.EASY, today).interval_days
            > review(state, Grade.GOOD, today).interval_days
            > review(state, Grade.HARD, today).interval_days
        )

    def test_lapse_resets_interval_but_keeps_history(self, today):
        state = ReviewState(question_id="m06-q01", interval_days=40, reps=6, lapses=1)
        result = review(state, Grade.AGAIN, today)
        assert result.interval_days == 0
        assert result.reps == 7
        assert result.lapses == 2


class TestSameDayRegrade:
    """The last grade of the day replaces earlier ones instead of compounding."""

    def test_repeat_good_does_not_double_advance(self, today):
        once = review(new_state("m06-q01"), Grade.GOOD, today)
        twice = review(once, Grade.GOOD, today)
        assert twice.interval_days == once.interval_days
        assert twice.due_date == once.due_date
        assert twice.reps == once.reps

    def test_five_grades_land_where_one_would(self, today):
        state = ReviewState(question_id="m06-q01", interval_days=6, reps=3)
        once = review(state, Grade.GOOD, today)
        repeated = state
        for _ in range(5):
            repeated = review(repeated, Grade.GOOD, today)
        assert repeated.interval_days == once.interval_days
        assert repeated.ease == pytest.approx(once.ease)
        assert repeated.reps == once.reps

    def test_later_grade_wins(self, today):
        """Failing then correcting to easy lands where a single easy would."""
        state = ReviewState(question_id="m06-q01", interval_days=6, reps=3)
        direct = review(state, Grade.EASY, today)
        corrected = review(review(state, Grade.AGAIN, today), Grade.EASY, today)
        assert corrected.interval_days == direct.interval_days
        assert corrected.ease == pytest.approx(direct.ease)
        assert corrected.lapses == direct.lapses == 0

    def test_next_day_review_compounds_normally(self, today):
        """The rewind applies only within a single date."""
        first = review(new_state("m06-q01"), Grade.GOOD, today)
        second = review(first, Grade.GOOD, today + timedelta(days=1))
        assert second.reps == 2
        assert second.interval_days > first.interval_days


class TestDueness:
    def test_new_card_is_due(self, today):
        assert new_state("m06-q01").is_due(today)
        assert new_state("m06-q01").is_new

    def test_passed_card_is_not_due_same_day(self, today):
        """The property behind: re-running a drill serves no repeat of a passed card."""
        result = review(new_state("m06-q01"), Grade.GOOD, today)
        assert not result.is_due(today)
        assert result.is_due(result.due_date)


class TestMastery:
    def test_threshold(self):
        assert is_mastered(ReviewState(question_id="q", interval_days=MASTERY_INTERVAL_DAYS))
        assert not is_mastered(
            ReviewState(question_id="q", interval_days=MASTERY_INTERVAL_DAYS - 1)
        )

    def test_fraction_counts_against_the_bank_not_the_reviewed_set(self):
        states = [ReviewState(question_id="q1", interval_days=MASTERY_INTERVAL_DAYS)]
        assert mastery_fraction(states, total_questions=4) == pytest.approx(0.25)

    def test_fraction_handles_empty_module(self):
        assert mastery_fraction([], total_questions=0) == 0.0
