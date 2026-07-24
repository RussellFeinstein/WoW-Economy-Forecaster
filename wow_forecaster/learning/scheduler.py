"""
Spaced-repetition scheduling (SM-2 variant).

Why SM-2 and not something newer: it is roughly thirty lines, deterministic,
needs no training data, and is the algorithm behind the tool most people have
actually used. FSRS is better calibrated but wants a review history to fit
against, which does not exist on day one.

Deviations from the original, each deliberate:

  - **Four grades, not six.** A self-grader cannot reliably tell "perfect
    recall" from "recall after hesitation" from "recall with serious
    difficulty". Collapsing to again / hard / good / easy removes distinctions
    that would be noise.
  - **Strictly growing intervals on a pass.** ``max(base + 1, round(base * m))``
    guarantees a passed card always moves further out. Plain SM-2 can stall a
    short-interval card at its current interval when the multiplier rounds down,
    so the card is served forever without ever being wrong.
  - **Same-day re-grades replace rather than compound.** See below.

Same-day re-grading
-------------------
Re-drilling a module the same day must not push a card months out just because
it was answered twice. ``ReviewState`` therefore keeps ``prev_ease`` and
``prev_interval_days``, the values as of before the most recent review. A second
review on the same date rewinds to those values first, so the last grade of the
day is the one that counts and grading a card five times in an afternoon lands
in the same place as grading it once.

The clock is injectable everywhere (``today: date | None = None``, resolving to
the real date on None), the same pattern the rollup step uses for ``run(now=...)``
and the health check for ``load_ingest_age_hours(now=...)``.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import date, timedelta

from wow_forecaster.learning.models import Grade, ReviewState

#: Ease floor. Below roughly 1.3 the interval stops growing meaningfully and the
#: card is served forever, which is a scheduling failure rather than difficulty.
MIN_EASE = 1.3

#: Ease ceiling. Without one, a long easy streak pushes a card past any horizon
#: where review is still useful.
MAX_EASE = 3.0

EASE_PENALTY_AGAIN = 0.20
EASE_PENALTY_HARD = 0.15
EASE_BONUS_EASY = 0.15

#: Extra multiplier on an EASY pass, on top of the ease factor.
EASY_BONUS = 1.3

#: Multiplier for a HARD pass. Below the ease floor on purpose: HARD should
#: advance the card, but barely.
HARD_MULTIPLIER = 1.2

#: Interval in days for the first passing review of a card.
FIRST_INTERVAL_DAYS: dict[Grade, int] = {Grade.HARD: 1, Grade.GOOD: 1, Grade.EASY: 4}

#: Interval in days for the second passing review, before ease takes over.
SECOND_INTERVAL_DAYS: dict[Grade, int] = {Grade.HARD: 3, Grade.GOOD: 6, Grade.EASY: 10}

#: Interval at which a question counts as mastered for reporting. Three weeks is
#: the point where a card stops appearing often enough to feel like study.
MASTERY_INTERVAL_DAYS = 21


def _resolve_today(today: date | None) -> date:
    """Return ``today``, defaulting to the real current date."""
    return today if today is not None else date.today()


def review(
    state: ReviewState,
    grade: Grade,
    today: date | None = None,
) -> ReviewState:
    """Apply one graded review and return the updated state.

    Pure: the same (state, grade, today) always produces the same result, which
    is what makes the scheduling testable without freezing the clock globally.

    Args:
        state: Current scheduling state for the question.
        grade: How well it was recalled.
        today: Review date. Defaults to the real current date.

    Returns:
        A new ``ReviewState``. The input is never mutated (models are frozen).
    """
    now = _resolve_today(today)

    # Rewind a same-day repeat so the last grade of the day is authoritative.
    if state.last_reviewed_at == now and state.reps > 0:
        base_ease = state.prev_ease
        base_interval = state.prev_interval_days
        base_reps = state.reps - 1
        base_lapses = state.lapses - (1 if state.last_grade is Grade.AGAIN else 0)
    else:
        base_ease = state.ease
        base_interval = state.interval_days
        base_reps = state.reps
        base_lapses = state.lapses

    if grade.is_lapse:
        ease = max(MIN_EASE, base_ease - EASE_PENALTY_AGAIN)
        interval = 0
        lapses = base_lapses + 1
        due = now
    else:
        if grade is Grade.HARD:
            ease = max(MIN_EASE, base_ease - EASE_PENALTY_HARD)
        elif grade is Grade.EASY:
            ease = min(MAX_EASE, base_ease + EASE_BONUS_EASY)
        else:
            ease = base_ease

        if base_reps == 0 or base_interval == 0:
            interval = FIRST_INTERVAL_DAYS[grade]
        elif base_interval <= 1:
            interval = SECOND_INTERVAL_DAYS[grade]
        else:
            multiplier = {
                Grade.HARD: HARD_MULTIPLIER,
                Grade.GOOD: ease,
                Grade.EASY: ease * EASY_BONUS,
            }[grade]
            # Strictly growing: never let rounding stall a passed card.
            interval = max(base_interval + 1, round(base_interval * multiplier))

        lapses = base_lapses
        due = now + timedelta(days=interval)

    return ReviewState(
        question_id=state.question_id,
        ease=round(ease, 4),
        interval_days=interval,
        due_date=due,
        reps=base_reps + 1,
        lapses=lapses,
        last_grade=grade,
        last_reviewed_at=now,
        prev_ease=round(base_ease, 4),
        prev_interval_days=base_interval,
    )


def new_state(question_id: str) -> ReviewState:
    """Return the starting state for a question that has never been reviewed."""
    return ReviewState(question_id=question_id)


def is_mastered(state: ReviewState) -> bool:
    """True when the card has reached the mastery interval without a pending lapse."""
    return state.interval_days >= MASTERY_INTERVAL_DAYS


def mastery_fraction(states: Iterable[ReviewState], total_questions: int) -> float:
    """Fraction of a module's questions that have reached the mastery interval.

    Args:
        states:          Review states for questions in the module. States for
                         questions that no longer exist are the caller's problem
                         to filter; this function only counts.
        total_questions: Question count in the module, the denominator. Counting
                         against the bank rather than against reviewed cards
                         keeps an untouched module at 0 percent instead of
                         undefined.

    Returns:
        A value in [0.0, 1.0]. Zero when the module has no questions.
    """
    if total_questions <= 0:
        return 0.0
    mastered = sum(1 for s in states if is_mastered(s))
    return min(1.0, mastered / total_questions)
