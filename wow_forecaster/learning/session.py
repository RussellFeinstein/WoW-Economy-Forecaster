"""
Question selection and exam scoring. Pure logic, no I/O.

Kept free of the store and the terminal for the same reason
``cloud_sync.select_objects_to_ingest`` is a pure function over key names: the
interesting rules are the selection rules, and they should be testable without a
database or a keyboard.

Drill order
-----------
Due reviews before new material, oldest-overdue first. New questions follow in
bank order, which is authored pedagogical order rather than anything computed.
The consequence worth knowing: a passing grade sets ``due_date`` at least one day
out, so re-running a drill the same day serves no repeat of a card graded ``good``
or better, while a card graded ``again`` stays due today and does come back. That
is relearning, not a bug.

Exam order
----------
Round-robin across modules rather than a flat random sample. A flat sample over
unevenly sized banks quietly over-weights the largest module; round-robin gives
every module its turn before any module gets a second question.
"""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date

from wow_forecaster.learning.models import Question, ReviewState
from wow_forecaster.learning.scheduler import new_state

#: Rubric fraction at or above which a free-response answer counts as passed in
#: the exam summary. A reporting threshold only: it never touches scheduling,
#: where the reader's own grade is authoritative.
RUBRIC_PASS_FRACTION = 0.6


@dataclass(frozen=True)
class DrillItem:
    """One question queued for review, with the state it will be graded against."""

    question: Question
    state: ReviewState
    is_new: bool


def select_drill(
    questions: Sequence[Question],
    states: Mapping[str, ReviewState],
    count: int,
    today: date,
    include_new: bool = True,
) -> tuple[DrillItem, ...]:
    """Choose the next questions to drill.

    Args:
        questions:   Candidate pool, in authored order.
        states:      Stored review states keyed by question id. Missing entries
                     are treated as never-reviewed.
        count:       Maximum number of items to return. Values below 1 return
                     nothing rather than raising, so ``--count 0`` is a legal
                     way to ask "show me nothing".
        today:       The review date, injected.
        include_new: When False, only cards already in rotation are served.
                     Useful for clearing a backlog without taking on new
                     material.

    Returns:
        Due reviews first (oldest due date, then question id), then new
        questions in authored order, capped at ``count``.
    """
    if count < 1:
        return ()

    due: list[DrillItem] = []
    fresh: list[DrillItem] = []

    for q in questions:
        state = states.get(q.id)
        if state is None or state.is_new:
            fresh.append(DrillItem(question=q, state=state or new_state(q.id), is_new=True))
        elif state.is_due(today):
            due.append(DrillItem(question=q, state=state, is_new=False))

    due.sort(key=lambda item: (item.state.due_date or today, item.question.id))

    ordered = due + (fresh if include_new else [])
    return tuple(ordered[:count])


def select_exam(
    banks: Mapping[str, Sequence[Question]],
    count: int,
    rng: random.Random | None = None,
) -> tuple[Question, ...]:
    """Choose a stratified exam sample across modules.

    Args:
        banks: Questions keyed by module id.
        count: Maximum number of questions. Values below 1 return nothing.
        rng:   Random source. Pass a seeded ``random.Random`` for a repeatable
               exam; None draws a fresh unseeded one.

    Returns:
        Questions ordered module-by-module in rotation, so coverage is even
        before depth. Fewer than ``count`` when the banks hold fewer questions.
    """
    if count < 1:
        return ()
    source = rng if rng is not None else random.Random()

    pools: dict[str, list[Question]] = {}
    for module_id in sorted(banks):
        pool = list(banks[module_id])
        source.shuffle(pool)
        if pool:
            pools[module_id] = pool

    picked: list[Question] = []
    while len(picked) < count and pools:
        for module_id in sorted(pools):
            if len(picked) >= count:
                break
            picked.append(pools[module_id].pop())
        for module_id in [m for m, p in pools.items() if not p]:
            del pools[module_id]
    return tuple(picked)


@dataclass(frozen=True)
class ExamAnswer:
    """One recorded exam response.

    Exactly one of ``selected_index`` and ``rubric_hits`` is meaningful,
    depending on the question kind. Selection kinds score inline during the
    exam; free response is collected and self-scored against its rubric after
    the final question, so nothing is revealed mid-exam.
    """

    question: Question
    selected_index: int | None = None
    rubric_hits: int = 0
    skipped: bool = False

    @property
    def rubric_total(self) -> int:
        return len(self.question.rubric)

    @property
    def score(self) -> float:
        """Credit earned, in [0.0, 1.0]."""
        if self.skipped:
            return 0.0
        if self.question.is_selection:
            return 1.0 if self.selected_index == self.question.correct_index else 0.0
        if self.rubric_total == 0:
            return 0.0
        return min(1.0, self.rubric_hits / self.rubric_total)

    @property
    def passed(self) -> bool:
        """Whether this answer counts as correct in the summary."""
        if self.skipped:
            return False
        if self.question.is_selection:
            return self.selected_index == self.question.correct_index
        return self.score >= RUBRIC_PASS_FRACTION


@dataclass(frozen=True)
class ExamReport:
    """Aggregated exam outcome."""

    answers: tuple[ExamAnswer, ...]

    @property
    def n_questions(self) -> int:
        return len(self.answers)

    @property
    def n_passed(self) -> int:
        return sum(1 for a in self.answers if a.passed)

    @property
    def total_score(self) -> float:
        """Sum of partial credit, so a half-remembered rubric is not a zero."""
        return sum(a.score for a in self.answers)

    @property
    def percent(self) -> float:
        """Partial-credit score as a percentage, 0.0 when the exam was empty."""
        if not self.answers:
            return 0.0
        return 100.0 * self.total_score / self.n_questions

    def misses(self) -> tuple[ExamAnswer, ...]:
        """Answers that did not pass, in the order they were asked."""
        return tuple(a for a in self.answers if not a.passed)

    def by_module(self) -> dict[str, tuple[int, int]]:
        """Per-module ``(passed, asked)`` counts, keyed by module id."""
        tally: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        for a in self.answers:
            entry = tally[a.question.module_id]
            entry[1] += 1
            if a.passed:
                entry[0] += 1
        return {mid: (p, asked) for mid, (p, asked) in sorted(tally.items())}
