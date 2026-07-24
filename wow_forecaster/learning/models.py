"""
Learning-track domain models.

All models are ``frozen=True``, matching the house convention: only
``RunMetadata`` in ``models/meta.py`` is mutable, because it carries a status
that changes mid-run. Nothing here does.

Why Pydantic rather than dataclasses: the question banks are hand-authored
TOML, so the parse boundary is where typos surface. A validation error naming
the field beats an ``AttributeError`` three calls later.

Source references
-----------------
Every ``Question`` cites its evidence as a ``source`` path plus a verbatim
single-line ``anchor``, never a line number. Line numbers rot the moment a line
is inserted above them; an anchor either still exists in the file or the
integrity test fails. The CLI resolves the anchor to a current line number at
display time, so the citation is correct without being stored.
"""

from __future__ import annotations

from datetime import date
from enum import IntEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

QuestionKind = Literal["recall", "mcq", "predict", "find"]

#: Kinds whose answer is chosen from a fixed option list, so a session can
#: score them without the reader grading themselves.
SELECTION_KINDS: frozenset[str] = frozenset({"mcq", "predict"})


class Grade(IntEnum):
    """Recall quality for one review, in SM-2 terms.

    The four-button scheme rather than SM-2's original 0-5: a self-grader
    cannot reliably distinguish six levels, and collapsing to four removes the
    distinctions nobody uses. AGAIN is the only failing grade.
    """

    AGAIN = 1
    HARD = 2
    GOOD = 3
    EASY = 4

    @property
    def is_lapse(self) -> bool:
        """True when this grade resets the interval."""
        return self is Grade.AGAIN


class LabStatus(IntEnum):
    """Where a lab stands. Ordered so ``max()`` picks the furthest progress."""

    NOT_STARTED = 0
    IN_PROGRESS = 1
    DONE = 2


class Option(BaseModel):
    """One choice for a selection-kind question.

    ``note`` carries why a wrong option is wrong. For multiple choice that is
    where most of the learning happens, so it is required on incorrect options
    and pointless on the correct one.
    """

    model_config = ConfigDict(frozen=True)

    text: str = Field(min_length=1)
    correct: bool = False
    note: str | None = None

    @model_validator(mode="after")
    def _wrong_options_explain_themselves(self) -> Option:
        if not self.correct and not self.note:
            raise ValueError(
                f"incorrect option {self.text!r} needs a note explaining why it is wrong"
            )
        return self


class Question(BaseModel):
    """One question, its answer, and the evidence backing it."""

    model_config = ConfigDict(frozen=True)

    id: str = Field(pattern=r"^m\d{2}-q\d{2}$")
    kind: QuestionKind
    prompt: str = Field(min_length=1)
    answer: str = Field(min_length=1)
    rubric: tuple[str, ...] = ()
    options: tuple[Option, ...] = ()
    source: str | None = None
    anchor: str | None = None
    see_also: tuple[str, ...] = ()
    lab: str | None = None
    commit: str | None = None

    @property
    def module_id(self) -> str:
        """The owning module id, derived from the question id (``m06-q03`` -> ``m06``)."""
        return self.id.split("-", 1)[0]

    @property
    def is_selection(self) -> bool:
        """True when the question can be auto-scored from an option list."""
        return self.kind in SELECTION_KINDS and bool(self.options)

    @property
    def correct_index(self) -> int | None:
        """Zero-based index of the correct option, or None for free response."""
        for i, opt in enumerate(self.options):
            if opt.correct:
                return i
        return None

    @field_validator("anchor")
    @classmethod
    def _anchor_is_single_line(cls, v: str | None) -> str | None:
        """Reject multi-line anchors.

        A multi-line anchor cannot match reliably: the file on disk may use CRLF
        while the TOML literal uses LF, so the substring search fails for a
        reason that has nothing to do with the code having changed. Single-line
        anchors are immune.
        """
        if v is not None and ("\n" in v or "\r" in v):
            raise ValueError("anchor must be a single line (no newlines)")
        return v

    @model_validator(mode="after")
    def _shape_matches_kind(self) -> Question:
        if self.kind == "recall" and not self.rubric:
            raise ValueError(f"{self.id}: recall questions need a non-empty rubric")
        if self.options:
            n_correct = sum(1 for o in self.options if o.correct)
            if len(self.options) < 3:
                raise ValueError(f"{self.id}: needs at least 3 options, got {len(self.options)}")
            if n_correct != 1:
                raise ValueError(
                    f"{self.id}: needs exactly 1 correct option, got {n_correct}"
                )
        if self.kind == "mcq" and not self.options:
            raise ValueError(f"{self.id}: mcq questions need an option list")
        if self.anchor is not None and self.source is None:
            raise ValueError(f"{self.id}: anchor given without a source path")
        return self


class Module(BaseModel):
    """One curriculum module: what it covers, what to read, and its lab."""

    model_config = ConfigDict(frozen=True)

    id: str = Field(pattern=r"^m\d{2}$")
    title: str = Field(min_length=1)
    part: str = Field(min_length=1)
    summary: str = ""
    objectives: tuple[str, ...] = ()
    reading: tuple[str, ...] = ()
    prereqs: tuple[str, ...] = ()
    lab: str | None = None

    @property
    def lesson_filename_stem(self) -> str:
        """Slug stem used to locate ``learning/modules/<stem>*.md``."""
        return self.id


class Curriculum(BaseModel):
    """The full module list, in study order."""

    model_config = ConfigDict(frozen=True)

    modules: tuple[Module, ...]

    @model_validator(mode="after")
    def _ids_unique_and_prereqs_resolve(self) -> Curriculum:
        ids = [m.id for m in self.modules]
        dupes = {i for i in ids if ids.count(i) > 1}
        if dupes:
            raise ValueError(f"duplicate module ids: {sorted(dupes)}")
        known = set(ids)
        for m in self.modules:
            unknown = set(m.prereqs) - known
            if unknown:
                raise ValueError(f"{m.id}: unknown prereqs {sorted(unknown)}")
        return self

    def by_id(self, module_id: str) -> Module:
        """Look up one module.

        Raises:
            KeyError: If no module carries that id, listing the valid ids.
        """
        for m in self.modules:
            if m.id == module_id:
                return m
        raise KeyError(
            f"unknown module {module_id!r}; known ids: {', '.join(m.id for m in self.modules)}"
        )


class ReviewState(BaseModel):
    """Scheduling state for one question.

    ``ease`` is the SM-2 ease factor: the multiplier applied to the interval on
    a passing review. It floors at 1.3, below which a card would repeat forever.

    ``prev_ease`` and ``prev_interval_days`` hold the values as of before the
    most recent review. They exist so a second grade on the same date can rewind
    and replace the first rather than compounding on top of it, which would push
    a re-drilled card months out for no reason. See ``scheduler.review``.
    """

    model_config = ConfigDict(frozen=True)

    question_id: str
    ease: float = 2.5
    interval_days: int = 0
    due_date: date | None = None
    reps: int = 0
    lapses: int = 0
    last_grade: Grade | None = None
    last_reviewed_at: date | None = None
    prev_ease: float = 2.5
    prev_interval_days: int = 0

    @property
    def is_new(self) -> bool:
        """True when this question has never been graded."""
        return self.reps == 0 and self.last_reviewed_at is None

    def is_due(self, today: date) -> bool:
        """True when the card is scheduled for review on or before ``today``."""
        if self.due_date is None:
            return True
        return self.due_date <= today


class LabState(BaseModel):
    """Progress on one lab."""

    model_config = ConfigDict(frozen=True)

    lab_id: str
    status: LabStatus = LabStatus.NOT_STARTED
    branch: str | None = None
    started_at: date | None = None
    completed_at: date | None = None
    notes: str | None = None
