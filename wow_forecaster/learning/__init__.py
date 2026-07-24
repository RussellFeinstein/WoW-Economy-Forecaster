"""
Learning and assessment track.

A syllabus and an examiner for this repo. ``docs/ROADMAP.md`` owns the research
arc and the issue numbering, ``PLAN.md`` owns the lifecycle and legibility arc,
and this package owns understanding.

Governing principle: the repo is the textbook, this track is the syllabus and
the exam. No module here restates what a docstring already explains. Each
module is framing, a reading list of real files, a question set, and a lab.

Split of responsibility:
  - Content lives in ``learning/`` at the repo root as TOML and Markdown, the
    same way ``config/events/*.json`` holds seed content for ``events/``.
  - Code lives here, because the CLI entry point has to import it.
  - Personal review state lives in its own SQLite file (default
    ``data/learn/progress.db``), never in the product database. That database
    is copied into every durable backup and feeds the M3 warehouse; review
    state belongs in neither.

Entry point: ``wowfc learn`` (see ``cli.py``).
"""

from __future__ import annotations

__all__ = [
    "Curriculum",
    "Grade",
    "LabState",
    "Module",
    "Question",
    "ReviewState",
]

from wow_forecaster.learning.models import (
    Curriculum,
    Grade,
    LabState,
    Module,
    Question,
    ReviewState,
)
