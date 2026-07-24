"""
Fixtures for the learning-track tests.

The autouse ``isolated_learn_db`` fixture is the important one: without it a test
that forgets ``--db-path`` writes ``data/learn/progress.db`` into the working
tree, which on CI means a test run mutating the checkout. Pinning
``WOWFC_LEARN_DB`` for every test in this package removes the possibility rather
than relying on each test to remember.
"""

from __future__ import annotations

from collections.abc import Generator
from datetime import date
from pathlib import Path

import pytest

from wow_forecaster.learning.loader import content_root
from wow_forecaster.learning.models import Option, Question
from wow_forecaster.learning.store import LEARN_DB_ENV, ProgressStore


@pytest.fixture(autouse=True)
def isolated_learn_db(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Generator[Path, None, None]:
    """Point the progress database at tmp_path for every test in this package."""
    db_path = tmp_path / "progress.db"
    monkeypatch.setenv(LEARN_DB_ENV, str(db_path))
    yield db_path


@pytest.fixture
def store(isolated_learn_db: Path) -> ProgressStore:
    """A ProgressStore on a fresh temporary database, schema applied."""
    s = ProgressStore()
    s.apply_schema()
    return s


@pytest.fixture
def repo_root() -> Path:
    """The real repo root, for tests that read the committed content tree."""
    return content_root()


@pytest.fixture
def today() -> date:
    """A fixed date, so nothing in these tests depends on the wall clock."""
    return date(2026, 7, 24)


def make_question(
    qid: str = "m06-q01",
    kind: str = "recall",
    **overrides,
) -> Question:
    """Build a valid Question with minimal ceremony.

    Defaults satisfy the model validators, so a test only states the field it
    actually cares about.
    """
    fields: dict = {
        "id": qid,
        "kind": kind,
        "prompt": "prompt",
        "answer": "answer",
    }
    if kind == "recall":
        fields["rubric"] = ("point one",)
    if kind == "mcq":
        fields["options"] = (
            Option(text="right", correct=True),
            Option(text="wrong one", note="because"),
            Option(text="wrong two", note="because"),
        )
    fields.update(overrides)
    return Question(**fields)
