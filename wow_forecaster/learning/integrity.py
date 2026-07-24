"""
Content integrity checks: the drift guard.

This is the mechanism that decides whether a hand-authored question bank is
worth writing at all. A question citing ``trainer.py:88-97`` is wrong the moment
a line is inserted above it, and a study guide that quietly describes last
month's code is worse than no study guide. So every citation is a path plus a
verbatim single-line anchor, and this module proves that every anchor still
exists in the file it claims.

``check_content()`` returns problems rather than raising, and both
``tests/test_learning/test_bank_integrity.py`` and ``wowfc learn validate`` call
it. One implementation, so the rule the test enforces and the rule the authoring
command reports are the same rule by construction.

Deliberately not checked
------------------------
GitHub issue numbers. Verifying them needs a network call, and a test that fails
because the network is down is a test that gets skipped and then ignored.

Commit SHAs are checked only when the clone is deep enough to resolve them.
``.github/workflows/ci.yml`` uses ``actions/checkout@v4`` with no ``fetch-depth``,
so CI gets a depth-1 clone where no historical SHA resolves. Asserting there
would fail every run for a reason unrelated to content accuracy.
"""

from __future__ import annotations

import logging
import subprocess
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from wow_forecaster.learning.loader import (
    LABS_DIRNAME,
    content_root,
    lab_path,
    load_all_banks,
    load_curriculum,
)
from wow_forecaster.learning.models import Question

logger = logging.getLogger(__name__)

CONTENT_DIRNAME = "learning"


@dataclass(frozen=True)
class Problem:
    """One integrity failure.

    Attributes:
        code:    Stable machine-readable category, e.g. ``anchor-missing``.
        subject: What is wrong: a question id, module id, or path.
        detail:  Human-readable explanation, including how to fix it.
    """

    code: str
    subject: str
    detail: str

    def __str__(self) -> str:
        return f"[{self.code}] {self.subject}: {self.detail}"


def _is_shallow_clone(root: Path) -> bool:
    """True when the git clone is shallow, so historical SHAs cannot resolve."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--is-shallow-repository"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        logger.debug("shallow check failed, assuming shallow: %s", exc)
        return True
    if result.returncode != 0:
        return True
    return result.stdout.strip() == "true"


def _commit_exists(root: Path, sha: str) -> bool:
    """True when ``sha`` resolves to an object in this clone."""
    try:
        result = subprocess.run(
            ["git", "cat-file", "-e", f"{sha}^{{commit}}"],
            cwd=root,
            capture_output=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0


def _heading_lines(path: Path) -> list[str]:
    """Markdown heading lines from a file, read as UTF-8."""
    return [
        line
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.lstrip().startswith("#")
    ]


def _check_source(root: Path, q: Question) -> list[Problem]:
    """Verify a question's ``source`` path and ``anchor`` still hold."""
    problems: list[Problem] = []
    if q.source is None:
        return problems

    path = root / q.source
    if not path.is_file():
        problems.append(
            Problem("source-missing", q.id, f"source path does not exist: {q.source}")
        )
        return problems

    if q.anchor is None:
        return problems

    # UTF-8 pinned: the docstrings these anchors quote contain real arrows and
    # plus-minus characters, and Path.read_text() defaults to the locale
    # codepage on Windows, which raises on them.
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        problems.append(
            Problem("source-unreadable", q.id, f"{q.source} is not valid UTF-8: {exc}")
        )
        return problems

    if q.anchor not in text:
        problems.append(
            Problem(
                "anchor-missing",
                q.id,
                f"anchor no longer present in {q.source}. The code moved; "
                f"update the question or re-anchor it. Anchor: {q.anchor!r}",
            )
        )
    return problems


def _check_see_also(root: Path, q: Question) -> list[Problem]:
    """Verify every ``see_also`` reference resolves, fragments included."""
    problems: list[Problem] = []
    for ref in q.see_also:
        rel, _, fragment = ref.partition("#")
        path = root / rel
        if not path.is_file():
            problems.append(
                Problem("see-also-missing", q.id, f"see_also path does not exist: {rel}")
            )
            continue
        if fragment and not any(fragment in line for line in _heading_lines(path)):
            problems.append(
                Problem(
                    "see-also-anchor-missing",
                    q.id,
                    f"no heading in {rel} contains {fragment!r}",
                )
            )
    return problems


def check_content(root: Path | None = None) -> list[Problem]:
    """Run every content integrity check.

    Args:
        root: Repo root override. Defaults to discovery via ``content_root()``.

    Returns:
        Problems found, in a stable order. Empty means the content is coherent
        with the code as it stands right now.
    """
    base = root or content_root()
    problems: list[Problem] = []

    curriculum = load_curriculum(base)
    declared = {m.id for m in curriculum.modules}
    banks = load_all_banks(base)

    # A declared module with no bank is legal: the curriculum lists the whole
    # track so its shape is visible, while banks land one part per PR. A bank
    # with no declared module is not legal, because nothing would ever serve it.
    for module_id in sorted(set(banks) - declared):
        problems.append(
            Problem(
                "bank-undeclared",
                module_id,
                f"learning/banks/{module_id}.toml has no [[module]] entry in curriculum.toml",
            )
        )

    labs_dir = base / CONTENT_DIRNAME / LABS_DIRNAME
    for module in curriculum.modules:
        if module.lab and lab_path(module.lab, base) is None:
            problems.append(
                Problem(
                    "lab-missing",
                    module.id,
                    f"declares lab {module.lab!r} but {labs_dir / (module.lab + '.md')} is absent",
                )
            )

    all_questions: list[Question] = [q for qs in banks.values() for q in qs]

    duplicate_ids = [qid for qid, n in Counter(q.id for q in all_questions).items() if n > 1]
    for qid in sorted(duplicate_ids):
        problems.append(Problem("duplicate-id", qid, "question id appears more than once"))

    shallow = _is_shallow_clone(base)
    for q in sorted(all_questions, key=lambda x: x.id):
        problems.extend(_check_source(base, q))
        problems.extend(_check_see_also(base, q))
        if q.lab and lab_path(q.lab, base) is None:
            problems.append(
                Problem("lab-missing", q.id, f"references lab {q.lab!r} with no brief on disk")
            )
        if q.commit and not shallow and not _commit_exists(base, q.commit):
            problems.append(
                Problem("commit-missing", q.id, f"commit {q.commit} does not resolve")
            )

    if shallow:
        logger.debug("shallow clone: commit SHA checks skipped")

    return problems
