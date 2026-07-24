"""
Content loading and source-reference resolution.

Content root discovery
----------------------
``learning/`` sits at the repo root, outside the installable package:
``pyproject.toml`` sets ``packages.find`` to ``wow_forecaster*`` only, the same
way ``dashboard/`` is deliberately excluded. So the content is present in a
source checkout (``pip install -e .``) and absent from a wheel. Rather than
failing later on a missing file, ``content_root()`` says so by name.

Anchor resolution
-----------------
Questions cite a file path plus a verbatim single-line anchor. The current line
number is resolved here at display time, so a citation is always correct without
a stored number that would go stale. Files are read with an explicit
``encoding="utf-8"`` because the docstrings being quoted contain real Unicode
(arrows, plus-minus, approximately-equal) and ``Path.read_text()`` on Windows
defaults to the locale codepage, which raises on those bytes.
"""

from __future__ import annotations

import logging
import os
import tomllib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from wow_forecaster.learning.models import Curriculum, Module, Question

logger = logging.getLogger(__name__)

#: Environment override for the repo root. Test seam, matching the existing
#: WOWFC_SCHTASKS / WOWFC_POWERCFG / WOWFC pattern in scripts/ and tests/.
REPO_ROOT_ENV = "WOWFC_REPO_ROOT"

CONTENT_DIRNAME = "learning"
CURRICULUM_FILENAME = "curriculum.toml"
BANKS_DIRNAME = "banks"
MODULES_DIRNAME = "modules"
LABS_DIRNAME = "labs"


class ContentNotFoundError(RuntimeError):
    """Raised when the ``learning/`` content tree cannot be located."""


class BankParseError(ValueError):
    """Raised when a bank or curriculum file is malformed.

    Carries the offending path so a hand-authoring typo is one message away
    from the file that needs editing.
    """


@dataclass(frozen=True)
class SourceRef:
    """A resolved citation.

    Attributes:
        path:        Repo-relative source path as written in the bank.
        line:        1-based line number of the first anchor match, or None
                     when the anchor is absent or no anchor was given.
        occurrences: How many lines contain the anchor. More than one is legal
                     but worth surfacing, since the citation is then ambiguous.
        exists:      Whether the path resolves to a file at all.
    """

    path: str
    line: int | None = None
    occurrences: int = 0
    exists: bool = True

    def display(self) -> str:
        """Render as ``path:line``, degrading to ``path`` when unresolved.

        An anchor that no longer matches degrades rather than raising: the
        integrity test is the enforcement point, and a drill session should not
        crash because a citation went stale between commits.
        """
        if not self.exists:
            return f"{self.path} (missing)"
        if self.line is None:
            return self.path
        if self.occurrences > 1:
            return f"{self.path}:{self.line} (+{self.occurrences - 1} more)"
        return f"{self.path}:{self.line}"


def content_root(start: Path | None = None) -> Path:
    """Locate the repo root holding both ``pyproject.toml`` and ``learning/``.

    Args:
        start: Directory to search upward from. Defaults to this module's
            location, so the answer does not depend on the working directory.

    Returns:
        The repo root as an absolute path.

    Raises:
        ContentNotFoundError: When no ancestor qualifies, which means the
            package was installed from a wheel without the content tree.
    """
    override = os.environ.get(REPO_ROOT_ENV)
    if override:
        root = Path(override).resolve()
        if not (root / CONTENT_DIRNAME).is_dir():
            raise ContentNotFoundError(
                f"{REPO_ROOT_ENV}={override} does not contain a {CONTENT_DIRNAME}/ directory"
            )
        return root

    here = (start or Path(__file__).resolve().parent).resolve()
    for candidate in (here, *here.parents):
        if (candidate / "pyproject.toml").is_file() and (candidate / CONTENT_DIRNAME).is_dir():
            return candidate

    raise ContentNotFoundError(
        f"could not locate the {CONTENT_DIRNAME}/ content tree above {here}. "
        "`wowfc learn` needs a source checkout: the content ships with the repo, "
        "not with the wheel. Install with `pip install -e .` from a clone, or set "
        f"{REPO_ROOT_ENV} to the repo root."
    )


def _read_toml(path: Path) -> dict:
    """Parse one TOML file, reporting the path on failure."""
    if not path.is_file():
        raise BankParseError(f"missing content file: {path}")
    try:
        with path.open("rb") as fh:
            return tomllib.load(fh)
    except tomllib.TOMLDecodeError as exc:
        raise BankParseError(f"{path}: malformed TOML: {exc}") from exc


def load_curriculum(root: Path | None = None) -> Curriculum:
    """Load and validate ``learning/curriculum.toml``.

    Raises:
        BankParseError: If the file is missing, malformed, or fails validation.
    """
    base = root or content_root()
    path = base / CONTENT_DIRNAME / CURRICULUM_FILENAME
    raw = _read_toml(path)
    entries = raw.get("module")
    if not entries:
        raise BankParseError(f"{path}: no [[module]] entries found")
    try:
        modules = tuple(Module(**e) for e in entries)
        return Curriculum(modules=modules)
    except (TypeError, ValueError) as exc:
        raise BankParseError(f"{path}: {exc}") from exc


def bank_path(module_id: str, root: Path | None = None) -> Path:
    """Path to one module's bank file, whether or not it exists yet."""
    base = root or content_root()
    return base / CONTENT_DIRNAME / BANKS_DIRNAME / f"{module_id}.toml"


def load_bank(module_id: str, root: Path | None = None) -> tuple[Question, ...]:
    """Load one module's questions.

    Question ids are checked against the module id here rather than in the
    model, because a ``Question`` knows its own module from its id but cannot
    know which file it was read from.

    Raises:
        BankParseError: If the file is missing, malformed, fails validation, or
            contains a question belonging to a different module.
    """
    path = bank_path(module_id, root)
    raw = _read_toml(path)
    entries = raw.get("question", [])
    if not entries:
        raise BankParseError(f"{path}: no [[question]] entries found")

    questions: list[Question] = []
    for entry in entries:
        try:
            q = Question(**entry)
        except (TypeError, ValueError) as exc:
            qid = entry.get("id", "<no id>")
            raise BankParseError(f"{path}: question {qid}: {exc}") from exc
        if q.module_id != module_id:
            raise BankParseError(
                f"{path}: question {q.id} belongs to module {q.module_id}, not {module_id}"
            )
        questions.append(q)
    return tuple(questions)


def authored_module_ids(root: Path | None = None) -> tuple[str, ...]:
    """Module ids that have a bank file on disk, in sorted order.

    The curriculum lists all twenty modules from the first phase onward so the
    shape of the track is visible, while banks land a part at a time. This is
    the difference between planned and available.
    """
    base = root or content_root()
    banks_dir = base / CONTENT_DIRNAME / BANKS_DIRNAME
    if not banks_dir.is_dir():
        return ()
    return tuple(sorted(p.stem for p in banks_dir.glob("m[0-9][0-9].toml")))


def load_all_banks(root: Path | None = None) -> dict[str, tuple[Question, ...]]:
    """Load every authored bank, keyed by module id."""
    base = root or content_root()
    return {mid: load_bank(mid, base) for mid in authored_module_ids(base)}


def lesson_path(module_id: str, root: Path | None = None) -> Path | None:
    """Locate ``learning/modules/<module_id>-*.md``, or None if not authored."""
    base = root or content_root()
    matches = sorted((base / CONTENT_DIRNAME / MODULES_DIRNAME).glob(f"{module_id}-*.md"))
    return matches[0] if matches else None


def lab_path(lab_id: str, root: Path | None = None) -> Path | None:
    """Locate ``learning/labs/<lab_id>.md``, or None if not authored."""
    base = root or content_root()
    candidate = base / CONTENT_DIRNAME / LABS_DIRNAME / f"{lab_id}.md"
    return candidate if candidate.is_file() else None


@lru_cache(maxsize=256)
def _read_lines(path: Path) -> tuple[str, ...]:
    """Read a source file as lines, UTF-8, cached within one process.

    Cached because a drill session resolves many anchors against the same few
    files, and because the integrity test resolves every anchor in the repo.
    """
    return tuple(path.read_text(encoding="utf-8").splitlines())


def resolve_source(
    source: str | None,
    anchor: str | None,
    root: Path | None = None,
) -> SourceRef | None:
    """Resolve a citation to a current line number.

    Args:
        source: Repo-relative path, or None when the question cites nothing.
        anchor: Verbatim single-line snippet to locate within ``source``.
        root:   Content root override.

    Returns:
        A ``SourceRef``, or None when ``source`` is None.
    """
    if source is None:
        return None
    base = root or content_root()
    path = base / source
    if not path.is_file():
        return SourceRef(path=source, exists=False)
    if anchor is None:
        return SourceRef(path=source)

    lines = _read_lines(path)
    hits = [i for i, line in enumerate(lines, start=1) if anchor in line]
    if not hits:
        logger.debug("anchor not found in %s: %r", source, anchor)
        return SourceRef(path=source, occurrences=0)
    return SourceRef(path=source, line=hits[0], occurrences=len(hits))
