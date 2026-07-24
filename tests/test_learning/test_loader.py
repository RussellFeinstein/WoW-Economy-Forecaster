"""
Tests for content discovery, parsing, and anchor resolution.

Most tests build a synthetic content tree under tmp_path rather than reading the
committed one, so a parse failure is attributable to the loader and not to a
typo in a real bank. The tests that do read the real tree say so.
"""

from __future__ import annotations

import pytest

from wow_forecaster.learning.loader import (
    REPO_ROOT_ENV,
    BankParseError,
    ContentNotFoundError,
    authored_module_ids,
    content_root,
    lab_path,
    lesson_path,
    load_bank,
    load_curriculum,
    resolve_source,
)

MINIMAL_MODULE = """
[[module]]
id = "m01"
title = "First"
part = "Part I"
"""

MINIMAL_BANK = """
[[question]]
id = "m01-q01"
kind = "recall"
prompt = "why"
answer = "because"
rubric = ["a point"]
"""


def build_tree(root, curriculum=MINIMAL_MODULE, banks=None):
    """Create a minimal learning/ tree under ``root`` and return the root."""
    (root / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
    content = root / "learning"
    (content / "banks").mkdir(parents=True)
    (content / "modules").mkdir()
    (content / "labs").mkdir()
    (content / "curriculum.toml").write_text(curriculum, encoding="utf-8")
    for module_id, text in (banks or {"m01": MINIMAL_BANK}).items():
        (content / "banks" / f"{module_id}.toml").write_text(text, encoding="utf-8")
    return root


class TestContentRoot:
    def test_finds_the_real_repo_root(self):
        root = content_root()
        assert (root / "pyproject.toml").is_file()
        assert (root / "learning" / "curriculum.toml").is_file()

    def test_env_override(self, tmp_path, monkeypatch):
        build_tree(tmp_path)
        monkeypatch.setenv(REPO_ROOT_ENV, str(tmp_path))
        assert content_root() == tmp_path.resolve()

    def test_override_without_content_raises(self, tmp_path, monkeypatch):
        monkeypatch.setenv(REPO_ROOT_ENV, str(tmp_path))
        with pytest.raises(ContentNotFoundError, match="learning/ directory"):
            content_root()

    def test_missing_tree_names_the_wheel_problem(self, tmp_path, monkeypatch):
        """A wheel install has no content, and the error has to say so."""
        monkeypatch.delenv(REPO_ROOT_ENV, raising=False)
        orphan = tmp_path / "no" / "repo" / "here"
        orphan.mkdir(parents=True)
        with pytest.raises(ContentNotFoundError, match="source checkout"):
            content_root(start=orphan)


class TestCurriculum:
    def test_real_curriculum_declares_twenty_modules(self):
        curriculum = load_curriculum()
        assert len(curriculum.modules) == 20
        assert [m.id for m in curriculum.modules] == [f"m{i:02d}" for i in range(1, 21)]

    def test_by_id_raises_with_known_ids_listed(self):
        with pytest.raises(KeyError, match="m06"):
            load_curriculum().by_id("m99")

    def test_malformed_toml_names_the_file(self, tmp_path):
        build_tree(tmp_path, curriculum="[[module]\nid = ")
        with pytest.raises(BankParseError, match="malformed TOML"):
            load_curriculum(tmp_path)

    def test_empty_curriculum_rejected(self, tmp_path):
        build_tree(tmp_path, curriculum="# nothing here\n")
        with pytest.raises(BankParseError, match="no \\[\\[module\\]\\] entries"):
            load_curriculum(tmp_path)

    def test_unknown_prereq_rejected(self, tmp_path):
        build_tree(
            tmp_path,
            curriculum=MINIMAL_MODULE + '\nprereqs = ["m99"]\n',
        )
        with pytest.raises(BankParseError, match="unknown prereqs"):
            load_curriculum(tmp_path)


class TestBanks:
    def test_loads_the_real_m06_bank(self):
        questions = load_bank("m06")
        assert len(questions) >= 15
        assert all(q.module_id == "m06" for q in questions)

    def test_missing_bank_names_the_path(self, tmp_path):
        build_tree(tmp_path)
        with pytest.raises(BankParseError, match="missing content file"):
            load_bank("m02", tmp_path)

    def test_question_id_must_match_its_file(self, tmp_path):
        build_tree(tmp_path, banks={"m01": MINIMAL_BANK.replace("m01-q01", "m02-q01")})
        with pytest.raises(BankParseError, match="belongs to module m02"):
            load_bank("m01", tmp_path)

    def test_validation_error_names_the_question(self, tmp_path):
        """A recall question without a rubric is a parse-time failure, not a runtime one."""
        broken = MINIMAL_BANK.replace('rubric = ["a point"]', "")
        build_tree(tmp_path, banks={"m01": broken})
        with pytest.raises(BankParseError, match="m01-q01"):
            load_bank("m01", tmp_path)

    def test_authored_ids_are_sorted_and_exclude_unwritten_modules(self):
        authored = authored_module_ids()
        assert "m06" in authored
        assert list(authored) == sorted(authored)
        assert len(authored) < 20


class TestPaths:
    def test_lesson_and_lab_resolve_for_m06(self):
        lesson = lesson_path("m06")
        assert lesson is not None and lesson.name.startswith("m06-")
        assert lab_path("lab-01-purge-embargo") is not None

    def test_unwritten_lesson_and_lab_return_none(self):
        assert lesson_path("m20") is None
        assert lab_path("lab-99-does-not-exist") is None


class TestResolveSource:
    def test_none_source_resolves_to_none(self):
        assert resolve_source(None, None) is None

    def test_anchor_resolves_to_a_current_line_number(self):
        ref = resolve_source(
            "wow_forecaster/ml/trainer.py",
            "val_split_date = date_strs[-(val_days + 1)]",
        )
        assert ref.exists and ref.line is not None and ref.occurrences == 1
        assert ref.display() == f"wow_forecaster/ml/trainer.py:{ref.line}"

    def test_unicode_anchor_reads_without_a_decode_error(self):
        """The quoted docstrings contain real Unicode; reads must pin UTF-8."""
        ref = resolve_source(
            "wow_forecaster/features/lag_rolling.py",
            "Lag and rolling features only access obs_date ≤ current date.",
        )
        assert ref.line is not None

    def test_missing_path_degrades_rather_than_raising(self):
        ref = resolve_source("wow_forecaster/does_not_exist.py", "anything")
        assert not ref.exists
        assert "missing" in ref.display()

    def test_absent_anchor_degrades_to_path_only(self):
        """A stale anchor must not crash a drill; the test suite is the enforcement point."""
        ref = resolve_source("wow_forecaster/ml/trainer.py", "this text is not in the file")
        assert ref.exists and ref.line is None and ref.occurrences == 0
        assert ref.display() == "wow_forecaster/ml/trainer.py"

    def test_source_without_anchor_reports_the_path(self):
        ref = resolve_source("wow_forecaster/ml/trainer.py", None)
        assert ref.display() == "wow_forecaster/ml/trainer.py"

    def test_multiple_matches_are_flagged(self, tmp_path, monkeypatch):
        build_tree(tmp_path)
        (tmp_path / "dup.py").write_text("x = 1\ny = 2\nx = 1\n", encoding="utf-8")
        monkeypatch.setenv(REPO_ROOT_ENV, str(tmp_path))
        ref = resolve_source("dup.py", "x = 1")
        assert ref.occurrences == 2
        assert ref.line == 1
        assert "+1 more" in ref.display()
