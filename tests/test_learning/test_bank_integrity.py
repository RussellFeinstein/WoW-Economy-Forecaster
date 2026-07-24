"""
The drift guard.

This is the test that makes a hand-authored question bank worth writing. Every
question cites a file path plus a verbatim single-line anchor, and this asserts
those citations still hold against the code as it stands right now. Edit a cited
line in ``trainer.py`` and this goes red in the same PR that made the edit.

``check_content()`` is shared with ``wowfc learn validate``, so the rule enforced
here and the rule reported while authoring cannot drift apart.
"""

from __future__ import annotations

import pytest

from wow_forecaster.learning.integrity import Problem, check_content
from wow_forecaster.learning.loader import (
    authored_module_ids,
    lab_path,
    load_all_banks,
    load_curriculum,
)
from wow_forecaster.learning.models import SELECTION_KINDS


class TestCommittedContent:
    """The real content tree must be coherent with the real code."""

    def test_every_citation_resolves(self, repo_root):
        problems = check_content(repo_root)
        assert problems == [], "\n".join(str(p) for p in problems)

    def test_question_ids_are_unique_across_all_banks(self):
        ids = [q.id for questions in load_all_banks().values() for q in questions]
        assert len(ids) == len(set(ids))

    def test_every_bank_is_declared_in_the_curriculum(self):
        declared = {m.id for m in load_curriculum().modules}
        assert set(authored_module_ids()) <= declared

    def test_every_declared_lab_has_a_brief(self):
        for module in load_curriculum().modules:
            if module.lab:
                assert lab_path(module.lab) is not None, f"{module.id} lab {module.lab}"

    def test_anchors_are_single_line(self):
        for questions in load_all_banks().values():
            for question in questions:
                if question.anchor:
                    assert "\n" not in question.anchor
                    assert "\r" not in question.anchor

    def test_selection_questions_have_exactly_one_correct_option(self):
        for questions in load_all_banks().values():
            for question in questions:
                if question.options:
                    assert sum(1 for o in question.options if o.correct) == 1, question.id
                    assert len(question.options) >= 3, question.id

    def test_wrong_options_explain_why_they_are_wrong(self):
        for questions in load_all_banks().values():
            for question in questions:
                for option in question.options:
                    if not option.correct:
                        assert option.note, f"{question.id}: {option.text}"

    def test_recall_questions_carry_a_rubric(self):
        for questions in load_all_banks().values():
            for question in questions:
                if question.kind == "recall":
                    assert question.rubric, question.id

    def test_authored_content_avoids_em_dashes(self):
        """Repo prose style: authored content ships under the user's name.

        Quoted code and docstrings may contain them, which is why this checks the
        authored prose fields and not the anchors.
        """
        for questions in load_all_banks().values():
            for question in questions:
                authored = [question.prompt, question.answer, *question.rubric]
                authored += [o.text for o in question.options]
                authored += [o.note or "" for o in question.options]
                for text in authored:
                    assert "—" not in text, question.id


class TestGuardDetectsDrift:
    """The guard has to actually fail when a citation goes stale."""

    def test_broken_anchor_is_reported_with_the_question_id(self, tmp_path):
        root = _synthetic_tree(tmp_path, anchor="this text is not in the file")
        problems = check_content(root)
        assert [p.code for p in problems] == ["anchor-missing"]
        assert problems[0].subject == "m01-q01"

    def test_missing_source_path_is_reported(self, tmp_path):
        root = _synthetic_tree(tmp_path, source="does/not/exist.py")
        assert [p.code for p in problems_of(root)] == ["source-missing"]

    def test_intact_anchor_passes(self, tmp_path):
        root = _synthetic_tree(tmp_path)
        assert check_content(root) == []

    def test_undeclared_bank_is_reported(self, tmp_path):
        root = _synthetic_tree(tmp_path)
        (root / "learning" / "banks" / "m09.toml").write_text(
            _bank_toml("m09-q01", "target.py", "needle"), encoding="utf-8"
        )
        assert "bank-undeclared" in {p.code for p in check_content(root)}

    def test_missing_see_also_path_is_reported(self, tmp_path):
        root = _synthetic_tree(tmp_path, see_also='see_also = ["NOPE.md"]')
        assert "see-also-missing" in {p.code for p in check_content(root)}

    def test_missing_see_also_heading_fragment_is_reported(self, tmp_path):
        root = _synthetic_tree(tmp_path, see_also='see_also = ["NOTES.md#Absent heading"]')
        (root / "NOTES.md").write_text("# Present heading\n\ntext\n", encoding="utf-8")
        assert "see-also-anchor-missing" in {p.code for p in check_content(root)}

    def test_present_see_also_heading_fragment_passes(self, tmp_path):
        root = _synthetic_tree(tmp_path, see_also='see_also = ["NOTES.md#Present heading"]')
        (root / "NOTES.md").write_text("# Present heading\n\ntext\n", encoding="utf-8")
        assert check_content(root) == []

    def test_unresolvable_lab_reference_is_reported(self, tmp_path):
        root = _synthetic_tree(tmp_path, extra='lab = "lab-99-nope"')
        assert "lab-missing" in {p.code for p in check_content(root)}

    def test_problem_renders_readably(self):
        rendered = str(Problem("anchor-missing", "m06-q03", "gone"))
        assert rendered == "[anchor-missing] m06-q03: gone"


class TestShallowCloneGuard:
    """CI clones at depth 1, where no historical SHA resolves."""

    def test_commit_check_is_skipped_on_a_shallow_clone(self, tmp_path, monkeypatch):
        root = _synthetic_tree(
            tmp_path, extra='commit = "0000000000000000000000000000000000000000"'
        )
        monkeypatch.setattr(
            "wow_forecaster.learning.integrity._is_shallow_clone", lambda _root: True
        )
        assert check_content(root) == []

    def test_bogus_commit_is_reported_on_a_full_clone(self, tmp_path, monkeypatch):
        root = _synthetic_tree(
            tmp_path, extra='commit = "0000000000000000000000000000000000000000"'
        )
        monkeypatch.setattr(
            "wow_forecaster.learning.integrity._is_shallow_clone", lambda _root: False
        )
        monkeypatch.setattr(
            "wow_forecaster.learning.integrity._commit_exists", lambda _root, _sha: False
        )
        assert [p.code for p in check_content(root)] == ["commit-missing"]


def test_selection_kinds_are_what_the_session_scores_inline():
    """A guard on the constant the exam flow branches on."""
    assert SELECTION_KINDS == frozenset({"mcq", "predict"})


# ── helpers ───────────────────────────────────────────────────────────────────

def problems_of(root):
    return check_content(root)


def _bank_toml(qid, source, anchor, see_also="", extra=""):
    return f"""
[[question]]
id = "{qid}"
kind = "recall"
prompt = "prompt"
answer = "answer"
rubric = ["a point"]
source = "{source}"
anchor = "{anchor}"
{see_also}
{extra}
"""


def _synthetic_tree(root, source="target.py", anchor="needle", see_also="", extra=""):
    """Build a self-contained content tree whose one citation is under test."""
    (root / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
    (root / "target.py").write_text("alpha\nneedle in here\nomega\n", encoding="utf-8")
    content = root / "learning"
    (content / "banks").mkdir(parents=True)
    (content / "modules").mkdir()
    (content / "labs").mkdir()
    (content / "curriculum.toml").write_text(
        '[[module]]\nid = "m01"\ntitle = "T"\npart = "P"\n', encoding="utf-8"
    )
    (content / "banks" / "m01.toml").write_text(
        _bank_toml("m01-q01", source, anchor, see_also, extra), encoding="utf-8"
    )
    return root


@pytest.fixture(autouse=True)
def _no_repo_root_override(monkeypatch):
    """Synthetic-tree tests pass root explicitly; make sure no env var overrides it."""
    monkeypatch.delenv("WOWFC_REPO_ROOT", raising=False)
