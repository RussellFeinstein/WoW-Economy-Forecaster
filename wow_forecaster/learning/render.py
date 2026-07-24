"""
Terminal rendering for the learning track.

ASCII only. Unicode arrows and box-drawing characters in ``typer.echo`` output
raise ``UnicodeEncodeError`` on the Windows console under the default codepage,
which is a documented gotcha in this project's CLAUDE.md and has bitten it
before. Docstrings may keep their arrows; anything printed uses ``->``.

Every function here returns a string rather than printing. That keeps the
formatting testable without capturing stdout, and keeps the decision about
where output goes in ``cli.py``.
"""

from __future__ import annotations

import textwrap
from collections.abc import Sequence
from dataclasses import dataclass

from wow_forecaster.learning.loader import SourceRef
from wow_forecaster.learning.models import Grade, LabState, LabStatus, Module, Question
from wow_forecaster.learning.session import ExamAnswer, ExamReport

WIDTH = 78

GRADE_KEYS: dict[str, Grade] = {
    "1": Grade.AGAIN,
    "2": Grade.HARD,
    "3": Grade.GOOD,
    "4": Grade.EASY,
}

GRADE_PROMPT = "Rate recall: [1] again  [2] hard  [3] good  [4] easy  [s] skip  [q] quit"

_LAB_LABEL: dict[LabStatus, str] = {
    LabStatus.NOT_STARTED: "not started",
    LabStatus.IN_PROGRESS: "in progress",
    LabStatus.DONE: "done",
}


def rule(char: str = "-", width: int = WIDTH) -> str:
    """A horizontal separator line."""
    return char * width


def wrap(text: str, indent: str = "", width: int = WIDTH) -> str:
    """Wrap prose to ``width``, preserving blank-line paragraph breaks."""
    out: list[str] = []
    for para in text.strip().split("\n\n"):
        collapsed = " ".join(para.split())
        if not collapsed:
            continue
        out.append(
            textwrap.fill(
                collapsed,
                width=width,
                initial_indent=indent,
                subsequent_indent=indent,
            )
        )
    return "\n\n".join(out)


def progress_bar(fraction: float, width: int = 20) -> str:
    """An ASCII progress bar, e.g. ``[########------------]``.

    Clamps rather than raising: a fraction slightly over 1.0 from rounding
    should render full, not crash a status report.
    """
    clamped = max(0.0, min(1.0, fraction))
    filled = round(clamped * width)
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def format_question(
    question: Question,
    position: int,
    total: int,
    module_title: str,
    due_label: str = "",
) -> str:
    """Render one question as it appears before the answer is revealed."""
    header = f"{question.module_id.upper()}  {module_title}"
    counter = f"[{position}/{total}]{('  ' + due_label) if due_label else ''}"
    pad = max(1, WIDTH - len(header) - len(counter))
    lines = [
        rule("="),
        f"{header}{' ' * pad}{counter}",
        rule("="),
        "",
        wrap(question.prompt, indent="  "),
        "",
    ]
    if question.options:
        for i, opt in enumerate(question.options):
            label = chr(ord("a") + i)
            lines.append(wrap(f"({label}) {opt.text}", indent="  "))
        lines.append("")
    return "\n".join(lines)


def format_answer(question: Question, ref: SourceRef | None) -> str:
    """Render the answer, rubric, citation, and further reading."""
    lines = [rule("-"), "ANSWER", "", wrap(question.answer, indent="  ")]

    if question.rubric:
        lines += ["", "  A complete answer covers:"]
        lines += [wrap(f"- {point}", indent="    ") for point in question.rubric]

    if question.options:
        correct = question.correct_index
        lines.append("")
        for i, opt in enumerate(question.options):
            label = chr(ord("a") + i)
            mark = "correct" if i == correct else "wrong"
            lines.append(wrap(f"({label}) {mark}. {opt.note or opt.text}", indent="    "))

    if ref is not None:
        lines += ["", f"  src  {ref.display()}"]
    if question.see_also:
        lines.append(f"  see  {' | '.join(question.see_also)}")
    if question.lab:
        lines.append(f"  lab  {question.lab}")
    lines.append("")
    return "\n".join(lines)


@dataclass(frozen=True)
class StatusRow:
    """One module's line in the status report."""

    module: Module
    n_questions: int
    n_seen: int
    n_due: int
    mastery: float
    authored: bool


def format_status(rows: Sequence[StatusRow], db_path: str) -> str:
    """Render the whole-track progress report."""
    lines = [
        rule("="),
        "WoW Economy Forecaster -- learning track",
        rule("="),
        "",
        f"{'':4} {'module':34} {'mastery':22} {'seen':>7} {'due':>5}",
        rule("-"),
    ]
    total_q = total_seen = total_due = 0
    for row in rows:
        title = row.module.title if len(row.module.title) <= 34 else row.module.title[:31] + "..."
        if row.authored:
            bar = f"{progress_bar(row.mastery)} {row.mastery * 100:3.0f}%"
            seen = f"{row.n_seen}/{row.n_questions}"
            due = str(row.n_due)
            total_q += row.n_questions
            total_seen += row.n_seen
            total_due += row.n_due
        else:
            bar = "not authored yet"
            seen = "-"
            due = "-"
        lines.append(f"{row.module.id:4} {title:34} {bar:22} {seen:>7} {due:>5}")

    seen_total = f"{total_seen}/{total_q}"
    lines += [
        rule("-"),
        f"{'':4} {'total (authored modules)':34} {'':22} {seen_total:>7} {total_due:>5}",
        "",
        f"progress db: {db_path}",
        "",
        "Next: wowfc learn next          drill what is due",
        "      wowfc learn module <id>   objectives, reading list, lab",
        "",
    ]
    return "\n".join(lines)


def format_module(
    module: Module,
    n_questions: int,
    lesson: str | None,
    lab: LabState | None,
) -> str:
    """Render a module briefing: objectives, reading list, lab, question count."""
    lines = [
        rule("="),
        f"{module.id.upper()}  {module.title}",
        f"{module.part}",
        rule("="),
        "",
    ]
    if module.summary:
        lines += [wrap(module.summary, indent="  "), ""]
    if module.objectives:
        lines.append("  After this module you can:")
        lines += [wrap(f"- {o}", indent="    ") for o in module.objectives]
        lines.append("")
    if module.reading:
        lines.append("  Read (the repo is the textbook):")
        lines += [f"    - {r}" for r in module.reading]
        lines.append("")
    if module.prereqs:
        lines += [f"  Prereqs: {', '.join(module.prereqs)}", ""]

    lines.append(f"  Questions: {n_questions if n_questions else 'not authored yet'}")
    if lesson:
        lines.append(f"  Lesson:    {lesson}")
    if module.lab:
        status = _LAB_LABEL[lab.status] if lab else _LAB_LABEL[LabStatus.NOT_STARTED]
        lines.append(f"  Lab:       {module.lab} ({status})")
    lines.append("")
    return "\n".join(lines)


def format_exam_report(
    report: ExamReport,
    refs: dict[str, SourceRef | None],
    module_titles: dict[str, str],
) -> str:
    """Render the post-exam scorecard and the list of misses."""
    lines = [
        "",
        rule("="),
        f"EXAM RESULT   {report.total_score:.1f} / {report.n_questions}"
        f"   ({report.percent:.0f}%)"
        f"   fully correct: {report.n_passed}/{report.n_questions}",
        rule("="),
        "",
    ]
    by_module = report.by_module()
    if by_module:
        lines.append("  By module:")
        for module_id, (passed, asked) in by_module.items():
            title = module_titles.get(module_id, module_id)
            bar = progress_bar(passed / asked if asked else 0.0, width=12)
            lines.append(f"    {module_id}  {bar} {passed}/{asked}  {title}")
        lines.append("")

    misses = report.misses()
    if not misses:
        lines += ["  No misses.", ""]
        return "\n".join(lines)

    lines.append("  Review these:")
    for answer in misses:
        q = answer.question
        ref = refs.get(q.id)
        detail = _miss_detail(answer)
        lines.append("")
        lines.append(wrap(f"{q.id}  {detail}", indent="    "))
        lines.append(wrap(q.prompt, indent="      "))
        if ref is not None:
            lines.append(f"      src  {ref.display()}")
    lines.append("")
    return "\n".join(lines)


def _miss_detail(answer: ExamAnswer) -> str:
    """One-line reason a miss is a miss."""
    if answer.skipped:
        return "skipped"
    if answer.question.is_selection:
        return "wrong option"
    return f"rubric {answer.rubric_hits}/{answer.rubric_total}"
