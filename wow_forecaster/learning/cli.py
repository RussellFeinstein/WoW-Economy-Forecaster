"""
``wowfc learn`` command group.

Registered as a Typer sub-app rather than six more flat commands on the main
app: ``cli.py`` is already 4,500 lines and PLAN.md flags its size, so a new
surface goes in its own module and its own namespace.

Import discipline: this module imports typer and stdlib only at import time,
because ``cli.py`` imports it eagerly to register the group, and every ``wowfc``
invocation would otherwise pay for pydantic model construction and content
parsing it does not need. The loader, store, and renderer are imported inside
command bodies, matching how ``_load_config_or_exit`` already defers
``load_config``.

Interactive commands accept ``--list`` so the queue can be inspected, and
scripted, without a terminal. That also makes them testable without feeding
stdin.
"""

from __future__ import annotations

from datetime import date

import typer

learn_app = typer.Typer(
    name="learn",
    help="Study and assessment track for this repo (syllabus, drills, exams, labs).",
    no_args_is_help=True,
)

_QUIT_KEYS = frozenset({"q", "quit"})
_SKIP_KEYS = frozenset({"s", "skip"})


def _abort_to_exit(message: str = "Stopped.") -> None:
    """Report a clean stop for a non-interactive or interrupted session."""
    typer.echo(f"\n{message}")


def _prompt(text: str) -> str | None:
    """Prompt for a line, returning None when input is unavailable or aborted.

    Piped or redirected stdin raises rather than blocking, and an interactive
    Ctrl+C should not print a traceback. Both become a clean stop, so a drill
    can be safely piped or interrupted. ``typer.Abort`` is click's Abort, which
    is what a prompt raises on EOF.
    """
    try:
        return typer.prompt(text, default="", show_default=False).strip().lower()
    except (EOFError, KeyboardInterrupt, typer.Abort):
        return None


@learn_app.command("status")
def status_command(
    db_path: str = typer.Option(None, "--db-path", help="Progress database path override."),
) -> None:
    """Show mastery, cards due, and lab state for every module."""
    from wow_forecaster.learning.loader import authored_module_ids, load_bank, load_curriculum
    from wow_forecaster.learning.render import StatusRow, format_status
    from wow_forecaster.learning.scheduler import mastery_fraction
    from wow_forecaster.learning.store import ProgressStore

    curriculum = load_curriculum()
    authored = set(authored_module_ids())
    store = ProgressStore(db_path)
    store.apply_schema()
    states = store.get_states()
    today = date.today()

    rows: list[StatusRow] = []
    for module in curriculum.modules:
        if module.id not in authored:
            rows.append(
                StatusRow(
                    module=module, n_questions=0, n_seen=0, n_due=0, mastery=0.0, authored=False
                )
            )
            continue
        questions = load_bank(module.id)
        ids = [q.id for q in questions]
        module_states = [states[i] for i in ids if i in states]
        n_due = sum(1 for s in module_states if s.is_due(today)) + sum(
            1 for i in ids if i not in states
        )
        rows.append(
            StatusRow(
                module=module,
                n_questions=len(questions),
                n_seen=len(module_states),
                n_due=n_due,
                mastery=mastery_fraction(module_states, len(questions)),
                authored=True,
            )
        )

    typer.echo(format_status(rows, store.db_path))


@learn_app.command("next")
def next_command(
    module: str = typer.Option(None, "--module", "-m", help="Restrict to one module id."),
    count: int = typer.Option(10, "--count", "-n", help="Maximum questions to serve."),
    reviews_only: bool = typer.Option(
        False, "--reviews-only", help="Serve only cards already in rotation, no new material."
    ),
    list_only: bool = typer.Option(
        False, "--list", help="Print the queue and exit without prompting or recording."
    ),
    db_path: str = typer.Option(None, "--db-path", help="Progress database path override."),
) -> None:
    """Drill what is due, then new material."""
    from wow_forecaster.learning.loader import (
        authored_module_ids,
        load_bank,
        load_curriculum,
        resolve_source,
    )
    from wow_forecaster.learning.render import (
        GRADE_KEYS,
        GRADE_PROMPT,
        format_answer,
        format_question,
    )
    from wow_forecaster.learning.scheduler import review
    from wow_forecaster.learning.session import select_drill
    from wow_forecaster.learning.store import ProgressStore

    curriculum = load_curriculum()
    authored = authored_module_ids()
    if module:
        module = module.lower()
        try:
            curriculum.by_id(module)
        except KeyError as exc:
            typer.echo(f"[ERROR] {exc}", err=True)
            raise typer.Exit(code=1) from None
        if module not in authored:
            typer.echo(
                f"[ERROR] module {module} has no question bank yet. "
                f"Authored so far: {', '.join(authored) or 'none'}",
                err=True,
            )
            raise typer.Exit(code=1) from None
        target_ids = [module]
    else:
        target_ids = list(authored)

    if not target_ids:
        typer.echo("[WARN] no question banks authored yet.")
        raise typer.Exit(code=0)

    questions = [q for mid in target_ids for q in load_bank(mid)]
    titles = {m.id: m.title for m in curriculum.modules}

    store = ProgressStore(db_path)
    store.apply_schema()
    today = date.today()
    items = select_drill(
        questions, store.get_states([q.id for q in questions]), count, today, not reviews_only
    )

    if not items:
        typer.echo("Nothing due. Run `wowfc learn status` to see when cards come back.")
        raise typer.Exit(code=0)

    if list_only:
        typer.echo(f"{len(items)} question(s) queued:")
        for item in items:
            tag = "new" if item.is_new else f"due {item.state.due_date}"
            typer.echo(f"  {item.question.id}  [{tag}]  {item.question.kind}")
        raise typer.Exit(code=0)

    graded = 0
    for position, item in enumerate(items, start=1):
        q = item.question
        tag = "new" if item.is_new else f"due {item.state.due_date}"
        typer.echo(format_question(q, position, len(items), titles.get(q.module_id, ""), tag))

        answer_hint = (
            "Your answer, then [enter] to reveal  ([s] skip, [q] quit)"
            if q.options
            else "[enter] to reveal  ([s] skip, [q] quit)"
        )
        response = _prompt(answer_hint)
        if response is None:
            _abort_to_exit()
            break
        if response in _QUIT_KEYS:
            break
        if response in _SKIP_KEYS:
            continue

        ref = resolve_source(q.source, q.anchor)
        typer.echo(format_answer(q, ref))

        key = _prompt(GRADE_PROMPT)
        if key is None:
            _abort_to_exit()
            break
        if key in _QUIT_KEYS:
            break
        if key in _SKIP_KEYS or key not in GRADE_KEYS:
            if key not in _SKIP_KEYS:
                typer.echo("  (unrecognized grade, skipped)")
            continue

        updated = review(item.state, GRADE_KEYS[key], today)
        store.save_state(updated)
        store.log_review(updated, mode="drill")
        graded += 1
        typer.echo(f"  -> next review in {updated.interval_days}d ({updated.due_date})\n")

    typer.echo(f"Graded {graded} of {len(items)} question(s).")


@learn_app.command("module")
def module_command(
    module_id: str = typer.Argument(..., help="Module id, e.g. m06."),
) -> None:
    """Show a module's objectives, reading list, and lab."""
    from wow_forecaster.learning.loader import (
        authored_module_ids,
        lesson_path,
        load_bank,
        load_curriculum,
    )
    from wow_forecaster.learning.render import format_module
    from wow_forecaster.learning.store import ProgressStore

    curriculum = load_curriculum()
    try:
        module = curriculum.by_id(module_id.lower())
    except KeyError as exc:
        typer.echo(f"[ERROR] {exc}", err=True)
        raise typer.Exit(code=1) from None

    n_questions = len(load_bank(module.id)) if module.id in authored_module_ids() else 0
    lesson = lesson_path(module.id)
    lab_state = None
    if module.lab:
        store = ProgressStore()
        store.apply_schema()
        lab_state = store.get_lab(module.lab)

    typer.echo(format_module(module, n_questions, str(lesson) if lesson else None, lab_state))


@learn_app.command("exam")
def exam_command(
    module: str = typer.Option(None, "--module", "-m", help="Restrict to one module id."),
    count: int = typer.Option(20, "--count", "-n", help="Number of questions."),
    seed: int = typer.Option(None, "--seed", help="Seed for a repeatable question sample."),
    record: bool = typer.Option(
        False, "--record", help="Write results into review state. Off by default."
    ),
    list_only: bool = typer.Option(
        False, "--list", help="Print the sampled questions and exit."
    ),
    db_path: str = typer.Option(None, "--db-path", help="Progress database path override."),
) -> None:
    """Sit a scored exam. Nothing is revealed until the last question."""
    import random

    from wow_forecaster.learning.loader import (
        authored_module_ids,
        load_bank,
        load_curriculum,
        resolve_source,
    )
    from wow_forecaster.learning.models import Grade
    from wow_forecaster.learning.render import format_exam_report, format_question, rule, wrap
    from wow_forecaster.learning.scheduler import review
    from wow_forecaster.learning.session import ExamAnswer, ExamReport, select_exam
    from wow_forecaster.learning.store import ProgressStore

    curriculum = load_curriculum()
    authored = authored_module_ids()
    target_ids = [module.lower()] if module else list(authored)
    for mid in target_ids:
        if mid not in authored:
            typer.echo(
                f"[ERROR] module {mid} has no question bank yet. "
                f"Authored so far: {', '.join(authored) or 'none'}",
                err=True,
            )
            raise typer.Exit(code=1) from None
    if not target_ids:
        typer.echo("[WARN] no question banks authored yet.")
        raise typer.Exit(code=0)

    banks = {mid: load_bank(mid) for mid in target_ids}
    rng = random.Random(seed) if seed is not None else None
    sampled = select_exam(banks, count, rng)
    titles = {m.id: m.title for m in curriculum.modules}

    if list_only:
        typer.echo(f"{len(sampled)} question(s) sampled:")
        for q in sampled:
            typer.echo(f"  {q.id}  {q.kind}")
        raise typer.Exit(code=0)

    # Pass one: ask everything, reveal nothing.
    answers: list[ExamAnswer] = []
    for position, q in enumerate(sampled, start=1):
        typer.echo(format_question(q, position, len(sampled), titles.get(q.module_id, "")))
        if q.is_selection:
            reply = _prompt("Your answer (letter), or [s] to skip")
            if reply is None:
                _abort_to_exit("Exam abandoned; nothing recorded.")
                raise typer.Exit(code=0)
            if reply in _QUIT_KEYS:
                _abort_to_exit("Exam abandoned; nothing recorded.")
                raise typer.Exit(code=0)
            selected = None
            if len(reply) == 1 and reply.isalpha():
                index = ord(reply) - ord("a")
                if 0 <= index < len(q.options):
                    selected = index
            answers.append(
                ExamAnswer(question=q, selected_index=selected, skipped=reply in _SKIP_KEYS)
            )
        else:
            reply = _prompt("Answer out loud or on paper, then [enter]  ([s] skip)")
            if reply is None:
                _abort_to_exit("Exam abandoned; nothing recorded.")
                raise typer.Exit(code=0)
            if reply in _QUIT_KEYS:
                _abort_to_exit("Exam abandoned; nothing recorded.")
                raise typer.Exit(code=0)
            answers.append(ExamAnswer(question=q, skipped=reply in _SKIP_KEYS))

    # Pass two: reveal free-response rubrics and self-score. Selection kinds are
    # already scored, so only free response needs this pass.
    scored: list[ExamAnswer] = []
    free_response = [a for a in answers if not a.question.is_selection and not a.skipped]
    if free_response:
        typer.echo(rule("="))
        typer.echo("SELF-SCORE   Count the rubric points your answer actually covered.")
        typer.echo(rule("="))
    for answer in answers:
        if answer.question.is_selection or answer.skipped:
            scored.append(answer)
            continue
        q = answer.question
        typer.echo("")
        typer.echo(wrap(f"{q.id}  {q.prompt}", indent="  "))
        typer.echo("")
        for point in q.rubric:
            typer.echo(wrap(f"- {point}", indent="    "))
        typer.echo("")
        n_points = len(q.rubric)
        reply = _prompt(f"How many of the {n_points} points did you cover? [0-{n_points}]")
        hits = int(reply) if reply and reply.isdigit() else 0
        scored.append(
            ExamAnswer(question=q, rubric_hits=min(hits, len(q.rubric)), skipped=False)
        )

    report = ExamReport(answers=tuple(scored))
    refs = {a.question.id: resolve_source(a.question.source, a.question.anchor) for a in scored}
    typer.echo(format_exam_report(report, refs, titles))

    if record:
        store = ProgressStore(db_path)
        store.apply_schema()
        today = date.today()
        states = store.get_states([a.question.id for a in scored])
        from wow_forecaster.learning.scheduler import new_state as fresh_state

        for answer in scored:
            grade = Grade.GOOD if answer.passed else Grade.AGAIN
            current = states.get(answer.question.id) or fresh_state(answer.question.id)
            updated = review(current, grade, today)
            store.save_state(updated)
            store.log_review(updated, mode="exam")
        typer.echo(f"Recorded {len(scored)} result(s) into review state.")


@learn_app.command("lab")
def lab_command(
    lab_id: str = typer.Argument(..., help="Lab id, e.g. lab-01-purge-embargo."),
    start: bool = typer.Option(False, "--start", help="Mark the lab in progress."),
    done: bool = typer.Option(False, "--done", help="Mark the lab done."),
    branch: str = typer.Option(None, "--branch", help="Record the working branch name."),
    db_path: str = typer.Option(None, "--db-path", help="Progress database path override."),
) -> None:
    """Print a lab brief and record progress on it."""
    from wow_forecaster.learning.loader import lab_path
    from wow_forecaster.learning.models import LabState, LabStatus
    from wow_forecaster.learning.store import ProgressStore

    if start and done:
        typer.echo("[ERROR] pass --start or --done, not both.", err=True)
        raise typer.Exit(code=1)

    path = lab_path(lab_id)
    if path is None:
        typer.echo(f"[ERROR] no brief found for lab {lab_id!r}.", err=True)
        raise typer.Exit(code=1)

    store = ProgressStore(db_path)
    store.apply_schema()
    state = store.get_lab(lab_id)

    if start or done:
        today = date.today()
        state = LabState(
            lab_id=lab_id,
            status=LabStatus.DONE if done else LabStatus.IN_PROGRESS,
            branch=branch or state.branch,
            started_at=state.started_at or today,
            completed_at=today if done else state.completed_at,
            notes=state.notes,
        )
        store.save_lab(state)
        typer.echo(f"[OK] lab {lab_id} -> {state.status.name.lower()}")
        return

    typer.echo(path.read_text(encoding="utf-8"))
    typer.echo(f"--- status: {state.status.name.lower()}"
               f"{f', branch {state.branch}' if state.branch else ''} ---")


@learn_app.command("validate")
def validate_command(
    module: str = typer.Option(
        None, "--module", "-m", help="Check one module only. Useful while authoring."
    ),
) -> None:
    """Check every question's citations against the current code. Exits 1 on failure."""
    from wow_forecaster.learning.integrity import check_content
    from wow_forecaster.learning.loader import BankParseError, authored_module_ids, load_bank

    module_ids = [module.lower()] if module else None
    try:
        problems = check_content(module_ids=module_ids)
        banks = {
            mid: load_bank(mid) for mid in (module_ids or authored_module_ids())
        }
    except BankParseError as exc:
        typer.echo(f"[FAIL] {exc}", err=True)
        raise typer.Exit(code=1) from None
    n_questions = sum(len(qs) for qs in banks.values())

    if not problems:
        typer.echo(
            f"[OK] {n_questions} question(s) across {len(banks)} "
            "module(s); every citation resolves."
        )
        return

    typer.echo(f"[FAIL] {len(problems)} problem(s):", err=True)
    for problem in problems:
        typer.echo(f"  {problem}", err=True)
    raise typer.Exit(code=1)


@learn_app.command("reset")
def reset_command(
    module: str = typer.Option(None, "--module", "-m", help="Reset one module only."),
    yes: bool = typer.Option(False, "--yes", help="Skip the confirmation prompt."),
    db_path: str = typer.Option(None, "--db-path", help="Progress database path override."),
) -> None:
    """Clear review progress. Scoped to one module with --module."""
    from wow_forecaster.learning.loader import authored_module_ids, load_bank
    from wow_forecaster.learning.store import ProgressStore

    store = ProgressStore(db_path)
    store.apply_schema()

    if module:
        module = module.lower()
        if module not in authored_module_ids():
            typer.echo(f"[ERROR] module {module} has no question bank.", err=True)
            raise typer.Exit(code=1)
        ids = [q.id for q in load_bank(module)]
        scope = f"module {module} ({len(ids)} question(s))"
    else:
        ids = None
        scope = "ALL modules, plus lab progress"

    if not yes and not typer.confirm(f"Clear review progress for {scope}?"):
        typer.echo("Cancelled.")
        raise typer.Exit(code=0)

    deleted = store.reset(ids)
    typer.echo(f"[OK] cleared {deleted} review state row(s) for {scope}.")
