"""
Progress persistence for the learning track.

Deliberately a separate SQLite file from the product database. The product DB
carries 24 tables and 9 migrations, is copied wholesale into every durable
backup, and is the upstream source for the M3 warehouse. Personal review state
belongs in none of those, and putting it there would mean every nightly
``backup-durable-db`` shipped a record of which flashcards were hard.

Default location ``data/learn/progress.db``, which the ``data/learn/`` gitignore
rule keeps out of the repo. ``WOWFC_LEARN_DB`` overrides it, matching the
existing ``WOWFC_SCHTASKS`` / ``WOWFC_POWERCFG`` / ``WOWFC`` seam convention.
Tests always set it, so a test run never writes into the working tree.

Connections come from ``db.connection.get_connection``, which already handles
parent-directory creation, foreign keys, row factory, and commit-or-rollback.
WAL is off here: nothing else touches this file, so the extra sidecar files
would be noise.
"""

from __future__ import annotations

import logging
import os
import sqlite3
from collections.abc import Iterable, Sequence
from datetime import date
from pathlib import Path

from wow_forecaster.db.connection import get_connection
from wow_forecaster.learning.models import Grade, LabState, LabStatus, ReviewState

logger = logging.getLogger(__name__)

#: Environment override for the progress database path. Test seam.
LEARN_DB_ENV = "WOWFC_LEARN_DB"

DEFAULT_DB_RELPATH = Path("data") / "learn" / "progress.db"

SCHEMA_VERSION = 1

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS learn_schema_version (
    version    INTEGER NOT NULL PRIMARY KEY,
    applied_at TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);

CREATE TABLE IF NOT EXISTS review_state (
    question_id         TEXT    NOT NULL PRIMARY KEY,
    ease                REAL    NOT NULL,
    interval_days       INTEGER NOT NULL,
    due_date            TEXT,
    reps                INTEGER NOT NULL,
    lapses              INTEGER NOT NULL,
    last_grade          INTEGER,
    last_reviewed_at    TEXT,
    prev_ease           REAL    NOT NULL,
    prev_interval_days  INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_review_state_due ON review_state(due_date);

CREATE TABLE IF NOT EXISTS review_log (
    log_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    question_id   TEXT    NOT NULL,
    reviewed_at   TEXT    NOT NULL,
    grade         INTEGER NOT NULL,
    mode          TEXT    NOT NULL,
    interval_days INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_review_log_question
    ON review_log(question_id, reviewed_at);

CREATE TABLE IF NOT EXISTS lab_progress (
    lab_id       TEXT    NOT NULL PRIMARY KEY,
    status       INTEGER NOT NULL,
    branch       TEXT,
    started_at   TEXT,
    completed_at TEXT,
    notes        TEXT
);
"""


def default_db_path(root: Path | None = None) -> Path:
    """Resolve the progress database path.

    ``WOWFC_LEARN_DB`` wins when set, and is honoured without needing a content
    root, so a test can point at ``tmp_path`` regardless of install shape.

    Args:
        root: Repo root override, used only when the env var is unset.

    Returns:
        Absolute path to the progress database. Parent directories are created
        lazily by ``get_connection``, not here.
    """
    override = os.environ.get(LEARN_DB_ENV)
    if override:
        return Path(override).expanduser()

    from wow_forecaster.learning.loader import content_root

    base = root or content_root()
    return base / DEFAULT_DB_RELPATH


def _to_iso(value: date | None) -> str | None:
    return value.isoformat() if value is not None else None


def _from_iso(value: str | None) -> date | None:
    return date.fromisoformat(value) if value else None


def _row_to_review_state(row: sqlite3.Row) -> ReviewState:
    return ReviewState(
        question_id=row["question_id"],
        ease=row["ease"],
        interval_days=row["interval_days"],
        due_date=_from_iso(row["due_date"]),
        reps=row["reps"],
        lapses=row["lapses"],
        last_grade=Grade(row["last_grade"]) if row["last_grade"] is not None else None,
        last_reviewed_at=_from_iso(row["last_reviewed_at"]),
        prev_ease=row["prev_ease"],
        prev_interval_days=row["prev_interval_days"],
    )


def _row_to_lab_state(row: sqlite3.Row) -> LabState:
    return LabState(
        lab_id=row["lab_id"],
        status=LabStatus(row["status"]),
        branch=row["branch"],
        started_at=_from_iso(row["started_at"]),
        completed_at=_from_iso(row["completed_at"]),
        notes=row["notes"],
    )


class ProgressStore:
    """Read and write learning progress.

    Every method opens and closes its own short connection. The volumes here are
    trivial (a few hundred rows), and a held connection would be one more thing
    to leak, which is the failure this project already paid for once.
    """

    def __init__(self, db_path: str | Path | None = None, root: Path | None = None) -> None:
        self.db_path = str(db_path) if db_path is not None else str(default_db_path(root))

    def _connect(self):
        return get_connection(self.db_path, wal_mode=False)

    def apply_schema(self) -> None:
        """Create tables and indexes if absent. Idempotent."""
        with self._connect() as conn:
            conn.executescript(_SCHEMA_SQL)
            conn.execute(
                "INSERT OR IGNORE INTO learn_schema_version (version) VALUES (?);",
                (SCHEMA_VERSION,),
            )

    # ── review state ──────────────────────────────────────────────────────────

    def get_state(self, question_id: str) -> ReviewState | None:
        """Return stored state for one question, or None if never reviewed."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM review_state WHERE question_id = ?;", (question_id,)
            ).fetchone()
        return _row_to_review_state(row) if row else None

    def get_states(self, question_ids: Sequence[str] | None = None) -> dict[str, ReviewState]:
        """Return stored states keyed by question id.

        Args:
            question_ids: Restrict to these ids. None returns every stored row.
                An empty sequence returns an empty dict without querying, which
                matters because ``IN ()`` is a SQLite syntax error.
        """
        if question_ids is not None and len(question_ids) == 0:
            return {}
        with self._connect() as conn:
            if question_ids is None:
                rows = conn.execute("SELECT * FROM review_state;").fetchall()
            else:
                placeholders = ",".join("?" for _ in question_ids)
                rows = conn.execute(
                    f"SELECT * FROM review_state WHERE question_id IN ({placeholders});",
                    tuple(question_ids),
                ).fetchall()
        return {r["question_id"]: _row_to_review_state(r) for r in rows}

    def save_state(self, state: ReviewState) -> None:
        """Upsert one review state. Re-saving the same state is a no-op in effect."""
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO review_state (
                    question_id, ease, interval_days, due_date, reps, lapses,
                    last_grade, last_reviewed_at, prev_ease, prev_interval_days
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(question_id) DO UPDATE SET
                    ease               = excluded.ease,
                    interval_days      = excluded.interval_days,
                    due_date           = excluded.due_date,
                    reps               = excluded.reps,
                    lapses             = excluded.lapses,
                    last_grade         = excluded.last_grade,
                    last_reviewed_at   = excluded.last_reviewed_at,
                    prev_ease          = excluded.prev_ease,
                    prev_interval_days = excluded.prev_interval_days;
                """,
                (
                    state.question_id,
                    state.ease,
                    state.interval_days,
                    _to_iso(state.due_date),
                    state.reps,
                    state.lapses,
                    int(state.last_grade) if state.last_grade is not None else None,
                    _to_iso(state.last_reviewed_at),
                    state.prev_ease,
                    state.prev_interval_days,
                ),
            )

    def log_review(self, state: ReviewState, mode: str) -> None:
        """Append one row to the review history.

        The log is append-only and never read by the scheduler. It exists so
        that "which questions do I keep failing" is answerable later without
        having to reconstruct it from the current state, which only remembers
        the last grade.
        """
        if state.last_grade is None or state.last_reviewed_at is None:
            raise ValueError(f"cannot log an ungraded state for {state.question_id}")
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO review_log (question_id, reviewed_at, grade, mode, interval_days)
                VALUES (?, ?, ?, ?, ?);
                """,
                (
                    state.question_id,
                    state.last_reviewed_at.isoformat(),
                    int(state.last_grade),
                    mode,
                    state.interval_days,
                ),
            )

    def review_counts_by_grade(self) -> dict[Grade, int]:
        """Total logged reviews per grade, across all questions and modes."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT grade, COUNT(*) AS n FROM review_log GROUP BY grade;"
            ).fetchall()
        return {Grade(r["grade"]): r["n"] for r in rows}

    def reset(self, question_ids: Iterable[str] | None = None) -> int:
        """Delete review state and history.

        Args:
            question_ids: Restrict the delete to these ids. None clears
                everything, labs included.

        Returns:
            Number of ``review_state`` rows deleted.
        """
        with self._connect() as conn:
            if question_ids is None:
                deleted = conn.execute("DELETE FROM review_state;").rowcount
                conn.execute("DELETE FROM review_log;")
                conn.execute("DELETE FROM lab_progress;")
                return deleted
            ids = tuple(question_ids)
            if not ids:
                return 0
            placeholders = ",".join("?" for _ in ids)
            deleted = conn.execute(
                f"DELETE FROM review_state WHERE question_id IN ({placeholders});", ids
            ).rowcount
            conn.execute(
                f"DELETE FROM review_log WHERE question_id IN ({placeholders});", ids
            )
            return deleted

    # ── labs ──────────────────────────────────────────────────────────────────

    def get_lab(self, lab_id: str) -> LabState:
        """Return lab progress, defaulting to NOT_STARTED when unrecorded."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM lab_progress WHERE lab_id = ?;", (lab_id,)
            ).fetchone()
        return _row_to_lab_state(row) if row else LabState(lab_id=lab_id)

    def all_labs(self) -> dict[str, LabState]:
        """Return every recorded lab state, keyed by lab id."""
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM lab_progress;").fetchall()
        return {r["lab_id"]: _row_to_lab_state(r) for r in rows}

    def save_lab(self, state: LabState) -> None:
        """Upsert one lab state."""
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO lab_progress (lab_id, status, branch, started_at, completed_at, notes)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(lab_id) DO UPDATE SET
                    status       = excluded.status,
                    branch       = excluded.branch,
                    started_at   = excluded.started_at,
                    completed_at = excluded.completed_at,
                    notes        = excluded.notes;
                """,
                (
                    state.lab_id,
                    int(state.status),
                    state.branch,
                    _to_iso(state.started_at),
                    _to_iso(state.completed_at),
                    state.notes,
                ),
            )
