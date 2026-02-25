"""
SQLite connection management.

Provides a context manager ``get_connection()`` that:
  - Enables foreign key enforcement (OFF by default in SQLite).
  - Enables WAL journal mode for concurrent reads during pipeline runs.
  - Sets a busy timeout to handle lock contention gracefully.
  - Uses ``sqlite3.Row`` factory so rows behave like dicts.
  - Commits on clean exit, rolls back on exception.

Usage::

    from wow_forecaster.db.connection import get_connection

    with get_connection("data/db/wow_forecaster.db") as conn:
        conn.execute("INSERT INTO ...")
"""

from __future__ import annotations

import logging
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

logger = logging.getLogger(__name__)


@contextmanager
def get_connection(
    db_path: str,
    wal_mode: bool = True,
    busy_timeout_ms: int = 5000,
) -> Generator[sqlite3.Connection, None, None]:
    """Context manager yielding a configured SQLite connection.

    The connection is committed on clean exit and rolled back on exception.
    The database file (and any parent directories) are created if they do
    not already exist.

    Args:
        db_path: Path to the SQLite database file. Use ``":memory:"`` for
            in-memory databases (useful in tests).
        wal_mode: If ``True``, enable WAL journal mode for better concurrency.
        busy_timeout_ms: Milliseconds to wait when the database is locked
            before raising ``OperationalError``.

    Yields:
        An open, configured ``sqlite3.Connection``.

    Raises:
        sqlite3.OperationalError: If the database cannot be opened or is locked.
    """
    if db_path != ":memory:":
        db_file = Path(db_path)
        db_file.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(db_path, timeout=busy_timeout_ms / 1000)
    conn.row_factory = sqlite3.Row

    try:
        # These pragmas must be set before any DML/DDL
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute(f"PRAGMA busy_timeout = {busy_timeout_ms};")

        if wal_mode:
            conn.execute("PRAGMA journal_mode = WAL;")

        yield conn
        conn.commit()

    except Exception:
        conn.rollback()
        raise

    finally:
        conn.close()
