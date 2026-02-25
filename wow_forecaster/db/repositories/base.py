"""
Base repository providing shared SQLite execution helpers.

All repositories inherit from ``BaseRepository`` and receive a
``sqlite3.Connection`` at construction time. The connection is assumed
to be opened and managed by the caller (typically via ``get_connection()``).

Design:
  - No ORM — all SQL is explicit and lives in repository methods.
  - Repositories speak Pydantic models, not raw dicts.
  - ``row_factory = sqlite3.Row`` (set by ``get_connection()``) gives
    dict-like row access throughout.
"""

from __future__ import annotations

import logging
import sqlite3
from typing import Any, Optional

logger = logging.getLogger(__name__)


class BaseRepository:
    """Shared SQL execution helpers for all repository classes.

    Attributes:
        conn: The active ``sqlite3.Connection``.
    """

    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn

    def execute(
        self,
        sql: str,
        params: tuple[Any, ...] | dict[str, Any] = (),
    ) -> sqlite3.Cursor:
        """Execute a single SQL statement.

        Args:
            sql: SQL string with ``?`` or ``:name`` placeholders.
            params: Positional tuple or named dict of parameters.

        Returns:
            The resulting ``sqlite3.Cursor``.
        """
        logger.debug("SQL: %s | params: %s", sql.strip(), params)
        return self.conn.execute(sql, params)

    def executemany(
        self,
        sql: str,
        params_list: list[tuple[Any, ...] | dict[str, Any]],
    ) -> sqlite3.Cursor:
        """Execute a SQL statement for each element in ``params_list``.

        Args:
            sql: SQL string with placeholders.
            params_list: List of parameter tuples or dicts.

        Returns:
            The resulting ``sqlite3.Cursor``.
        """
        logger.debug("SQL (many): %s | count: %d", sql.strip(), len(params_list))
        return self.conn.executemany(sql, params_list)

    def fetchone(
        self,
        sql: str,
        params: tuple[Any, ...] | dict[str, Any] = (),
    ) -> Optional[sqlite3.Row]:
        """Execute a query and return the first row, or ``None``.

        Args:
            sql: SELECT SQL string.
            params: Query parameters.

        Returns:
            First ``sqlite3.Row`` or ``None``.
        """
        return self.execute(sql, params).fetchone()

    def fetchall(
        self,
        sql: str,
        params: tuple[Any, ...] | dict[str, Any] = (),
    ) -> list[sqlite3.Row]:
        """Execute a query and return all rows.

        Args:
            sql: SELECT SQL string.
            params: Query parameters.

        Returns:
            List of ``sqlite3.Row`` objects.
        """
        return self.execute(sql, params).fetchall()

    def last_insert_rowid(self) -> int:
        """Return the rowid of the last successful INSERT.

        Returns:
            Integer rowid.
        """
        row = self.fetchone("SELECT last_insert_rowid() AS rowid;")
        assert row is not None
        return int(row["rowid"])
