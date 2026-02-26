"""
Repositories for items and item categories.
"""

from __future__ import annotations

import logging
import sqlite3
from typing import Optional

from wow_forecaster.db.repositories.base import BaseRepository
from wow_forecaster.models.item import Item, ItemCategory

logger = logging.getLogger(__name__)


class ItemCategoryRepository(BaseRepository):
    """Read/write access to the ``item_categories`` table."""

    def insert(self, category: ItemCategory) -> int:
        """Insert a new item category.

        Args:
            category: The ``ItemCategory`` to persist.

        Returns:
            The newly assigned ``category_id``.
        """
        self.execute(
            """
            INSERT INTO item_categories (slug, display_name, parent_slug, archetype_tag, expansion_slug)
            VALUES (?, ?, ?, ?, ?);
            """,
            (
                category.slug,
                category.display_name,
                category.parent_slug,
                category.archetype_tag,
                category.expansion_slug,
            ),
        )
        return self.last_insert_rowid()

    def get_by_slug(self, slug: str) -> Optional[ItemCategory]:
        """Fetch a category by its slug.

        Args:
            slug: Unique category slug.

        Returns:
            ``ItemCategory`` or ``None``.
        """
        row = self.fetchone("SELECT * FROM item_categories WHERE slug = ?;", (slug,))
        return _row_to_category(row) if row else None

    def get_by_id(self, category_id: int) -> Optional[ItemCategory]:
        """Fetch a category by primary key.

        Args:
            category_id: The category_id to look up.

        Returns:
            ``ItemCategory`` or ``None``.
        """
        row = self.fetchone(
            "SELECT * FROM item_categories WHERE category_id = ?;", (category_id,)
        )
        return _row_to_category(row) if row else None

    def get_children(self, parent_slug: str) -> list[ItemCategory]:
        """Fetch all direct children of a category.

        Args:
            parent_slug: The slug of the parent category.

        Returns:
            List of child ``ItemCategory`` objects.
        """
        rows = self.fetchall(
            "SELECT * FROM item_categories WHERE parent_slug = ? ORDER BY slug;",
            (parent_slug,),
        )
        return [_row_to_category(r) for r in rows]

    def count(self) -> int:
        """Return total number of categories."""
        row = self.fetchone("SELECT COUNT(*) AS n FROM item_categories;")
        assert row is not None
        return int(row["n"])


class ItemRepository(BaseRepository):
    """Read/write access to the ``items`` table."""

    def insert(self, item: Item) -> int:
        """Insert a new item record.

        Uses ``item.item_id`` as the primary key (not auto-incremented).

        Args:
            item: The ``Item`` to persist.

        Returns:
            The ``item_id`` (same as ``item.item_id``).
        """
        self.execute(
            """
            INSERT INTO items (
                item_id, name, category_id, archetype_id,
                expansion_slug, quality, is_crafted, is_boe, ilvl, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            (
                item.item_id,
                item.name,
                item.category_id,
                item.archetype_id,
                item.expansion_slug,
                item.quality,
                int(item.is_crafted),
                int(item.is_boe),
                item.ilvl,
                item.notes,
            ),
        )
        return item.item_id

    def get_by_id(self, item_id: int) -> Optional[Item]:
        """Fetch an item by its WoW canonical ID.

        Args:
            item_id: WoW item ID.

        Returns:
            ``Item`` or ``None``.
        """
        row = self.fetchone("SELECT * FROM items WHERE item_id = ?;", (item_id,))
        return _row_to_item(row) if row else None

    def get_by_expansion(self, expansion_slug: str) -> list[Item]:
        """Fetch all items for a given expansion.

        Args:
            expansion_slug: Expansion identifier string.

        Returns:
            List of ``Item`` objects.
        """
        rows = self.fetchall(
            "SELECT * FROM items WHERE expansion_slug = ? ORDER BY item_id;",
            (expansion_slug,),
        )
        return [_row_to_item(r) for r in rows]

    def get_by_archetype(self, archetype_id: int) -> list[Item]:
        """Fetch all items assigned to a given archetype.

        Args:
            archetype_id: The archetype_id FK.

        Returns:
            List of ``Item`` objects.
        """
        rows = self.fetchall(
            "SELECT * FROM items WHERE archetype_id = ? ORDER BY item_id;",
            (archetype_id,),
        )
        return [_row_to_item(r) for r in rows]

    def get_all_item_ids(self) -> set[int]:
        """Return the set of all item IDs currently in the registry.

        Used by ingestion to guard against FK violations: observations for
        unknown items are skipped rather than inserted.

        Returns:
            Set of integer ``item_id`` values.
        """
        rows = self.fetchall("SELECT item_id FROM items;")
        return {int(row["item_id"]) for row in rows}

    def count(self) -> int:
        """Return total number of registered items."""
        row = self.fetchone("SELECT COUNT(*) AS n FROM items;")
        assert row is not None
        return int(row["n"])


# ── Private helpers ────────────────────────────────────────────────────────────

def _row_to_category(row: sqlite3.Row) -> ItemCategory:
    return ItemCategory(
        category_id=row["category_id"],
        slug=row["slug"],
        display_name=row["display_name"],
        parent_slug=row["parent_slug"],
        archetype_tag=row["archetype_tag"],
        expansion_slug=row["expansion_slug"],
    )


def _row_to_item(row: sqlite3.Row) -> Item:
    return Item(
        item_id=row["item_id"],
        name=row["name"],
        category_id=row["category_id"],
        archetype_id=row["archetype_id"],
        expansion_slug=row["expansion_slug"],
        quality=row["quality"],
        is_crafted=bool(row["is_crafted"]),
        is_boe=bool(row["is_boe"]),
        ilvl=row["ilvl"],
        notes=row["notes"],
    )
