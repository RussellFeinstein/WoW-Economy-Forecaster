"""
Repositories for economic archetypes and archetype mappings.
"""

from __future__ import annotations

import logging
import sqlite3
from typing import Optional

from wow_forecaster.db.repositories.base import BaseRepository
from wow_forecaster.models.archetype import ArchetypeMapping, EconomicArchetype
from wow_forecaster.taxonomy.archetype_taxonomy import ArchetypeCategory, ArchetypeTag

logger = logging.getLogger(__name__)


class ArchetypeRepository(BaseRepository):
    """Read/write access to the ``economic_archetypes`` table."""

    def insert(self, archetype: EconomicArchetype) -> int:
        """Insert a new archetype and return its ``archetype_id``.

        Args:
            archetype: The ``EconomicArchetype`` to persist.

        Returns:
            The newly assigned ``archetype_id``.
        """
        self.execute(
            """
            INSERT INTO economic_archetypes (
                slug, display_name, category_tag, sub_tag,
                description, is_transferable, transfer_confidence, transfer_notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?);
            """,
            (
                archetype.slug,
                archetype.display_name,
                archetype.category_tag.value,
                archetype.sub_tag.value if archetype.sub_tag else None,
                archetype.description,
                int(archetype.is_transferable),
                archetype.transfer_confidence,
                archetype.transfer_notes,
            ),
        )
        return self.last_insert_rowid()

    def get_by_id(self, archetype_id: int) -> Optional[EconomicArchetype]:
        """Fetch an archetype by primary key.

        Args:
            archetype_id: The ``archetype_id`` to look up.

        Returns:
            ``EconomicArchetype`` or ``None``.
        """
        row = self.fetchone(
            "SELECT * FROM economic_archetypes WHERE archetype_id = ?;", (archetype_id,)
        )
        return _row_to_archetype(row) if row else None

    def get_by_slug(self, slug: str) -> Optional[EconomicArchetype]:
        """Fetch an archetype by its unique slug.

        Args:
            slug: Archetype slug, e.g. ``"consumable.flask.stat"``.

        Returns:
            ``EconomicArchetype`` or ``None``.
        """
        row = self.fetchone(
            "SELECT * FROM economic_archetypes WHERE slug = ?;", (slug,)
        )
        return _row_to_archetype(row) if row else None

    def get_by_category(self, category_tag: ArchetypeCategory) -> list[EconomicArchetype]:
        """Fetch all archetypes in a category.

        Args:
            category_tag: The ``ArchetypeCategory`` to filter by.

        Returns:
            List of ``EconomicArchetype`` objects, ordered by slug.
        """
        rows = self.fetchall(
            "SELECT * FROM economic_archetypes WHERE category_tag = ? ORDER BY slug;",
            (category_tag.value,),
        )
        return [_row_to_archetype(r) for r in rows]

    def get_transferable(self) -> list[EconomicArchetype]:
        """Fetch all archetypes flagged as transferable.

        Returns:
            List of ``EconomicArchetype`` objects where ``is_transferable = 1``.
        """
        rows = self.fetchall(
            "SELECT * FROM economic_archetypes WHERE is_transferable = 1 ORDER BY slug;",
        )
        return [_row_to_archetype(r) for r in rows]

    def count(self) -> int:
        """Return total number of archetypes."""
        row = self.fetchone("SELECT COUNT(*) AS n FROM economic_archetypes;")
        assert row is not None
        return int(row["n"])


class ArchetypeMappingRepository(BaseRepository):
    """Read/write access to the ``archetype_mappings`` table."""

    def insert(self, mapping: ArchetypeMapping) -> int:
        """Insert a new archetype mapping.

        Args:
            mapping: The ``ArchetypeMapping`` to persist.

        Returns:
            The newly assigned ``mapping_id``.
        """
        self.execute(
            """
            INSERT INTO archetype_mappings (
                source_archetype_id, target_archetype_id,
                source_expansion, target_expansion,
                confidence_score, mapping_rationale, created_by
            ) VALUES (?, ?, ?, ?, ?, ?, ?);
            """,
            (
                mapping.source_archetype_id,
                mapping.target_archetype_id,
                mapping.source_expansion,
                mapping.target_expansion,
                mapping.confidence_score,
                mapping.mapping_rationale,
                mapping.created_by,
            ),
        )
        return self.last_insert_rowid()

    def get_by_source(
        self,
        source_archetype_id: int,
        source_expansion: str = "tww",
    ) -> list[ArchetypeMapping]:
        """Fetch all mappings from a given source archetype.

        Args:
            source_archetype_id: The source archetype FK.
            source_expansion: Source expansion slug.

        Returns:
            List of ``ArchetypeMapping`` objects, sorted by confidence descending.
        """
        rows = self.fetchall(
            """
            SELECT * FROM archetype_mappings
            WHERE source_archetype_id = ? AND source_expansion = ?
            ORDER BY confidence_score DESC;
            """,
            (source_archetype_id, source_expansion),
        )
        return [_row_to_mapping(r) for r in rows]

    def count(self) -> int:
        """Return total number of archetype mappings."""
        row = self.fetchone("SELECT COUNT(*) AS n FROM archetype_mappings;")
        assert row is not None
        return int(row["n"])


# ── Private helpers ────────────────────────────────────────────────────────────

def _row_to_archetype(row: sqlite3.Row) -> EconomicArchetype:
    sub_tag_val = row["sub_tag"]
    return EconomicArchetype(
        archetype_id=row["archetype_id"],
        slug=row["slug"],
        display_name=row["display_name"],
        category_tag=ArchetypeCategory(row["category_tag"]),
        sub_tag=ArchetypeTag(sub_tag_val) if sub_tag_val else None,
        description=row["description"],
        is_transferable=bool(row["is_transferable"]),
        transfer_confidence=row["transfer_confidence"],
        transfer_notes=row["transfer_notes"],
    )


def _row_to_mapping(row: sqlite3.Row) -> ArchetypeMapping:
    return ArchetypeMapping(
        mapping_id=row["mapping_id"],
        source_archetype_id=row["source_archetype_id"],
        target_archetype_id=row["target_archetype_id"],
        source_expansion=row["source_expansion"],
        target_expansion=row["target_expansion"],
        confidence_score=row["confidence_score"],
        mapping_rationale=row["mapping_rationale"],
        created_by=row["created_by"],
    )
