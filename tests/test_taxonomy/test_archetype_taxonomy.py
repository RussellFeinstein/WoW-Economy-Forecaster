"""Tests for archetype taxonomy integrity — CATEGORY_TAG_MAP contract."""

from __future__ import annotations

import re

import pytest

from wow_forecaster.taxonomy.archetype_taxonomy import (
    CATEGORY_TAG_MAP,
    ArchetypeCategory,
    ArchetypeTag,
)


class TestArchetypeCategoryEnum:
    def test_all_values_are_strings(self):
        for member in ArchetypeCategory:
            assert isinstance(member.value, str)

    def test_no_duplicate_values(self):
        values = [m.value for m in ArchetypeCategory]
        assert len(values) == len(set(values)), "ArchetypeCategory has duplicate values"

    def test_key_categories_exist(self):
        required = {"consumable", "mat", "gear", "enchant", "gem"}
        actual = {m.value for m in ArchetypeCategory}
        missing = required - actual
        assert not missing, f"Required ArchetypeCategory values missing: {missing}"


class TestArchetypeTagEnum:
    def test_all_values_are_strings(self):
        for member in ArchetypeTag:
            assert isinstance(member.value, str)

    def test_no_duplicate_values(self):
        values = [m.value for m in ArchetypeTag]
        assert len(values) == len(set(values)), "ArchetypeTag has duplicate values"

    def test_tag_slug_format(self):
        """All tags should match pattern: lowercase, dot-delimited, at least 2 parts."""
        pattern = re.compile(r"^[a-z_]+(\.[a-z_]+){1,}$")
        for member in ArchetypeTag:
            assert pattern.match(member.value), (
                f"ArchetypeTag.{member.name} = '{member.value}' "
                "does not match expected slug format"
            )

    def test_key_tags_exist(self):
        required = {
            "consumable.flask.stat",
            "consumable.potion.combat",
            "mat.ore.common",
            "mat.herb.common",
            "gear.boe.endgame",
            "enchant.weapon",
        }
        actual = {m.value for m in ArchetypeTag}
        missing = required - actual
        assert not missing, f"Required ArchetypeTag values missing: {missing}"


class TestCategoryTagMap:
    def test_all_categories_have_entry(self):
        """Every ArchetypeCategory must appear as a key in CATEGORY_TAG_MAP."""
        for category in ArchetypeCategory:
            assert category in CATEGORY_TAG_MAP, (
                f"ArchetypeCategory.{category.name} is missing from CATEGORY_TAG_MAP"
            )

    def test_no_extra_keys(self):
        """CATEGORY_TAG_MAP must not have keys outside ArchetypeCategory."""
        valid_categories = set(ArchetypeCategory)
        for key in CATEGORY_TAG_MAP:
            assert key in valid_categories, (
                f"CATEGORY_TAG_MAP has unknown key: {key}"
            )

    def test_every_tag_in_exactly_one_category(self):
        """Each ArchetypeTag must appear in exactly one category list."""
        all_tags: list[ArchetypeTag] = []
        for tags in CATEGORY_TAG_MAP.values():
            all_tags.extend(tags)

        # All listed tags are unique (no tag in two categories)
        assert len(all_tags) == len(set(all_tags)), (
            "Some ArchetypeTag appears in more than one category in CATEGORY_TAG_MAP"
        )

    def test_every_defined_tag_appears_in_map(self):
        """Every ArchetypeTag must appear in at least one category list."""
        mapped_tags: set[ArchetypeTag] = set()
        for tags in CATEGORY_TAG_MAP.values():
            mapped_tags.update(tags)

        all_defined_tags = set(ArchetypeTag)
        unmapped = all_defined_tags - mapped_tags
        assert not unmapped, (
            f"These ArchetypeTag values are defined but not in CATEGORY_TAG_MAP: "
            f"{[t.value for t in unmapped]}"
        )

    def test_consumable_tags_have_correct_prefix(self):
        """Tags under CONSUMABLE category must start with 'consumable.'"""
        for tag in CATEGORY_TAG_MAP.get(ArchetypeCategory.CONSUMABLE, []):
            assert tag.value.startswith("consumable."), (
                f"Tag '{tag.value}' under CONSUMABLE does not start with 'consumable.'"
            )

    def test_crafting_mat_tags_have_correct_prefix(self):
        """Tags under CRAFTING_MAT category must start with 'mat.'"""
        for tag in CATEGORY_TAG_MAP.get(ArchetypeCategory.CRAFTING_MAT, []):
            assert tag.value.startswith("mat."), (
                f"Tag '{tag.value}' under CRAFTING_MAT does not start with 'mat.'"
            )

    def test_gear_tags_have_correct_prefix(self):
        """Tags under GEAR category must start with 'gear.'"""
        for tag in CATEGORY_TAG_MAP.get(ArchetypeCategory.GEAR, []):
            assert tag.value.startswith("gear."), (
                f"Tag '{tag.value}' under GEAR does not start with 'gear.'"
            )

    def test_no_cross_category_contamination(self):
        """No tag from category A should start with a different category's prefix."""
        # Build prefix → category mapping
        prefix_to_cat = {
            "consumable.": ArchetypeCategory.CONSUMABLE,
            "mat.": ArchetypeCategory.CRAFTING_MAT,
            "gear.": ArchetypeCategory.GEAR,
            "enchant.": ArchetypeCategory.ENCHANT,
            "gem.": ArchetypeCategory.GEM,
            "prof_tool.": ArchetypeCategory.PROFESSION_TOOL,
            "reagent.": ArchetypeCategory.REAGENT,
            "trade_good.": ArchetypeCategory.TRADE_GOOD,
        }

        for category, tags in CATEGORY_TAG_MAP.items():
            for tag in tags:
                for prefix, expected_cat in prefix_to_cat.items():
                    if tag.value.startswith(prefix):
                        assert category == expected_cat, (
                            f"Tag '{tag.value}' starts with '{prefix}' prefix "
                            f"but is listed under {category.value} "
                            f"(expected {expected_cat.value})"
                        )
