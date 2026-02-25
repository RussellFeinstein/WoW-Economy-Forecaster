"""
Data quality checks for the assembled feature dataset.

Purpose
-------
Before writing Parquet files, ``build_quality_report()`` inspects the complete
list of assembled feature rows and produces a ``DataQualityReport`` describing:

- Missingness fraction per feature column.
- Duplicate (archetype_id, realm_slug, obs_date) rows.
- Time-series continuity gaps (calendar gaps > 1 day in any series).
- Leakage heuristic warnings (``event_days_to_next < 0`` implies a past event
  is being treated as "next", which indicates a logic error).
- Volume proxy prevalence.
- Cold-start archetype prevalence.
- Items excluded from features because their archetype_id was NULL.

``is_clean`` is set to False only for hard errors (duplicates, leakage warnings)
that indicate data integrity problems.  High missingness in lag/rolling columns
is expected for the first N rows of each series and does NOT mark the report
as unclean.

Usage
-----
``build_quality_report()`` operates on plain Python dicts (the feature rows
assembled by ``dataset_builder``), so it can be called without a DB connection
and is straightforward to unit-test with synthetic data.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

from wow_forecaster.features.registry import feature_names


@dataclass
class DataQualityReport:
    """Summary of data quality checks on the assembled feature rows.

    Attributes:
        total_rows:                  Total rows in the dataset.
        total_archetypes:            Distinct archetype_id values.
        total_realms:                Distinct realm_slug values.
        date_range_start:            Earliest obs_date, or None if empty.
        date_range_end:              Latest obs_date, or None if empty.
        missingness:                 Col name → fraction of null values [0.0, 1.0].
        high_missingness_cols:       Cols where missingness > ``missingness_threshold``.
        duplicate_key_count:         Count of duplicate (archetype_id, realm, date) rows.
        date_gap_series_count:       Count of (archetype, realm) series with any
                                     calendar gap > 1 day.
        leakage_warnings:            List of human-readable leakage warning strings.
        volume_proxy_pct:            Fraction of rows where is_volume_proxy is True.
        cold_start_pct:              Fraction of rows where is_cold_start is True.
        items_excluded_no_archetype: Items in the items table without an archetype.
        is_clean:                    False if duplicate_key_count > 0 OR
                                     leakage_warnings is non-empty.
    """

    total_rows: int
    total_archetypes: int
    total_realms: int
    date_range_start: date | None
    date_range_end: date | None
    missingness: dict[str, float]
    high_missingness_cols: list[str]
    duplicate_key_count: int
    date_gap_series_count: int
    leakage_warnings: list[str]
    volume_proxy_pct: float
    cold_start_pct: float
    items_excluded_no_archetype: int
    is_clean: bool


def build_quality_report(
    rows: list[dict[str, Any]],
    items_excluded: int = 0,
    missingness_threshold: float = 0.30,
) -> DataQualityReport:
    """Build a quality report for a list of assembled feature rows.

    Args:
        rows:                  Assembled feature dicts (output of dataset_builder).
        items_excluded:        Pre-computed count of items excluded due to NULL
                               archetype_id; pass the result of
                               ``archetype_features.count_items_without_archetype()``.
        missingness_threshold: Cols with null fraction > this appear in
                               ``high_missingness_cols``.

    Returns:
        A ``DataQualityReport`` instance.
    """
    if not rows:
        return DataQualityReport(
            total_rows=0,
            total_archetypes=0,
            total_realms=0,
            date_range_start=None,
            date_range_end=None,
            missingness={},
            high_missingness_cols=[],
            duplicate_key_count=0,
            date_gap_series_count=0,
            leakage_warnings=[],
            volume_proxy_pct=0.0,
            cold_start_pct=0.0,
            items_excluded_no_archetype=items_excluded,
            is_clean=True,
        )

    n = len(rows)
    all_cols = feature_names()

    # ── Missingness ────────────────────────────────────────────────────────────
    missingness: dict[str, float] = {}
    for col in all_cols:
        null_count = sum(1 for r in rows if r.get(col) is None)
        missingness[col] = null_count / n

    high_missingness_cols = [
        col for col, frac in missingness.items()
        if frac > missingness_threshold
    ]

    # ── Duplicate key detection ────────────────────────────────────────────────
    seen_keys: set[tuple] = set()
    duplicate_key_count = 0
    for r in rows:
        key = (r.get("archetype_id"), r.get("realm_slug"), r.get("obs_date"))
        if key in seen_keys:
            duplicate_key_count += 1
        seen_keys.add(key)

    # ── Time-series continuity ─────────────────────────────────────────────────
    # Group obs_date values by (archetype_id, realm_slug) and check for gaps.
    series_dates: dict[tuple, list[date]] = defaultdict(list)
    for r in rows:
        key = (r.get("archetype_id"), r.get("realm_slug"))
        obs = r.get("obs_date")
        if obs is not None:
            series_dates[key].append(obs)

    date_gap_series_count = 0
    for dates in series_dates.values():
        sorted_dates = sorted(dates)
        for i in range(1, len(sorted_dates)):
            gap = (sorted_dates[i] - sorted_dates[i - 1]).days
            if gap > 1:
                date_gap_series_count += 1
                break  # count each series at most once

    # ── Leakage heuristic ──────────────────────────────────────────────────────
    leakage_warnings: list[str] = []
    for r in rows:
        days_to_next = r.get("event_days_to_next")
        if days_to_next is not None and days_to_next < 0.0:
            obs_date = r.get("obs_date")
            arch_id  = r.get("archetype_id")
            leakage_warnings.append(
                f"event_days_to_next={days_to_next:.1f} < 0 for "
                f"archetype_id={arch_id} obs_date={obs_date} — "
                "a past event may be incorrectly labelled as 'next upcoming'."
            )

    # ── Volume proxy and cold-start prevalence ─────────────────────────────────
    proxy_count = sum(1 for r in rows if r.get("is_volume_proxy") is True)
    cold_count  = sum(1 for r in rows if r.get("is_cold_start") is True)
    volume_proxy_pct = proxy_count / n
    cold_start_pct   = cold_count  / n

    # ── Aggregates ─────────────────────────────────────────────────────────────
    archetypes = {r.get("archetype_id") for r in rows if r.get("archetype_id") is not None}
    realms     = {r.get("realm_slug")   for r in rows if r.get("realm_slug")   is not None}
    dates      = [r.get("obs_date")     for r in rows if r.get("obs_date")     is not None]

    date_range_start = min(dates) if dates else None
    date_range_end   = max(dates) if dates else None

    is_clean = duplicate_key_count == 0 and len(leakage_warnings) == 0

    return DataQualityReport(
        total_rows=n,
        total_archetypes=len(archetypes),
        total_realms=len(realms),
        date_range_start=date_range_start,
        date_range_end=date_range_end,
        missingness=missingness,
        high_missingness_cols=high_missingness_cols,
        duplicate_key_count=duplicate_key_count,
        date_gap_series_count=date_gap_series_count,
        leakage_warnings=leakage_warnings,
        volume_proxy_pct=volume_proxy_pct,
        cold_start_pct=cold_start_pct,
        items_excluded_no_archetype=items_excluded,
        is_clean=is_clean,
    )
