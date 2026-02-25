"""
Feature selection and encoding for the LightGBM forecaster.

Defines which columns from the training/inference Parquet are model inputs,
and how string/categorical columns are encoded to integers for LightGBM.

Why a separate module?
----------------------
The ML layer must not depend on the taxonomy or models packages (avoids
circular imports and keeps the ML code runnable in isolation for debugging).
This module is the single place to update if feature names or category slugs
change.

Excluded columns (never model inputs)
--------------------------------------
- archetype_id   : Would memorise TWW archetypes; prevents cold-start transfer.
- realm_slug     : String identifier; for V1 we train a separate model per realm.
- obs_date       : Temporal patterns captured via day_of_week / week_of_year.
- archetype_sub_tag : High-cardinality string; overlaps with archetype_category.
- target_price_* : Forward-looking labels — must never be model features.
- is_volume_proxy: Boolean flag with low independent signal.
"""

from __future__ import annotations

from typing import Any

# ── Category encoding ──────────────────────────────────────────────────────────
# Matches ArchetypeCategory slug values defined in archetype_taxonomy.py.
# Encoded as sorted ordinals (1-based; 0 = unknown/None).

CATEGORY_ENCODING: dict[str | None, int] = {
    None:          0,
    "collection":  1,
    "consumable":  2,
    "enchant":     3,
    "gear":        4,
    "gem":         5,
    "mat":         6,
    "prof_tool":   7,
    "reagent":     8,
    "service":     9,
    "trade_good": 10,
}

# ── Severity encoding ──────────────────────────────────────────────────────────
SEVERITY_ENCODING: dict[str | None, int] = {
    None:           0,
    "minor":        1,
    "moderate":     2,
    "major":        3,
    "critical":     4,
    "catastrophic": 5,
}

# ── Impact direction encoding ──────────────────────────────────────────────────
IMPACT_ENCODING: dict[str | None, int] = {
    None:       0,
    "neutral":  0,
    "mixed":    0,
    "positive": 1,
    "negative": -1,
}

# ── Training feature columns ───────────────────────────────────────────────────
# These are the columns passed to LightGBM as model inputs.
# All "_enc" / "_int" columns are produced by encode_row() below.
# Order matches what LightGBM expects (consistent between train and inference).

TRAINING_FEATURE_COLS: list[str] = [
    # price summary
    "price_mean",
    "price_min",
    "price_max",
    "market_value_mean",
    "historical_value_mean",
    "obs_count",
    # volume / velocity
    "quantity_sum",
    "auctions_sum",
    # lag features
    "price_lag_1d",
    "price_lag_3d",
    "price_lag_7d",
    "price_lag_14d",
    "price_lag_28d",
    # rolling stats
    "price_roll_mean_7d",
    "price_roll_std_7d",
    "price_roll_mean_14d",
    "price_roll_std_14d",
    "price_roll_mean_28d",
    "price_roll_std_28d",
    # momentum
    "price_pct_change_7d",
    "price_pct_change_14d",
    "price_pct_change_28d",
    # temporal
    "day_of_week",
    "day_of_month",
    "week_of_year",
    "days_since_expansion",
    # event features (encoded)
    "event_active_int",
    "event_days_to_next",
    "event_days_since_last",
    "event_severity_enc",
    "event_archetype_impact_enc",
    # archetype features (encoded)
    "archetype_category_enc",
    "is_transferable_int",
    "is_cold_start_int",
    "item_count_in_archetype",
    # transfer features
    "has_transfer_mapping_int",
    "transfer_confidence",
]

# Columns in TRAINING_FEATURE_COLS treated as LightGBM categorical features.
# These must be non-negative integers; LightGBM handles non-ordinal splits.
CATEGORICAL_FEATURE_COLS: list[str] = [
    "archetype_category_enc",
    "event_severity_enc",
    "day_of_week",
]

# Maps horizon_days (int) to the corresponding target column name in the Parquet.
TARGET_COL_MAP: dict[int, str] = {
    1:  "target_price_1d",
    7:  "target_price_7d",
    28: "target_price_28d",
}


# ── Row encoding ───────────────────────────────────────────────────────────────


def encode_row(row: dict[str, Any]) -> dict[str, Any]:
    """Encode string/bool fields to integer/numeric.

    Creates a copy of ``row`` with additional ``_enc`` / ``_int`` keys that
    LightGBM can consume. The original raw keys are preserved for traceability.

    Args:
        row: A dict representing one row from the training or inference Parquet.

    Returns:
        New dict with extra encoded keys added.
    """
    encoded = dict(row)

    # Bool → int
    encoded["event_active_int"]         = int(bool(row.get("event_active", False)))
    encoded["is_transferable_int"]      = int(bool(row.get("is_transferable", True)))
    encoded["is_cold_start_int"]        = int(bool(row.get("is_cold_start", False)))
    encoded["has_transfer_mapping_int"] = int(bool(row.get("has_transfer_mapping", False)))

    # String → ordinal int
    encoded["event_severity_enc"]          = SEVERITY_ENCODING.get(
        row.get("event_severity_max"), 0
    )
    encoded["event_archetype_impact_enc"]  = IMPACT_ENCODING.get(
        row.get("event_archetype_impact"), 0
    )
    encoded["archetype_category_enc"]      = CATEGORY_ENCODING.get(
        row.get("archetype_category"), 0
    )

    return encoded


def to_float(v: Any) -> float:
    """Convert a value to float, returning NaN for None or non-numeric values.

    LightGBM propagates NaN as missing — never pass Python None to lgb.Dataset.
    """
    if v is None:
        return float("nan")
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def build_feature_matrix(
    rows: list[dict[str, Any]],
    feature_cols: list[str] | None = None,
) -> list[list[float]]:
    """Build a float matrix from a list of encoded row dicts.

    Args:
        rows: Pre-encoded row dicts (from ``encode_row()``).
        feature_cols: Column names to extract. Defaults to TRAINING_FEATURE_COLS.

    Returns:
        List of float lists (outer = rows, inner = feature values).
        None / non-numeric values become NaN.
    """
    cols = feature_cols or TRAINING_FEATURE_COLS
    return [[to_float(row.get(c)) for c in cols] for row in rows]
