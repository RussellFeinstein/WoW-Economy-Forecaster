"""
Feature registry for the WoW Economy Forecaster dataset builder.

This module is the single source of truth for every column that appears in the
training and inference Parquet files.  All other feature modules produce dicts
whose keys MUST match names declared here; ``dataset_builder`` enforces this
contract when constructing the PyArrow schema.

Design rationale
----------------
A plain list of ``FeatureSpec`` dataclasses is deliberately simple:

* Easy to read — no framework magic, no decorators.
* Easy to add or remove a feature — edit one line here.
* Provides type metadata consumed by ``dataset_builder.build_parquet_schema()``.
* Can be queried programmatically (e.g. ``feature_names(group="lag")``).

Groups
------
price       Daily OHLC-style summaries aggregated to (archetype, realm, date).
volume      Quantity / auction count fields with proxy indicator.
lag         Lookback price at N calendar days prior.
rolling     Rolling mean and std over N-day windows.
momentum    Percentage price change over N-day windows.
temporal    Calendar features and content-cycle indicators.
event       Event proximity and severity features (leakage-safe).
archetype   Category encoding, cold-start flag, transfer metadata.
transfer    Cross-expansion transfer mapping features.
target      Forward-looking price labels (training dataset only).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class FeatureSpec:
    """Specification for a single feature (or identifier/target) column.

    Attributes:
        name: Column name in the Parquet file.
        pa_type: PyArrow type string recognised by ``_pa_type_from_str()``.
        group: Logical group for filtering and documentation.
        description: Human-readable explanation of what the feature captures.
        is_nullable: Whether the column may contain null values.
        is_proxy: True if the column is a fallback approximation (e.g. volume
            proxy) rather than a direct measurement.
        is_target: True for forward-looking labels. Target columns are excluded
            from the inference Parquet and must never be used as model features.
        requires_history_days: Minimum calendar days of prior history needed
            for this feature to produce a non-null value.
    """

    name: str
    pa_type: str       # "int32", "float32", "bool", "utf8", "date32"
    group: str
    description: str
    is_nullable: bool = True
    is_proxy: bool = False
    is_target: bool = False
    requires_history_days: int = 0


# ── Registry ──────────────────────────────────────────────────────────────────
# Order here determines column order in the Parquet file.

FEATURE_REGISTRY: list[FeatureSpec] = [

    # ── Identifiers / keys ─────────────────────────────────────────────────
    FeatureSpec("archetype_id",   "int32",   "price", "Economic archetype primary key.",        is_nullable=False),
    FeatureSpec("realm_slug",     "utf8",    "price", "Blizzard realm slug (e.g. 'area-52').",  is_nullable=False),
    FeatureSpec("obs_date",       "date32",  "price", "Calendar date of the daily aggregate.",   is_nullable=False),

    # ── Price summary ──────────────────────────────────────────────────────
    FeatureSpec("price_mean",              "float32", "price",
                "Mean non-outlier min-buyout price in gold across all items in the archetype."),
    FeatureSpec("price_min",               "float32", "price",
                "Lowest min-buyout price in gold on this date (price floor)."),
    FeatureSpec("price_max",               "float32", "price",
                "Highest min-buyout price in gold on this date (price ceiling)."),
    FeatureSpec("market_value_mean",       "float32", "price",
                "Mean TSM-style market value in gold (nullable; not all sources provide it)."),
    FeatureSpec("historical_value_mean",   "float32", "price",
                "Mean historical value in gold (long-run average from source API)."),
    FeatureSpec("obs_count",               "int32",   "price",
                "Number of normalised price observations on this date.",
                is_nullable=False),

    # ── Volume / velocity ──────────────────────────────────────────────────
    FeatureSpec("quantity_sum",    "float32", "volume",
                "Total units listed on the AH across all observations."),
    FeatureSpec("auctions_sum",    "float32", "volume",
                "Total individual auction listings across all observations."),
    FeatureSpec("is_volume_proxy", "bool",    "volume",
                "True when quantity_listed was unavailable; obs_count used as velocity proxy.",
                is_nullable=False, is_proxy=True),

    # ── Lag features ───────────────────────────────────────────────────────
    FeatureSpec("price_lag_1d",  "float32", "lag", "price_mean N=1  calendar day prior.",  requires_history_days=1),
    FeatureSpec("price_lag_3d",  "float32", "lag", "price_mean N=3  calendar days prior.", requires_history_days=3),
    FeatureSpec("price_lag_7d",  "float32", "lag", "price_mean N=7  calendar days prior.", requires_history_days=7),
    FeatureSpec("price_lag_14d", "float32", "lag", "price_mean N=14 calendar days prior.", requires_history_days=14),
    FeatureSpec("price_lag_28d", "float32", "lag", "price_mean N=28 calendar days prior.", requires_history_days=28),

    # ── Rolling stats ──────────────────────────────────────────────────────
    FeatureSpec("price_roll_mean_7d",  "float32", "rolling",
                "Rolling mean of price_mean over the past 7 calendar days.",  requires_history_days=7),
    FeatureSpec("price_roll_std_7d",   "float32", "rolling",
                "Rolling std-dev of price_mean over the past 7 calendar days.",  requires_history_days=7),
    FeatureSpec("price_roll_mean_14d", "float32", "rolling",
                "Rolling mean of price_mean over the past 14 calendar days.", requires_history_days=14),
    FeatureSpec("price_roll_std_14d",  "float32", "rolling",
                "Rolling std-dev of price_mean over the past 14 calendar days.", requires_history_days=14),
    FeatureSpec("price_roll_mean_28d", "float32", "rolling",
                "Rolling mean of price_mean over the past 28 calendar days.", requires_history_days=28),
    FeatureSpec("price_roll_std_28d",  "float32", "rolling",
                "Rolling std-dev of price_mean over the past 28 calendar days.", requires_history_days=28),

    # ── Momentum / pct-change ──────────────────────────────────────────────
    FeatureSpec("price_pct_change_7d",  "float32", "momentum",
                "% change in price_mean vs 7 days ago: (today - 7d) / 7d.",  requires_history_days=7),
    FeatureSpec("price_pct_change_14d", "float32", "momentum",
                "% change in price_mean vs 14 days ago.",                     requires_history_days=14),
    FeatureSpec("price_pct_change_28d", "float32", "momentum",
                "% change in price_mean vs 28 days ago.",                     requires_history_days=28),

    # ── Temporal / calendar ────────────────────────────────────────────────
    FeatureSpec("day_of_week",           "int32", "temporal",
                "ISO weekday 1=Mon … 7=Sun.", is_nullable=False),
    FeatureSpec("day_of_month",          "int32", "temporal",
                "Day of month 1–31.", is_nullable=False),
    FeatureSpec("week_of_year",          "int32", "temporal",
                "ISO week number 1–53.", is_nullable=False),
    FeatureSpec("days_since_expansion",  "int32", "temporal",
                "Calendar days since the active expansion's launch event. "
                "Null if launch event is absent from wow_events table."),

    # ── Event features ─────────────────────────────────────────────────────
    # All event features are computed using only events where
    # announced_at <= obs_date (strict leakage guard).
    FeatureSpec("event_active",          "bool",    "event",
                "True if any known event is active on this date.",
                is_nullable=False),
    FeatureSpec("event_days_to_next",    "float32", "event",
                "Days until the start of the next known future event. "
                "Null if no future event is known."),
    FeatureSpec("event_days_since_last", "float32", "event",
                "Days since the end of the most recently completed known event. "
                "Null if no past event is known."),
    FeatureSpec("event_severity_max",    "utf8",    "event",
                "Maximum EventSeverity string of currently-active known events. "
                "Null if no event is active."),
    FeatureSpec("event_archetype_impact", "utf8",   "event",
                "Impact direction for this archetype from event_archetype_impacts "
                "for the most recently started active event. "
                "Null if no impact record exists."),

    # ── Archetype / category encoding ──────────────────────────────────────
    FeatureSpec("archetype_category",      "utf8",    "archetype",
                "ArchetypeCategory slug (e.g. 'consumable', 'mat').", is_nullable=False),
    FeatureSpec("archetype_sub_tag",       "utf8",    "archetype",
                "Full ArchetypeTag slug (e.g. 'consumable.flask.stat'). "
                "Null if archetype has no sub_tag."),
    FeatureSpec("is_transferable",         "bool",    "archetype",
                "True if this archetype is expected to have a meaningful analogue "
                "in the transfer-target expansion.", is_nullable=False),
    FeatureSpec("is_cold_start",           "bool",    "archetype",
                "True if expansion_slug == transfer_target AND total obs < cold_start_threshold.",
                is_nullable=False),
    FeatureSpec("item_count_in_archetype", "int32",   "archetype",
                "Number of distinct items contributing to this archetype series "
                "on this realm.", is_nullable=False),

    # ── Transfer learning features ─────────────────────────────────────────
    FeatureSpec("has_transfer_mapping",        "bool",    "transfer",
                "True if an archetype_mapping exists for this archetype "
                "(source_expansion → transfer_target).", is_nullable=False),
    FeatureSpec("transfer_confidence",         "float32", "transfer",
                "Max confidence_score of any archetype_mapping for this archetype. "
                "Null if has_transfer_mapping=False."),

    # ── Forward-looking targets (training dataset only) ────────────────────
    # These columns are deliberately forward-looking — they are the labels the
    # model learns to predict.  They are EXCLUDED from the inference Parquet.
    # They must never be used as input features during model training.
    FeatureSpec("target_price_1d",  "float32", "target",
                "price_mean 1  calendar day forward. Null at the series tail.",
                is_target=True, requires_history_days=0),
    FeatureSpec("target_price_7d",  "float32", "target",
                "price_mean 7  calendar days forward. Null at the series tail.",
                is_target=True, requires_history_days=0),
    FeatureSpec("target_price_28d", "float32", "target",
                "price_mean 28 calendar days forward. Null at the series tail.",
                is_target=True, requires_history_days=0),
]

# ── Registry helpers ───────────────────────────────────────────────────────────


def feature_names(group: str | None = None) -> list[str]:
    """Return column names, optionally filtered to a single group."""
    if group is None:
        return [f.name for f in FEATURE_REGISTRY]
    return [f.name for f in FEATURE_REGISTRY if f.group == group]


def training_feature_names() -> list[str]:
    """All column names included in the training Parquet (45 columns)."""
    return [f.name for f in FEATURE_REGISTRY]


def inference_feature_names() -> list[str]:
    """All column names included in the inference Parquet (no target group)."""
    return [f.name for f in FEATURE_REGISTRY if not f.is_target]


def target_feature_names() -> list[str]:
    """Forward-looking target column names only."""
    return [f.name for f in FEATURE_REGISTRY if f.is_target]


def get_spec(name: str) -> FeatureSpec:
    """Return the FeatureSpec for ``name``.

    Raises:
        KeyError: If ``name`` is not in the registry.
    """
    for spec in FEATURE_REGISTRY:
        if spec.name == name:
            return spec
    raise KeyError(f"Feature '{name}' not found in FEATURE_REGISTRY.")


def feature_groups() -> list[str]:
    """Return unique group names in registry order (no duplicates)."""
    seen: set[str] = set()
    result: list[str] = []
    for f in FEATURE_REGISTRY:
        if f.group not in seen:
            seen.add(f.group)
            result.append(f.group)
    return result
