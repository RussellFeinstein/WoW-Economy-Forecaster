"""Feature engineering package for the WoW Economy Forecaster.

Modules
-------
registry        — FeatureSpec dataclass + FEATURE_REGISTRY (schema source of truth)
daily_agg       — SQL daily aggregation from market_observations_normalized
lag_rolling     — Lag, rolling-window, momentum, and forward-looking target features
event_features  — Event proximity features with strict is_known_at() leakage guard
archetype_features — Category / transfer encoding and cold-start detection
quality         — DataQualityReport for assembled feature rows
dataset_builder — Top-level orchestrator: Parquet + JSON manifest output
"""
