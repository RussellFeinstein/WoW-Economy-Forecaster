"""
Dataset assembly: Parquet files + JSON manifest.

Purpose
-------
``build_datasets()`` is the top-level orchestrator called by ``FeatureBuildStage``.
It coordinates all feature modules in sequence and writes:

    data/processed/features/training/train_{realm}_{start}_{end}.parquet
    data/processed/features/inference/inference_{realm}_{date}.parquet
    data/processed/features/manifests/manifest_{realm}_{date}.json

One set of files is produced per realm slug.  When multiple realms are requested,
``build_datasets()`` loops over them and accumulates the total row count for
``RunMetadata.rows_processed``.

Feature assembly pipeline (step-by-step)
-----------------------------------------
1.  ``daily_agg.fetch_daily_agg()``
        SQL → list[DailyAggRow]  (archetype × realm × date, with date spine)

2.  ``lag_rolling.compute_lag_rolling_features()``
        Adds: price_lag_*d, price_roll_mean/std_*d, price_pct_change_*d,
              target_price_*d (forward-looking labels)

3.  ``event_features.load_known_events()`` + ``load_archetype_impacts()``
        Loaded once per call (not per row).

4.  For each (archetype_id, realm_slug) group:
        ``event_features.compute_event_features()``
        Adds: event_active, event_days_to_next/since_last, event_severity_max,
              event_archetype_impact

5.  ``archetype_features.load_archetype_metadata()`` + ``count_obs_per_archetype_realm()``
        Loaded once per call.

6.  For each (archetype_id, realm_slug) group:
        ``archetype_features.compute_archetype_features()``
        Adds: archetype_category, archetype_sub_tag, is_transferable,
              is_cold_start, item_count_in_archetype,
              has_transfer_mapping, transfer_confidence

7.  Temporal features (pure date arithmetic, inline):
        Adds: day_of_week, day_of_month, week_of_year, days_since_expansion

8.  ``quality.build_quality_report()``

9.  ``write_training_parquet()`` + ``write_inference_parquet()``

10. ``write_manifest()``

Parquet schema
--------------
Derived from ``registry.FEATURE_REGISTRY``.  Training Parquet: 45 columns.
Inference Parquet: 42 columns (``target`` group excluded).
Both use Snappy compression.  The schema uses float32 for price/stat columns
(sufficient for gold prices; reduces file size).

Assumptions
-----------
- ``market_observations_normalized.archetype_id`` may be NULL everywhere.
  The daily aggregation JOIN goes through ``items.archetype_id`` instead.
- Items with NULL archetype_id in the items table are excluded from all outputs.
  Their count is recorded in the quality report and manifest.
- An empty result (no normalised observations for a realm) produces no Parquet
  files for that realm, and the realm is skipped with a logged warning.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from wow_forecaster.config import AppConfig
from wow_forecaster.features.archetype_features import (
    compute_archetype_features,
    count_items_per_archetype,
    count_items_without_archetype,
    count_obs_per_archetype_realm,
    load_archetype_metadata,
)
from wow_forecaster.features.daily_agg import fetch_daily_agg
from wow_forecaster.features.event_features import (
    compute_event_features,
    load_archetype_impacts,
    load_known_events,
)
from wow_forecaster.features.lag_rolling import compute_lag_rolling_features
from wow_forecaster.features.quality import build_quality_report
from wow_forecaster.features.registry import (
    FEATURE_REGISTRY,
    FeatureSpec,
    feature_groups,
    feature_names,
    inference_feature_names,
    target_feature_names,
    training_feature_names,
)
from wow_forecaster.models.meta import RunMetadata

log = logging.getLogger(__name__)

# ── PyArrow type map ───────────────────────────────────────────────────────────

_PA_TYPE_MAP: dict[str, pa.DataType] = {
    "int32":   pa.int32(),
    "float32": pa.float32(),
    "bool":    pa.bool_(),
    "utf8":    pa.string(),
    "date32":  pa.date32(),
}


def _pa_field(spec: FeatureSpec) -> pa.Field:
    pa_type = _PA_TYPE_MAP.get(spec.pa_type)
    if pa_type is None:
        raise ValueError(f"Unknown pa_type '{spec.pa_type}' for feature '{spec.name}'.")
    return pa.field(spec.name, pa_type, nullable=spec.is_nullable)


def build_parquet_schema(include_targets: bool = True) -> pa.Schema:
    """Build the PyArrow schema from FEATURE_REGISTRY.

    Args:
        include_targets: If False, target group columns are excluded (for
                         the inference Parquet).

    Returns:
        A ``pa.Schema`` with fields in registry order.
    """
    fields = [
        _pa_field(spec)
        for spec in FEATURE_REGISTRY
        if include_targets or not spec.is_target
    ]
    return pa.schema(fields)


# ── Parquet assembly ───────────────────────────────────────────────────────────

def rows_to_parquet_table(
    rows: list[dict[str, Any]],
    schema: pa.Schema,
) -> pa.Table:
    """Convert a list of feature dicts to a PyArrow Table with the given schema.

    For each field in the schema:
    - Extracts the column values from the rows.
    - Converts Python ``datetime.date`` objects for ``date32`` fields.
    - Relies on PyArrow's native None handling for nullable columns.

    Args:
        rows:   Feature rows; each dict must contain all field names in schema.
        schema: PyArrow schema (from ``build_parquet_schema()``).

    Returns:
        A ``pa.Table`` ready for ``pq.write_table()``.
    """
    arrays: dict[str, pa.Array] = {}
    for field in schema:
        values = [r.get(field.name) for r in rows]
        if field.type == pa.date32():
            # Ensure values are datetime.date, not str or datetime.
            values = [
                v if isinstance(v, date) else date.fromisoformat(v) if isinstance(v, str) else None
                for v in values
            ]
        arrays[field.name] = pa.array(values, type=field.type)
    return pa.table(arrays, schema=schema)


def write_training_parquet(rows: list[dict[str, Any]], path: Path) -> int:
    """Write the training Parquet file (all 45 columns including targets).

    Returns the number of rows written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    schema = build_parquet_schema(include_targets=True)
    table = rows_to_parquet_table(rows, schema)
    pq.write_table(table, str(path), compression="snappy")
    log.info("Training Parquet written: %s (%d rows)", path.name, len(rows))
    return len(rows)


def write_inference_parquet(rows: list[dict[str, Any]], path: Path) -> int:
    """Write the inference Parquet file (42 columns, no target columns).

    Uses the last available row per (archetype_id, realm_slug) as the
    inference row — i.e., the most recent features available for prediction.

    Returns the number of rows written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    # Keep only the most recent date per (archetype_id, realm_slug).
    latest: dict[tuple, dict[str, Any]] = {}
    for r in rows:
        key = (r.get("archetype_id"), r.get("realm_slug"))
        obs_date = r.get("obs_date")
        if obs_date is None:
            continue
        existing = latest.get(key)
        if existing is None or obs_date > existing["obs_date"]:
            latest[key] = r

    inference_rows = list(latest.values())
    schema = build_parquet_schema(include_targets=False)
    table = rows_to_parquet_table(inference_rows, schema)
    pq.write_table(table, str(path), compression="snappy")
    log.info("Inference Parquet written: %s (%d rows)", path.name, len(inference_rows))
    return len(inference_rows)


# ── Output path helpers ────────────────────────────────────────────────────────

def make_output_paths(
    processed_dir: str,
    realm_slug: str,
    start_date: date,
    end_date: date,
) -> dict[str, Path]:
    """Build deterministic output file paths for one realm.

    Returns a dict with keys: ``training``, ``inference``, ``manifest``.
    """
    base = Path(processed_dir) / "features"
    slug = realm_slug.replace("-", "_")
    today = date.today().isoformat()
    return {
        "training":  base / "training"  / f"train_{slug}_{start_date}_{end_date}.parquet",
        "inference": base / "inference" / f"inference_{slug}_{today}.parquet",
        "manifest":  base / "manifests" / f"manifest_{slug}_{today}.json",
    }


def _hash_file(path: Path) -> str:
    """Return the SHA-256 hex digest of a file's contents."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# ── Manifest ───────────────────────────────────────────────────────────────────

def build_manifest(
    realm_slug: str,
    start_date: date,
    end_date: date,
    run_slug: str,
    training_path: Path,
    inference_path: Path,
    training_rows: int,
    inference_rows: int,
    quality: Any,   # DataQualityReport
    config: AppConfig,
) -> dict[str, Any]:
    """Build the manifest dict for a single realm dataset build."""
    groups = feature_groups()
    feature_cols: dict[str, list[str]] = {g: feature_names(group=g) for g in groups}

    manifest: dict[str, Any] = {
        "schema_version": "1.0",
        "built_at":   datetime.now(tz=timezone.utc).isoformat(),
        "run_slug":   run_slug,
        "realm_slug": realm_slug,
        "date_range": {
            "start": start_date.isoformat(),
            "end":   end_date.isoformat(),
        },
        "files": {
            "training": {
                "path":             str(training_path),
                "sha256":           _hash_file(training_path)  if training_path.exists() else None,
                "rows":             training_rows,
                "includes_targets": True,
                "compression":      "snappy",
            },
            "inference": {
                "path":             str(inference_path),
                "sha256":           _hash_file(inference_path) if inference_path.exists() else None,
                "rows":             inference_rows,
                "includes_targets": False,
                "compression":      "snappy",
            },
        },
        "feature_columns": feature_cols,
        "quality": {
            "total_rows":                  quality.total_rows,
            "total_archetypes":            quality.total_archetypes,
            "duplicate_key_count":         quality.duplicate_key_count,
            "leakage_warnings":            quality.leakage_warnings,
            "volume_proxy_pct":            round(quality.volume_proxy_pct, 4),
            "cold_start_pct":              round(quality.cold_start_pct, 4),
            "items_excluded_no_archetype": quality.items_excluded_no_archetype,
            "is_clean":                    quality.is_clean,
        },
        "config_snapshot": {
            "features":   config.features.model_dump(),
            "expansions": config.expansions.model_dump(),
            "realms":     config.realms.model_dump(),
        },
    }
    return manifest


def write_manifest(manifest: dict[str, Any], path: Path) -> None:
    """Write the manifest dict as pretty-printed JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, default=str)
    log.info("Manifest written: %s", path.name)


# ── Temporal features (inline) ─────────────────────────────────────────────────

def _add_temporal_features(
    rows: list[dict[str, Any]],
    expansion_launch_date: date | None,
) -> list[dict[str, Any]]:
    """Add day_of_week, day_of_month, week_of_year, days_since_expansion."""
    result: list[dict[str, Any]] = []
    for row in rows:
        d: date = row["obs_date"]
        iso = d.isocalendar()
        days_since = (d - expansion_launch_date).days if expansion_launch_date else None
        result.append({
            **row,
            "day_of_week":          iso[2],           # 1=Mon … 7=Sun
            "day_of_month":         d.day,
            "week_of_year":         iso[1],
            "days_since_expansion": days_since,
        })
    return result


def _find_expansion_launch(events: Any) -> date | None:
    """Return the start_date of the EXPANSION_LAUNCH event, or None."""
    from wow_forecaster.taxonomy.event_taxonomy import EventType
    for e in events:
        if e.event_type == EventType.EXPANSION_LAUNCH:
            return e.start_date
    return None


# ── Main orchestrator ──────────────────────────────────────────────────────────

def build_datasets(
    conn: sqlite3.Connection,
    config: AppConfig,
    run: RunMetadata,
    realm_slugs: list[str],
    start_date: date,
    end_date: date,
    build_training: bool = True,
    build_inference: bool = True,
) -> int:
    """Build training + inference Parquet datasets and manifest for each realm.

    This is the entry point called by ``FeatureBuildStage._execute()``.

    Args:
        conn:           Open SQLite connection.
        config:         Full application config.
        run:            Current RunMetadata (for run_slug in manifest).
        realm_slugs:    Realms to build datasets for.
        start_date:     Earliest date in the training window.
        end_date:       Latest date in the training window (today for live runs).
        build_training: Write training Parquet if True (default True).
        build_inference: Write inference Parquet if True (default True).

    Returns:
        Total rows written across all realms and files (for RunMetadata.rows_processed).
    """
    cfg_feat = config.features
    cfg_exp  = config.expansions

    # ── Load shared data once (not per realm, not per row) ────────────────────
    events  = load_known_events(conn)
    impacts = load_archetype_impacts(conn)
    arch_meta = load_archetype_metadata(conn, cfg_exp.active, cfg_exp.transfer_target)
    items_excluded = count_items_without_archetype(conn)
    expansion_launch = _find_expansion_launch(events)

    total_rows = 0

    for realm_slug in realm_slugs:
        log.info("Building features for realm=%s  window=%s → %s", realm_slug, start_date, end_date)

        # Step 1: Daily aggregation.
        agg_rows = fetch_daily_agg(conn, realm_slug, start_date, end_date)
        if not agg_rows:
            log.warning("No normalised observations for realm=%s — skipping.", realm_slug)
            continue

        # Step 2: Lag / rolling / momentum / targets.
        feature_rows = compute_lag_rolling_features(agg_rows, cfg_feat)

        # Step 3–4: Event features (per archetype group).
        groups_by_arch: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for r in feature_rows:
            groups_by_arch[r["archetype_id"]].append(r)

        event_rows: list[dict[str, Any]] = []
        for arch_id, arch_rows in groups_by_arch.items():
            event_rows.extend(
                compute_event_features(arch_rows, events, impacts, arch_id)
            )

        # Step 5–6: Archetype / transfer features.
        cold_start_counts = count_obs_per_archetype_realm(
            conn, realm_slug, cfg_exp.transfer_target
        )
        item_counts = count_items_per_archetype(conn, realm_slug)

        arch_rows_out: list[dict[str, Any]] = []
        # Re-group event_rows by archetype_id.
        event_by_arch: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for r in event_rows:
            event_by_arch[r["archetype_id"]].append(r)

        for arch_id, arch_rows in event_by_arch.items():
            arch_rows_out.extend(
                compute_archetype_features(
                    arch_rows,
                    arch_id,
                    realm_slug,
                    arch_meta,
                    cold_start_counts,
                    item_counts,
                    cfg_feat.cold_start_threshold,
                    cfg_exp.transfer_target,
                )
            )

        # Step 7: Temporal features (pure date arithmetic).
        final_rows = _add_temporal_features(arch_rows_out, expansion_launch)

        # Sort output by (archetype_id, obs_date) for deterministic output.
        final_rows.sort(key=lambda r: (r["archetype_id"], r["obs_date"]))

        # Step 8: Quality report.
        quality = build_quality_report(final_rows, items_excluded=items_excluded)
        if not quality.is_clean:
            log.warning(
                "Quality issues for realm=%s: %d duplicates, %d leakage warnings",
                realm_slug, quality.duplicate_key_count, len(quality.leakage_warnings),
            )

        # Step 9: Write Parquet files.
        paths = make_output_paths(config.data.processed_dir, realm_slug, start_date, end_date)
        train_count = 0
        infer_count = 0

        if build_training:
            train_count = write_training_parquet(final_rows, paths["training"])
            total_rows += train_count

        if build_inference:
            infer_count = write_inference_parquet(final_rows, paths["inference"])

        # Step 10: Manifest.
        manifest = build_manifest(
            realm_slug=realm_slug,
            start_date=start_date,
            end_date=end_date,
            run_slug=run.run_slug,
            training_path=paths["training"],
            inference_path=paths["inference"],
            training_rows=train_count,
            inference_rows=infer_count,
            quality=quality,
            config=config,
        )
        write_manifest(manifest, paths["manifest"])

        log.info(
            "realm=%s done | training_rows=%d  inference_rows=%d  is_clean=%s",
            realm_slug, train_count, infer_count, quality.is_clean,
        )

    return total_rows
