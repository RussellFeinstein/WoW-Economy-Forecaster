"""
Dashboard data loader.

All functions are decorated with ``@st.cache_data`` so Streamlit only
re-reads files when they change on disk.  This avoids re-reading on every
widget interaction.

If ``wow_forecaster`` is installed (``pip install -e .``), the reader module
is used for file discovery.  If not, a lightweight fallback implementation
scans the output directories directly.

Functions return ``None`` (rather than raising) when no files are found so
every Streamlit view can show a graceful "no data yet" message.
"""

from __future__ import annotations

import csv
import json
import os
import sqlite3
from pathlib import Path

try:
    import streamlit as st
    _CACHE = st.cache_data
except ImportError:
    # Allow importing outside Streamlit context (e.g. tests).
    def _CACHE(fn):  # type: ignore[misc]
        return fn


# ── Internal helpers ─────────────────────────────────────────────────────────

def _find_latest(directory: Path, pattern: str) -> Path | None:
    """Return the most-recently-modified file matching ``pattern``."""
    if not directory.exists():
        return None
    matches = list(directory.glob(pattern))
    return max(matches, key=lambda p: p.stat().st_mtime) if matches else None


def _load_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _load_csv(path: Path) -> list[dict]:
    rows: list[dict] = []
    try:
        with path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                rows.append(dict(row))
    except Exception:
        pass
    return rows


# ── Loaders ──────────────────────────────────────────────────────────────────


@_CACHE(ttl=300)
def load_recommendations(realm: str, output_dir: str) -> dict | None:
    """Load the latest recommendations JSON for *realm*.

    TTL: 5 minutes (re-reads after a new pipeline run).
    """
    path = _find_latest(Path(output_dir), f"recommendations_{realm}_*.json")
    return _load_json(path) if path else None


@_CACHE(ttl=300)
def load_forecasts(realm: str, output_dir: str) -> list[dict]:
    """Load the latest forecast CSV rows for *realm*.

    Returns an empty list (not None) so callers can call ``len()`` safely.
    """
    path = _find_latest(Path(output_dir), f"forecast_{realm}_*.csv")
    return _load_csv(path) if path else []


@_CACHE(ttl=300)
def load_drift(realm: str, output_dir: str) -> dict | None:
    """Load the latest drift status JSON for *realm*."""
    path = _find_latest(Path(output_dir), f"drift_status_{realm}_*.json")
    return _load_json(path) if path else None


@_CACHE(ttl=300)
def load_health(realm: str, output_dir: str) -> dict | None:
    """Load the latest model health JSON for *realm*."""
    path = _find_latest(Path(output_dir), f"model_health_{realm}_*.json")
    return _load_json(path) if path else None


@_CACHE(ttl=300)
def load_provenance(realm: str, output_dir: str) -> dict | None:
    """Load the latest provenance JSON for *realm*."""
    path = _find_latest(Path(output_dir), f"provenance_{realm}_*.json")
    return _load_json(path) if path else None


@_CACHE(ttl=300)
def load_historical_prices(
    db_path: str,
    realm_slug: str,
    archetype_id: str,
    days: int = 90,
) -> list[dict]:
    """Query normalized historical daily prices for one archetype from SQLite.

    Returns a list of ``{"date": str, "avg_price_gold": float}`` rows
    sorted by date ascending.  Used for the forecast vs actual chart.
    """
    rows: list[dict] = []
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cur = conn.execute(
            """
            SELECT date(observed_at) AS obs_date,
                   AVG(price_gold)   AS avg_price_gold
            FROM   market_observations_normalized n
            JOIN   items i ON n.item_id = i.item_id
            WHERE  i.archetype_id  = ?
              AND  n.realm_slug    = ?
              AND  n.is_outlier    = 0
              AND  date(observed_at) >= date('now', ? || ' days')
            GROUP  BY obs_date
            ORDER  BY obs_date
            """,
            (archetype_id, realm_slug, f"-{days}"),
        )
        for row in cur.fetchall():
            rows.append({"date": row["obs_date"], "avg_price_gold": row["avg_price_gold"]})
        conn.close()
    except Exception:
        pass
    return rows


@_CACHE(ttl=600)
def load_events(db_path: str, days_ahead: int = 30) -> list[dict]:
    """Return upcoming and active WoW events for event annotation on charts."""
    rows: list[dict] = []
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cur = conn.execute(
            """
            SELECT slug, display_name, start_date, end_date,
                   event_type, severity
            FROM   wow_events
            WHERE  date(start_date) <= date('now', ? || ' days')
              AND  (end_date IS NULL OR date(end_date) >= date('now', '-7 days'))
            ORDER  BY start_date
            """,
            (days_ahead,),
        )
        for row in cur.fetchall():
            rows.append(dict(row))
        conn.close()
    except Exception:
        pass
    return rows


def report_file_age_hours(realm: str, pattern: str, output_dir: str) -> float | None:
    """Return file age in hours for the most-recently-modified matching file."""
    path = _find_latest(Path(output_dir), pattern.replace("{realm}", realm))
    if path is None:
        return None
    return (os.path.getmtime.__func__ if hasattr(os.path.getmtime, "__func__")
            else os.path.getmtime)(str(path))


def file_age_hours(realm: str, pattern: str, output_dir: str) -> float | None:
    """Return age in hours of the most-recently-modified file matching *pattern*.

    Pattern may contain ``{realm}`` which is substituted.
    """
    import time as _time
    path = _find_latest(Path(output_dir), pattern.replace("{realm}", realm))
    if path is None:
        return None
    return (_time.time() - os.path.getmtime(str(path))) / 3600.0
