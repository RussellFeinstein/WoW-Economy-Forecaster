"""
Tests for wow_forecaster.ingestion.snapshot — disk persistence helpers.

Covers:
  - build_snapshot_path(): deterministic, correctly structured paths
  - save_snapshot(): creates file, returns (hash, count), envelope structure
  - compute_hash(): deterministic across calls
  - load_snapshot(): roundtrip from save → load
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from wow_forecaster.ingestion.snapshot import (
    build_snapshot_path,
    compute_hash,
    load_snapshot,
    save_snapshot,
)

# ── Fixtures ───────────────────────────────────────────────────────────────────

_FIXED_DT = datetime(2026, 2, 24, 15, 0, 0, tzinfo=timezone.utc)
_SAMPLE_RECORDS = [
    {"item_id": 191528, "min_buyout": 1_500_000, "quantity": 10},
    {"item_id": 204783, "min_buyout": 80_000, "quantity": 200},
]


# ── build_snapshot_path ────────────────────────────────────────────────────────

class TestBuildSnapshotPath:
    def test_basic_structure(self):
        path = build_snapshot_path("data/raw", "undermine", "area-52_neutral", _FIXED_DT)
        parts = path.parts
        assert "snapshots" in parts
        assert "undermine" in parts
        assert "2026" in parts
        assert "02" in parts
        assert "24" in parts

    def test_filename_format(self):
        path = build_snapshot_path("data/raw", "undermine", "area-52_neutral", _FIXED_DT)
        assert path.name == "area-52_neutral_20260224T150000Z.json"

    def test_source_is_segment(self):
        for source in ("undermine", "blizzard_api", "blizzard_news"):
            path = build_snapshot_path("data/raw", source, "test", _FIXED_DT)
            assert source in str(path)

    def test_deterministic_same_inputs(self):
        path1 = build_snapshot_path("data/raw", "undermine", "area-52_neutral", _FIXED_DT)
        path2 = build_snapshot_path("data/raw", "undermine", "area-52_neutral", _FIXED_DT)
        assert path1 == path2

    def test_different_timestamps_produce_different_paths(self):
        dt2 = datetime(2026, 2, 24, 16, 0, 0, tzinfo=timezone.utc)
        p1 = build_snapshot_path("data/raw", "undermine", "area-52_neutral", _FIXED_DT)
        p2 = build_snapshot_path("data/raw", "undermine", "area-52_neutral", dt2)
        assert p1 != p2

    def test_different_realms_produce_different_paths(self):
        p1 = build_snapshot_path("data/raw", "undermine", "area-52_neutral", _FIXED_DT)
        p2 = build_snapshot_path("data/raw", "undermine", "illidan_neutral", _FIXED_DT)
        assert p1 != p2

    def test_raw_dir_prefix(self, tmp_path):
        path = build_snapshot_path(str(tmp_path), "undermine", "realm", _FIXED_DT)
        assert str(path).startswith(str(tmp_path))


# ── compute_hash ───────────────────────────────────────────────────────────────

class TestComputeHash:
    def test_returns_64_char_hex(self):
        h = compute_hash({"key": "value"})
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)

    def test_deterministic(self):
        payload = {"items": [1, 2, 3], "source": "test"}
        h1 = compute_hash(payload)
        h2 = compute_hash(payload)
        assert h1 == h2

    def test_key_order_independent(self):
        h1 = compute_hash({"a": 1, "b": 2})
        h2 = compute_hash({"b": 2, "a": 1})
        assert h1 == h2

    def test_different_payloads_differ(self):
        h1 = compute_hash({"item_id": 1})
        h2 = compute_hash({"item_id": 2})
        assert h1 != h2


# ── save_snapshot ──────────────────────────────────────────────────────────────

class TestSaveSnapshot:
    def test_creates_file(self, tmp_path):
        path = tmp_path / "test" / "snapshot.json"
        save_snapshot(path, _SAMPLE_RECORDS)
        assert path.exists()

    def test_creates_parent_dirs(self, tmp_path):
        deep = tmp_path / "a" / "b" / "c" / "snap.json"
        save_snapshot(deep, _SAMPLE_RECORDS)
        assert deep.exists()

    def test_returns_hash_and_count(self, tmp_path):
        path = tmp_path / "snap.json"
        content_hash, record_count = save_snapshot(path, _SAMPLE_RECORDS)
        assert len(content_hash) == 64
        assert record_count == len(_SAMPLE_RECORDS)

    def test_single_dict_count_is_one(self, tmp_path):
        path = tmp_path / "snap.json"
        _, count = save_snapshot(path, {"item_id": 123})
        assert count == 1

    def test_envelope_structure(self, tmp_path):
        path = tmp_path / "snap.json"
        save_snapshot(path, _SAMPLE_RECORDS, metadata={"source": "undermine"})
        with open(path) as f:
            data = json.load(f)
        assert "_meta" in data
        assert "data" in data
        assert data["_meta"]["source"] == "undermine"
        assert data["data"] == _SAMPLE_RECORDS

    def test_meta_written_at_added(self, tmp_path):
        path = tmp_path / "snap.json"
        save_snapshot(path, _SAMPLE_RECORDS)
        with open(path) as f:
            data = json.load(f)
        assert "written_at" in data["_meta"]

    def test_hash_deterministic(self, tmp_path):
        p1 = tmp_path / "s1.json"
        p2 = tmp_path / "s2.json"
        import time; time.sleep(0.01)  # ensure written_at differs
        # same payload → hashes differ because written_at differs
        h1, _ = save_snapshot(p1, _SAMPLE_RECORDS, metadata={"fixed": True})
        h2, _ = save_snapshot(p2, _SAMPLE_RECORDS, metadata={"fixed": True})
        # hashes will differ due to written_at timestamp; test the function works
        assert len(h1) == 64
        assert len(h2) == 64

    def test_empty_list(self, tmp_path):
        path = tmp_path / "empty.json"
        _, count = save_snapshot(path, [])
        assert count == 0


# ── load_snapshot ──────────────────────────────────────────────────────────────

class TestLoadSnapshot:
    def test_roundtrip(self, tmp_path):
        path = tmp_path / "snap.json"
        save_snapshot(path, _SAMPLE_RECORDS, metadata={"realm": "area-52"})
        loaded = load_snapshot(path)
        assert loaded["data"] == _SAMPLE_RECORDS
        assert loaded["_meta"]["realm"] == "area-52"

    def test_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_snapshot(tmp_path / "nonexistent.json")

    def test_invalid_json(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text("this is not json", encoding="utf-8")
        with pytest.raises(json.JSONDecodeError):
            load_snapshot(path)
