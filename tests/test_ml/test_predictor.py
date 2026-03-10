"""Integration tests for run_inference() — cold-start prediction blending."""

from __future__ import annotations

import tempfile
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from wow_forecaster.ml.predictor import run_inference
from wow_forecaster.models.forecast import ForecastOutput
from wow_forecaster.models.meta import RunMetadata


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_config(confidence_pct: float = 0.80) -> MagicMock:
    cfg = MagicMock()
    cfg.forecast.confidence_pct = confidence_pct
    return cfg


def _make_run(run_id: int = 1) -> RunMetadata:
    from datetime import datetime, timezone
    import uuid
    run = RunMetadata(
        run_slug=str(uuid.uuid4()),
        pipeline_stage="forecast",
        realm_slug="us",
        config_snapshot={},
        started_at=datetime.now(tz=timezone.utc),
    )
    run.run_id = run_id
    return run


def _make_forecaster(prediction: float = 100.0) -> MagicMock:
    """Return a mock LightGBMForecaster that always predicts ``prediction``."""
    fc = MagicMock()
    fc.MODEL_VERSION = "v0.5.0"
    # predict() takes encoded_rows list, returns list of floats
    fc.predict.side_effect = lambda rows: [prediction] * len(rows)
    return fc


def _write_parquet(rows: list[dict], path: Path) -> None:
    """Write a list of dicts to a Parquet file using pyarrow."""
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, str(path))


def _archetype_row(
    archetype_id: int = 42,
    price: float = 80.0,
    is_cold_start: bool = False,
    has_transfer_mapping: bool = False,
    transfer_confidence: float | None = None,
    rolling_std: float | None = 5.0,
) -> dict:
    """Build a minimal inference-Parquet row for an archetype."""
    return {
        "archetype_id": archetype_id,
        "realm_slug": "us",
        "price_mean": price,
        "price_gold": price,
        "price_roll_std_7d": rolling_std,
        "is_cold_start": is_cold_start,
        "has_transfer_mapping": has_transfer_mapping,
        "transfer_confidence": transfer_confidence,
        # Required feature cols with minimal values
        "archetype_category_enc": 6,
        "event_severity_enc": 0,
        "day_of_week": 1,
        "week_of_year": 10,
        "is_cold_start_int": int(is_cold_start),
        "is_transferable_int": 1,
        "event_active_int": 0,
        "has_transfer_mapping_int": int(has_transfer_mapping),
        "price_roll_mean_7d": price,
        "price_roll_mean_28d": price,
        "quantity_sum_7d": 200.0,
        "quantity_roll_mean_7d": 28.0,
        "volume_score": 0.4,
        "obs_count_7d": 7,
        "obs_count_28d": 28,
        "archetype_obs_count": 100,
        "price_mean_7d_lag": price,
        "price_mean_14d_lag": price,
        "price_change_7d": 0.0,
        "price_change_14d": 0.0,
        "volatility_7d": 0.05,
        "volatility_28d": 0.05,
        "event_impact_mean": 0.0,
        "event_count_active": 0,
        "event_max_severity": 0,
        "event_days_since_last": 30,
        "event_days_until_next": 30,
        "event_overlap_count": 0,
        "event_impact_sum": 0.0,
        "event_impact_max": 0.0,
        "momentum_7d": 0.0,
        "momentum_28d": 0.0,
        "price_percentile_7d": 50.0,
        "price_percentile_28d": 50.0,
    }


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestRunInferenceBlending:
    """Tests verifying cold-start blending in run_inference()."""

    def test_warm_archetype_not_blended(self, tmp_path):
        """Warm (non-cold-start) archetypes are never blended."""
        parquet_path = tmp_path / "inference_us_test.parquet"
        _write_parquet([_archetype_row(archetype_id=10, price=80.0, is_cold_start=False)], parquet_path)

        forecaster_prediction = 120.0
        fc = _make_forecaster(forecaster_prediction)
        blend_data = {10: (50.0, 0.5)}  # would blend to 85 if applied

        outputs = run_inference(
            config=_make_config(),
            run=_make_run(),
            forecasters={7: fc},
            inference_parquet_path=parquet_path,
            realm_slug="us",
            cold_start_blend=blend_data,
        )

        assert len(outputs) == 1
        # Warm archetype — blending must NOT be applied
        assert outputs[0].predicted_price_gold == pytest.approx(forecaster_prediction, abs=0.01)

    def test_cold_start_with_blend_data_produces_blended_prediction(self, tmp_path):
        """Cold-start archetype WITH blend data gets a blended prediction."""
        parquet_path = tmp_path / "inference_us_test.parquet"
        cold_row = _archetype_row(
            archetype_id=99,
            price=80.0,
            is_cold_start=True,
            has_transfer_mapping=True,
            transfer_confidence=0.7,
        )
        _write_parquet([cold_row], parquet_path)

        model_pred = 100.0
        source_price = 50.0
        confidence = 0.7
        # expected blend = 0.7*100 + 0.3*50 = 70 + 15 = 85
        expected_blended = confidence * model_pred + (1.0 - confidence) * source_price

        fc = _make_forecaster(model_pred)
        blend_data = {99: (source_price, confidence)}

        outputs = run_inference(
            config=_make_config(),
            run=_make_run(),
            forecasters={7: fc},
            inference_parquet_path=parquet_path,
            realm_slug="us",
            cold_start_blend=blend_data,
        )

        assert len(outputs) == 1
        assert outputs[0].predicted_price_gold == pytest.approx(expected_blended, abs=0.01)

    def test_cold_start_without_blend_data_uses_raw_prediction(self, tmp_path):
        """Cold-start archetype WITHOUT a matching blend_data entry uses raw prediction."""
        parquet_path = tmp_path / "inference_us_test.parquet"
        cold_row = _archetype_row(
            archetype_id=55,
            price=80.0,
            is_cold_start=True,
            has_transfer_mapping=False,
            transfer_confidence=None,
        )
        _write_parquet([cold_row], parquet_path)

        model_pred = 100.0
        fc = _make_forecaster(model_pred)

        outputs = run_inference(
            config=_make_config(),
            run=_make_run(),
            forecasters={7: fc},
            inference_parquet_path=parquet_path,
            realm_slug="us",
            cold_start_blend=None,
        )

        assert len(outputs) == 1
        assert outputs[0].predicted_price_gold == pytest.approx(model_pred, abs=0.01)

    def test_cold_start_blend_none_uses_raw_prediction(self, tmp_path):
        """Passing cold_start_blend=None for a cold-start archetype uses raw prediction."""
        parquet_path = tmp_path / "inference_us_test.parquet"
        cold_row = _archetype_row(
            archetype_id=77,
            price=80.0,
            is_cold_start=True,
            has_transfer_mapping=True,
            transfer_confidence=0.8,
        )
        _write_parquet([cold_row], parquet_path)

        model_pred = 100.0
        fc = _make_forecaster(model_pred)

        outputs = run_inference(
            config=_make_config(),
            run=_make_run(),
            forecasters={7: fc},
            inference_parquet_path=parquet_path,
            realm_slug="us",
            cold_start_blend=None,  # explicitly no blend data
        )

        assert len(outputs) == 1
        assert outputs[0].predicted_price_gold == pytest.approx(model_pred, abs=0.01)

    def test_blending_applied_before_ci_computation(self, tmp_path):
        """CI is centred on the blended value, not the raw model prediction."""
        parquet_path = tmp_path / "inference_us_test.parquet"
        cold_row = _archetype_row(
            archetype_id=33,
            price=80.0,
            is_cold_start=True,
            has_transfer_mapping=True,
            transfer_confidence=0.5,
            rolling_std=None,  # forces _DEFAULT_UNCERTAINTY_FRAC path
        )
        _write_parquet([cold_row], parquet_path)

        model_pred = 200.0
        source_price = 100.0
        confidence = 0.5
        expected_blended = 0.5 * model_pred + 0.5 * source_price  # = 150.0

        fc = _make_forecaster(model_pred)
        blend_data = {33: (source_price, confidence)}

        outputs = run_inference(
            config=_make_config(confidence_pct=0.80),
            run=_make_run(),
            forecasters={7: fc},
            inference_parquet_path=parquet_path,
            realm_slug="us",
            cold_start_blend=blend_data,
        )

        assert len(outputs) == 1
        out = outputs[0]
        assert out.predicted_price_gold == pytest.approx(expected_blended, abs=0.01)
        # CI must be centred on blended value (lower <= predicted <= upper)
        assert out.confidence_lower <= out.predicted_price_gold
        assert out.confidence_upper >= out.predicted_price_gold

    def test_model_slug_transfer_suffix_for_blended_cold_start(self, tmp_path):
        """Cold-start archetype with transfer mapping gets _transfer suffix in model_slug."""
        parquet_path = tmp_path / "inference_us_test.parquet"
        cold_row = _archetype_row(
            archetype_id=44,
            price=80.0,
            is_cold_start=True,
            has_transfer_mapping=True,
            transfer_confidence=0.8,
        )
        _write_parquet([cold_row], parquet_path)

        fc = _make_forecaster(100.0)
        blend_data = {44: (60.0, 0.8)}

        outputs = run_inference(
            config=_make_config(),
            run=_make_run(),
            forecasters={7: fc},
            inference_parquet_path=parquet_path,
            realm_slug="us",
            cold_start_blend=blend_data,
        )

        assert len(outputs) == 1
        assert outputs[0].model_slug.endswith("_transfer")

    def test_multiple_archetypes_only_cold_start_blended(self, tmp_path):
        """In a mixed batch, only cold-start archetypes with blend entries get blended."""
        parquet_path = tmp_path / "inference_us_test.parquet"
        warm_row = _archetype_row(archetype_id=1, price=100.0, is_cold_start=False)
        cold_row = _archetype_row(
            archetype_id=2,
            price=100.0,
            is_cold_start=True,
            has_transfer_mapping=True,
            transfer_confidence=0.6,
        )
        _write_parquet([warm_row, cold_row], parquet_path)

        model_pred = 100.0
        source_price = 40.0
        confidence = 0.6
        expected_blended = confidence * model_pred + (1.0 - confidence) * source_price

        fc = _make_forecaster(model_pred)
        blend_data = {2: (source_price, confidence)}

        outputs = run_inference(
            config=_make_config(),
            run=_make_run(),
            forecasters={7: fc},
            inference_parquet_path=parquet_path,
            realm_slug="us",
            cold_start_blend=blend_data,
        )

        assert len(outputs) == 2
        by_arch = {o.archetype_id: o for o in outputs}

        # Warm archetype: raw prediction
        assert by_arch[1].predicted_price_gold == pytest.approx(model_pred, abs=0.01)
        # Cold-start archetype: blended prediction
        assert by_arch[2].predicted_price_gold == pytest.approx(expected_blended, abs=0.01)
