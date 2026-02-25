"""Tests for pipeline stage abstract base and stub implementations."""

from __future__ import annotations

import pytest

from wow_forecaster.pipeline.base import PipelineStage
from wow_forecaster.pipeline.feature_build import FeatureBuildStage
from wow_forecaster.pipeline.forecast import ForecastStage
from wow_forecaster.pipeline.ingest import IngestStage
from wow_forecaster.pipeline.normalize import NormalizeStage
from wow_forecaster.pipeline.recommend import RecommendStage
from wow_forecaster.pipeline.train import TrainStage


class TestPipelineStageABC:
    def test_cannot_instantiate_base_directly(self):
        """PipelineStage is abstract and cannot be instantiated without _execute."""
        with pytest.raises(TypeError):
            PipelineStage(config=None)  # type: ignore

    def test_concrete_subclass_without_execute_raises(self):
        """A subclass that doesn't implement _execute should raise TypeError on construction."""
        class IncompleteStage(PipelineStage):
            stage_name = "ingest"
            # Missing _execute

        with pytest.raises(TypeError):
            IncompleteStage(config=None)  # type: ignore


class TestAllStubsHaveCorrectStageName:
    def test_ingest_stage_name(self):
        assert IngestStage.stage_name == "ingest"

    def test_normalize_stage_name(self):
        assert NormalizeStage.stage_name == "normalize"

    def test_feature_build_stage_name(self):
        assert FeatureBuildStage.stage_name == "feature_build"

    def test_train_stage_name(self):
        assert TrainStage.stage_name == "train"

    def test_forecast_stage_name(self):
        assert ForecastStage.stage_name == "forecast"

    def test_recommend_stage_name(self):
        assert RecommendStage.stage_name == "recommend"


class TestStubStagesRaiseNotImplemented:
    """All stub stages should raise NotImplementedError when _execute() is called directly."""

    @pytest.fixture
    def minimal_config(self):
        """Create a minimal AppConfig for testing."""
        from wow_forecaster.config import AppConfig
        return AppConfig()

    def test_ingest_execute_raises(self, minimal_config):
        stage = IngestStage(config=minimal_config)
        from wow_forecaster.models.meta import RunMetadata
        from wow_forecaster.utils.time_utils import utcnow
        run = RunMetadata(
            run_slug="test-slug",
            pipeline_stage="ingest",
            config_snapshot={},
            started_at=utcnow(),
        )
        with pytest.raises(NotImplementedError):
            stage._execute(run=run)

    def test_normalize_execute_raises(self, minimal_config):
        stage = NormalizeStage(config=minimal_config)
        from wow_forecaster.models.meta import RunMetadata
        from wow_forecaster.utils.time_utils import utcnow
        run = RunMetadata(
            run_slug="test-slug",
            pipeline_stage="normalize",
            config_snapshot={},
            started_at=utcnow(),
        )
        with pytest.raises(NotImplementedError):
            stage._execute(run=run)

    def test_feature_build_execute_raises(self, minimal_config):
        stage = FeatureBuildStage(config=minimal_config)
        from wow_forecaster.models.meta import RunMetadata
        from wow_forecaster.utils.time_utils import utcnow
        run = RunMetadata(
            run_slug="test-slug",
            pipeline_stage="feature_build",
            config_snapshot={},
            started_at=utcnow(),
        )
        with pytest.raises(NotImplementedError):
            stage._execute(run=run)

    def test_all_stage_names_are_valid_pipeline_stages(self, minimal_config):
        """All stage names must match VALID_PIPELINE_STAGES in RunMetadata."""
        from wow_forecaster.models.meta import VALID_PIPELINE_STAGES
        stages = [
            IngestStage, NormalizeStage, FeatureBuildStage,
            TrainStage, ForecastStage, RecommendStage,
        ]
        for stage_cls in stages:
            assert stage_cls.stage_name in VALID_PIPELINE_STAGES, (
                f"{stage_cls.__name__}.stage_name = '{stage_cls.stage_name}' "
                f"not in VALID_PIPELINE_STAGES = {VALID_PIPELINE_STAGES}"
            )
