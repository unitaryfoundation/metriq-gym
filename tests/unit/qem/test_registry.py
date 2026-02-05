"""Tests for the QEM technique registry."""

import pytest

from metriq_gym.qem.registry import (
    TECHNIQUE_REGISTRY,
    build_pipeline,
    register_technique,
    _ensure_techniques_loaded,
)
from metriq_gym.qem.technique import MitigationTechnique, MitigationMetadata


class TestRegistry:
    def test_zne_registered(self):
        """ZNE should be auto-registered when techniques are loaded."""
        _ensure_techniques_loaded()
        assert "zne" in TECHNIQUE_REGISTRY

    def test_register_technique_decorator(self):
        """The @register_technique decorator should add to TECHNIQUE_REGISTRY."""

        @register_technique("test_dummy")
        class DummyTechnique(MitigationTechnique):
            name = "test_dummy"

            def transform_circuits(self, circuits, shots):
                return circuits, MitigationMetadata(technique_name="test_dummy")

            def postprocess_counts(self, counts, metadata):
                return counts

        assert "test_dummy" in TECHNIQUE_REGISTRY
        assert TECHNIQUE_REGISTRY["test_dummy"] is DummyTechnique
        # Cleanup
        del TECHNIQUE_REGISTRY["test_dummy"]


class TestBuildPipeline:
    def test_build_pipeline_zne(self):
        pipeline = build_pipeline([
            {"technique": "zne", "scale_factors": [1.0, 2.0, 3.0]}
        ])
        assert len(pipeline.techniques) == 1
        assert pipeline.techniques[0].name == "zne"

    def test_build_pipeline_empty(self):
        pipeline = build_pipeline([])
        assert pipeline.is_empty
        assert len(pipeline.techniques) == 0

    def test_build_pipeline_unknown_technique(self):
        with pytest.raises(ValueError, match="Unknown mitigation technique"):
            build_pipeline([{"technique": "nonexistent"}])

    def test_build_pipeline_missing_technique_key(self):
        with pytest.raises(ValueError, match="missing 'technique' key"):
            build_pipeline([{"scale_factors": [1.0, 2.0]}])

    def test_build_pipeline_multiple_techniques(self):
        """Building a pipeline with multiple ZNE instances (for composition testing)."""
        pipeline = build_pipeline([
            {"technique": "zne", "scale_factors": [1.0, 2.0]},
            {"technique": "zne", "scale_factors": [1.0, 3.0]},
        ])
        assert len(pipeline.techniques) == 2
