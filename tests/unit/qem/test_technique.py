"""Tests for MitigationTechnique, MitigationMetadata, and MitigationPipeline."""

import pytest
from qiskit import QuantumCircuit

from metriq_gym.qem.technique import MitigationMetadata, MitigationTechnique
from metriq_gym.qem.pipeline import MitigationPipeline


class IdentityTechnique(MitigationTechnique):
    """A no-op technique for testing pipeline composition."""

    name = "identity"

    def __init__(self, config=None):
        super().__init__(config or {})
        self.transform_called = False
        self.postprocess_called = False

    def transform_circuits(self, circuits, shots):
        self.transform_called = True
        return circuits, MitigationMetadata(technique_name="identity", data={"tag": "id"})

    def postprocess_counts(self, counts, metadata):
        self.postprocess_called = True
        return counts


class DoublingTechnique(MitigationTechnique):
    """A technique that doubles each circuit for testing expansion."""

    name = "doubler"

    def __init__(self, config=None):
        super().__init__(config or {})

    def transform_circuits(self, circuits, shots):
        doubled = []
        for c in circuits:
            doubled.append(c.copy())
            doubled.append(c.copy())
        return doubled, MitigationMetadata(
            technique_name="doubler",
            data={"original_count": len(circuits)},
        )

    def postprocess_counts(self, counts, metadata):
        original_count = metadata.data["original_count"]
        # Average pairs of counts
        result = []
        for i in range(original_count):
            c1 = counts[2 * i]
            c2 = counts[2 * i + 1]
            all_keys = set(c1.keys()) | set(c2.keys())
            merged = {k: c1.get(k, 0) + c2.get(k, 0) for k in all_keys}
            result.append(merged)
        return result


class TestMitigationMetadata:
    def test_creation(self):
        md = MitigationMetadata(technique_name="test", data={"key": "value"})
        assert md.technique_name == "test"
        assert md.data == {"key": "value"}

    def test_default_data(self):
        md = MitigationMetadata(technique_name="test")
        assert md.data == {}


class TestMitigationPipeline:
    def test_empty_pipeline(self):
        pipeline = MitigationPipeline([])
        assert pipeline.is_empty

        qc = QuantumCircuit(1)
        qc.h(0)
        circuits, metadata = pipeline.transform_circuits([qc], 100)
        assert len(circuits) == 1
        assert metadata == []

        counts = [{"0": 50, "1": 50}]
        result = pipeline.postprocess_counts(counts, [])
        assert result == counts

    def test_single_technique(self):
        technique = IdentityTechnique()
        pipeline = MitigationPipeline([technique])
        assert not pipeline.is_empty

        qc = QuantumCircuit(1)
        circuits, metadata = pipeline.transform_circuits([qc], 100)
        assert technique.transform_called
        assert len(metadata) == 1
        assert metadata[0].technique_name == "identity"

        counts = [{"0": 50, "1": 50}]
        result = pipeline.postprocess_counts(counts, metadata)
        assert technique.postprocess_called
        assert result == counts

    def test_composition_order(self):
        """Transforms apply in order, post-processing in reverse."""
        call_order = []

        class OrderTracker(MitigationTechnique):
            name = "tracker"

            def __init__(self, tag):
                super().__init__({})
                self.tag = tag

            def transform_circuits(self, circuits, shots):
                call_order.append(f"transform_{self.tag}")
                return circuits, MitigationMetadata(
                    technique_name=self.tag, data={}
                )

            def postprocess_counts(self, counts, metadata):
                call_order.append(f"postprocess_{self.tag}")
                return counts

        pipeline = MitigationPipeline([OrderTracker("A"), OrderTracker("B")])
        qc = QuantumCircuit(1)
        circuits, metadata = pipeline.transform_circuits([qc], 100)
        pipeline.postprocess_counts([{"0": 100}], metadata)

        assert call_order == [
            "transform_A",
            "transform_B",
            "postprocess_B",
            "postprocess_A",
        ]

    def test_circuit_expansion_and_reduction(self):
        """Doubling technique doubles circuits, then merges counts."""
        pipeline = MitigationPipeline([DoublingTechnique()])

        qc = QuantumCircuit(1)
        circuits, metadata = pipeline.transform_circuits([qc], 100)
        assert len(circuits) == 2  # 1 circuit doubled

        counts = [{"0": 50, "1": 50}, {"0": 60, "1": 40}]
        result = pipeline.postprocess_counts(counts, metadata)
        assert len(result) == 1  # Back to 1
        assert result[0] == {"0": 110, "1": 90}
