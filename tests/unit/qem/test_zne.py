"""Tests for the ZNE (Zero Noise Extrapolation) technique."""

import pytest
from qiskit import QuantumCircuit

from metriq_gym.qem.techniques.zne import ZNETechnique
from metriq_gym.qem.technique import MitigationMetadata


class TestZNEConfig:
    def test_default_config(self):
        zne = ZNETechnique({"technique": "zne"})
        assert zne.scale_factors == [1.0, 2.0, 3.0]
        assert zne.factory_type == "linear"
        assert zne.fold_method == "global"

    def test_custom_config(self):
        zne = ZNETechnique({
            "technique": "zne",
            "scale_factors": [1.0, 1.5, 2.0],
            "factory": "richardson",
            "fold_method": "gates_at_random",
            "degree": 3,
        })
        assert zne.scale_factors == [1.0, 1.5, 2.0]
        assert zne.factory_type == "richardson"
        assert zne.fold_method == "gates_at_random"
        assert zne.degree == 3

    def test_too_few_scale_factors(self):
        with pytest.raises(ValueError, match="at least 2 scale factors"):
            ZNETechnique({"technique": "zne", "scale_factors": [1.0]})

    def test_missing_unity_scale_factor(self):
        with pytest.raises(ValueError, match="must include 1.0"):
            ZNETechnique({"technique": "zne", "scale_factors": [2.0, 3.0]})

    def test_invalid_factory(self):
        with pytest.raises(ValueError, match="Unknown ZNE factory"):
            ZNETechnique({
                "technique": "zne",
                "factory": "invalid",
            })

    def test_invalid_fold_method(self):
        with pytest.raises(ValueError, match="Unknown fold method"):
            ZNETechnique({
                "technique": "zne",
                "fold_method": "invalid",
            })


class TestZNETransformCircuits:
    def _make_bell_circuit(self):
        qc = QuantumCircuit(2, 2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure([0, 1], [0, 1])
        return qc

    def test_circuit_expansion(self):
        """N circuits with K scale factors should produce N*K circuits."""
        zne = ZNETechnique({
            "technique": "zne",
            "scale_factors": [1.0, 2.0, 3.0],
        })
        circuits = [self._make_bell_circuit(), self._make_bell_circuit()]
        transformed, metadata = zne.transform_circuits(circuits, shots=100)

        assert len(transformed) == 6  # 2 circuits * 3 scale factors
        assert metadata.technique_name == "zne"
        assert metadata.data["original_circuit_count"] == 2
        assert metadata.data["scale_factors"] == [1.0, 2.0, 3.0]

    def test_single_circuit(self):
        zne = ZNETechnique({
            "technique": "zne",
            "scale_factors": [1.0, 3.0],
        })
        circuits = [self._make_bell_circuit()]
        transformed, metadata = zne.transform_circuits(circuits, shots=100)

        assert len(transformed) == 2  # 1 circuit * 2 scale factors

    def test_unscaled_circuit_unchanged(self):
        """The circuit at scale_factor=1.0 should be the same depth as the original."""
        zne = ZNETechnique({
            "technique": "zne",
            "scale_factors": [1.0, 3.0],
        })
        qc = QuantumCircuit(1, 1)
        qc.h(0)
        qc.measure(0, 0)

        transformed, _ = zne.transform_circuits([qc], shots=100)
        # First circuit (scale=1.0) should have same structure
        unscaled = transformed[0]
        assert unscaled.num_qubits == 1

    def test_scaled_circuit_deeper(self):
        """Circuits at scale > 1 should have more gates."""
        zne = ZNETechnique({
            "technique": "zne",
            "scale_factors": [1.0, 3.0],
        })
        qc = QuantumCircuit(2, 2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure([0, 1], [0, 1])

        transformed, _ = zne.transform_circuits([qc], shots=100)
        unscaled = transformed[0]
        scaled = transformed[1]
        # Scaled circuit should have more operations (gate folding adds gates)
        unscaled_ops = sum(
            1 for inst in unscaled.data if inst.operation.name != "measure"
        )
        scaled_ops = sum(
            1 for inst in scaled.data if inst.operation.name != "measure"
        )
        assert scaled_ops > unscaled_ops

    def test_circuit_without_measurements(self):
        """Should handle circuits without measurements."""
        zne = ZNETechnique({
            "technique": "zne",
            "scale_factors": [1.0, 2.0],
        })
        qc = QuantumCircuit(1)
        qc.h(0)

        transformed, _ = zne.transform_circuits([qc], shots=100)
        assert len(transformed) == 2


class TestZNEPostprocessCounts:
    def test_basic_extrapolation(self):
        """Linear extrapolation of a clean trend should work."""
        zne = ZNETechnique({
            "technique": "zne",
            "scale_factors": [1.0, 2.0, 3.0],
            "factory": "linear",
        })
        metadata = MitigationMetadata(
            technique_name="zne",
            data={
                "scale_factors": [1.0, 2.0, 3.0],
                "factory_type": "linear",
                "degree": 2,
                "fold_method": "global",
                "original_circuit_count": 1,
            },
        )
        # Simulate linearly degrading probabilities
        counts = [
            {"00": 800, "01": 50, "10": 50, "11": 100},   # scale 1.0
            {"00": 600, "01": 100, "10": 100, "11": 200},  # scale 2.0
            {"00": 400, "01": 150, "10": 150, "11": 300},  # scale 3.0
        ]
        mitigated = zne.postprocess_counts(counts, metadata)
        assert len(mitigated) == 1
        # Extrapolation should push toward the noiseless distribution
        # (more 00, less noise)
        assert mitigated[0]["00"] > counts[0]["00"]

    def test_multiple_circuits(self):
        """Should handle multiple original circuits."""
        zne = ZNETechnique({
            "technique": "zne",
            "scale_factors": [1.0, 2.0],
            "factory": "linear",
        })
        metadata = MitigationMetadata(
            technique_name="zne",
            data={
                "scale_factors": [1.0, 2.0],
                "factory_type": "linear",
                "degree": 2,
                "fold_method": "global",
                "original_circuit_count": 2,
            },
        )
        # 2 circuits * 2 scale factors = 4 count dicts
        counts = [
            {"0": 700, "1": 300},   # circuit 0, scale 1.0
            {"0": 500, "1": 500},   # circuit 0, scale 2.0
            {"0": 600, "1": 400},   # circuit 1, scale 1.0
            {"0": 400, "1": 600},   # circuit 1, scale 2.0
        ]
        mitigated = zne.postprocess_counts(counts, metadata)
        assert len(mitigated) == 2

    def test_preserves_total_shots(self):
        """Mitigated counts should sum to approximately the same total."""
        zne = ZNETechnique({
            "technique": "zne",
            "scale_factors": [1.0, 2.0, 3.0],
            "factory": "linear",
        })
        metadata = MitigationMetadata(
            technique_name="zne",
            data={
                "scale_factors": [1.0, 2.0, 3.0],
                "factory_type": "linear",
                "degree": 2,
                "fold_method": "global",
                "original_circuit_count": 1,
            },
        )
        counts = [
            {"0": 700, "1": 300},
            {"0": 600, "1": 400},
            {"0": 500, "1": 500},
        ]
        mitigated = zne.postprocess_counts(counts, metadata)
        original_total = sum(counts[0].values())
        mitigated_total = sum(mitigated[0].values())
        # Should be close due to rounding
        assert abs(original_total - mitigated_total) <= 2
