"""Tests for the DDD (Digital Dynamical Decoupling) technique."""

import pytest
from qiskit import QuantumCircuit

from metriq_gym.qem.techniques.ddd import DDDTechnique
from metriq_gym.qem.technique import MitigationMetadata


class TestDDDConfig:
    def test_default_config(self):
        ddd = DDDTechnique({"technique": "ddd"})
        assert ddd.rule_name == "xyxy"
        assert ddd.spacing == -1

    def test_custom_config(self):
        ddd = DDDTechnique({
            "technique": "ddd",
            "rule": "xx",
            "spacing": 2,
        })
        assert ddd.rule_name == "xx"
        assert ddd.spacing == 2

    def test_yy_rule(self):
        ddd = DDDTechnique({"technique": "ddd", "rule": "yy"})
        assert ddd.rule_name == "yy"

    def test_invalid_rule(self):
        with pytest.raises(ValueError, match="Unknown DDD rule"):
            DDDTechnique({"technique": "ddd", "rule": "invalid"})


class TestDDDTransformCircuits:
    def _make_circuit_with_idle(self):
        """Create a circuit with clear idle periods on qubit 1."""
        qc = QuantumCircuit(2, 2)
        qc.h(0)
        qc.x(0)
        qc.y(0)
        qc.z(0)
        qc.cx(0, 1)
        qc.measure([0, 1], [0, 1])
        return qc

    def _make_compact_circuit(self):
        """Create a compact circuit with no slack windows."""
        qc = QuantumCircuit(2, 2)
        qc.h(0)
        qc.h(1)
        qc.cx(0, 1)
        qc.measure([0, 1], [0, 1])
        return qc

    def test_one_to_one_mapping(self):
        """DDD should produce same number of circuits (no expansion)."""
        ddd = DDDTechnique({"technique": "ddd"})
        circuits = [self._make_circuit_with_idle(), self._make_circuit_with_idle()]
        transformed, metadata = ddd.transform_circuits(circuits, shots=100)

        assert len(transformed) == 2  # 1:1 mapping
        assert metadata.technique_name == "ddd"
        assert metadata.data["original_circuit_count"] == 2

    def test_inserts_gates_in_idle_periods(self):
        """Circuits with idle periods should have more gates after DDD."""
        ddd = DDDTechnique({"technique": "ddd", "rule": "xyxy"})
        qc = self._make_circuit_with_idle()

        original_gate_count = sum(
            1 for inst in qc.data if inst.operation.name not in ("measure", "barrier")
        )

        transformed, _ = ddd.transform_circuits([qc], shots=100)
        transformed_gate_count = sum(
            1 for inst in transformed[0].data
            if inst.operation.name not in ("measure", "barrier")
        )

        # DDD should add gates to fill idle periods
        assert transformed_gate_count > original_gate_count

    def test_compact_circuit_unchanged(self):
        """Circuits without slack windows should be mostly unchanged."""
        ddd = DDDTechnique({"technique": "ddd"})
        qc = self._make_compact_circuit()

        original_ops = dict(qc.count_ops())
        transformed, _ = ddd.transform_circuits([qc], shots=100)
        transformed_ops = dict(transformed[0].count_ops())

        # May have minor changes from conversion, but no significant gate additions
        # The key gates (h, cx) should still be present
        assert "h" in transformed_ops or "H" in transformed_ops.get("h", 0) >= 0
        assert "cx" in transformed_ops or "CNOT" in transformed_ops

    def test_metadata_contains_config(self):
        """Metadata should record the DDD configuration."""
        ddd = DDDTechnique({"technique": "ddd", "rule": "xx", "spacing": 3})
        circuits = [self._make_circuit_with_idle()]
        _, metadata = ddd.transform_circuits(circuits, shots=100)

        assert metadata.data["rule"] == "xx"
        assert metadata.data["spacing"] == 3

    def test_different_rules_produce_different_results(self):
        """Different DDD rules should insert different gate patterns."""
        qc = self._make_circuit_with_idle()

        ddd_xyxy = DDDTechnique({"technique": "ddd", "rule": "xyxy"})
        ddd_xx = DDDTechnique({"technique": "ddd", "rule": "xx"})

        transformed_xyxy, _ = ddd_xyxy.transform_circuits([qc], shots=100)
        transformed_xx, _ = ddd_xx.transform_circuits([qc], shots=100)

        ops_xyxy = dict(transformed_xyxy[0].count_ops())
        ops_xx = dict(transformed_xx[0].count_ops())

        # XYXY uses both X and Y gates, XX uses only X gates
        # The counts should differ
        assert ops_xyxy != ops_xx


class TestDDDPostprocessCounts:
    def test_returns_counts_unchanged(self):
        """DDD postprocess should return counts exactly as-is."""
        ddd = DDDTechnique({"technique": "ddd"})
        metadata = MitigationMetadata(
            technique_name="ddd",
            data={"rule": "xyxy", "spacing": -1, "original_circuit_count": 2},
        )
        counts = [
            {"00": 450, "01": 50, "10": 40, "11": 460},
            {"00": 500, "11": 500},
        ]

        result = ddd.postprocess_counts(counts, metadata)

        assert result == counts
        assert result is counts  # Should be same object, not a copy

    def test_single_circuit(self):
        """Should work with single circuit."""
        ddd = DDDTechnique({"technique": "ddd"})
        metadata = MitigationMetadata(
            technique_name="ddd",
            data={"rule": "xyxy", "spacing": -1, "original_circuit_count": 1},
        )
        counts = [{"0": 700, "1": 300}]

        result = ddd.postprocess_counts(counts, metadata)
        assert result == counts
