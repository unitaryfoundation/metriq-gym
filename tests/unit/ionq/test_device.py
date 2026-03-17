"""Tests for the IonQ patches that fix qBraid's IonQ result processing."""

from unittest.mock import MagicMock
import types

from qiskit import QuantumCircuit
from qbraid.runtime import IonQDevice

from metriq_gym.ionq.device import (
    _circuits_to_qasm2,
    _extract_measurement_map,
    _marginalize_to_clbits,
    _pad_counts,
)


class TestCircuitsToQasm2:
    def test_converts_single_circuit(self):
        qc = QuantumCircuit(2, 2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure([0, 1], [0, 1])

        result = _circuits_to_qasm2(qc)
        assert isinstance(result, str)
        assert "OPENQASM 2.0" in result
        assert "h q[0]" in result

    def test_converts_list_of_circuits(self):
        qc1 = QuantumCircuit(2)
        qc1.h(0)

        qc2 = QuantumCircuit(2)
        qc2.x(0)

        result = _circuits_to_qasm2([qc1, qc2])
        assert isinstance(result, list)
        assert len(result) == 2
        assert all(isinstance(s, str) for s in result)

    def test_passes_through_non_qiskit_input(self):
        qasm_str = "OPENQASM 2.0; qreg q[2]; h q[0];"
        assert _circuits_to_qasm2(qasm_str) == qasm_str

    def test_passes_through_list_of_strings(self):
        inputs = ["OPENQASM 2.0;", "OPENQASM 2.0;"]
        assert _circuits_to_qasm2(inputs) == inputs


class TestPatchIonqDevice:
    def test_patched_run_converts_circuit_to_qasm(self):
        """Patched device.run() should convert QuantumCircuit to QASM2 string."""
        original_run = MagicMock()

        device = MagicMock(spec=IonQDevice)
        device.run = original_run

        # Manually set up the same way patch_ionq_device does, using the mock as original
        def run_with_qasm_conversion(self, run_input, *args, **kwargs):
            kwargs.pop("gateset", None)
            kwargs.pop("ionq_compiler_synthesis", None)
            return original_run(_circuits_to_qasm2(run_input), *args, **kwargs)

        device.run = types.MethodType(run_with_qasm_conversion, device)

        qc = QuantumCircuit(2, 2)
        qc.h(0)
        qc.measure([0, 1], [0, 1])

        device.run(qc, shots=100)

        # Verify original_run received a QASM string, not a QuantumCircuit
        call_args = original_run.call_args
        run_input_received = call_args[0][0]
        assert isinstance(run_input_received, str)
        assert "OPENQASM 2.0" in run_input_received


class TestExtractMeasurementMap:
    def test_full_measurement(self):
        qc = QuantumCircuit(3, 3)
        qc.measure([0, 1, 2], [0, 1, 2])
        meas_map = _extract_measurement_map(qc)
        assert meas_map == {"0": 0, "1": 1, "2": 2}

    def test_partial_measurement(self):
        """WIT-style: 7 qubits, only qubit 5 measured into clbit 0."""
        qc = QuantumCircuit(7, 1)
        qc.h(0)
        qc.measure(5, 0)
        meas_map = _extract_measurement_map(qc)
        assert meas_map == {"5": 0}

    def test_no_measurements(self):
        qc = QuantumCircuit(2)
        meas_map = _extract_measurement_map(qc)
        assert meas_map == {}


class TestPadCounts:
    def test_pads_short_keys(self):
        assert _pad_counts({"0": 8192}, 7) == {"0000000": 8192}

    def test_already_correct_width(self):
        counts = {"0000000": 4096, "1111111": 4096}
        assert _pad_counts(counts, 7) == counts

    def test_mixed_lengths(self):
        counts = {"0": 100, "101": 200}
        assert _pad_counts(counts, 5) == {"00000": 100, "00101": 200}


class TestMarginalizeToClbits:
    def test_wit_single_qubit_measurement(self):
        """7-qubit circuit measuring only qubit 5 → 1 classical bit."""
        meas_map = {"5": 0}
        # In "0000000", qubit 5 (index 5) = '0'
        # In "0000010", qubit 5 (index 5) = '1'
        counts = {"0000000": 4000, "0000010": 4000}
        result = _marginalize_to_clbits(counts, meas_map, num_clbits=1)
        assert result == {"0": 4000, "1": 4000}

    def test_full_measurement_is_identity(self):
        """When all qubits are measured, marginalization is a no-op (reorder)."""
        meas_map = {"0": 0, "1": 1}
        counts = {"00": 50, "01": 30, "10": 15, "11": 5}
        result = _marginalize_to_clbits(counts, meas_map, num_clbits=2)
        assert result == counts

    def test_two_qubit_partial(self):
        """4-qubit circuit measuring qubits 1 and 3 into clbits 0 and 1."""
        meas_map = {"1": 0, "3": 1}
        # bitstring: q0 q1 q2 q3
        # "0000" → q1='0', q3='0' → clbits="00"
        # "0101" → q1='1', q3='1' → clbits="11"
        # "0100" → q1='1', q3='0' → clbits="10"
        counts = {"0000": 40, "0101": 30, "0100": 20, "1010": 10}
        result = _marginalize_to_clbits(counts, meas_map, num_clbits=2)
        # "1010" → q1='0', q3='0' → "00"
        assert result == {"00": 50, "11": 30, "10": 20}

    def test_aggregates_degenerate_states(self):
        """Multiple all-qubit states that map to the same classical state."""
        meas_map = {"0": 0}
        counts = {"00": 25, "01": 25, "10": 25, "11": 25}
        result = _marginalize_to_clbits(counts, meas_map, num_clbits=1)
        assert result == {"0": 50, "1": 50}
