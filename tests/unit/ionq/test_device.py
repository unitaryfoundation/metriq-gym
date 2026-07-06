"""Tests for the IonQ patches that fix qBraid's IonQ result processing."""

import json
from unittest.mock import MagicMock

from qiskit import QuantumCircuit

from metriq_gym.ionq.device import (
    _META_MEAS_MAPS,
    _META_NUM_CLBITS,
    _META_NUM_QUBITS,
    _extract_measurement_map,
    _marginalize_to_clbits,
    _pad_counts,
    _reshape_counts,
    patch_ionq_device,
)


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
    def test_full_measurement_is_identity(self):
        """An identity qubit->clbit map leaves big-endian bitstrings unchanged."""
        meas_map = {"0": 0, "1": 1}
        counts = {"00": 50, "01": 30, "10": 15, "11": 5}
        result = _marginalize_to_clbits(counts, meas_map, num_clbits=2)
        assert result == counts

    def test_wit_single_qubit_measurement(self):
        """7-qubit circuit measuring only qubit 5 into 1 classical bit.

        Big-endian bitstrings: qubit q is at string index (n-1-q), so qubit 5
        of a 7-char string is index 1. "0100000" has '1' there -> qubit 5 = 1.
        """
        meas_map = {"5": 0}
        counts = {"0000000": 4000, "0100000": 4000}
        result = _marginalize_to_clbits(counts, meas_map, num_clbits=1)
        assert result == {"0": 4000, "1": 4000}

    def test_two_qubit_partial_preserves_clbit_order(self):
        """4 qubits, measuring qubit 1 -> clbit 0 and qubit 3 -> clbit 1.

        Big-endian 4-char string: qubit q at index (3-q); clbit c at output
        index (1-c). Qubit 1 -> clbit 0 (rightmost), qubit 3 -> clbit 1 (left).
        """
        meas_map = {"1": 0, "3": 1}
        # "1010": q1=index2='1' -> clbit0; q3=index0='1' -> clbit1 => "11"
        # "0010": q1=index2='1' -> clbit0; q3=index0='0' -> clbit1 => "01"
        # "1001": q1=index2='0' -> clbit0; q3=index0='1' -> clbit1 => "10"
        counts = {"0000": 40, "1010": 30, "0010": 20, "1001": 10}
        result = _marginalize_to_clbits(counts, meas_map, num_clbits=2)
        assert result == {"00": 40, "11": 30, "01": 20, "10": 10}

    def test_aggregates_degenerate_states(self):
        """States that share the measured qubit's value merge together."""
        meas_map = {"0": 0}
        counts = {"00": 25, "01": 25, "10": 25, "11": 25}
        result = _marginalize_to_clbits(counts, meas_map, num_clbits=1)
        assert result == {"0": 50, "1": 50}


class TestReshapeCounts:
    def test_returns_unchanged_without_metadata(self):
        counts = {"1": 100}
        assert _reshape_counts(counts, None, None, None) == counts

    def test_single_circuit_pads_and_marginalizes(self):
        # Raw value 4 -> "100"; padded to 4 -> "0100"; identity map -> "0100".
        counts = {"100": 200}
        out = _reshape_counts(counts, [{"0": 0, "1": 1, "2": 2, "3": 3}], [4], [4])
        assert out == {"0100": 200}

    def test_multi_circuit_reshapes_each_independently(self):
        counts = [{"1": 200}, {"100": 200}]
        maps = [{"0": 0, "1": 1, "2": 2, "3": 3}] * 2
        out = _reshape_counts(counts, maps, [4, 4], [4, 4])
        assert out == [{"0001": 200}, {"0100": 200}]

    def test_partial_measurement_reduces_width(self):
        # 7 qubits, only qubit 5 measured -> 1 clbit.
        counts = {"0100000": 500}
        out = _reshape_counts(counts, [{"5": 0}], [7], [1])
        assert out == {"1": 500}


class TestPatchIonqDevice:
    def test_run_converts_to_qasm_and_records_metadata(self):
        """Dispatch converts qiskit circuits to QASM3 and records reshape metadata."""
        captured = {}

        class FakeIonQDevice:
            def run(self, run_input, *args, **kwargs):
                captured["run_input"] = run_input
                captured["metadata"] = kwargs.get("metadata")
                return MagicMock()

        device = FakeIonQDevice()
        patch_ionq_device(device)

        c0 = QuantumCircuit(4, 4)
        c0.x(0)
        c0.measure(range(4), range(4))
        c1 = QuantumCircuit(4, 1)
        c1.h(2)
        c1.measure(2, 0)

        device.run([c0, c1], shots=100)

        # Circuits handed to qBraid are OpenQASM 3 strings, not qiskit circuits.
        assert all(isinstance(x, str) and "OPENQASM 3" in x for x in captured["run_input"])

        meta = captured["metadata"]
        assert json.loads(meta[_META_NUM_QUBITS]) == [4, 4]
        assert json.loads(meta[_META_NUM_CLBITS]) == [4, 1]
        assert json.loads(meta[_META_MEAS_MAPS]) == [
            {"0": 0, "1": 1, "2": 2, "3": 3},
            {"2": 0},
        ]

    def test_run_preexisting_metadata_preserved(self):
        captured = {}

        class FakeIonQDevice:
            def run(self, run_input, *args, **kwargs):
                captured["metadata"] = kwargs.get("metadata")
                return MagicMock()

        device = FakeIonQDevice()
        patch_ionq_device(device)

        qc = QuantumCircuit(1, 1)
        qc.measure(0, 0)
        device.run(qc, shots=10, metadata={"user_tag": "abc"})

        assert captured["metadata"]["user_tag"] == "abc"
        assert _META_NUM_QUBITS in captured["metadata"]
