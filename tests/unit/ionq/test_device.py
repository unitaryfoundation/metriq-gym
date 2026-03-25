"""Tests for the IonQ patches that fix qBraid's IonQ result processing."""

from qiskit import QuantumCircuit

from metriq_gym.ionq.device import (
    _extract_measurement_map,
    _marginalize_to_clbits,
    _pad_counts,
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
    def test_wit_single_qubit_measurement(self):
        """7-qubit circuit measuring only qubit 5 → 1 classical bit.

        Big-endian bitstrings: qubit q is at string index (n-1-q).
        Qubit 5 in 7-char string → index 1.
        "0100000" has '1' at index 1 → qubit 5 = |1⟩.
        """
        meas_map = {"5": 0}
        counts = {"0000000": 4000, "0100000": 4000}
        result = _marginalize_to_clbits(counts, meas_map, num_clbits=1)
        assert result == {"0": 4000, "1": 4000}

    def test_full_measurement_is_identity(self):
        """When all qubits are measured, marginalization reverses to little-endian.

        Big-endian "01": index 0 = qubit 1, index 1 = qubit 0.
        So qubit 0 → clbit 0 reads index 1, qubit 1 → clbit 1 reads index 0.
        "01" → clbit0=bitstring[1]='1', clbit1=bitstring[0]='0' → "10"
        """
        meas_map = {"0": 0, "1": 1}
        counts = {"00": 50, "01": 30, "10": 15, "11": 5}
        result = _marginalize_to_clbits(counts, meas_map, num_clbits=2)
        # Big-endian to little-endian reversal:
        # "00" → "00", "01" → "10", "10" → "01", "11" → "11"
        assert result == {"00": 50, "10": 30, "01": 15, "11": 5}

    def test_two_qubit_partial(self):
        """4-qubit circuit measuring qubits 1 and 3 into clbits 0 and 1.

        Big-endian 4-char string: qubit q at index (3-q).
        Qubit 1 → index 2, qubit 3 → index 0.
        """
        meas_map = {"1": 0, "3": 1}
        # "0000": q1=bitstring[2]='0', q3=bitstring[0]='0' → "00"
        # "1010": q1=bitstring[2]='1', q3=bitstring[0]='1' → "11"
        # "0010": q1=bitstring[2]='1', q3=bitstring[0]='0' → "10"
        counts = {"0000": 40, "1010": 30, "0010": 20, "1001": 10}
        result = _marginalize_to_clbits(counts, meas_map, num_clbits=2)
        # "1001": q1=bitstring[2]='0', q3=bitstring[0]='1' → "01"
        assert result == {"00": 40, "11": 30, "10": 20, "01": 10}

    def test_aggregates_degenerate_states(self):
        """Multiple all-qubit states that map to the same classical state.

        Qubit 0 in 2-char big-endian string → index 1.
        "00" → q0='0', "01" → q0='1', "10" → q0='0', "11" → q0='1'
        """
        meas_map = {"0": 0}
        counts = {"00": 25, "01": 25, "10": 25, "11": 25}
        result = _marginalize_to_clbits(counts, meas_map, num_clbits=1)
        assert result == {"0": 50, "1": 50}
