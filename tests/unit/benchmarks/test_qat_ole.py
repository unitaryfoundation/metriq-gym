from datetime import datetime
from unittest.mock import MagicMock

import pytest

from metriq_gym.benchmarks.benchmark import BenchmarkScore
import random

from metriq_gym.benchmarks.qat_ole import (
    QATOLEResult,
    _build_ole_circuits,
    _initial_state_sign,
    _load_qasm_source,
    _ole_estimate,
    _pauli_z_product_expectation,
    _sample_initial_states,
)
from metriq_gym.constants import JobType
from metriq_gym.exporters.base_exporter import BaseExporter
from metriq_gym.job_manager import MetriqGymJob


class _DummyExporter(BaseExporter):
    def export(self) -> None:  # pragma: no cover
        raise NotImplementedError


def _build_metriq_job() -> MetriqGymJob:
    return MetriqGymJob(
        id="test-job",
        job_type=JobType.QAT_OLE,
        params={"benchmark_name": "QAT OLE", "circuit": "49Q_L3", "shots": 1000},
        data={"provider_job_ids": ["qid"], "observable_qubits": [52, 59, 72], "shots": 1000},
        provider_name="provider",
        device_name="device",
        dispatch_time=datetime.now(),
    )


def _make_result(value: float = 0.85, uncertainty: float = 0.02) -> QATOLEResult:
    return QATOLEResult(
        observable_value=BenchmarkScore(value=value, uncertainty=uncertainty),
        circuit_id="49Q_L3",
    )


# --- Pauli-Z product expectation ---


def test_pauli_z_product_expectation_all_even():
    # All outcomes have even parity: expectation = 1.0
    counts = {"000": 500, "011": 300, "101": 150, "110": 50}
    value, uncertainty = _pauli_z_product_expectation(counts)
    assert value == pytest.approx(1.0)
    assert uncertainty == pytest.approx(0.0)


def test_pauli_z_product_expectation_all_odd():
    # All outcomes have odd parity: expectation = -1.0
    counts = {"001": 400, "010": 300, "100": 200, "111": 100}
    value, uncertainty = _pauli_z_product_expectation(counts)
    assert value == pytest.approx(-1.0)
    assert uncertainty == pytest.approx(0.0)


def test_pauli_z_product_expectation_equal_split():
    # Equal even/odd split: expectation = 0.0
    counts = {"000": 500, "001": 500}
    value, uncertainty = _pauli_z_product_expectation(counts)
    assert value == pytest.approx(0.0)
    assert uncertainty == pytest.approx(2.0 * (0.5 * 0.5 / 1000) ** 0.5)


def test_pauli_z_product_expectation_empty_counts():
    value, uncertainty = _pauli_z_product_expectation({})
    assert value == pytest.approx(0.0)
    assert uncertainty == pytest.approx(0.0)


def test_pauli_z_product_expectation_single_qubit_one():
    # Single-qubit case: outcome "1" is odd, "0" is even
    counts = {"0": 700, "1": 300}
    value, uncertainty = _pauli_z_product_expectation(counts)
    # P(even) = 0.7, expectation = 2*0.7 - 1 = 0.4
    assert value == pytest.approx(0.4)
    assert uncertainty == pytest.approx(2.0 * (0.7 * 0.3 / 1000) ** 0.5)


def test_pauli_z_product_expectation_ghz_two_qubit():
    # GHZ state on 2 qubits: |00> + |11>; both outcomes have even parity
    counts = {"00": 512, "11": 488}
    value, uncertainty = _pauli_z_product_expectation(counts)
    assert value == pytest.approx(1.0)
    assert uncertainty == pytest.approx(0.0)


# --- QATOLEResult model ---


def test_qat_ole_result_score_unset():
    # score is intentionally unset pending a reference-based definition
    result = _make_result(0.85, 0.02)
    assert result.score is None


def test_qat_ole_result_observable_value():
    result = _make_result(0.72, 0.03)
    assert result.observable_value.value == pytest.approx(0.72)
    assert result.observable_value.uncertainty == pytest.approx(0.03)
    assert result.circuit_id == "49Q_L3"


def test_qat_ole_result_values_and_uncertainties():
    result = _make_result(0.72, 0.03)
    # observable_value is the only metric; resource fields live in job data now
    assert result.values["observable_value"] == pytest.approx(0.72)
    assert result.uncertainties["observable_value"] == pytest.approx(0.03)
    assert "shots" not in result.values
    assert "num_qubits" not in result.values
    assert "num_gates" not in result.values


def test_qat_ole_result_exporter_payload():
    job = _build_metriq_job()
    result = _make_result(0.91, 0.01)
    exporter = _DummyExporter(job, result)

    payload = exporter.as_dict()
    assert payload["results"]["observable_value"]["value"] == pytest.approx(0.91)
    assert payload["results"]["observable_value"]["uncertainty"] == pytest.approx(0.01)
    assert payload["platform"] == {"provider": "provider", "device": "device"}


def test_qat_ole_result_score_keys_match():
    result = _make_result(0.5, 0.05)
    assert "observable_value" in result.values
    assert "circuit_id" not in result.values  # string field, not a metric


# --- Input validation ---


def test_load_qasm_source_mutual_exclusion():
    params = MagicMock()
    params.circuit = "49Q_L3"
    params.qasm_path = "some/path.qasm"
    params.observable_qubits = None
    with pytest.raises(ValueError, match="mutually exclusive"):
        _load_qasm_source(params)


def test_load_qasm_source_missing_both():
    params = MagicMock()
    params.circuit = None
    params.qasm_path = None
    params.observable_qubits = None
    with pytest.raises(ValueError, match="Either"):
        _load_qasm_source(params)


def test_build_ole_circuits_rejects_existing_clbits():
    # QASM with an existing classical register should be rejected
    qasm = """
OPENQASM 3.0;
qubit[2] q;
bit[1] c;
c[0] = measure q[0];
"""
    with pytest.raises(ValueError, match="classical bits"):
        _build_ole_circuits(qasm, [0], ["00"])


def test_build_ole_circuits_rejects_out_of_range_qubit():
    qasm = "OPENQASM 3.0;\nqubit[3] q;\n"
    with pytest.raises(ValueError, match="out of range"):
        _build_ole_circuits(qasm, [0, 5], ["000"])


def test_build_ole_circuits_rejects_duplicate_qubits():
    qasm = "OPENQASM 3.0;\nqubit[3] q;\n"
    with pytest.raises(ValueError, match="duplicates"):
        _build_ole_circuits(qasm, [0, 1, 0], ["000"])


def test_build_ole_circuits_one_circuit_per_initial_state():
    qasm = "OPENQASM 3.0;\nqubit[3] q;\n"
    circuits = _build_ole_circuits(qasm, [0, 1], ["000", "110", "011"])
    assert len(circuits) == 3
    # The "110" preparation should contain exactly two X gates
    x_counts = [sum(1 for instr in qc.data if instr.operation.name == "x") for qc in circuits]
    assert x_counts == [0, 2, 2]
    # Each circuit measures both observable qubits
    for qc in circuits:
        assert sum(1 for instr in qc.data if instr.operation.name == "measure") == 2


# --- Initial-state sampling and sign factor ---


def test_sample_initial_states_shape_and_alphabet():
    rng = random.Random(7)
    states = _sample_initial_states(num_qubits=5, num_states=4, rng=rng)
    assert len(states) == 4
    assert all(len(s) == 5 and set(s) <= {"0", "1"} for s in states)


def test_sample_initial_states_reproducible_with_seed():
    states_a = _sample_initial_states(6, 8, random.Random(42))
    states_b = _sample_initial_states(6, 8, random.Random(42))
    assert states_a == states_b


def test_initial_state_sign():
    # Even parity on the observable qubits → +1, odd → −1
    assert _initial_state_sign("000", [0, 1, 2]) == 1
    assert _initial_state_sign("110", [0, 1, 2]) == 1
    assert _initial_state_sign("100", [0, 1, 2]) == -1
    assert _initial_state_sign("111", [0, 1, 2]) == -1
    # Only the observable qubits count toward the parity
    assert _initial_state_sign("100", [1, 2]) == 1
    assert _initial_state_sign("010", [1, 2]) == -1


# --- OLE estimate over initial states ---


def test_ole_estimate_perfect_echo():
    # A perfect echo returns each initial state unchanged: the measured parity
    # always equals the initial-state parity, so sign * m = +1 for every state.
    counts_list = [
        {"000": 100},  # initial 000: even parity, sign +1, m = +1
        {"101": 100},  # initial 101 measured as-is on qubits [0,1,2]
    ]
    initial_states = ["000", "101"]
    value, uncertainty = _ole_estimate(counts_list, initial_states, [0, 1, 2])
    assert value == pytest.approx(1.0)
    assert uncertainty == pytest.approx(0.0)


def test_ole_estimate_sign_weighting():
    # Both runs measure even parity (m = +1), but the second initial state has
    # odd parity (sign −1), so the two contributions cancel.
    counts_list = [{"00": 100}, {"00": 100}]
    initial_states = ["00", "10"]
    value, _ = _ole_estimate(counts_list, initial_states, [0, 1])
    assert value == pytest.approx(0.0)


def test_ole_estimate_includes_sampling_spread():
    # Per-state estimates are exact (no shot noise) but disagree, so the
    # uncertainty must reflect the spread across initial states.
    counts_list = [{"00": 100}, {"01": 100}]
    initial_states = ["00", "00"]
    value, uncertainty = _ole_estimate(counts_list, initial_states, [0, 1])
    assert value == pytest.approx(0.0)
    assert uncertainty > 0.5  # sample std of [+1, −1] over sqrt(2)


def test_ole_estimate_length_mismatch():
    with pytest.raises(ValueError, match="initial states"):
        _ole_estimate([{"0": 1}], ["0", "1"], [0])
