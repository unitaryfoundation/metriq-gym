from datetime import datetime
from unittest.mock import MagicMock

import pytest

from metriq_gym.benchmarks.benchmark import BenchmarkScore
import random

from metriq_gym.benchmarks.qat_ole import (
    _OBSERVABLE_QUBITS,
    QATOLEResult,
    _active_qubits,
    _build_ole_circuits,
    _initial_state_sign,
    _load_qasm_source,
    _noiseless_reference,
    _ole_estimate,
    _parse_and_validate,
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


def _make_result(
    value: float = 0.85, uncertainty: float = 0.02, reference: float | None = None
) -> QATOLEResult:
    return QATOLEResult(
        observable_value=BenchmarkScore(value=value, uncertainty=uncertainty),
        noiseless_reference=reference,
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


def test_qat_ole_result_score_unset_without_reference():
    # No classical reference (e.g. the named 156-qubit circuits): score unset
    result = _make_result(0.85, 0.02, reference=None)
    assert result.score is None


def test_qat_ole_result_score_is_ratio_to_reference():
    result = _make_result(0.85, 0.02, reference=0.94)
    assert result.score is not None
    assert result.score.value == pytest.approx(0.85 / 0.94)
    assert result.score.uncertainty == pytest.approx(0.02 / 0.94)


def test_qat_ole_result_score_unset_for_near_zero_reference():
    # A vanishing ideal echo makes the ratio meaningless
    result = _make_result(0.01, 0.02, reference=1e-12)
    assert result.score is None


def test_qat_ole_result_observable_value():
    result = _make_result(0.72, 0.03)
    assert result.observable_value.value == pytest.approx(0.72)
    assert result.observable_value.uncertainty == pytest.approx(0.03)


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
    result = _make_result(0.5, 0.05, reference=0.9)
    assert "observable_value" in result.values
    assert "noiseless_reference" in result.values
    # circuit_id lives in the job data/params, not the result
    assert "circuit_id" not in QATOLEResult.model_fields


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


def test_load_qasm_source_unknown_circuit_name():
    # An unknown name should fail with the supported IDs, before any fetch
    params = MagicMock()
    params.circuit = "99Q_L9"
    params.qasm_path = None
    params.observable_qubits = None
    with pytest.raises(ValueError, match="49Q_L3"):
        _load_qasm_source(params)


def test_load_qasm_source_returns_observable_qubits_copy(monkeypatch):
    # Mutating the returned list must not corrupt the module-level constant
    monkeypatch.setattr("metriq_gym.benchmarks.qat_ole._fetch_qasm", lambda name: "")
    params = MagicMock()
    params.circuit = "49Q_L3"
    params.qasm_path = None
    params.observable_qubits = None
    _, _, obs_qubits = _load_qasm_source(params)
    obs_qubits.append(999)
    assert _OBSERVABLE_QUBITS == [52, 59, 72]


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


def test_ole_estimate_uncertainty_is_sem_of_states():
    # The uncertainty is the standard error of the mean of the per-state
    # estimates: sample std of [+1, −1] is sqrt(2), over sqrt(n=2) gives 1.0.
    # No separate shot term is added — the spread already contains shot noise.
    counts_list = [{"00": 100}, {"01": 100}]
    initial_states = ["00", "00"]
    value, uncertainty = _ole_estimate(counts_list, initial_states, [0, 1])
    assert value == pytest.approx(0.0)
    assert uncertainty == pytest.approx(1.0)


def test_ole_estimate_single_state_uses_shot_noise():
    # With one initial state there is no spread to measure, so the propagated
    # binomial shot noise of that single estimate is used.
    counts = {"00": 70, "01": 30}
    value, uncertainty = _ole_estimate([counts], ["00"], [0, 1])
    _, expected_u = _pauli_z_product_expectation(counts)
    assert value == pytest.approx(0.4)
    assert uncertainty == pytest.approx(expected_u)


def test_ole_estimate_length_mismatch():
    with pytest.raises(ValueError, match="initial states"):
        _ole_estimate([{"0": 1}], ["0", "1"], [0])


def test_sample_initial_states_respects_active_qubits():
    # Idle qubits must stay '0' across every sampled state; active ones vary.
    rng = random.Random(3)
    active = {1, 3}
    states = _sample_initial_states(num_qubits=5, num_states=20, rng=rng, active_qubits=active)
    for s in states:
        assert s[0] == "0" and s[2] == "0" and s[4] == "0"
    assert any(s[1] == "1" for s in states)
    assert any(s[3] == "1" for s in states)


def test_noiseless_reference_perfect_echo_is_one():
    # An empty base circuit is a perfect echo: each initial state comes back
    # unchanged, so sign * <Z...Z> = +1 for every sampled state.
    base = _parse_and_validate("OPENQASM 3.0;\nqubit[3] q;\n", [0, 1, 2])
    ref = _noiseless_reference(base, ["000", "101", "110"], [0, 1, 2])
    assert ref == pytest.approx(1.0)


def test_noiseless_reference_parity_flip_is_minus_one():
    # An X on an observable qubit flips the parity for every initial state
    qasm = 'OPENQASM 3.0;\ninclude "stdgates.inc";\nqubit[2] q;\nx q[0];\n'
    base = _parse_and_validate(qasm, [0])
    ref = _noiseless_reference(base, ["00", "10"], [0])
    assert ref == pytest.approx(-1.0)


def test_noiseless_reference_none_above_qubit_limit():
    from qiskit import QuantumCircuit

    base = QuantumCircuit(21)
    assert _noiseless_reference(base, ["0" * 21], [0]) is None


def test_active_qubits_ignores_barriers_and_idle():
    qasm = """
OPENQASM 3.0;
include "stdgates.inc";
qubit[6] q;
h q[1];
cx q[1], q[3];
barrier q;
"""
    base = _parse_and_validate(qasm, [1])
    assert _active_qubits(base) == {1, 3}
