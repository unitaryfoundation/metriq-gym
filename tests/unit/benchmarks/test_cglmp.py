import math

import pytest
from qiskit import QuantumCircuit

from metriq_gym.benchmarks.cglmp import (
    CLASSICAL_BOUND,
    SUPPORTED_D,
    build_cglmp_circuits,
    cglmp_score,
    counts_to_probs,
    ideal_quantum_probs,
    num_qubits_per_party,
    optimal_alice_phases,
    optimal_bob_phases,
    quantum_bound,
)


def test_num_qubits_per_party_for_supported_d():
    assert num_qubits_per_party(2) == 1
    assert num_qubits_per_party(4) == 2


def test_unsupported_d_raises():
    with pytest.raises(NotImplementedError):
        num_qubits_per_party(3)
    with pytest.raises(NotImplementedError):
        quantum_bound(3)


def test_optimal_phases():
    assert optimal_alice_phases() == (0.0, 0.5)
    assert optimal_bob_phases() == (0.25, -0.25)


def test_classical_and_quantum_bounds():
    assert CLASSICAL_BOUND == 2.0
    assert math.isclose(quantum_bound(2), 2.0 * math.sqrt(2.0))
    assert math.isclose(quantum_bound(4), 2.896194, rel_tol=1e-5)


@pytest.mark.parametrize("d", SUPPORTED_D)
def test_ideal_probs_normalised_per_setting(d):
    probs = ideal_quantum_probs(d)
    for x in (1, 2):
        for y in (1, 2):
            total = sum(probs[(x, y, a, b)] for a in range(d) for b in range(d))
            assert math.isclose(total, 1.0, rel_tol=1e-9)


@pytest.mark.parametrize("d", SUPPORTED_D)
def test_cglmp_score_on_ideal_probs_matches_quantum_bound(d):
    # Both the analytic probs formula and the tabulated bound represent
    # the same quantum maximum on the maximally entangled state. Tolerance
    # is loose because the tabulated values are 6-digit rounded literature
    # numbers, not exact.
    probs = ideal_quantum_probs(d)
    score = cglmp_score(probs, d)
    assert math.isclose(score, quantum_bound(d), rel_tol=1e-4)


def test_build_cglmp_circuits_d2():
    circuits = build_cglmp_circuits(
        d=2, alice_qubits=[0], bob_qubits=[1], num_qubits=2
    )
    assert len(circuits) == 4
    for qc in circuits:
        assert isinstance(qc, QuantumCircuit)
        assert qc.num_qubits == 2
        assert qc.num_clbits == 2


def test_build_cglmp_circuits_d4():
    circuits = build_cglmp_circuits(
        d=4, alice_qubits=[0, 1], bob_qubits=[2, 3], num_qubits=4
    )
    assert len(circuits) == 4
    for qc in circuits:
        assert isinstance(qc, QuantumCircuit)
        assert qc.num_qubits == 4
        assert qc.num_clbits == 4


def test_counts_to_probs_d2_round_trip():
    # All shots in "00" → A=0, B=0 with probability 1.
    out = counts_to_probs({"00": 1024}, d=2, x=1, y=1)
    assert out[(1, 1, 0, 0)] == 1.0
    # All shots in "10" → in Qiskit convention, bit 0 = right-most. So
    # "10" has classical bit 0 = '0' (Alice) and bit 1 = '1' (Bob),
    # giving Alice=0, Bob=1.
    out = counts_to_probs({"10": 1024}, d=2, x=1, y=2)
    assert out[(1, 2, 0, 1)] == 1.0
