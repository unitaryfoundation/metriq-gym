import math

from qiskit import QuantumCircuit

from metriq_gym.benchmarks.svetlichny import (
    CLASSICAL_BOUND,
    QUANTUM_BOUND,
    SVETLICHNY_COEFFICIENTS,
    build_svetlichny_circuits,
    correlator_from_counts,
    optimal_angles,
    prepare_ghz3,
)


def test_svetlichny_coefficients_match_canonical_form():
    # S_3 = A_1 B_1 C_1 + A_1 B_1 C_2 + A_1 B_2 C_1 - A_1 B_2 C_2
    #     + A_2 B_1 C_1 - A_2 B_1 C_2 - A_2 B_2 C_1 - A_2 B_2 C_2
    assert SVETLICHNY_COEFFICIENTS == {
        (0, 0, 0): +1.0,
        (0, 0, 1): +1.0,
        (0, 1, 0): +1.0,
        (0, 1, 1): -1.0,
        (1, 0, 0): +1.0,
        (1, 0, 1): -1.0,
        (1, 1, 0): -1.0,
        (1, 1, 1): -1.0,
    }


def test_bounds_match_known_values():
    assert CLASSICAL_BOUND == 4.0
    assert math.isclose(QUANTUM_BOUND, 4.0 * math.sqrt(2.0))


def test_optimal_angles_saturate_quantum_bound():
    # On GHZ_3, <A_1(s_1) A_2(s_2) A_3(s_3)> = cos(sum alpha_j + |s|*pi/2)
    # with primed = unprimed + pi/2. Sum the polynomial against these
    # correlators and verify it equals 4*sqrt(2).
    theta, theta_prime = optimal_angles()
    score = 0.0
    for setting, coef in SVETLICHNY_COEFFICIENTS.items():
        phase = sum(theta[j] if s == 0 else theta_prime[j] for j, s in enumerate(setting))
        score += coef * math.cos(phase)
    assert math.isclose(score, QUANTUM_BOUND, rel_tol=1e-12)


def test_prepare_ghz3_gate_sequence():
    qc = prepare_ghz3(num_qubits=4, qubits=(0, 1, 2))
    gate_names = [instr.operation.name for instr in qc.data]
    assert gate_names == ["h", "cx", "cx"]


def test_build_svetlichny_circuits_count_and_shape():
    circuits, settings = build_svetlichny_circuits(qubits=(0, 1, 2), num_qubits=3)
    assert len(circuits) == 8
    assert len(settings) == 8
    assert set(settings) == set(SVETLICHNY_COEFFICIENTS.keys())
    for qc in circuits:
        assert isinstance(qc, QuantumCircuit)
        assert qc.num_qubits == 3
        assert qc.num_clbits == 3


def test_correlator_from_counts_parity():
    # All-zeros bitstring has even parity → +1.
    assert correlator_from_counts({"000": 1024}) == 1.0
    # Single-one bitstring (odd parity) → -1.
    assert correlator_from_counts({"001": 512}) == -1.0
    assert correlator_from_counts({"010": 512}) == -1.0
    assert correlator_from_counts({"100": 512}) == -1.0
    # Two ones (even parity) → +1.
    assert correlator_from_counts({"110": 512}) == 1.0
    # Three ones (odd parity) → -1.
    assert correlator_from_counts({"111": 512}) == -1.0
    # Equal split → 0.
    assert correlator_from_counts({"000": 500, "001": 500}) == 0.0
