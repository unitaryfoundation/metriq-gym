import math

from qiskit import QuantumCircuit

from metriq_gym.benchmarks.semi_brukner import (
    SEMI_BRUKNER_TERMS,
    build_semi_brukner_circuits,
    correlator_from_counts,
    optimal_angles,
    prepare_singlet,
)


def test_optimal_angles_saturate_quantum_bound():
    # On the singlet, <A(alpha) (x) B(gamma)> = -cos(alpha - gamma), so
    # tr(B |psi-><psi-|) reduces to a sum of cosines that should hit 2*sqrt(2)
    # with the chosen angles.
    angles = optimal_angles()
    score = 0.0
    for a_label, b_label, coef in SEMI_BRUKNER_TERMS:
        score += coef * (-math.cos(angles[a_label] - angles[b_label]))
    assert math.isclose(score, 2 * math.sqrt(2), rel_tol=1e-12)


def test_optimal_angles_keys():
    angles = optimal_angles()
    assert set(angles.keys()) == {"A1", "A3", "B2", "B3"}


def test_prepare_singlet_gate_sequence():
    qc = prepare_singlet(num_qubits=3, qubits=(0, 1))
    # Expect exactly: X(0), H(0), X(1), CX(0, 1).
    gate_names = [instr.operation.name for instr in qc.data]
    assert gate_names == ["x", "h", "x", "cx"]


def test_build_semi_brukner_circuits_count_and_shape():
    qubits = (0, 1)
    circuits = build_semi_brukner_circuits(qubits, num_qubits=2)
    assert len(circuits) == 4
    for qc in circuits:
        assert isinstance(qc, QuantumCircuit)
        assert qc.num_qubits == 2
        assert qc.num_clbits == 2


def test_correlator_from_counts_parity():
    assert correlator_from_counts({"00": 1024}) == 1.0
    assert correlator_from_counts({"11": 512}) == 1.0  # parity 0
    assert correlator_from_counts({"01": 512}) == -1.0
    assert correlator_from_counts({"10": 512}) == -1.0
    assert correlator_from_counts({"00": 500, "01": 500}) == 0.0
