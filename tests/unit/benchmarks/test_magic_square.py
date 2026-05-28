from qiskit import QuantumCircuit

from metriq_gym.benchmarks.magic_square import (
    CLASSICAL_BOUND,
    QUANTUM_BOUND,
    SHARED_CELL_SIGN,
    build_magic_square_circuits,
    prepare_two_bell_pairs,
    win_probability_from_counts,
)


def test_bounds():
    assert CLASSICAL_BOUND == 8.0 / 9.0
    assert QUANTUM_BOUND == 1.0


def test_shared_cell_sign_matches_y_parity():
    # +1 when the operator has an even number of Y's; -1 otherwise.
    assert SHARED_CELL_SIGN[(1, 1)] == +1  # I (x) X
    assert SHARED_CELL_SIGN[(1, 3)] == +1  # X (x) X
    assert SHARED_CELL_SIGN[(2, 1)] == -1  # Y (x) I
    assert SHARED_CELL_SIGN[(2, 3)] == +1  # Y (x) Y (two Y's)
    assert SHARED_CELL_SIGN[(3, 3)] == +1  # Z (x) Z
    assert SHARED_CELL_SIGN[(3, 1)] == -1  # Y (x) X


def test_prepare_two_bell_pairs_gate_sequence():
    qc = prepare_two_bell_pairs(num_qubits=4, alice_qubits=(0, 1), bob_qubits=(2, 3))
    gate_names = [instr.operation.name for instr in qc.data]
    # H + CNOT for the first pair, then H + CNOT for the second.
    assert gate_names == ["h", "cx", "h", "cx"]


def test_build_magic_square_circuits_count_and_shape():
    circuits = build_magic_square_circuits(
        alice_qubits=(0, 1), bob_qubits=(2, 3), num_qubits=4
    )
    assert len(circuits) == 9
    for qc in circuits:
        assert isinstance(qc, QuantumCircuit)
        assert qc.num_qubits == 4
        assert qc.num_clbits == 4


def test_win_probability_full_win_on_correlated_cell():
    # (r=1, c=1): I (x) X, symmetric -> Alice's q_second must equal
    # Bob's q_second.
    #
    # Qiskit prints classical bits right-to-left, so for a 4-bit string
    # "d c b a" we have classical bit 0 = 'a', bit 1 = 'b', bit 2 = 'c',
    # bit 3 = 'd'. Classical bits 0, 1 are Alice's q_first, q_second;
    # bits 2, 3 are Bob's q_first, q_second.
    #
    # For (r=1, c=1) shared cell value: a_c = (-1)^bit_1 (Alice q_second)
    # and b_r = (-1)^bit_3 (Bob q_second), with sigma = +1, so win iff
    # bit_1 == bit_3.
    #
    # "0000" -> bit_1 = bit_3 = 0 -> wins.
    # "0010" -> bit_1 = 1, bit_3 = 0 -> loses.
    counts = {"0000": 600, "0010": 400}
    p = win_probability_from_counts(counts, r=1, c=1)
    assert p == 0.6


def test_win_probability_uses_sigma_for_anti_correlated_cell():
    # (r=2, c=1): Y (x) I (anti-correlated), sigma = -1.
    # Row 2 decoder: a = (v0, v1, v0*v1). Shared cell c=1 -> a_1 = v0.
    # Col 1 decoder: b = (v1, v0, v0*v1). Shared cell r=2 -> b_2 = v0.
    # Win iff a_c == sigma * b_r, i.e., v0_alice == -v0_bob, i.e.,
    # bit_0_alice != bit_0_bob.
    counts = {"0100": 1000}  # b0=0, b1=0, b2=1, b3=0
    # bit_0_alice = b0 = 0, bit_0_bob = b2 = 1 -> they differ -> win.
    p = win_probability_from_counts(counts, r=2, c=1)
    assert p == 1.0


def test_win_probability_loses_when_correlation_violates_sigma():
    # Same (r=2, c=1) but with bit_0_alice == bit_0_bob -> always lose.
    counts = {"0000": 1000}  # all bits 0 -> a_1 = +1, b_2 = +1; sigma = -1 -> not equal
    p = win_probability_from_counts(counts, r=2, c=1)
    assert p == 0.0
