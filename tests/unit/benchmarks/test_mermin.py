import math

import rustworkx as rx
from qiskit import QuantumCircuit

from metriq_gym.benchmarks.mermin import (
    build_mermin_circuits,
    correlator_from_counts,
    find_connected_path,
    mermin_polynomial,
    optimal_angles,
)


def test_mermin_polynomial_n2_recovers_chsh():
    poly = mermin_polynomial(2)
    assert poly == {(0, 0): 1.0, (0, 1): 1.0, (1, 0): 1.0, (1, 1): -1.0}


def test_mermin_polynomial_n3_matches_standard_form():
    # M_3 = A_1 A_2 A'_3 + A_1 A'_2 A_3 + A'_1 A_2 A_3 - A'_1 A'_2 A'_3,
    # the 4 nonzero terms left after BK cancellation.
    poly = mermin_polynomial(3)
    assert poly == {
        (0, 0, 1): 1.0,
        (0, 1, 0): 1.0,
        (1, 0, 0): 1.0,
        (1, 1, 1): -1.0,
    }


def test_optimal_angles_match_closed_form():
    n = 4
    theta, theta_prime = optimal_angles(n)
    assert len(theta) == n and len(theta_prime) == n
    for j in range(n):
        assert math.isclose(theta[j], -j * math.pi / (2 * n))
        assert math.isclose(theta_prime[j], theta[j] + math.pi / 2)


def test_find_connected_path_complete_graph():
    graph = rx.generators.complete_graph(5)
    path = find_connected_path(graph, 4)
    assert path is not None
    assert len(path) == 4
    assert len(set(path)) == 4


def test_find_connected_path_returns_none_when_too_long():
    graph = rx.generators.path_graph(3)
    assert find_connected_path(graph, 5) is None


def test_build_mermin_circuits_count_matches_polynomial():
    n = 3
    qubits = [0, 1, 2]
    circuits, settings = build_mermin_circuits(n, qubits, num_qubits=n)
    poly = mermin_polynomial(n)
    assert len(circuits) == len(poly) == len(settings)
    for qc in circuits:
        assert isinstance(qc, QuantumCircuit)
        assert qc.num_qubits == n
        assert qc.num_clbits == n


def test_correlator_from_counts_parity():
    # All shots in "000" -> parity +1.
    counts = {"000": 1024}
    assert correlator_from_counts(counts, 3) == 1.0
    # All shots in "001" (odd parity) -> -1.
    counts = {"001": 512}
    assert correlator_from_counts(counts, 3) == -1.0
    # Equal mix -> 0.
    counts = {"000": 500, "001": 500}
    assert correlator_from_counts(counts, 3) == 0.0
