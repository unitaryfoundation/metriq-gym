import rustworkx as rx
from qiskit import QuantumCircuit
from metriq_gym.benchmarks.bseq import build_bseq_circuits
from metriq_gym.helpers.graph_helpers import GraphColoring


def test_build_bseq_circuits_complete_graph():
    num_nodes = 4
    graph = rx.generators.complete_graph(num_nodes)

    circuit_sets, coloring = build_bseq_circuits(graph)

    assert isinstance(coloring, GraphColoring)
    assert coloring.num_nodes == num_nodes
    # Complete graph K4 should need 3 colors, but the misra-gries only guarantees 4 colors found
    assert coloring.num_colors == num_nodes

    assert isinstance(circuit_sets, list)
    assert len(circuit_sets) == coloring.num_colors

    for circuit_group in circuit_sets:
        assert isinstance(circuit_group, list)
        # Each group has 4 measurement bases
        assert len(circuit_group) == 4
        for qc in circuit_group:
            assert isinstance(qc, QuantumCircuit)
            assert qc.num_qubits == num_nodes

    # Now check if we had limited to 2 max_colors, we get only 2 sets

    max_colors = 2
    circuit_sets, coloring = build_bseq_circuits(graph, max_colors=max_colors)

    assert coloring.num_colors == max_colors
    assert len(circuit_sets) == max_colors


def test_build_bseq_circuits_linear_graph():
    num_nodes = 5
    graph = rx.generators.path_graph(num_nodes)

    circuit_sets, coloring = build_bseq_circuits(graph)

    assert isinstance(coloring, GraphColoring)
    assert coloring.num_nodes == num_nodes
    assert coloring.num_colors == 2

    assert len(circuit_sets) == 2


def test_build_bseq_circuits_heavy_hex_like():
    degree = 3
    graph = rx.generators.heavy_hex_graph(3)

    circuit_sets, coloring = build_bseq_circuits(graph)

    assert isinstance(coloring, GraphColoring)
    assert coloring.num_nodes == graph.num_nodes()
    # Graph degree is 3, so should be 3 colors
    assert coloring.num_colors == degree

    assert len(circuit_sets) == coloring.num_colors
