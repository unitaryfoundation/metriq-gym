import pytest
import networkx as nx
import random
from metriq_gym.benchmarks.lr_qaoa import prepare_qaoa_circuit


@pytest.mark.parametrize("num_qubits, p_layers", [(5, [10]), [10, [5, 7]]])
def test_prepare_lrqaoa_circuits(num_qubits, p_layers):
    # p_layers = [5,10]
    graph_1D = nx.Graph()
    graph_1D.add_nodes_from(range(num_qubits))
    possible_weights = [0.1, 0.2, 0.3, 0.5, 1.0]
    graph_1D.add_weighted_edges_from(
        [[i, i + 1, random.choice(possible_weights)] for i in range(num_qubits - 1)]
    )
    circuits = prepare_qaoa_circuit(graph_1D, p_layers, "1D", "Direct")
    assert len(circuits) == len(p_layers)
    assert all(circ.num_qubits == num_qubits for circ in circuits)
