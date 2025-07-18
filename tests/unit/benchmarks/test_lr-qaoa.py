import pytest
import networkx as nx
import random
from metriq_gym.benchmarks.lr_qaoa import prepare_qaoa_circuit, LinearRampQAOAData, calc_stats


@pytest.mark.parametrize("num_qubits, p_layers", [(5, [10]), [10, [5, 7]]])
def test_prepare_1d_lrqaoa_circuits(num_qubits, p_layers):
    graph = nx.Graph()
    graph.add_nodes_from(range(num_qubits))
    possible_weights = [0.1, 0.2, 0.3, 0.5, 1.0]
    graph.add_weighted_edges_from(
        [[i, i + 1, random.choice(possible_weights)] for i in range(num_qubits - 1)]
    )
    circuits = prepare_qaoa_circuit(graph, p_layers, "1D", "Direct")
    assert len(circuits) == len(p_layers)
    assert all(circ.num_qubits == num_qubits for circ in circuits)
    for circ_i, p_layer in zip(circuits, p_layers):
        assert (num_qubits - 1) * p_layer == circ_i.count_ops()["rzz"]


@pytest.mark.parametrize("grid_size, p_layers", [(5, [10]), [10, [5, 7]]])
def test_prepare_native_layout_lrqaoa_circuits(grid_size, p_layers):
    graph = nx.grid_2d_graph(grid_size, grid_size)
    graph = nx.convert_node_labels_to_integers(graph, ordering="sorted")
    possible_weights = [0.1, 0.2, 0.3, 0.5, 1.0]
    for u, v in graph.edges():
        graph[u][v]["weight"] = random.choice(possible_weights)
    circuits = prepare_qaoa_circuit(graph, p_layers, "NL", "Direct")
    assert len(circuits) == len(p_layers)
    assert all(circ.num_qubits == grid_size**2 for circ in circuits)
    for circ_i, p_layer in zip(circuits, p_layers):
        assert p_layer * graph.number_of_edges() == circ_i.count_ops()["rzz"]
    for circ_i, p_layer in zip(circuits, p_layers):
        assert p_layer * grid_size**2 == circ_i.count_ops()["rx"]


@pytest.mark.parametrize("num_qubits, p_layers", [(5, [10]), (10, [5, 7]), (17, [3, 8, 15])])
def test_prepare_fully_connected_layout_lrqaoa_circuits(num_qubits, p_layers):
    graph = nx.complete_graph(num_qubits)
    possible_weights = [0.1, 0.2, 0.3, 0.5, 1.0]
    for u, v in graph.edges():
        graph[u][v]["weight"] = random.choice(possible_weights)
    circuits = prepare_qaoa_circuit(graph, p_layers, "FC", "Direct")
    assert len(circuits) == len(p_layers)
    assert all(circ.num_qubits == num_qubits for circ in circuits)
    for circ_i, p_layer in zip(circuits, p_layers):
        assert p_layer * num_qubits * (num_qubits - 1) / 2 == circ_i.count_ops()["rzz"]
    for circ_i, p_layer in zip(circuits, p_layers):
        assert p_layer * num_qubits == circ_i.count_ops()["rx"]

    circuits = prepare_qaoa_circuit(graph, p_layers, "FC", "SWAP")
    assert len(circuits) == len(p_layers)
    assert all(circ.num_qubits == num_qubits for circ in circuits)
    for circ_i, p_layer in zip(circuits, p_layers):
        assert p_layer * (num_qubits - 1) == circ_i.count_ops()["rzz"]
    for circ_i, p_layer in zip(circuits, p_layers):
        assert (
            3 * p_layer * (graph.number_of_edges() - (num_qubits - 1)) == circ_i.count_ops()["cx"]
        )
    for circ_i, p_layer in zip(circuits, p_layers):
        assert p_layer * num_qubits == circ_i.count_ops()["rx"]


def test_calc_stats_pass():
    num_qubits = 5
    graph = nx.Graph()
    graph.add_nodes_from(range(num_qubits))
    possible_weights = [0.1, 0.2, 0.3, 0.5, 1.0]
    graph.add_weighted_edges_from(
        [[i, i + 1, random.choice(possible_weights)] for i in range(num_qubits - 1)]
    )
    p_layers = [3, 5, 7]
    trials = 2
    job_data = LinearRampQAOAData(
        provider_job_ids=["test_job_id"],
        num_qubits=5,
        graph=graph,
        graph_type="1D",
        optimal_sol="10101",
        trials=trials,
        num_random_trials=10,
        confidence_level=0.99,
        shots=100,
        p_layers=p_layers,
        seed=123,
        delta_beta=1.0,
        delta_gamma=1.0,
        circuit_encoding="Direct",
    )
    counts = [
        {"01010": 40, "10101": 30, "11000": 20},
        {"01010": 50, "10101": 50},
        {"00000": 40, "10101": 30, "01010": 20},
        {"11111": 50, "10101": 50},
        {"01010": 50, "10101": 50},
        {"11111": 50, "00000": 50},
    ]
    stats = calc_stats(job_data, counts)
    assert stats.confidence_pass == [True, True, False]  # All trials pass confidence level
    assert len(stats.confidence_pass) == len(p_layers)
