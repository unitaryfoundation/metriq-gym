import pytest
import networkx as nx
import random
from metriq_gym.benchmarks.lr_qaoa import (
    prepare_qaoa_circuit,
    calc_stats,
    cost_maxcut,
    weighted_maxcut_solver,
    calc_random_stats,
)
from metriq_gym.benchmarks.lr_qaoa import LinearRampQAOAData
from metriq_gym.circuits import distribute_edges, SWAP_pairs


@pytest.mark.parametrize("num_qubits, qaoa_layers", [(5, [10]), [10, [5, 7]]])
def test_prepare_1d_lrqaoa_circuits(num_qubits, qaoa_layers):
    graph = nx.Graph()
    graph.add_nodes_from(range(num_qubits))
    possible_weights = [0.1, 0.2, 0.3, 0.5, 1.0]
    graph.add_weighted_edges_from(
        [[i, i + 1, random.choice(possible_weights)] for i in range(num_qubits - 1)]
    )
    circuits = prepare_qaoa_circuit(graph, qaoa_layers, "1D", "Direct")
    assert len(circuits) == len(qaoa_layers)
    assert all(circ.num_qubits == num_qubits for circ in circuits)
    for circ_i, p_layer in zip(circuits, qaoa_layers):
        assert (num_qubits - 1) * p_layer == circ_i.count_ops()["rzz"]


@pytest.mark.parametrize("grid_size, qaoa_layers", [(5, [10]), [10, [5, 7]]])
def test_prepare_native_layout_lrqaoa_circuits(grid_size, qaoa_layers):
    graph = nx.grid_2d_graph(grid_size, grid_size)
    graph = nx.convert_node_labels_to_integers(graph, ordering="sorted")
    possible_weights = [0.1, 0.2, 0.3, 0.5, 1.0]
    for u, v in graph.edges():
        graph[u][v]["weight"] = random.choice(possible_weights)
    circuits = prepare_qaoa_circuit(graph, qaoa_layers, "NL", "Direct")
    assert len(circuits) == len(qaoa_layers)
    assert all(circ.num_qubits == grid_size**2 for circ in circuits)
    for circ_i, p_layer in zip(circuits, qaoa_layers):
        assert p_layer * graph.number_of_edges() == circ_i.count_ops()["rzz"]
    for circ_i, p_layer in zip(circuits, qaoa_layers):
        assert p_layer * grid_size**2 == circ_i.count_ops()["rx"]


@pytest.mark.parametrize("num_qubits, qaoa_layers", [(5, [10]), (10, [5, 7]), (17, [3, 8, 15])])
def test_prepare_fully_connected_layout_lrqaoa_circuits(num_qubits, qaoa_layers):
    graph = nx.complete_graph(num_qubits)
    possible_weights = [0.1, 0.2, 0.3, 0.5, 1.0]
    for u, v in graph.edges():
        graph[u][v]["weight"] = random.choice(possible_weights)
    circuits = prepare_qaoa_circuit(graph, qaoa_layers, "FC", "Direct")
    assert len(circuits) == len(qaoa_layers)
    assert all(circ.num_qubits == num_qubits for circ in circuits)
    for circ_i, p_layer in zip(circuits, qaoa_layers):
        assert p_layer * num_qubits * (num_qubits - 1) / 2 == circ_i.count_ops()["rzz"]
    for circ_i, p_layer in zip(circuits, qaoa_layers):
        assert p_layer * num_qubits == circ_i.count_ops()["rx"]

    circuits = prepare_qaoa_circuit(graph, qaoa_layers, "FC", "SWAP")
    assert len(circuits) == len(qaoa_layers)
    assert all(circ.num_qubits == num_qubits for circ in circuits)
    for circ_i, p_layer in zip(circuits, qaoa_layers):
        assert p_layer * (num_qubits - 1) == circ_i.count_ops()["rzz"]
    for circ_i, p_layer in zip(circuits, qaoa_layers):
        assert (
            3 * p_layer * (graph.number_of_edges() - (num_qubits - 1)) == circ_i.count_ops()["cx"]
        )
    for circ_i, p_layer in zip(circuits, qaoa_layers):
        assert p_layer * num_qubits == circ_i.count_ops()["rx"]


def test_calc_stats_pass():
    random.seed(123)
    num_qubits = 5
    shots = 100
    graph = nx.Graph()
    graph.add_nodes_from(range(num_qubits))
    possible_weights = [0.1, 0.2, 0.3, 0.5, 1.0]
    graph_info = [[i, i + 1, random.choice(possible_weights)] for i in range(num_qubits - 1)]
    graph.add_weighted_edges_from(graph_info)
    qaoa_layers = [3, 5, 7]
    trials = 2
    num_random_trials = 10
    optimal_sol = "10101"
    approx_ratio_random_mean, approx_ratio_random_std = calc_random_stats(
        num_qubits, graph_info, shots, num_random_trials, optimal_sol
    )
    job_data = LinearRampQAOAData(
        provider_job_ids=["test_job_id"],
        num_qubits=5,
        graph_info=graph_info,
        graph_type="1D",
        optimal_sol=optimal_sol,
        trials=trials,
        num_random_trials=num_random_trials,
        confidence_level=0.99,
        num_shots=100,
        qaoa_layers=qaoa_layers,
        seed=123,
        delta_beta=1.0,
        delta_gamma=1.0,
        approx_ratio_random_mean=approx_ratio_random_mean,
        approx_ratio_random_std=approx_ratio_random_std,
        circuit_encoding="Direct",
    )
    counts = [
        {"01010": 40, "10101": 30, "11000": 30},
        {"01010": 50, "10101": 50},
        {"00000": 50, "10101": 30, "01010": 20},
        {"11111": 50, "10101": 50},
        {"01010": 50, "10101": 50},
        {"11111": 50, "00000": 50},
    ]
    stats = calc_stats(job_data, counts)
    assert stats.confidence_pass == [True, True, False]
    assert len(stats.confidence_pass) == len(qaoa_layers)
    assert [round(i, 6) for i in stats.optimal_probability] == [0.6, 1.0, 0.25]
    assert [round(i, 6) for i in stats.approx_ratio] == [0.645, 1.0, 0.25]
    assert round(approx_ratio_random_mean, 1) == 0.5


@pytest.mark.parametrize("width, length", [(5, 5), (7, 7), (3, 3)])
def test_distributed_edges(width, length):
    # Heavy-Hex layout
    graph = nx.hexagonal_lattice_graph(width, length)
    layers = distribute_edges(graph)
    assert len(layers) == 3
    # Square layout
    graph = nx.grid_2d_graph(width, length)
    layers = distribute_edges(graph)
    assert len(layers) == 4


@pytest.mark.parametrize("nq", [6, 10, 16])
def test_simulated_annealing_solver(nq):
    graph = nx.random_regular_graph(3, nq)
    possible_weights = [0.1, 0.2, 0.3, 0.5, 1.0]
    for u, v in graph.edges():
        graph[u][v]["weight"] = random.choice(possible_weights)

    optimal_solution = weighted_maxcut_solver(graph)
    assert len(optimal_solution) == nq
    assert all(bit in "01" for bit in optimal_solution)

    # Check if the solution is valid
    cost = cost_maxcut(optimal_solution, graph)
    assert cost >= 0


@pytest.mark.parametrize("nq", [5, 10, 15, 20])
def test_swap_network(nq):
    list_pairs = SWAP_pairs(nq)
    assert len(list_pairs) == nq
    assert [len(layer) for layer in list_pairs] == (
        len(list_pairs) * [nq // 2]
        if nq % 2 == 1
        else [(nq // 2 if i % 2 == 0 else nq // 2 - 1) for i in range(len(list_pairs))]
    )
