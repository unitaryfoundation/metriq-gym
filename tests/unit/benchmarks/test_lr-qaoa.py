import pytest
import networkx as nx
import random
import rustworkx as rx
from types import SimpleNamespace
from unittest.mock import Mock, patch

from metriq_gym.benchmarks.lr_qaoa import (
    LinearRampQAOA,
    prepare_qaoa_circuit,
    calc_stats,
    cost_maxcut,
    plant_partition,
    weighted_maxcut_solver,
    calc_random_stats,
)
from metriq_gym.benchmarks.lr_qaoa import LinearRampQAOAData
from metriq_gym.circuits import distribute_edges, SWAP_pairs
from metriq_gym.exceptions import DeviceCapacityError


def test_lr_qaoa_rejects_oversized_workload_before_connectivity_lookup():
    benchmark = LinearRampQAOA(
        SimpleNamespace(),
        SimpleNamespace(num_qubits=50),
    )
    device = SimpleNamespace(
        id="arn:aws:braket:us-east-1::device/qpu/ionq/Forte-1",
        num_qubits=36,
    )

    with (
        patch("metriq_gym.benchmarks.lr_qaoa.connectivity_graph") as connectivity_mock,
        pytest.raises(DeviceCapacityError, match="Requested 50 qubits"),
    ):
        benchmark._build_circuits(device)

    connectivity_mock.assert_not_called()


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
        shots=100,
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
    # effective_approx_ratio is floored at zero when approx_ratio is below the
    # random baseline (third layer: 0.25 < ~0.5), and positive otherwise.
    assert stats.effective_approx_ratio[0] > 0
    assert stats.effective_approx_ratio[1] == 1.0
    assert stats.effective_approx_ratio[2] == 0.0
    assert all(r >= 0 for r in stats.effective_approx_ratio)


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


@pytest.mark.parametrize(
    "num_qubits, expected_cost", [(10, 3.0), (20, 6.9), (50, 18.6), (100, 43.0)]
)
def test_chain_reference_normalizes_optimal_samples(num_qubits, expected_cost, capsys):
    # These widths and graph seed match the Metriq Score 1.0 suite. Previously
    # annealing could underestimate the optimum and make perfect samples score >1.
    params = SimpleNamespace(
        num_qubits=num_qubits,
        graph_type="1D",
        qaoa_layers=[10],
        trials=1,
        shots=1000,
        num_random_trials=25,
        confidence_level=0.999,
        delta_beta=0.3,
        delta_gamma=0.6,
        seed=123,
    )
    benchmark = LinearRampQAOA(SimpleNamespace(), params)
    device = SimpleNamespace(
        id="test-device",
        num_qubits=num_qubits,
        run=Mock(return_value=SimpleNamespace(id="test-job")),
    )
    with (
        patch(
            "metriq_gym.benchmarks.lr_qaoa.connectivity_graph",
            return_value=rx.generators.path_graph(num_qubits),
        ),
        patch(
            "metriq_gym.benchmarks.lr_qaoa.SimulatedAnnealingSampler",
            side_effect=AssertionError("Chains should not need annealing"),
        ),
    ):
        data = benchmark.dispatch_handler(device)

    graph = nx.Graph()
    graph.add_weighted_edges_from(data.graph_info)
    assert cost_maxcut(data.optimal_sol, graph) == pytest.approx(expected_cost)

    exact_solution = "01" * (num_qubits // 2)
    stats = calc_stats(data, [{exact_solution: params.shots}])
    assert stats.approx_ratio == pytest.approx([1.0])
    assert stats.effective_approx_ratio == pytest.approx([1.0])
    assert stats.optimal_probability == pytest.approx([1.0])
    assert "There is a better cost" not in capsys.readouterr().out


def test_bipartite_reference_preserves_node_indices_across_components():
    graph = nx.Graph()
    graph.add_nodes_from([3, 1, 0, 4, 2, 5])
    graph.add_weighted_edges_from([(0, 2, 0.1), (2, 1, 1.0), (3, 4, 0.0)])
    with patch(
        "metriq_gym.benchmarks.lr_qaoa.SimulatedAnnealingSampler",
        side_effect=AssertionError("Nonnegative bipartite graphs should not need annealing"),
    ):
        solution = weighted_maxcut_solver(graph)

    assert len(solution) == 6
    assert set(solution) <= {"0", "1"}
    assert cost_maxcut(solution, graph) == pytest.approx(1.1)


@pytest.mark.parametrize(
    "edges, sample, expected_solution",
    [
        ([(0, 1, 1.0), (1, 2, 1.0), (0, 2, 1.0)], {0: -1, 1: 1, 2: -1}, "010"),
        ([(0, 1, 1.0), (1, 2, -1.0), (0, 2, -1.0)], {0: -1, 1: 1, 2: 1}, "011"),
    ],
    ids=["odd-cycle", "unbalanced-signs"],
)
def test_other_graphs_keep_annealing_reference(edges, sample, expected_solution):
    graph = nx.Graph()
    graph.add_weighted_edges_from(edges)
    with patch("metriq_gym.benchmarks.lr_qaoa.SimulatedAnnealingSampler") as sampler:
        sampler.return_value.sample.return_value = [sample]
        solution = weighted_maxcut_solver(graph)

    sampler.return_value.sample.assert_called_once()
    assert solution == expected_solution


def _dispatch_without_annealing(graph_type, device_graph, seed=123):
    num_qubits = device_graph.num_nodes()
    params = SimpleNamespace(
        num_qubits=num_qubits,
        graph_type=graph_type,
        qaoa_layers=[1],
        trials=1,
        shots=100,
        num_random_trials=5,
        confidence_level=0.999,
        delta_beta=0.3,
        delta_gamma=0.6,
        seed=seed,
    )
    benchmark = LinearRampQAOA(SimpleNamespace(), params)
    device = SimpleNamespace(
        id="test-device",
        num_qubits=num_qubits,
        run=Mock(return_value=SimpleNamespace(id="test-job")),
    )
    with (
        patch("metriq_gym.benchmarks.lr_qaoa.connectivity_graph", return_value=device_graph),
        patch(
            "metriq_gym.benchmarks.lr_qaoa.SimulatedAnnealingSampler",
            side_effect=AssertionError("Generated instances should not need annealing"),
        ),
    ):
        data = benchmark.dispatch_handler(device)
    graph = nx.Graph()
    graph.add_nodes_from(range(num_qubits))
    graph.add_weighted_edges_from(data.graph_info)
    return data, graph


def _brute_force_maxcut(graph):
    n = graph.number_of_nodes()
    return max(cost_maxcut(format(k, f"0{n}b"), graph) for k in range(2**n))


@pytest.mark.parametrize(
    "device_graph",
    [
        rx.generators.heavy_hex_graph(3),
        rx.generators.heavy_square_graph(3),
        rx.generators.grid_graph(4, 4),
    ],
    ids=["heavy-hex", "heavy-square", "square"],
)
def test_bipartite_native_layout_reference_cuts_every_edge(device_graph):
    data, graph = _dispatch_without_annealing("NL", device_graph)

    weights = [w for _, _, w in data.graph_info]
    assert all(w > 0 for w in weights)
    assert cost_maxcut(data.optimal_sol, graph) == pytest.approx(sum(weights))


def test_non_bipartite_native_layout_plants_reference():
    device_graph = rx.generators.grid_graph(3, 3)
    device_graph.add_edge(0, 4, None)  # A diagonal coupler creates a triangle.
    data, graph = _dispatch_without_annealing("NL", device_graph)

    assert any(w < 0 for _, _, w in data.graph_info)
    assert cost_maxcut(data.optimal_sol, graph) == pytest.approx(_brute_force_maxcut(graph))


@pytest.mark.parametrize("num_qubits", [3, 6, 10])
@pytest.mark.parametrize("seed", [0, 1, 123])
@pytest.mark.parametrize("encoding", ["Direct", "SWAP"])
def test_fully_connected_reference_is_planted_optimum(num_qubits, seed, encoding):
    device_graph = (
        rx.generators.complete_graph(num_qubits)
        if encoding == "Direct"
        else rx.generators.path_graph(num_qubits)
    )
    data, graph = _dispatch_without_annealing("FC", device_graph, seed)

    assert data.circuit_encoding == encoding
    positive_weight = sum(w for _, _, w in data.graph_info if w > 0)
    assert positive_weight > 0
    assert cost_maxcut(data.optimal_sol, graph) == pytest.approx(positive_weight)
    assert cost_maxcut(data.optimal_sol, graph) == pytest.approx(_brute_force_maxcut(graph))

    stats = calc_stats(data, [{data.optimal_sol: data.shots}])
    assert stats.approx_ratio == pytest.approx([1.0])
    assert stats.optimal_probability == pytest.approx([1.0])


def test_plant_partition_keeps_both_sides_nonempty():
    graph_info = [(0, 1, 0.5)]
    for seed in range(50):
        random.seed(seed)
        assert plant_partition(2, graph_info) == [(0, 1, 0.5)]


@pytest.mark.parametrize("nq", [5, 10, 15, 20])
def test_swap_network(nq):
    list_pairs = SWAP_pairs(nq)
    assert len(list_pairs) == nq
    assert [len(layer) for layer in list_pairs] == (
        len(list_pairs) * [nq // 2]
        if nq % 2 == 1
        else [(nq // 2 if i % 2 == 0 else nq // 2 - 1) for i in range(len(list_pairs))]
    )
