# File: tests/benchmarks/test_mirror_circuits.py
"""
Unit tests for mirror circuits benchmark.

Updated to work without qiskit_aer dependency, using qiskit.quantum_info.Statevector
for more efficient and lightweight simulation in testing.
"""

import pytest
from qbraid import QuantumJob
import rustworkx as rx
import numpy as np
from unittest.mock import MagicMock, patch

from metriq_gym.benchmarks.mirror_circuits import (
    MirrorCircuits,
    MirrorCircuitsData,
    MirrorCircuitsResult,
    TwoQubitGateType,
    generate_mirror_circuit,
    random_paulis,
    random_single_cliffords,
    edge_grab,
    random_cliffords,
    select_optimal_qubit_subset,
    create_subgraph_from_qubits,
)
from qbraid.runtime.result_data import MeasCount, GateModelResultData


def _edge_pairs(graph: rx.PyGraph) -> set[tuple[int, int]]:
    """Normalize edge tuples into unordered pairs for assertions.

    rustworkx edges may appear as ``(u, v)`` or ``(u, v, data)`` tuples. Only the endpoints
    matter for these topology checks, so extra elements (edge weights/data) are intentionally
    ignored.
    """
    pairs: set[tuple[int, int]] = set()
    for edge in graph.edge_list():
        if len(edge) == 2:
            u, v = edge
        elif len(edge) >= 3:
            # rustworkx includes edge data at index 2; discard it for endpoint comparisons
            u, v = edge[0], edge[1]
        else:
            raise AssertionError(f"Unexpected edge format: {edge}")
        pairs.add((min(u, v), max(u, v)))
    return pairs


class TestMirrorCircuitGeneration:
    def test_random_paulis(self):
        graph = rx.PyGraph()
        graph.add_nodes_from([0, 1, 2])
        random_state = np.random.RandomState(42)

        circuit = random_paulis(graph, random_state)

        assert circuit.num_qubits == 3
        assert circuit.depth() >= 0
        assert circuit.size() >= 0

    def test_random_paulis_empty_graph(self):
        graph = rx.PyGraph()
        random_state = np.random.RandomState(42)

        circuit = random_paulis(graph, random_state)

        assert circuit.num_qubits == 0

    def test_random_single_cliffords(self):
        graph = rx.PyGraph()
        graph.add_nodes_from([0, 1, 2])
        random_state = np.random.RandomState(42)

        circuit = random_single_cliffords(graph, random_state)

        assert circuit.num_qubits == 3
        assert circuit.depth() >= 0
        assert circuit.size() >= 0

        # Test determinism with same seed
        random_state2 = np.random.RandomState(42)
        circuit2 = random_single_cliffords(graph, random_state2, base_seed=42)

        # With same seed, should produce equivalent circuits
        assert circuit.num_qubits == circuit2.num_qubits

    def test_random_single_cliffords_empty_graph(self):
        graph = rx.PyGraph()
        random_state = np.random.RandomState(42)

        circuit = random_single_cliffords(graph, random_state)

        assert circuit.num_qubits == 0

    def test_edge_grab(self):
        graph = rx.PyGraph()
        # First add nodes, then edges (RustWorkX requirement)
        graph.add_nodes_from([0, 1, 2, 3])
        graph.add_edges_from([(0, 1, None), (1, 2, None), (2, 3, None)])
        random_state = np.random.RandomState(42)

        selected_edges = edge_grab(0.5, graph, random_state)

        assert isinstance(selected_edges, rx.PyGraph)
        assert len(selected_edges.node_indices()) >= 0

        # Test select_optimal_qubit_subset
        subset = select_optimal_qubit_subset(graph, 2)
        assert len(subset) == 2
        assert all(qubit in graph.node_indices() for qubit in subset)

        # Test create_subgraph_from_qubits
        subgraph = create_subgraph_from_qubits(graph, subset)
        assert len(subgraph.node_indices()) == 2
        assert all(node in range(len(subset)) for node in subgraph.node_indices())

        # Test edge cases for subset selection
        full_subset = select_optimal_qubit_subset(graph, 10)
        assert len(full_subset) == 4

        single_subset = select_optimal_qubit_subset(graph, 1)
        assert len(single_subset) == 1

    def test_edge_grab_zero_probability(self):
        graph = rx.PyGraph()
        # First add nodes, then edges (RustWorkX requirement)
        graph.add_nodes_from([0, 1, 2])
        graph.add_edges_from([(0, 1, None), (1, 2, None)])
        random_state = np.random.RandomState(42)

        selected_edges = edge_grab(0.0, graph, random_state)

        assert len(selected_edges.edge_list()) == 0

    def test_random_cliffords(self):
        graph = rx.PyGraph()
        # First add nodes, then edges (RustWorkX requirement)
        graph.add_nodes_from([0, 1, 2])
        graph.add_edge(0, 1, None)
        # Node 2 is already added and will be isolated
        random_state = np.random.RandomState(42)

        circuit = random_cliffords(graph, random_state, "CNOT")

        assert circuit.num_qubits == 3
        assert circuit.depth() >= 1

        # Should contain a CNOT gate on edge (0,1)
        gate_counts = circuit.count_ops()
        assert "cx" in gate_counts or "cnot" in gate_counts

    def test_random_cliffords_invalid_gate(self):
        graph = rx.PyGraph()
        # First add nodes, then edges (RustWorkX requirement)
        graph.add_nodes_from([0, 1])
        graph.add_edge(0, 1, None)
        random_state = np.random.RandomState(42)

        with pytest.raises(ValueError, match="Unsupported two-qubit gate"):
            random_cliffords(graph, random_state, "INVALID")

    def test_random_cliffords_empty_graph(self):
        graph = rx.PyGraph()
        random_state = np.random.RandomState(42)

        circuit = random_cliffords(graph, random_state, "CNOT")

        assert circuit.num_qubits == 0


    def test_generate_mirror_circuit(self):
        # 2-node connected graph
        graph = rx.PyGraph()
        graph.add_nodes_from([0, 1])
        graph.add_edges_from([(0, 1, None)])
    
        # Deterministic generation with a seed
        circuit_1, expected_1 = generate_mirror_circuit(
            num_layers=2,
            two_qubit_gate_prob=0.5,
            connectivity_graph=graph,
            two_qubit_gate_name="CNOT",
            seed=42,
        )
        assert circuit_1.num_qubits == 2
        assert isinstance(expected_1, str) and set(expected_1) <= {"0", "1"} and len(expected_1) == 2
    
        # Same seed should give same expected bitstring
        circuit_2, expected_2 = generate_mirror_circuit(
            num_layers=2,
            two_qubit_gate_prob=0.5,
            connectivity_graph=graph,
            two_qubit_gate_name="CNOT",
            seed=42,
        )
        assert expected_2 == expected_1
    
        # Also test a 3-qubit subgraph path taken from a larger graph
        big = rx.PyGraph()
        big.add_nodes_from([0, 1, 2, 3, 4])
        big.add_edges_from([(0, 1, None), (1, 2, None), (2, 3, None), (3, 4, None)])
        subset = select_optimal_qubit_subset(big, 3)
        subgraph_3 = create_subgraph_from_qubits(big, subset)
        assert len(subgraph_3.node_indices()) == 3
    
        circuit_sub, bitstring_sub = generate_mirror_circuit(
            num_layers=1,
            two_qubit_gate_prob=0.5,
            connectivity_graph=subgraph_3,
            seed=42,
        )
        assert circuit_sub.num_qubits == 3
        assert isinstance(bitstring_sub, str) and len(bitstring_sub) == 3 and set(bitstring_sub) <= {"0", "1"}

    def test_generate_mirror_circuit_invalid_prob(self):
        graph = rx.PyGraph()
        graph.add_node(0)

        with pytest.raises(ValueError, match="two_qubit_gate_prob must be between 0 and 1"):
            generate_mirror_circuit(num_layers=1, two_qubit_gate_prob=1.5, connectivity_graph=graph)

    def test_generate_mirror_circuit_invalid_gate(self):
        graph = rx.PyGraph()
        graph.add_node(0)

        with pytest.raises(ValueError, match="two_qubit_gate_name must be one of"):
            generate_mirror_circuit(
                num_layers=1,
                two_qubit_gate_prob=0.5,
                connectivity_graph=graph,
                two_qubit_gate_name="INVALID",
            )
            
    def test_generate_mirror_circuit_empty_graph(self):
        graph = rx.PyGraph()  # empty graph
    
        circuit, expected_bitstring = generate_mirror_circuit(
            num_layers=1,
            two_qubit_gate_prob=0.5,
            connectivity_graph=graph,
            seed=99,
        )
    
        # By convention in the benchmark, an empty graph falls back to a single-qubit circuit.
        assert circuit.num_qubits == 1
        assert expected_bitstring == "0"

class TestTwoQubitGateType:
    def test_enum_values(self):
        assert TwoQubitGateType.CNOT == "CNOT"
        assert TwoQubitGateType.CZ == "CZ"
        assert list(TwoQubitGateType) == ["CNOT", "CZ"]

    def test_enum_validation(self):
        assert TwoQubitGateType("CNOT") == TwoQubitGateType.CNOT
        assert TwoQubitGateType("CZ") == TwoQubitGateType.CZ

        with pytest.raises(ValueError):
            TwoQubitGateType("INVALID")


class TestMirrorCircuitsBenchmark:
    @pytest.fixture
    def mock_device(self):
        device = MagicMock()
        device.num_qubits = 3
        return device

    @pytest.fixture
    def mock_params(self):
        params = MagicMock()
        params.num_layers = 2
        params.two_qubit_gate_prob = 0.5
        params.two_qubit_gate_name = "CNOT"
        params.num_shots = 100
        params.num_circuits = 5
        params.seed = 42
        return params

    @pytest.fixture
    def mock_params_minimal(self):
        params = MagicMock()
        params.num_layers = 1
        params.two_qubit_gate_prob = 0.5
        params.two_qubit_gate_name = "CNOT"
        params.num_shots = 100
        params.num_circuits = 1
        params.seed = None
        return params

    @pytest.fixture
    def benchmark(self, mock_params):
        args = MagicMock()
        return MirrorCircuits(args, mock_params)

    @pytest.fixture
    def benchmark_minimal(self, mock_params_minimal):
        args = MagicMock()
        return MirrorCircuits(args, mock_params_minimal)

    @patch("metriq_gym.benchmarks.mirror_circuits.connectivity_graph")
    @patch("metriq_gym.benchmarks.mirror_circuits.generate_mirror_circuit")
    def test_dispatch_handler(
        self,
        mock_generate_circuit,
        mock_connectivity_graph,
        benchmark,
        mock_device,
    ):
        # Mock connectivity graph
        mock_graph = rx.PyGraph()
        mock_graph.add_nodes_from([0, 1, 2])
        mock_connectivity_graph.return_value = mock_graph

        # Mock circuit generation to return consistent expected_bitstring
        mock_circuit = MagicMock()
        mock_generate_circuit.return_value = (mock_circuit, "001")

        # Mock device.run
        mock_job = MagicMock(spec=QuantumJob)
        mock_job.id = "test_job_id"
        mock_device.run.return_value = mock_job

        result = benchmark.dispatch_handler(mock_device)

        NUM_CIRCUITS = 5
        assert isinstance(result, MirrorCircuitsData)
        assert result.num_layers == 2
        assert result.two_qubit_gate_prob == 0.5
        assert result.two_qubit_gate_name == "CNOT"
        assert result.num_shots == 100
        assert result.num_circuits == NUM_CIRCUITS
        assert result.seed == 42
        assert result.expected_bitstrings == ["001"] * NUM_CIRCUITS
        assert result.provider_job_ids == ["test_job_id"]

        # Verify that generate_mirror_circuit was called correctly
        assert mock_generate_circuit.call_count == 5  # num_circuits times
        for call in mock_generate_circuit.call_args_list:
            line_graph = call.kwargs["connectivity_graph"]
            assert isinstance(line_graph, rx.PyGraph)
            assert set(line_graph.node_indices()) == {0, 1, 2}
            assert _edge_pairs(line_graph) == {(0, 1), (1, 2)}

        # Test width parameter functionality
        mock_params_with_width = MagicMock()
        mock_params_with_width.num_layers = 2
        mock_params_with_width.two_qubit_gate_prob = 0.5
        mock_params_with_width.two_qubit_gate_name = "CNOT"
        mock_params_with_width.shots = 100
        mock_params_with_width.num_circuits = 1
        mock_params_with_width.seed = 42
        mock_params_with_width.width = 2

        benchmark_with_width = MirrorCircuits(MagicMock(), mock_params_with_width)

        mock_generate_circuit.reset_mock()

        result_with_width = benchmark_with_width.dispatch_handler(mock_device)

        assert result_with_width.num_qubits == 2
        mock_generate_circuit.assert_called_once()
        width_call_graph = mock_generate_circuit.call_args.kwargs["connectivity_graph"]
        assert set(width_call_graph.node_indices()) == {0, 1}
        assert _edge_pairs(width_call_graph) == {(0, 1)}

        # Test width parameter validation
        mock_params_invalid = MagicMock()
        mock_params_invalid.num_layers = 1
        mock_params_invalid.two_qubit_gate_prob = 0.5
        mock_params_invalid.two_qubit_gate_name = "CNOT"
        mock_params_invalid.shots = 100
        mock_params_invalid.num_circuits = 1
        mock_params_invalid.seed = 42
        mock_params_invalid.width = 10

        benchmark_invalid = MirrorCircuits(MagicMock(), mock_params_invalid)

        with pytest.raises(ValueError, match="exceeds device capacity"):
            benchmark_invalid.dispatch_handler(mock_device)

    def test_poll_handler_perfect_success(self, benchmark):
        job_data = MirrorCircuitsData(
            provider_job_ids=["test_job"],
            num_layers=2,
            two_qubit_gate_prob=0.5,
            two_qubit_gate_name="CNOT",
            num_shots=100,
            num_qubits=2,
            num_circuits=2,
            seed=42,
            expected_bitstrings=["01"],
        )

        # Mock perfect results (all measurements return expected bitstring)
        counts1 = MeasCount({"01": 100})  # Matches expected_bitstrings
        counts2 = MeasCount({"01": 100})
        result_data = [GateModelResultData(measurement_counts=[counts1, counts2])]
        quantum_jobs = [MagicMock()]

        result = benchmark.poll_handler(job_data, result_data, quantum_jobs)

        assert isinstance(result, MirrorCircuitsResult)
        assert result.success_probability.value == 1.0
        assert result.binary_success is True
        # Polarization should be 1.0 for perfect success
        expected_polarization = (1.0 - 0.25) / (1.0 - 0.25)  # (S - 1/4) / (1 - 1/4)
        assert result.polarization.value == expected_polarization

    def test_poll_handler_partial_success(self, benchmark):
        job_data = MirrorCircuitsData(
            provider_job_ids=["test_job"],
            num_layers=2,
            two_qubit_gate_prob=0.5,
            two_qubit_gate_name="CNOT",
            num_shots=100,
            num_qubits=2,
            num_circuits=1,
            seed=42,
            expected_bitstrings=["11"],
        )

        # Mock partial success (70% success rate for "11")
        counts = MeasCount({"11": 70, "01": 10, "10": 10, "00": 10})
        result_data = [GateModelResultData(measurement_counts=counts)]
        quantum_jobs = [MagicMock()]

        result = benchmark.poll_handler(job_data, result_data, quantum_jobs)

        assert isinstance(result, MirrorCircuitsResult)
        assert result.success_probability.value == 0.7
        assert result.binary_success is True  # 0.7 > 2/3

        # Calculate expected polarization
        baseline = 0.25  # 1/4 for 2 qubits
        expected_polarization = (0.7 - baseline) / (1.0 - baseline)
        assert abs(result.polarization.value - expected_polarization) < 1e-10

    def test_poll_handler_failure(self, benchmark):
        job_data = MirrorCircuitsData(
            provider_job_ids=["test_job"],
            num_layers=2,
            two_qubit_gate_prob=0.5,
            two_qubit_gate_name="CNOT",
            num_shots=100,
            num_qubits=2,
            num_circuits=1,
            seed=42,
            expected_bitstrings=["10"],
        )

        # Mock low success (10% success rate for "10", below 1/e threshold)
        counts = MeasCount({"10": 10, "01": 30, "11": 30, "00": 30})
        result_data = [GateModelResultData(measurement_counts=counts)]
        quantum_jobs = [MagicMock()]

        result = benchmark.poll_handler(job_data, result_data, quantum_jobs)

        assert isinstance(result, MirrorCircuitsResult)
        assert result.success_probability.value == 0.1
        assert result.binary_success is False  # 0.1 < 1/e

    def test_poll_handler_zero_qubits_error(self, benchmark):
        job_data = MirrorCircuitsData(
            provider_job_ids=["test_job"],
            num_layers=1,
            two_qubit_gate_prob=0.5,
            two_qubit_gate_name="CNOT",
            num_shots=100,
            num_qubits=0,
            num_circuits=1,
            seed=42,
            expected_bitstrings=[""],
        )

        result_data = []
        quantum_jobs = []

        with pytest.raises(ValueError, match="Mirror circuits benchmark requires at least 1 qubit"):
            benchmark.poll_handler(job_data, result_data, quantum_jobs)

    def test_poll_handler_multiple_circuits(self, benchmark):
        job_data = MirrorCircuitsData(
            provider_job_ids=["test_job1", "test_job2"],
            num_layers=1,
            two_qubit_gate_prob=0.5,
            two_qubit_gate_name="CNOT",
            num_shots=100,
            num_qubits=2,
            num_circuits=2,
            seed=42,
            expected_bitstrings=["01", "01"],
        )

        # Mock results from two circuits - both looking for "01"
        counts1 = MeasCount({"01": 80, "00": 20})  # 80% success
        counts2 = MeasCount({"01": 60, "11": 40})  # 60% success
        result_data = [
            GateModelResultData(measurement_counts=counts1),
            GateModelResultData(measurement_counts=counts2),
        ]
        quantum_jobs = [MagicMock(), MagicMock()]

        result = benchmark.poll_handler(job_data, result_data, quantum_jobs)

        # Average success should be (80 + 60) / (100 + 100) = 70%
        assert isinstance(result, MirrorCircuitsResult)
        assert result.success_probability.value == 0.7
        assert result.binary_success is True  # 0.7 > 2/3
