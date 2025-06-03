
"""
Unit tests for mirror circuits benchmark.
"""

import pytest
import networkx as nx
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
    random_cliffords
)
from qbraid.runtime.result_data import MeasCount, GateModelResultData


class TestMirrorCircuitGeneration:
    """Test mirror circuit generation functions."""

    def test_random_paulis(self):
        """Test random Pauli gate generation."""
        graph = nx.Graph()
        graph.add_nodes_from([0, 1, 2])
        random_state = np.random.RandomState(42)
        
        circuit = random_paulis(graph, random_state)
        
        assert circuit.num_qubits == 3
        assert circuit.depth() >= 0

    def test_random_paulis_empty_graph(self):
        """Test random Pauli gate generation with empty graph."""
        graph = nx.Graph()
        random_state = np.random.RandomState(42)
        
        circuit = random_paulis(graph, random_state)
        
        assert circuit.num_qubits == 0

    def test_random_single_cliffords(self):
        """Test random single-qubit Clifford gate generation."""
        graph = nx.Graph()
        graph.add_nodes_from([0, 1, 2])
        random_state = np.random.RandomState(42)
        
        circuit = random_single_cliffords(graph, random_state)
        
        assert circuit.num_qubits == 3
        assert circuit.depth() >= 0

    def test_random_single_cliffords_empty_graph(self):
        """Test random single-qubit Clifford generation with empty graph."""
        graph = nx.Graph()
        random_state = np.random.RandomState(42)
        
        circuit = random_single_cliffords(graph, random_state)
        
        assert circuit.num_qubits == 0

    def test_edge_grab(self):
        """Test edge selection for two-qubit gates."""
        graph = nx.Graph()
        graph.add_edges_from([(0, 1), (1, 2), (2, 3)])
        random_state = np.random.RandomState(42)
        
        selected_edges = edge_grab(0.5, graph, random_state)
        
        assert isinstance(selected_edges, nx.Graph)
        assert len(selected_edges.nodes) == 4

    def test_edge_grab_zero_probability(self):
        """Test edge selection with zero probability."""
        graph = nx.Graph()
        graph.add_edges_from([(0, 1), (1, 2)])
        random_state = np.random.RandomState(42)
        
        selected_edges = edge_grab(0.0, graph, random_state)
        
        assert len(selected_edges.edges) == 0
        assert len(selected_edges.nodes) == 3

    def test_random_cliffords(self):
        """Test random Clifford gate generation."""
        graph = nx.Graph()
        graph.add_edge(0, 1)
        random_state = np.random.RandomState(42)
        
        circuit = random_cliffords(graph, random_state, "CNOT")
        
        assert circuit.num_qubits == 2

    def test_random_cliffords_invalid_gate(self):
        """Test random Clifford generation with invalid gate name."""
        graph = nx.Graph()
        graph.add_edge(0, 1)
        random_state = np.random.RandomState(42)
        
        with pytest.raises(ValueError, match="Unsupported two-qubit gate"):
            random_cliffords(graph, random_state, "INVALID")

    def test_random_cliffords_empty_graph(self):
        """Test random Clifford generation with empty graph."""
        graph = nx.Graph()
        random_state = np.random.RandomState(42)
        
        circuit = random_cliffords(graph, random_state, "CNOT")
        
        assert circuit.num_qubits == 0

    def test_generate_mirror_circuit(self):
        """Test complete mirror circuit generation."""
        graph = nx.Graph()
        graph.add_edges_from([(0, 1), (1, 2)])
        
        circuit, expected_bitstring = generate_mirror_circuit(
            nlayers=2,
            two_qubit_gate_prob=0.5,
            connectivity_graph=graph,
            two_qubit_gate_name="CNOT",
            seed=42
        )
        
        assert circuit.num_qubits == 3
        assert expected_bitstring == "000"
        assert circuit.depth() > 0

    def test_generate_mirror_circuit_invalid_prob(self):
        """Test mirror circuit generation with invalid probability."""
        graph = nx.Graph()
        graph.add_node(0)
        
        with pytest.raises(ValueError, match="two_qubit_gate_prob must be between 0 and 1"):
            generate_mirror_circuit(
                nlayers=1,
                two_qubit_gate_prob=1.5,
                connectivity_graph=graph
            )

    def test_generate_mirror_circuit_invalid_gate(self):
        """Test mirror circuit generation with invalid gate name."""
        graph = nx.Graph()
        graph.add_node(0)
        
        with pytest.raises(ValueError, match="two_qubit_gate_name must be one of"):
            generate_mirror_circuit(
                nlayers=1,
                two_qubit_gate_prob=0.5,
                connectivity_graph=graph,
                two_qubit_gate_name="INVALID"
            )

    def test_generate_mirror_circuit_empty_graph(self):
        """Test mirror circuit generation with empty graph."""
        graph = nx.Graph()
        
        circuit, expected_bitstring = generate_mirror_circuit(
            nlayers=1,
            two_qubit_gate_prob=0.5,
            connectivity_graph=graph
        )
        
        assert circuit.num_qubits == 1
        assert expected_bitstring == "0"


class TestTwoQubitGateType:
    """Test TwoQubitGateType enum."""

    def test_enum_values(self):
        """Test enum contains expected values."""
        assert TwoQubitGateType.CNOT == "CNOT"
        assert TwoQubitGateType.CZ == "CZ"
        assert list(TwoQubitGateType) == ["CNOT", "CZ"]

    def test_enum_validation(self):
        """Test enum validation."""
        assert TwoQubitGateType("CNOT") == TwoQubitGateType.CNOT
        assert TwoQubitGateType("CZ") == TwoQubitGateType.CZ
        
        with pytest.raises(ValueError):
            TwoQubitGateType("INVALID")


class TestMirrorCircuitsBenchmark:
    """Test mirror circuits benchmark class."""

    @pytest.fixture
    def mock_device(self):
        """Create a mock quantum device."""
        device = MagicMock()
        device.num_qubits = 3
        return device

    @pytest.fixture
    def mock_params(self):
        """Create mock benchmark parameters."""
        params = MagicMock()
        params.nlayers = 2
        params.two_qubit_gate_prob = 0.5
        params.two_qubit_gate_name = "CNOT"
        params.shots = 100
        params.num_circuits = 5
        params.seed = 42
        return params

    @pytest.fixture
    def mock_params_minimal(self):
        """Create minimal mock benchmark parameters."""
        params = MagicMock()
        params.nlayers = 1
        params.two_qubit_gate_prob = 0.5
        params.two_qubit_gate_name = "CNOT"
        params.shots = 100
        # num_circuits and seed should not exist to test defaults
        del params.num_circuits
        del params.seed
        return params

    @pytest.fixture
    def benchmark(self, mock_params):
        """Create a MirrorCircuits benchmark instance."""
        args = MagicMock()
        return MirrorCircuits(args, mock_params)

    @pytest.fixture 
    def benchmark_minimal(self, mock_params_minimal):
        """Create a MirrorCircuits benchmark instance with minimal params."""
        args = MagicMock()
        return MirrorCircuits(args, mock_params_minimal)

    @patch('metriq_gym.benchmarks.mirror_circuits.connectivity_graph')
    @patch('metriq_gym.benchmarks.mirror_circuits.flatten_job_ids')
    def test_dispatch_handler(self, mock_flatten_job_ids, mock_connectivity_graph, benchmark, mock_device):
        """Test the dispatch handler."""
        # Mock connectivity graph
        mock_graph = MagicMock()
        mock_graph.node_indices.return_value = [0, 1, 2]
        mock_graph.edge_list.return_value = [(0, 1), (1, 2)]
        mock_connectivity_graph.return_value = mock_graph
        
        # Mock device.run
        mock_job = MagicMock()
        mock_job.id = "test_job_id"
        mock_device.run.return_value = mock_job
        
        # Mock flatten_job_ids to return the expected result
        mock_flatten_job_ids.return_value = ["test_job_id"]
        
        result = benchmark.dispatch_handler(mock_device)
        
        assert isinstance(result, MirrorCircuitsData)
        assert result.nlayers == 2
        assert result.two_qubit_gate_prob == 0.5
        assert result.two_qubit_gate_name == "CNOT"
        assert result.shots == 100
        assert result.num_circuits == 5
        assert result.seed == 42
        assert result.expected_bitstring == "000"
        assert result.provider_job_ids == ["test_job_id"]
        
        # Verify that flatten_job_ids was called with the mock job
        mock_flatten_job_ids.assert_called_once_with(mock_job)

    @patch('metriq_gym.benchmarks.mirror_circuits.connectivity_graph')
    @patch('metriq_gym.benchmarks.mirror_circuits.flatten_job_ids')
    def test_dispatch_handler_defaults(self, mock_flatten_job_ids, mock_connectivity_graph, benchmark_minimal, mock_device):
        """Test the dispatch handler with default parameters."""
        # Mock connectivity graph
        mock_graph = MagicMock()
        mock_graph.node_indices.return_value = [0, 1]
        mock_graph.edge_list.return_value = [(0, 1)]
        mock_connectivity_graph.return_value = mock_graph
        
        # Mock device.run
        mock_job = MagicMock()
        mock_job.id = "test_job_id"
        mock_device.run.return_value = mock_job
        
        # Mock flatten_job_ids
        mock_flatten_job_ids.return_value = ["test_job_id"]
        
        result = benchmark_minimal.dispatch_handler(mock_device)
        
        assert result.num_circuits == 1  # Default value
        assert result.seed is None  # Default value

    def test_poll_handler_perfect_success(self, benchmark):
        """Test poll handler with perfect success rate."""
        job_data = MirrorCircuitsData(
            provider_job_ids=["test_job"],
            nlayers=2,
            two_qubit_gate_prob=0.5,
            two_qubit_gate_name="CNOT",
            shots=100,
            num_qubits=2,
            num_circuits=2,
            seed=42,
            expected_bitstring="00"
        )
        
        # Mock perfect results (all measurements return expected bitstring)
        counts1 = MeasCount({"00": 100})
        counts2 = MeasCount({"00": 100})
        result_data = [GateModelResultData(measurement_counts=[counts1, counts2])]
        quantum_jobs = [MagicMock()]
        
        result = benchmark.poll_handler(job_data, result_data, quantum_jobs)
        
        assert isinstance(result, MirrorCircuitsResult)
        assert result.success_probability == 1.0
        assert result.binary_success is True
        # Polarization should be 1.0 for perfect success
        expected_polarization = (1.0 - 0.25) / (1.0 - 0.25)  # (S - 1/4) / (1 - 1/4)
        assert result.polarization == expected_polarization

    def test_poll_handler_partial_success(self, benchmark):
        """Test poll handler with partial success rate."""
        job_data = MirrorCircuitsData(
            provider_job_ids=["test_job"],
            nlayers=2,
            two_qubit_gate_prob=0.5,
            two_qubit_gate_name="CNOT",
            shots=100,
            num_qubits=2,
            num_circuits=1,
            seed=42,
            expected_bitstring="00"
        )
        
        # Mock partial success (70% success rate)
        counts = MeasCount({"00": 70, "01": 10, "10": 10, "11": 10})
        result_data = [GateModelResultData(measurement_counts=counts)]
        quantum_jobs = [MagicMock()]
        
        result = benchmark.poll_handler(job_data, result_data, quantum_jobs)
        
        assert isinstance(result, MirrorCircuitsResult)
        assert result.success_probability == 0.7
        assert result.binary_success is True  # 0.7 > 2/3
        
        # Calculate expected polarization
        baseline = 0.25  # 1/4 for 2 qubits
        expected_polarization = (0.7 - baseline) / (1.0 - baseline)
        assert abs(result.polarization - expected_polarization) < 1e-10

    def test_poll_handler_failure(self, benchmark):
        """Test poll handler with low success rate."""
        job_data = MirrorCircuitsData(
            provider_job_ids=["test_job"],
            nlayers=2,
            two_qubit_gate_prob=0.5,
            two_qubit_gate_name="CNOT",
            shots=100,
            num_qubits=2,
            num_circuits=1,
            seed=42,
            expected_bitstring="00"
        )
        
        # Mock low success (50% success rate, below 2/3 threshold)
        counts = MeasCount({"00": 50, "01": 20, "10": 20, "11": 10})
        result_data = [GateModelResultData(measurement_counts=counts)]
        quantum_jobs = [MagicMock()]
        
        result = benchmark.poll_handler(job_data, result_data, quantum_jobs)
        
        assert isinstance(result, MirrorCircuitsResult)
        assert result.success_probability == 0.5
        assert result.binary_success is False  # 0.5 < 2/3

    def test_poll_handler_zero_qubits_error(self, benchmark):
        """Test poll handler raises error for zero qubits."""
        job_data = MirrorCircuitsData(
            provider_job_ids=["test_job"],
            nlayers=1,
            two_qubit_gate_prob=0.5,
            two_qubit_gate_name="CNOT",
            shots=100,
            num_qubits=0,
            num_circuits=1,
            seed=42,
            expected_bitstring=""
        )
        
        result_data = []
        quantum_jobs = []
        
        with pytest.raises(ValueError, match="Mirror circuits benchmark requires at least 1 qubit"):
            benchmark.poll_handler(job_data, result_data, quantum_jobs)

    def test_poll_handler_multiple_circuits(self, benchmark):
        """Test poll handler with multiple circuit repetitions."""
        job_data = MirrorCircuitsData(
            provider_job_ids=["test_job1", "test_job2"],
            nlayers=1,
            two_qubit_gate_prob=0.5,
            two_qubit_gate_name="CNOT",
            shots=100,
            num_qubits=2,
            num_circuits=2,
            seed=42,
            expected_bitstring="00"
        )
        
        # Mock results from two circuits
        counts1 = MeasCount({"00": 80, "01": 20})  # 80% success
        counts2 = MeasCount({"00": 60, "01": 40})  # 60% success
        result_data = [
            GateModelResultData(measurement_counts=counts1),
            GateModelResultData(measurement_counts=counts2)
        ]
        quantum_jobs = [MagicMock(), MagicMock()]
        
        result = benchmark.poll_handler(job_data, result_data, quantum_jobs)
        
        # Average success should be (80 + 60) / (100 + 100) = 70%
        assert isinstance(result, MirrorCircuitsResult)
        assert result.success_probability == 0.7
        assert result.binary_success is True  # 0.7 > 2/3

