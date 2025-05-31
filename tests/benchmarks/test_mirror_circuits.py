
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
    generate_mirror_circuit,
    random_paulis,
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
        assert circuit.depth() >= 0  # Should have some gates

    def test_edge_grab(self):
        """Test edge selection for two-qubit gates."""
        graph = nx.Graph()
        graph.add_edges_from([(0, 1), (1, 2), (2, 3)])
        random_state = np.random.RandomState(42)
        
        selected_edges = edge_grab(0.5, graph, random_state)
        
        assert isinstance(selected_edges, nx.Graph)
        assert len(selected_edges.nodes) == 4

    def test_random_cliffords(self):
        """Test random Clifford gate generation."""
        graph = nx.Graph()
        graph.add_edge(0, 1)
        random_state = np.random.RandomState(42)
        
        circuit = random_cliffords(graph, random_state, "CNOT")
        
        assert circuit.num_qubits == 2

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
        
        with pytest.raises(ValueError, match="two_qubit_gate_name must be 'CNOT' or 'CZ'"):
            generate_mirror_circuit(
                nlayers=1,
                two_qubit_gate_prob=0.5,
                connectivity_graph=graph,
                two_qubit_gate_name="INVALID"
            )


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
        params.seed = 42
        return params

    @pytest.fixture
    def benchmark(self, mock_params):
        """Create a MirrorCircuits benchmark instance."""
        args = MagicMock()
        return MirrorCircuits(args, mock_params)

    @patch('metriq_gym.benchmarks.mirror_circuits.connectivity_graph')
    def test_dispatch_handler(self, mock_connectivity_graph, benchmark, mock_device):
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
        
        result = benchmark.dispatch_handler(mock_device)
        
        assert isinstance(result, MirrorCircuitsData)
        assert result.nlayers == 2
        assert result.two_qubit_gate_prob == 0.5
        assert result.two_qubit_gate_name == "CNOT"
        assert result.shots == 100
        assert result.seed == 42
        assert result.expected_bitstring == "000"
        assert result.provider_job_ids == ["test_job_id"]

    def test_poll_handler_perfect_success(self, benchmark):
        """Test poll handler with perfect success rate."""
        job_data = MirrorCircuitsData(
            provider_job_ids=["test_job"],
            nlayers=2,
            two_qubit_gate_prob=0.5,
            two_qubit_gate_name="CNOT",
            shots=100,
            num_qubits=2,
            seed=42,
            expected_bitstring="00"
        )
        
        # Mock perfect results (all measurements return expected bitstring)
        counts = MeasCount({"00": 100})
        result_data = [GateModelResultData(measurement_counts=counts)]
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

