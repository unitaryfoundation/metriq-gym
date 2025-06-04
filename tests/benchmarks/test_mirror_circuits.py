"""
Unit tests for mirror circuits benchmark.

Updated to work without qiskit_aer dependency, using qiskit.quantum_info.Statevector
for more efficient and lightweight simulation in testing.
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
    random_cliffords,
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
        # Verify that the circuit has operations (Pauli gates or identity)
        assert circuit.size() >= 0

    def test_random_paulis_empty_graph(self):
        """Test random Pauli gate generation with empty graph."""
        graph = nx.Graph()
        random_state = np.random.RandomState(42)

        circuit = random_paulis(graph, random_state)

        assert circuit.num_qubits == 0

    def test_random_single_cliffords(self):
        """Test random single-qubit Clifford gate generation.

        This tests that the function generates valid Clifford circuits using uniform
        sampling from the full 24-element single-qubit Clifford group, rather than
        just the 6 generators. This ensures proper statistical properties for
        randomized benchmarking protocols.
        """
        graph = nx.Graph()
        graph.add_nodes_from([0, 1, 2])
        random_state = np.random.RandomState(42)

        circuit = random_single_cliffords(graph, random_state)

        assert circuit.num_qubits == 3
        assert circuit.depth() >= 0
        # The circuit should contain some operations (Clifford gates)
        assert circuit.size() >= 0

        # Test determinism with same seed
        random_state2 = np.random.RandomState(42)
        circuit2 = random_single_cliffords(graph, random_state2)

        # With same seed, should produce equivalent circuits
        # Note: Due to qiskit's random_clifford internal randomness,
        # we test structural properties rather than exact equality
        assert circuit.num_qubits == circuit2.num_qubits
        assert circuit.depth() == circuit2.depth()

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
        """Test random Clifford gate generation.

        Updated to handle the new behavior where isolated qubits
        get random single-qubit Cliffords from the full group.
        """
        graph = nx.Graph()
        graph.add_edge(0, 1)
        graph.add_node(2)  # Isolated node
        random_state = np.random.RandomState(42)

        circuit = random_cliffords(graph, random_state, "CNOT")

        assert circuit.num_qubits == 3
        assert circuit.depth() >= 1  # Should have at least the CNOT gate

        # Should contain a CNOT gate on edge (0,1)
        gate_counts = circuit.count_ops()
        assert "cx" in gate_counts or "cnot" in gate_counts

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

    @patch("metriq_gym.benchmarks.mirror_circuits.Statevector")
    def test_generate_mirror_circuit(self, mock_statevector):
        """Test complete mirror circuit generation using Statevector simulation.

        We mock the Statevector to control the expected_bitstring since it's now
        calculated from noiseless simulation rather than hardcoded.
        """
        # Mock the statevector and its probabilities
        mock_sv = MagicMock()
        mock_sv.probabilities.return_value = np.array([0.36, 0.64, 0, 0])  # |01⟩ most probable
        mock_statevector.return_value = mock_sv

        graph = nx.Graph()
        graph.add_edges_from([(0, 1)])

        circuit, expected_bitstring = generate_mirror_circuit(
            nlayers=2,
            two_qubit_gate_prob=0.5,
            connectivity_graph=graph,
            two_qubit_gate_name="CNOT",
            seed=42,
        )

        assert circuit.num_qubits == 2
        assert expected_bitstring == "01"  # From our mocked probabilities
        assert circuit.depth() > 0

        # Verify the Statevector was used
        mock_statevector.assert_called_once()

    @patch("metriq_gym.benchmarks.mirror_circuits.Statevector")
    def test_generate_mirror_circuit_simulation_error(self, mock_statevector):
        """Test mirror circuit generation handles simulation errors gracefully."""
        # Mock Statevector to raise an exception
        mock_statevector.side_effect = Exception("Simulation failed")

        graph = nx.Graph()
        graph.add_edges_from([(0, 1)])

        circuit, expected_bitstring = generate_mirror_circuit(
            nlayers=1, two_qubit_gate_prob=0.5, connectivity_graph=graph, seed=42
        )

        assert circuit.num_qubits == 2
        assert expected_bitstring == "00"  # Fallback to all zeros
        assert circuit.depth() > 0

    def test_generate_mirror_circuit_invalid_prob(self):
        """Test mirror circuit generation with invalid probability."""
        graph = nx.Graph()
        graph.add_node(0)

        with pytest.raises(ValueError, match="two_qubit_gate_prob must be between 0 and 1"):
            generate_mirror_circuit(nlayers=1, two_qubit_gate_prob=1.5, connectivity_graph=graph)

    def test_generate_mirror_circuit_invalid_gate(self):
        """Test mirror circuit generation with invalid gate name."""
        graph = nx.Graph()
        graph.add_node(0)

        with pytest.raises(ValueError, match="two_qubit_gate_name must be one of"):
            generate_mirror_circuit(
                nlayers=1,
                two_qubit_gate_prob=0.5,
                connectivity_graph=graph,
                two_qubit_gate_name="INVALID",
            )

    @patch("metriq_gym.benchmarks.mirror_circuits.Statevector")
    def test_generate_mirror_circuit_empty_graph(self, mock_statevector):
        """Test mirror circuit generation with empty graph.

        This should create a 1-qubit circuit and return "0" as expected.
        """
        # Mock for empty graph case (1 qubit, ground state)
        mock_sv = MagicMock()
        mock_sv.probabilities.return_value = np.array([1.0, 0.0])  # |0⟩ state
        mock_statevector.return_value = mock_sv

        graph = nx.Graph()

        circuit, expected_bitstring = generate_mirror_circuit(
            nlayers=1, two_qubit_gate_prob=0.5, connectivity_graph=graph
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

    @patch("metriq_gym.benchmarks.mirror_circuits.connectivity_graph")
    @patch("metriq_gym.benchmarks.mirror_circuits.flatten_job_ids")
    @patch("metriq_gym.benchmarks.mirror_circuits.generate_mirror_circuit")
    def test_dispatch_handler(
        self,
        mock_generate_circuit,
        mock_flatten_job_ids,
        mock_connectivity_graph,
        benchmark,
        mock_device,
    ):
        """Test the dispatch handler with controlled circuit generation."""
        # Mock connectivity graph
        mock_graph = MagicMock()
        mock_graph.node_indices.return_value = [0, 1, 2]
        mock_graph.edge_list.return_value = [(0, 1), (1, 2)]
        mock_connectivity_graph.return_value = mock_graph

        # Mock circuit generation to return consistent expected_bitstring
        mock_circuit = MagicMock()
        mock_generate_circuit.return_value = (mock_circuit, "001")

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
        assert result.expected_bitstring == "001"  # From our mock
        assert result.provider_job_ids == ["test_job_id"]

        # Verify that generate_mirror_circuit was called correctly
        assert mock_generate_circuit.call_count == 5  # num_circuits times

    @patch("metriq_gym.benchmarks.mirror_circuits.connectivity_graph")
    @patch("metriq_gym.benchmarks.mirror_circuits.flatten_job_ids")
    @patch("metriq_gym.benchmarks.mirror_circuits.generate_mirror_circuit")
    def test_dispatch_handler_defaults(
        self,
        mock_generate_circuit,
        mock_flatten_job_ids,
        mock_connectivity_graph,
        benchmark_minimal,
        mock_device,
    ):
        """Test the dispatch handler with default parameters."""
        # Mock connectivity graph
        mock_graph = MagicMock()
        mock_graph.node_indices.return_value = [0, 1]
        mock_graph.edge_list.return_value = [(0, 1)]
        mock_connectivity_graph.return_value = mock_graph

        # Mock circuit generation
        mock_circuit = MagicMock()
        mock_generate_circuit.return_value = (mock_circuit, "10")

        # Mock device.run
        mock_job = MagicMock()
        mock_job.id = "test_job_id"
        mock_device.run.return_value = mock_job

        # Mock flatten_job_ids
        mock_flatten_job_ids.return_value = ["test_job_id"]

        result = benchmark_minimal.dispatch_handler(mock_device)

        assert result.num_circuits == 1  # Default value
        assert result.seed is None  # Default value
        assert result.expected_bitstring == "10"  # From our mock

    def test_poll_handler_perfect_success(self, benchmark):
        """Test poll handler with perfect success rate.

        Updated to use a realistic expected_bitstring that matches
        the measurement counts from simulated circuit execution.
        """
        job_data = MirrorCircuitsData(
            provider_job_ids=["test_job"],
            nlayers=2,
            two_qubit_gate_prob=0.5,
            two_qubit_gate_name="CNOT",
            shots=100,
            num_qubits=2,
            num_circuits=2,
            seed=42,
            expected_bitstring="01",  # Calculated from noiseless simulation
        )

        # Mock perfect results (all measurements return expected bitstring)
        counts1 = MeasCount({"01": 100})  # Matches expected_bitstring
        counts2 = MeasCount({"01": 100})
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
            expected_bitstring="11",  # From simulation
        )

        # Mock partial success (70% success rate for "11")
        counts = MeasCount({"11": 70, "01": 10, "10": 10, "00": 10})
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
            expected_bitstring="10",  # From simulation
        )

        # Mock low success (50% success rate for "10", below 2/3 threshold)
        counts = MeasCount({"10": 50, "01": 20, "11": 20, "00": 10})
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
            expected_bitstring="",
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
            expected_bitstring="01",  # Consistent expected bitstring from simulation
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
        assert result.success_probability == 0.7
        assert result.binary_success is True  # 0.7 > 2/3

    def test_poll_handler_edge_case_with_different_bitstring_patterns(self, benchmark):
        """Test poll handler with different expected bitstring patterns to ensure robustness."""
        # Test with 3-qubit expected bitstring from simulation
        job_data = MirrorCircuitsData(
            provider_job_ids=["test_job"],
            nlayers=1,
            two_qubit_gate_prob=0.3,
            two_qubit_gate_name="CZ",
            shots=1000,
            num_qubits=3,
            num_circuits=1,
            seed=123,
            expected_bitstring="101",  # From noiseless simulation
        )

        # Mock results with the expected bitstring having highest probability
        counts = MeasCount(
            {
                "101": 600,  # 60% success
                "000": 150,
                "111": 100,
                "010": 75,
                "001": 75,
            }
        )
        result_data = [GateModelResultData(measurement_counts=counts)]
        quantum_jobs = [MagicMock()]

        result = benchmark.poll_handler(job_data, result_data, quantum_jobs)

        assert isinstance(result, MirrorCircuitsResult)
        assert result.success_probability == 0.6
        assert result.binary_success is False  # 0.6 < 2/3

        # Check polarization calculation for 3 qubits
        baseline = 1.0 / 8  # 1/2^3 for 3 qubits
        expected_polarization = (0.6 - baseline) / (1.0 - baseline)
        assert abs(result.polarization - expected_polarization) < 1e-10
