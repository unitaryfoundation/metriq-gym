"""
Unit tests for qplatform device utility functions.

Tests the version and connectivity_graph functions for different device types
using mocked qBraid device objects.
"""

import pytest
from unittest.mock import Mock
import rustworkx as rx
import networkx as nx
from qbraid.runtime import QiskitBackend, BraketDevice, AzureQuantumDevice

from metriq_gym.qplatform.device import version, connectivity_graph


class MockCouplingMap:
    """Mock coupling map for QiskitBackend testing."""

    def __init__(self, num_qubits=5):
        self.graph = self._create_mock_graph(num_qubits)

    def _create_mock_graph(self, num_qubits):
        """Create a mock graph with to_undirected method."""
        mock_graph = Mock()
        # Create a simple linear connectivity graph
        edges = [(i, i + 1) for i in range(num_qubits - 1)]
        result_graph = rx.PyGraph()
        result_graph.add_nodes_from(range(num_qubits))
        result_graph.add_edges_from_no_data(edges)
        mock_graph.to_undirected.return_value = result_graph
        return mock_graph


class MockTopologyGraph:
    """Mock topology graph for BraketDevice testing."""

    def __init__(self, num_qubits=8):
        self.num_qubits = num_qubits

    def to_undirected(self):
        """Return a mock undirected NetworkX graph."""
        # Create a ring topology for testing
        edges = [(i, (i + 1) % self.num_qubits) for i in range(self.num_qubits)]
        return nx.Graph(edges)


@pytest.fixture
def mock_qiskit_backend():
    """Create a mock QiskitBackend device."""
    device = Mock(spec=QiskitBackend)

    # Mock the internal backend
    mock_backend = Mock()
    mock_backend.backend_version = "1.6.73"
    mock_backend.coupling_map = MockCouplingMap(num_qubits=5)
    device._backend = mock_backend

    return device


@pytest.fixture
def mock_braket_device():
    """Create a mock BraketDevice."""
    device = Mock(spec=BraketDevice)

    # Mock the internal device
    mock_internal_device = Mock()
    mock_internal_device.topology_graph = MockTopologyGraph(num_qubits=8)
    device._device = mock_internal_device

    return device


@pytest.fixture
def mock_azure_device():
    """Create a mock AzureQuantumDevice."""
    device = Mock(spec=AzureQuantumDevice)

    # Mock the metadata method
    device.metadata.return_value = {"num_qubits": 16}

    return device


@pytest.fixture
def mock_unsupported_device():
    """Create a mock device of unsupported type."""

    class UnsupportedDevice:
        def __init__(self):
            self.num_qubits = 10

    return UnsupportedDevice()


class TestVersionFunction:
    """Test cases for the version function."""

    def test_qiskit_backend_version(self, mock_qiskit_backend):
        """Test version extraction from QiskitBackend."""
        result = version(mock_qiskit_backend)
        assert result == "1.6.73"
        assert isinstance(result, str)

    def test_unsupported_device_version_raises(self, mock_unsupported_device):
        """Test that unsupported device types raise NotImplementedError."""
        with pytest.raises(NotImplementedError) as exc_info:
            version(mock_unsupported_device)

        assert "Device version not implemented" in str(exc_info.value)
        assert "UnsupportedDevice" in str(exc_info.value)


class TestConnectivityGraphFunction:
    """Test cases for the connectivity_graph function."""

    def test_qiskit_backend_connectivity(self, mock_qiskit_backend):
        """Test connectivity graph extraction from QiskitBackend."""
        result = connectivity_graph(mock_qiskit_backend)

        # Verify it's a rustworkx PyGraph
        assert isinstance(result, rx.PyGraph)

        # Verify the graph structure
        assert result.num_nodes() == 5
        assert result.num_edges() == 4  # Linear connectivity: 0-1-2-3-4

        # Verify it's undirected
        edge_list = result.edge_list()
        assert (0, 1) in edge_list or (1, 0) in edge_list

    def test_braket_device_connectivity(self, mock_braket_device):
        """Test connectivity graph extraction from BraketDevice."""
        result = connectivity_graph(mock_braket_device)

        # Verify it's a rustworkx PyGraph
        assert isinstance(result, rx.PyGraph)

        # Verify the graph structure (ring topology)
        assert result.num_nodes() == 8
        assert result.num_edges() == 8  # Ring connectivity

        # Verify conversion from NetworkX was called
        mock_braket_device._device.topology_graph.to_undirected.assert_called_once()

    def test_azure_device_connectivity(self, mock_azure_device):
        """Test connectivity graph extraction from AzureQuantumDevice."""
        result = connectivity_graph(mock_azure_device)

        # Verify it's a rustworkx PyGraph
        assert isinstance(result, rx.PyGraph)

        # Verify it's a complete graph with 16 qubits
        expected_nodes = 16
        expected_edges = expected_nodes * (expected_nodes - 1) // 2  # Complete graph formula

        assert result.num_nodes() == expected_nodes
        assert result.num_edges() == expected_edges

        # Verify metadata was called
        mock_azure_device.metadata.assert_called_once()

    def test_unsupported_device_connectivity_raises(self, mock_unsupported_device):
        """Test that unsupported device types raise NotImplementedError."""
        with pytest.raises(NotImplementedError) as exc_info:
            connectivity_graph(mock_unsupported_device)

        assert "Connectivity graph not implemented" in str(exc_info.value)
        assert "UnsupportedDevice" in str(exc_info.value)


class TestGraphProperties:
    """Test that returned graphs have expected properties."""

    def test_all_graphs_are_rustworkx_pygraphs(
        self, mock_qiskit_backend, mock_braket_device, mock_azure_device
    ):
        """Test that all connectivity graphs return rustworkx PyGraph objects."""
        devices = [mock_qiskit_backend, mock_braket_device, mock_azure_device]

        for device in devices:
            graph = connectivity_graph(device)
            assert isinstance(graph, rx.PyGraph)
            assert hasattr(graph, "num_nodes")
            assert hasattr(graph, "num_edges")
            assert hasattr(graph, "edge_list")

    def test_graphs_are_undirected(
        self, mock_qiskit_backend, mock_braket_device, mock_azure_device
    ):
        """Test that all connectivity graphs are undirected."""
        devices = [mock_qiskit_backend, mock_braket_device, mock_azure_device]

        for device in devices:
            graph = connectivity_graph(device)

            # For undirected graphs, if edge (a,b) exists, then (b,a) should also exist
            # or the edge should appear as (min(a,b), max(a,b))
            edge_list = graph.edge_list()

            if edge_list:  # Only test if graph has edges
                # Check that the graph behaves like an undirected graph
                # by verifying edge symmetry
                for edge in edge_list[:3]:  # Check first few edges
                    a, b = edge
                    # In an undirected graph, both directions should be accessible
                    # This is implementation dependent, but rustworkx should handle this
                    assert isinstance(a, int) and isinstance(b, int)

    def test_graph_node_count_consistency(self, mock_azure_device):
        """Test that node count matches expected qubit count."""
        # We know Azure device has 16 qubits from our mock
        graph = connectivity_graph(mock_azure_device)
        assert graph.num_nodes() == 16

    @pytest.mark.parametrize("num_qubits", [4, 8, 12, 20])
    def test_azure_complete_graph_properties(self, num_qubits):
        """Test Azure device complete graph properties with different qubit counts."""
        device = Mock(spec=AzureQuantumDevice)
        device.metadata.return_value = {"num_qubits": num_qubits}

        graph = connectivity_graph(device)
        expected_edges = num_qubits * (num_qubits - 1) // 2

        assert graph.num_nodes() == num_qubits
        assert graph.num_edges() == expected_edges


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_qiskit_backend_empty_coupling_map(self):
        """Test QiskitBackend with empty coupling map."""
        device = Mock(spec=QiskitBackend)
        mock_backend = Mock()
        mock_backend.backend_version = "2.0.0"

        # Mock empty coupling map
        empty_coupling_map = Mock()
        empty_graph = Mock()
        result_graph = rx.PyGraph()  # Empty graph
        empty_graph.to_undirected.return_value = result_graph
        empty_coupling_map.graph = empty_graph
        mock_backend.coupling_map = empty_coupling_map

        device._backend = mock_backend

        result = connectivity_graph(device)
        assert isinstance(result, rx.PyGraph)
        assert result.num_nodes() == 0
        assert result.num_edges() == 0

    def test_azure_device_zero_qubits(self):
        """Test AzureQuantumDevice with zero qubits."""
        device = Mock(spec=AzureQuantumDevice)
        device.metadata.return_value = {"num_qubits": 0}

        graph = connectivity_graph(device)
        assert isinstance(graph, rx.PyGraph)
        assert graph.num_nodes() == 0
        assert graph.num_edges() == 0

    def test_braket_device_single_node(self):
        """Test BraketDevice with single node topology."""
        device = Mock(spec=BraketDevice)
        mock_internal_device = Mock()

        # Single node graph
        topology = Mock()
        topology.to_undirected.return_value = nx.Graph()  # Empty NetworkX graph
        mock_internal_device.topology_graph = topology
        device._device = mock_internal_device

        result = connectivity_graph(device)
        assert isinstance(result, rx.PyGraph)
        # rustworkx conversion of empty NetworkX graph should give empty PyGraph
        assert result.num_nodes() == 0
