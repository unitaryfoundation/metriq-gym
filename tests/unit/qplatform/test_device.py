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

from metriq_gym.local.provider import LocalProvider
from metriq_gym.qplatform.device import version, connectivity_graph


class MockCouplingMap:
    """Mock coupling map for QiskitBackend testing."""

    def __init__(self, num_qubits=5):
        self.graph = self._create_mock_graph(num_qubits)

    def _create_mock_graph(self, num_qubits):
        mock_graph = Mock()
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
        self.to_undirected = Mock(return_value=self._create_nx_graph())

    def _create_nx_graph(self):
        edges = [(i, (i + 1) % self.num_qubits) for i in range(self.num_qubits)]
        return nx.Graph(edges)


@pytest.fixture
def mock_qiskit_backend():
    device = Mock(spec=QiskitBackend)
    mock_backend = Mock()
    mock_backend.backend_version = "1.6.73"
    mock_backend.coupling_map = MockCouplingMap(num_qubits=5)
    device._backend = mock_backend
    return device


@pytest.fixture
def mock_braket_device():
    device = Mock(spec=BraketDevice)
    mock_internal_device = Mock()
    mock_internal_device.topology_graph = MockTopologyGraph(num_qubits=8)
    device._device = mock_internal_device
    device._provider_name = "Rigetti"
    device.num_qubits = 8
    return device


@pytest.fixture
def mock_azure_device():
    device = Mock(spec=AzureQuantumDevice)
    device.metadata.return_value = {"num_qubits": 16}
    return device


@pytest.fixture
def mock_unsupported_device():
    class UnsupportedDevice:
        def __init__(self):
            self.num_qubits = 10

    return UnsupportedDevice()


class TestVersionFunction:
    """Test cases for the version function."""

    def test_qiskit_backend_version(self, mock_qiskit_backend):
        result = version(mock_qiskit_backend)
        assert result == "1.6.73"
        assert isinstance(result, str)

    def test_unsupported_device_version_raises(self, mock_unsupported_device):
        with pytest.raises(NotImplementedError) as exc_info:
            version(mock_unsupported_device)
        assert "Device version not implemented" in str(exc_info.value)
        assert "UnsupportedDevice" in str(exc_info.value)

    def test_local_aer_device_version(self):
        provider = LocalProvider()
        device = provider.get_device("aer_simulator")
        assert isinstance(version(device), str)


class TestConnectivityGraphFunction:
    """Test cases for the connectivity_graph function."""

    def test_qiskit_backend_connectivity(self, mock_qiskit_backend):
        result = connectivity_graph(mock_qiskit_backend)
        assert isinstance(result, rx.PyGraph)
        assert result.num_nodes() == 5
        assert result.num_edges() == 4
        edge_list = result.edge_list()
        assert (0, 1) in edge_list or (1, 0) in edge_list

    def test_braket_device_connectivity(self, mock_braket_device):
        result = connectivity_graph(mock_braket_device)
        assert isinstance(result, rx.PyGraph)
        assert result.num_nodes() == 8
        assert result.num_edges() == 8
        mock_braket_device._device.topology_graph.to_undirected.assert_called_once()

    def test_braket_device_all_to_all_connectivity_amazon_braket(self):
        mock_num_qubits = 4
        device = Mock(spec=BraketDevice)
        device._provider_name = "Amazon Braket"
        device.num_qubits = mock_num_qubits

        result = connectivity_graph(device)
        assert isinstance(result, rx.PyGraph)
        assert result.num_nodes() == mock_num_qubits
        # All-to-all connectivity: n*(n-1)/2 edges
        expected_edges = mock_num_qubits * (mock_num_qubits - 1) // 2
        assert result.num_edges() == expected_edges

    def test_braket_device_all_to_all_connectivity_ionq(self):
        mock_num_qubits = 3
        device = Mock(spec=BraketDevice)
        device._provider_name = "IonQ"
        device.num_qubits = mock_num_qubits

        result = connectivity_graph(device)
        assert isinstance(result, rx.PyGraph)
        assert result.num_nodes() == mock_num_qubits
        # All-to-all connectivity: n*(n-1)/2 edges
        expected_edges = mock_num_qubits * (mock_num_qubits - 1) // 2
        assert result.num_edges() == expected_edges

    def test_azure_device_connectivity(self, mock_azure_device):
        result = connectivity_graph(mock_azure_device)
        assert isinstance(result, rx.PyGraph)
        expected_nodes = 16
        expected_edges = expected_nodes * (expected_nodes - 1) // 2
        assert result.num_nodes() == expected_nodes
        assert result.num_edges() == expected_edges
        mock_azure_device.metadata.assert_called_once()

    def test_unsupported_device_connectivity_raises(self, mock_unsupported_device):
        with pytest.raises(NotImplementedError) as exc_info:
            connectivity_graph(mock_unsupported_device)
        assert "Connectivity graph not implemented" in str(exc_info.value)
        assert "UnsupportedDevice" in str(exc_info.value)


class TestGraphProperties:
    """Test that returned graphs have expected properties."""

    def test_all_graphs_are_rustworkx_pygraphs(
        self, mock_qiskit_backend, mock_braket_device, mock_azure_device
    ):
        devices = [mock_qiskit_backend, mock_braket_device, mock_azure_device]
        for device in devices:
            graph = connectivity_graph(device)
            assert isinstance(graph, rx.PyGraph)
            assert hasattr(graph, "num_nodes")
            assert hasattr(graph, "num_edges")
            assert hasattr(graph, "edge_list")

    @pytest.mark.parametrize(
        "fixture_name",
        ["mock_qiskit_backend", "mock_braket_device", "mock_azure_device"],
    )
    def test_graphs_are_undirected(self, request, fixture_name):
        device = request.getfixturevalue(fixture_name)
        graph = connectivity_graph(device)
        for u, v in graph.edge_list():
            assert isinstance(u, int) and isinstance(v, int)

    def test_graph_node_count_consistency(self, mock_azure_device):
        graph = connectivity_graph(mock_azure_device)
        assert graph.num_nodes() == 16

    @pytest.mark.parametrize("num_qubits", [4, 8, 12, 20])
    def test_azure_complete_graph_properties(self, num_qubits):
        device = Mock(spec=AzureQuantumDevice)
        device.metadata.return_value = {"num_qubits": num_qubits}
        graph = connectivity_graph(device)
        expected_edges = num_qubits * (num_qubits - 1) // 2
        assert graph.num_nodes() == num_qubits
        assert graph.num_edges() == expected_edges


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_qiskit_backend_empty_coupling_map(self):
        device = Mock(spec=QiskitBackend)
        mock_backend = Mock()
        mock_backend.backend_version = "2.0.0"
        empty_coupling_map = Mock()
        empty_graph = Mock()
        result_graph = rx.PyGraph()
        empty_graph.to_undirected.return_value = result_graph
        empty_coupling_map.graph = empty_graph
        mock_backend.coupling_map = empty_coupling_map
        device._backend = mock_backend

        result = connectivity_graph(device)
        assert isinstance(result, rx.PyGraph)
        assert result.num_nodes() == 0
        assert result.num_edges() == 0

    def test_azure_device_zero_qubits(self):
        device = Mock(spec=AzureQuantumDevice)
        device.metadata.return_value = {"num_qubits": 0}
        graph = connectivity_graph(device)
        assert isinstance(graph, rx.PyGraph)
        assert graph.num_nodes() == 0
        assert graph.num_edges() == 0

    def test_braket_device_single_node(self):
        device = Mock(spec=BraketDevice)
        mock_internal_device = Mock()
        topology = Mock()
        topology.to_undirected = Mock(return_value=nx.Graph())
        mock_internal_device.topology_graph = topology
        device._device = mock_internal_device
        device._provider_name = "Rigetti"  # non-all-to-all device
        device.num_qubits = 0

        result = connectivity_graph(device)
        assert isinstance(result, rx.PyGraph)
        assert result.num_nodes() == 0
