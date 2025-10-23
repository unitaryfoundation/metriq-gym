"""
Unit tests for qplatform device utility functions.

Tests the version and connectivity_graph functions for different device types
using mocked qBraid device objects.
"""

import types
from unittest.mock import Mock

import pytest
import rustworkx as rx
import networkx as nx
from qbraid.runtime import QiskitBackend, BraketDevice, AzureQuantumDevice

from metriq_gym.local.provider import LocalProvider
from metriq_gym.origin.device import OriginDevice
from metriq_gym.qplatform.device import version, connectivity_graph, normalized_metadata


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


def _make_origin_device(
    *, high=None, available=None, edges=None, double_edges=None, num_qubits=102
):
    class StubDoubleInfo:
        def __init__(self, qubits):
            self._qubits = qubits

        def get_qubits(self):
            return self._qubits

    class StubChipInfo:
        def __init__(self):
            self._high = [] if high is None else list(high)
            self._available = [] if available is None else list(available)
            self._edges = [] if edges is None else list(edges)

        def high_frequency_qubits(self):
            return self._high

        def available_qubits(self):
            return self._available

        def get_chip_topology(self):
            return self._edges

        def double_qubits_info(self):
            if not double_edges:
                return []
            return [StubDoubleInfo(pair) for pair in double_edges]

        def qubits_num(self):
            return num_qubits

        def get_basic_gates(self):
            return ["x", "cx"]

    class StubBackend:
        def __init__(self):
            self._chip_info = StubChipInfo()

        def chip_info(self):
            return self._chip_info

    backend = StubBackend()
    device = OriginDevice(
        provider=types.SimpleNamespace(),
        device_id="WK_C102_400",
        backend=backend,
        backend_name="WK_C102_400",
    )
    return device


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

    def test_braket_device_all_to_all_connectivity_amazon_braket_simulators(self):
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

    def test_azure_device_connectivity(self, mock_azure_device):
        result = connectivity_graph(mock_azure_device)
        assert isinstance(result, rx.PyGraph)
        expected_nodes = 16
        expected_edges = expected_nodes * (expected_nodes - 1) // 2
        assert result.num_nodes() == expected_nodes
        assert result.num_edges() == expected_edges
        mock_azure_device.metadata.assert_called_once()

    def test_origin_device_connectivity_reindexes_labels(self):
        device = _make_origin_device(
            high=[4, 18, 21], available=[4, 18, 21], edges=[(4, 18), (18, 21), (4, 21)]
        )

        graph = connectivity_graph(device)

        assert isinstance(graph, rx.PyGraph)
        assert graph.num_nodes() == 3
        remapped_edges = {tuple(sorted(edge[:2])) for edge in graph.edge_list()}
        assert remapped_edges == {(0, 1), (0, 2), (1, 2)}

    def test_origin_device_connectivity_filters_unused_available_qubits(self):
        device = _make_origin_device(high=[1, 9], available=[0, 1, 7, 9, 23], edges=[(1, 9)])

        graph = connectivity_graph(device)

        assert isinstance(graph, rx.PyGraph)
        assert graph.num_nodes() == 2
        remapped_edges = {tuple(sorted(edge[:2])) for edge in graph.edge_list()}
        assert remapped_edges == {(0, 1)}

    def test_origin_device_connectivity_prefers_high_frequency_subset(self):
        device = _make_origin_device(
            high=[30, 41], available=list(range(42)), edges=[], double_edges=[(3, 30), (3, 41)]
        )

        graph = connectivity_graph(device)

        assert isinstance(graph, rx.PyGraph)
        assert graph.num_nodes() == 3
        remapped_edges = {tuple(sorted(edge[:2])) for edge in graph.edge_list()}
        assert remapped_edges == {(0, 1), (0, 2)}

    def test_origin_device_connectivity_uses_double_qubits_info(self):
        device = _make_origin_device(
            high=[5, 7, 9], available=[5, 6, 7, 8, 9], edges=[], double_edges=[(5, 7), (7, 9)]
        )

        graph = connectivity_graph(device)

        assert isinstance(graph, rx.PyGraph)
        assert graph.num_nodes() == 3
        remapped_edges = {tuple(sorted(edge[:2])) for edge in graph.edge_list()}
        assert remapped_edges == {(0, 1), (1, 2)}

    def test_origin_device_connectivity_uses_available_qubits_without_edges(self):
        device = _make_origin_device(
            high=[7, 9, 11], available=[7, 9, 11, 15], edges=[], num_qubits=12
        )

        graph = connectivity_graph(device)

        assert isinstance(graph, rx.PyGraph)
        assert graph.num_nodes() == 3
        assert graph.num_edges() == 0

    def test_origin_device_connectivity_falls_back_to_complete_graph(self):
        device = _make_origin_device(high=[], available=[], edges=[], num_qubits=6)

        graph = connectivity_graph(device)

        assert isinstance(graph, rx.PyGraph)
        assert graph.num_nodes() == 6
        expected_edges = 6 * 5 // 2
        assert graph.num_edges() == expected_edges

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


class TestNormalizedMetadata:
    """Tests for normalized_metadata() helper."""

    def test_qiskit_backend_metadata(self, mock_qiskit_backend):
        meta = normalized_metadata(mock_qiskit_backend)
        # Only version is known for this mock; simulator/num_qubits absent
        assert isinstance(meta, dict)
        assert meta.get("version") == "1.6.73"
        assert "num_qubits" not in meta
        assert "simulator" not in meta

    def test_local_aer_device_metadata(self):
        provider = LocalProvider()
        device = provider.get_device("aer_simulator")
        meta = normalized_metadata(device)
        # Local Aer should report simulator flag and a version string; num_qubits should be int
        assert isinstance(meta, dict)
        assert meta.get("simulator") is True
        assert isinstance(meta.get("version"), str) and meta["version"]
        assert isinstance(meta.get("num_qubits"), int)

    def test_braket_device_metadata(self, mock_braket_device):
        meta = normalized_metadata(mock_braket_device)
        # For mocked Braket device, only num_qubits is set
        assert isinstance(meta, dict)
        assert meta.get("num_qubits") == 8
        assert "version" not in meta
        assert "simulator" not in meta

    def test_unsupported_device_metadata(self):
        class Unsupported:
            pass

        meta = normalized_metadata(Unsupported())
        assert meta == {}

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
