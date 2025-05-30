from unittest.mock import MagicMock
import pytest
from qbraid import QuantumDevice
from qbraid.runtime import QiskitBackend

from rustworkx import PyGraph
import rustworkx as rx
from metriq_gym.qplatform.device import connectivity_graph


def test_device_connectivity_graph_qiskit_backend():
    mock_backend = MagicMock()
    mock_graph = PyGraph()
    mock_graph.add_nodes_from(range(3))
    mock_graph.add_edge(0, 1, None)
    mock_graph.add_edge(1, 2, None)
    mock_backend.coupling_map.graph.to_undirected.return_value = mock_graph

    mock_device = MagicMock(spec=QiskitBackend)
    mock_device._backend = mock_backend

    graph = connectivity_graph(mock_device)

    assert isinstance(graph, PyGraph)
    assert set(graph.nodes()) == {0, 1, 2}
    assert set(graph.edge_list()) == {(0, 1), (1, 2)}


def test_device_connectivity_graph_invalid_device():
    mock_device = MagicMock(spec=QuantumDevice)  # Mock an unknown QuantumDevice
    with pytest.raises(NotImplementedError, match="Connectivity graph not implemented for device"):
        connectivity_graph(mock_device)
class TestLocalSimulators:
    """Test suite for local simulator functionality - adapter agnostic."""

    @pytest.fixture
    def mock_simulator_adapter(self):
        """Generic mock adapter that works with any simulator type."""
        mock_adapter = MagicMock()
        mock_adapter.get_backend_info.return_value = {
            'num_qubits': 32,
            'basis_gates': ['u1', 'u2', 'u3', 'cx', 'id', 'x', 'y', 'z', 'h'],
            'coupling_map': None,
            'method': 'automatic',
            'backend_name': 'local_simulator',
        }
        mock_adapter.get_version.return_value = "1.0.0"
        return mock_adapter

    def test_local_device_connectivity_graph_bipartite(self, mock_simulator_adapter):
        """Test that ANY local simulator generates bipartite connectivity graphs (critical for BSEQ)."""
        from metriq_gym.qplatform.device import LocalDevice
        
        # Test with generic device - should work with any simulator
        device = LocalDevice("any.simulator.method", mock_simulator_adapter)
        graph = connectivity_graph(device)
        
        assert isinstance(graph, PyGraph)
        assert graph.num_nodes() > 0
        
        # CRITICAL: Graph must be bipartite for BSEQ benchmark to work
        try:
            edge_coloring = rx.graph_bipartite_edge_color(graph)
            assert isinstance(edge_coloring, dict)
        except rx.GraphNotBipartite:
            pytest.fail("Local simulator connectivity graph must be bipartite for BSEQ compatibility")

    def test_local_device_universal_creation_and_metadata(self, mock_simulator_adapter):
        """Test that LocalDevice works universally with any simulator adapter."""
        from metriq_gym.qplatform.device import LocalDevice
        
        # Test with generic device spec - should work with any simulator
        device = LocalDevice("any.simulator.method", mock_simulator_adapter)
        
        # Verify universal device properties
        assert hasattr(device, 'id')
        assert device.device_type == "SIMULATOR"
        assert isinstance(device.num_qubits, int)
        assert device.num_qubits > 0
        assert hasattr(device, 'profile')
        
        # Verify metadata structure (universal for all simulators)
        metadata = device.metadata()
        required_fields = ['device_id', 'device_type', 'num_qubits', 'status', 'local', 'simulator']
        for field in required_fields:
            assert field in metadata
        
        assert metadata['device_type'] == "SIMULATOR"
        assert metadata['status'] == 'ONLINE'
        assert metadata['local'] is True
        assert metadata['simulator'] is True

    def test_available_simulators_auto_discovery(self):
        """Test that system automatically discovers and supports available simulators."""
        from metriq_gym.qplatform.device import get_available_simulators, create_local_device
        
        # Get whatever simulators are available in the current environment
        available_simulators = get_available_simulators()
        
        # Should return a dictionary of available simulators
        assert isinstance(available_simulators, dict)
        
        # If any simulators are available, test that they can be created
        if available_simulators:
            # Test first available simulator (adapter agnostic)
            first_simulator_id = next(iter(available_simulators.keys()))
            first_description = available_simulators[first_simulator_id]
            
            # Verify structure
            assert isinstance(first_simulator_id, str)
            assert isinstance(first_description, str)
            assert "." in first_simulator_id  # Should be in format "backend.simulator[.method]"
            
            # Test that it can be created (this validates the entire plugin architecture)
            try:
                device = create_local_device(first_simulator_id)
                assert hasattr(device, 'id')
                assert device.id == first_simulator_id
            except ImportError:
                # Simulator not installed in test environment - that's OK
                pytest.skip(f"Simulator {first_simulator_id} not available in test environment")
        else:
            # No simulators available - that's OK for pure test environments
            pytest.skip("No local simulators available in test environment")
