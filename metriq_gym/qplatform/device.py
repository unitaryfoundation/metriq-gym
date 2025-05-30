
"""Device introspection and metadata utilities for quantum computing platforms.

This module provides standardized, device- and provider-agnostic access to metadata
and introspection utilities for quantum devices, including both remote devices (via qBraid)
and local simulators through a plugin architecture.
"""

from functools import singledispatch
from typing import cast, Dict, Any, Protocol
import tempfile
import json
import uuid
import os
from abc import ABC, abstractmethod

import networkx as nx
from qbraid import QuantumDevice
from qbraid.runtime import BraketDevice, QiskitBackend
import rustworkx as rx


class SimulatorAdapter(Protocol):
    """Protocol defining the interface for simulator adapters."""
    
    def run_circuits(self, circuits, shots: int) -> Any:
        """Run circuits on the simulator backend."""
        ...
    
    def get_backend_info(self) -> Dict[str, Any]:
        """Get backend information (num_qubits, basis_gates, etc.)."""
        ...
    
    def get_version(self) -> str:
        """Get simulator version."""
        ...


class QiskitAerAdapter:
    """Adapter for Qiskit Aer simulators."""
    
    def __init__(self, method: str = "automatic"):
        """Initialize Qiskit Aer adapter.
        
        Args:
            method: Simulation method ('automatic', 'statevector', 'stabilizer', etc.)
        """
        try:
            from qiskit_aer import AerSimulator
            
            if method == "automatic":
                self._backend = AerSimulator()
            else:
                self._backend = AerSimulator(method=method)
                
            self.method = method
            
        except ImportError:
            raise ImportError(
                "Qiskit Aer is required for Qiskit simulators. "
                "Install with: pip install qiskit-aer"
            )
    
    def run_circuits(self, circuits, shots: int):
        """Run circuits on Qiskit Aer."""
        if isinstance(circuits, list):
            results = []
            for circuit in circuits:
                result = self._backend.run(circuit, shots=shots).result()
                results.append(result)
            return results
        else:
            return [self._backend.run(circuits, shots=shots).result()]
    
    def get_backend_info(self) -> Dict[str, Any]:
        """Get Qiskit backend information."""
        config = self._backend.configuration()
        return {
            'num_qubits': getattr(config, 'n_qubits', 64),
            'basis_gates': getattr(config, 'basis_gates', [
                'u1', 'u2', 'u3', 'cx', 'id', 'x', 'y', 'z', 'h', 's', 't', 'sdg', 'tdg'
            ]),
            'coupling_map': getattr(config, 'coupling_map', None),
            'method': self.method,
            'backend_name': config.backend_name,
        }
    
    def get_version(self) -> str:
        """Get Qiskit Aer version."""
        try:
            import qiskit_aer
            return qiskit_aer.__version__
        except (ImportError, AttributeError):
            return "unknown"


# Registry of available simulator adapters
SIMULATOR_ADAPTERS = {
    "qiskit.aer": QiskitAerAdapter,
}

# Topology configurations
TOPOLOGY_CONFIGS = {
    "line": {
        "type": "line",
        "bipartite": True,
        "description": "Linear chain topology (always bipartite)",
    },
    "ring": {
        "type": "ring", 
        "bipartite": lambda n: n % 2 == 0,  # Only even-sized rings are bipartite
        "description": "Ring topology (bipartite only for even number of qubits)",
    },
    "grid": {
        "type": "grid",
        "bipartite": True,
        "description": "2D grid topology (always bipartite)",
    },
    "heavy_hex": {
        "type": "heavy_hex",
        "bipartite": True,
        "description": "IBM heavy hexagon topology",
    },
    "all_to_all": {
        "type": "complete",
        "bipartite": lambda n: n <= 2,  # Only for very small systems
        "description": "Complete graph (rarely bipartite)",
    },
}


def create_topology_graph(topology_type: str, num_qubits: int) -> rx.PyGraph:
    """Create a topology graph based on configuration.
    
    Args:
        topology_type: Type of topology ('line', 'ring', 'grid', etc.)
        num_qubits: Number of qubits
        
    Returns:
        rx.PyGraph: The topology graph
    """
    graph = rx.PyGraph()
    graph.add_nodes_from(range(num_qubits))
    
    if topology_type == "line":
        # Linear chain: 0-1-2-3-...
        for i in range(num_qubits - 1):
            graph.add_edge(i, i + 1, None)
    
    elif topology_type == "ring":
        # Ring: 0-1-2-3-...-0
        for i in range(num_qubits - 1):
            graph.add_edge(i, i + 1, None)
        if num_qubits > 2:
            graph.add_edge(num_qubits - 1, 0, None)
    
    elif topology_type == "grid":
        # 2D grid topology
        import math
        side_length = int(math.sqrt(num_qubits))
        for i in range(num_qubits):
            row, col = divmod(i, side_length)
            # Connect to right neighbor
            if col < side_length - 1:
                graph.add_edge(i, i + 1, None)
            # Connect to bottom neighbor  
            if row < side_length - 1:
                graph.add_edge(i, i + side_length, None)
    
    elif topology_type == "all_to_all":
        # Complete graph (use sparingly)
        for i in range(num_qubits):
            for j in range(i + 1, num_qubits):
                graph.add_edge(i, j, None)
    
    else:
        # Default to line topology for unknown types
        for i in range(num_qubits - 1):
            graph.add_edge(i, i + 1, None)
    
    return graph


def get_available_simulators() -> Dict[str, str]:
    """Get available local simulators.
    
    Returns:
        Dict mapping device IDs to descriptions
    """
    available = {}
    
    # Check Qiskit Aer
    try:
        from qiskit_aer import AerSimulator
        available.update({
            "qiskit.aer.automatic": "Qiskit Aer automatic method selection",
            "qiskit.aer.statevector": "Qiskit Aer statevector simulator",
            "qiskit.aer.stabilizer": "Qiskit Aer stabilizer simulator", 
            "qiskit.aer.density_matrix": "Qiskit Aer density matrix simulator",
        })
    except ImportError:
        pass
    
    # Add other simulators here as they become available
    # try:
    #     import qrack
    #     available.update({
    #         "qrack.cpu": "Qrack CPU simulator",
    #         "qrack.gpu": "Qrack GPU simulator",
    #     })
    # except ImportError:
    #     pass
    
    return available


def create_local_device(device_spec: str) -> 'LocalDevice':
    """Factory function to create local devices based on specification.
    
    Args:
        device_spec: Device specification like 'qiskit.aer.statevector'
        
    Returns:
        LocalDevice: Configured local device
    """
    parts = device_spec.split('.')
    if len(parts) < 2:
        raise ValueError(f"Invalid device spec: {device_spec}. Expected format: 'backend.simulator[.method]'")
    
    backend = parts[0]
    simulator = parts[1]
    method = parts[2] if len(parts) > 2 else "automatic"
    
    adapter_key = f"{backend}.{simulator}"
    if adapter_key not in SIMULATOR_ADAPTERS:
        raise ValueError(f"Unsupported simulator: {adapter_key}")
    
    adapter_class = SIMULATOR_ADAPTERS[adapter_key]
    adapter = adapter_class(method)
    
    return LocalDevice(device_spec, adapter)


class LocalDevice:
    """Local quantum device wrapper using adapter pattern.
    
    This class provides a qBraid-compatible interface for local quantum simulators
    through pluggable adapters, making it extensible to multiple simulator backends.
    """
    
    def __init__(self, device_id: str, adapter: SimulatorAdapter):
        """Initialize a local device with a simulator adapter.
        
        Args:
            device_id: The identifier for the local device
            adapter: The simulator adapter to use
        """
        self.id = device_id
        self.device_type = "SIMULATOR"
        self._adapter = adapter
        self._setup_device()
        
        # Store results cache directory for this session
        self._results_cache = tempfile.mkdtemp(prefix="metriq_local_")
    
    def _setup_device(self):
        """Setup device properties from adapter."""
        backend_info = self._adapter.get_backend_info()
        self.num_qubits = backend_info.get('num_qubits', 64)
        self.profile = LocalDeviceProfile(backend_info)
    
    def run(self, circuits, shots: int = 1000):
        """Run quantum circuits on the local simulator.
        
        Args:
            circuits: Quantum circuit(s) to execute
            shots: Number of measurement shots
            
        Returns:
            LocalJob: A job object containing the execution results
        """
        from metriq_gym.qplatform.job import LocalJob
        
        # Execute circuits immediately (synchronous)
        try:
            results = self._adapter.run_circuits(circuits, shots)
            
            # Create a local job with immediate results
            job_id = str(uuid.uuid4())
            return LocalJob(job_id, results, self._results_cache, self._adapter)
            
        except Exception as e:
            raise RuntimeError(f"Local simulator execution failed: {e}")
    
    def metadata(self):
        """Get device metadata.
        
        Returns:
            dict: Device metadata including type, qubits, and status
        """
        backend_info = self._adapter.get_backend_info()
        return {
            'device_id': self.id,
            'device_type': self.device_type,
            'num_qubits': self.num_qubits,
            'status': 'ONLINE',
            'queue_depth': 0,
            'local': True,
            'simulator': True,
            'method': backend_info.get('method', 'unknown'),
            'backend_name': backend_info.get('backend_name', 'local_simulator'),
        }


class LocalDeviceProfile:
    """Profile for local device containing configuration information."""
    
    def __init__(self, backend_info: Dict[str, Any]):
        """Initialize profile from backend info.
        
        Args:
            backend_info: Backend information dictionary
        """
        self.basis_gates = backend_info.get('basis_gates', [
            'u1', 'u2', 'u3', 'cx', 'id', 'x', 'y', 'z', 'h', 's', 't', 'sdg', 'tdg'
        ])
        self.coupling_map = backend_info.get('coupling_map', None)
        self.n_qubits = backend_info.get('num_qubits', 64)
        self.method = backend_info.get('method', 'automatic')
        self.local = True
        self.simulator = True


### Version of a device backend (e.g. ibm_sherbrooke --> '1.6.73') ###
@singledispatch
def version(device: QuantumDevice) -> str:
    raise NotImplementedError(f"Device version not implemented for device of type {type(device)}")


@version.register
def _(device: QiskitBackend) -> str:
    return device._backend.backend_version


@version.register
def _(device: LocalDevice) -> str:
    """Get version information for local devices.
    
    Args:
        device: LocalDevice instance
        
    Returns:
        str: Version string for the local simulator
    """
    return device._adapter.get_version()


@singledispatch
def connectivity_graph(device: QuantumDevice) -> rx.PyGraph:
    raise NotImplementedError(
        f"Connectivity graph not implemented for device of type {type(device)}"
    )


@connectivity_graph.register
def _(device: QiskitBackend) -> rx.PyGraph:
    return device._backend.coupling_map.graph.to_undirected(multigraph=False)


@connectivity_graph.register
def _(device: BraketDevice) -> rx.PyGraph:
    return cast(
        rx.PyGraph,
        rx.networkx_converter(nx.Graph(device._device.topology_graph.to_undirected())),
    )


@connectivity_graph.register
def _(device: LocalDevice) -> rx.PyGraph:
    """Get connectivity graph for local devices.
    
    Args:
        device: LocalDevice instance
        
    Returns:
        rx.PyGraph: Connectivity graph for the device
    """
    num_qubits = device.num_qubits
    
    # Limit to reasonable size for benchmarks
    if num_qubits > 127:
        num_qubits = 20  # Reasonable default for benchmarking
    
    # Check if backend has specific coupling map
    if hasattr(device.profile, 'coupling_map') and device.profile.coupling_map:
        graph = rx.PyGraph()
        graph.add_nodes_from(range(num_qubits))
        for edge in device.profile.coupling_map:
            if len(edge) == 2 and edge[0] < num_qubits and edge[1] < num_qubits:
                graph.add_edge(edge[0], edge[1], None)
        return graph
    
    # Default topology selection based on device characteristics
    working_qubits = min(num_qubits, 10)  # Conservative default
    
    # Choose topology that's likely to be bipartite (required by BSEQ)
    # Line topology is always bipartite and works well for most benchmarks
    topology_type = "line"
    
    # For very specific needs, could choose based on device type
    backend_info = device._adapter.get_backend_info()
    method = backend_info.get('method', 'automatic')
    
    # Some methods might work better with different topologies
    if method == "stabilizer":
        topology_type = "line"  # Stabilizer works well with limited connectivity
    elif method == "density_matrix" and working_qubits <= 6:
        topology_type = "grid"  # Small grids for density matrix
    else:
        topology_type = "line"  # Safe default
    
    return create_topology_graph(topology_type, working_qubits)
