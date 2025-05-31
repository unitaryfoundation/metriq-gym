
"""Device introspection and metadata utilities for quantum computing platforms.

This module provides standardized, device- and provider-agnostic access to metadata
and introspection utilities for quantum devices, including both remote devices (via qBraid)
and local simulators.
"""

from functools import singledispatch
from typing import cast

import networkx as nx
from qbraid import QuantumDevice
from qbraid.runtime import BraketDevice, QiskitBackend
import rustworkx as rx


### Version of a device backend (e.g. ibm_sherbrooke --> '1.6.73') ###
@singledispatch
def version(device: QuantumDevice) -> str:
    """Get version information for a quantum device.
    
    Args:
        device: Quantum device instance
        
    Returns:
        Version string for the device
        
    Raises:
        NotImplementedError: If version not implemented for device type
    """
    raise NotImplementedError(f"Device version not implemented for device of type {type(device)}")


@version.register
def _(device: QiskitBackend) -> str:
    """Get version for QiskitBackend devices."""
    return device._backend.backend_version


def _get_local_device_version(device) -> str:
    """Get version for local devices using adapter pattern.
    
    Args:
        device: Device instance with _adapter attribute
        
    Returns:
        Version string from adapter or default
    """
    if hasattr(device, "_adapter") and hasattr(device._adapter, "get_version"):
        return device._adapter.get_version()
    return "local-1.0.0"


@singledispatch
def connectivity_graph(device: QuantumDevice) -> rx.PyGraph:
    """Get connectivity graph for a quantum device.
    
    Args:
        device: Quantum device instance
        
    Returns:
        Connectivity graph showing qubit connections
        
    Raises:
        NotImplementedError: If connectivity graph not implemented for device type
    """
    raise NotImplementedError(
        f"Connectivity graph not implemented for device of type {type(device)}"
    )


@connectivity_graph.register
def _(device: QiskitBackend) -> rx.PyGraph:
    """Get connectivity graph for QiskitBackend devices."""
    return device._backend.coupling_map.graph.to_undirected(multigraph=False)


@connectivity_graph.register
def _(device: BraketDevice) -> rx.PyGraph:
    """Get connectivity graph for BraketDevice devices."""
    return cast(
        rx.PyGraph,
        rx.networkx_converter(nx.Graph(device._device.topology_graph.to_undirected())),
    )


def _get_local_device_connectivity_graph(device) -> rx.PyGraph:
    """Get connectivity graph for local devices.
    
    Args:
        device: Local device instance
        
    Returns:
        Bipartite connectivity graph suitable for benchmarks
    """
    # Get device properties safely
    num_qubits = getattr(device, "num_qubits", 64)
    
    # Apply reasonable limits for benchmarking
    if num_qubits > 127:
        working_qubits = 20  # Conservative default for benchmarks
    else:
        working_qubits = min(num_qubits, 10)  # Conservative default
    
    # Check if device has specific coupling map
    if (hasattr(device, "profile") and 
        hasattr(device.profile, "coupling_map") and 
        device.profile.coupling_map):
        graph = rx.PyGraph()
        graph.add_nodes_from(range(working_qubits))
        for edge in device.profile.coupling_map:
            if len(edge) == 2 and edge[0] < working_qubits and edge[1] < working_qubits:
                graph.add_edge(edge[0], edge[1], None)
        return graph
    
    # Default: Create line topology (always bipartite - required for BSEQ)
    graph = rx.PyGraph()
    graph.add_nodes_from(range(working_qubits))
    for i in range(working_qubits - 1):
        graph.add_edge(i, i + 1, None)
    
    return graph

def _register_local_device_support():
    """Register local device support dynamically to avoid import issues.
    
    This function registers singledispatch functions for local devices
    when they are available, avoiding circular import problems.
    """
    try:
        # Try to import LocalDevice from the simulators module
        from metriq_gym.simulators.adapters import LocalDevice
        
        # Register version function for LocalDevice
        @version.register(LocalDevice)
        def _(device: LocalDevice) -> str:
            return _get_local_device_version(device)
        
        # Register connectivity_graph function for LocalDevice  
        @connectivity_graph.register(LocalDevice)
        def _(device: LocalDevice) -> rx.PyGraph:
            return _get_local_device_connectivity_graph(device)
            
    except ImportError:
        # LocalDevice not available - this is OK, just means no local simulator support
        pass


# Try to register local device support when module is imported
_register_local_device_support()

def get_device_version(device) -> str:
    """Get version for any device type with fallback.
    
    Args:
        device: Any device instance
        
    Returns:
        Version string
    """
    try:
        return version(device)
    except NotImplementedError:
        # Check if it's a local device (duck typing)
        if hasattr(device, "_adapter"):
            return _get_local_device_version(device)
        # Fallback for unknown devices
        return "unknown"


def get_device_connectivity_graph(device) -> rx.PyGraph:
    """Get connectivity graph for any device type with fallback.
    
    Args:
        device: Any device instance
        
    Returns:
        Connectivity graph
    """
    try:
        return connectivity_graph(device)
    except NotImplementedError:
        # Check if it's a local device (duck typing)
        if hasattr(device, "_adapter") or hasattr(device, "num_qubits"):
            return _get_local_device_connectivity_graph(device)
        # Fallback: minimal graph
        graph = rx.PyGraph()
        graph.add_nodes_from(range(2))
        graph.add_edge(0, 1, None)
        return graph