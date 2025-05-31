
"""Device introspection and metadata utilities for quantum computing platforms.

This module provides standardized, device- and provider-agnostic access to metadata
and introspection utilities for quantum devices, including both remote devices (via qBraid)
and local simulators.
"""

import json
import math
import os
from enum import StrEnum
from functools import singledispatch
from typing import cast

import networkx as nx
import rustworkx as rx
from qbraid import QuantumDevice
from qbraid.runtime import BraketDevice, QiskitBackend


class TopologyType(StrEnum):
    """Enumeration of supported topology types."""
    LINE = "line"
    RING = "ring" 
    GRID = "grid"
    HEAVY_HEX = "heavy_hex"
    ALL_TO_ALL = "all_to_all"


def _load_config_file(filename: str) -> dict:
    """Load configuration from JSON file.
    
    Args:
        filename: Name of the configuration file
        
    Returns:
        Configuration dictionary
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, "..", "config", filename)
    
    try:
        with open(config_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        # Return default configuration if file not found
        return _get_default_config()


def _get_default_config() -> dict:
    """Get default configuration when config file is not available.
    
    Returns:
        Default configuration dictionary
    """
    return {
        "topology_defaults": {
            "max_qubits": 10,
            "default_topology": TopologyType.LINE
        },
        "topologies": {
            TopologyType.LINE: {
                "bipartite": True,
                "description": "Linear chain topology (always bipartite)"
            },
            TopologyType.RING: {
                "bipartite_condition": "even_nodes",
                "description": "Ring topology (bipartite only for even number of qubits)"
            },
            TopologyType.GRID: {
                "bipartite": True,
                "description": "2D grid topology (always bipartite)"
            },
            TopologyType.HEAVY_HEX: {
                "bipartite": True,
                "description": "IBM heavy hexagon topology"
            },
            TopologyType.ALL_TO_ALL: {
                "bipartite_condition": "small_only",
                "description": "Complete graph (rarely bipartite)"
            }
        }
    }


def create_topology_graph(topology_type: str, num_qubits: int) -> rx.PyGraph:
    """Create a topology graph based on configuration.
    
    Args:
        topology_type: Type of topology from TopologyType enum
        num_qubits: Number of qubits
        
    Returns:
        rx.PyGraph: The topology graph
    """
    graph = rx.PyGraph()
    graph.add_nodes_from(range(num_qubits))
    
    if topology_type == TopologyType.LINE:
        # Linear chain: 0-1-2-3-...
        for i in range(num_qubits - 1):
            graph.add_edge(i, i + 1, None)
    
    elif topology_type == TopologyType.RING:
        # Ring: 0-1-2-3-...-0
        for i in range(num_qubits - 1):
            graph.add_edge(i, i + 1, None)
        if num_qubits > 2:
            graph.add_edge(num_qubits - 1, 0, None)
    
    elif topology_type == TopologyType.GRID:
        # 2D grid topology
        side_length = int(math.sqrt(num_qubits))
        for i in range(num_qubits):
            row, col = divmod(i, side_length)
            # Connect to right neighbor
            if col < side_length - 1:
                graph.add_edge(i, i + 1, None)
            # Connect to bottom neighbor  
            if row < side_length - 1:
                graph.add_edge(i, i + side_length, None)
    
    elif topology_type == TopologyType.ALL_TO_ALL:
        # Complete graph (use sparingly)
        for i in range(num_qubits):
            for j in range(i + 1, num_qubits):
                graph.add_edge(i, j, None)
    
    else:
        # Default to line topology for unknown types
        for i in range(num_qubits - 1):
            graph.add_edge(i, i + 1, None)
    
    return graph


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


@version.register
def _(device):
    """Get version for LocalDevice instances.
    
    This handles LocalDevice instances by checking for the adapter attribute.
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


@connectivity_graph.register
def _(device):
    """Get connectivity graph for LocalDevice instances.
    
    This handles LocalDevice instances by creating appropriate topology graphs
    based on device configuration and loaded settings.
    """
    # Load configuration
    config = _load_config_file("device_config.json")
    topology_defaults = config.get("topology_defaults", {})
    
    # Get device properties
    num_qubits = getattr(device, "num_qubits", 64)
    max_qubits = topology_defaults.get("max_qubits", 10)
    
    # Apply qubit limit with warning (not silent)
    if num_qubits > max_qubits:
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(
            f"Device reports {num_qubits} qubits, but limiting to {max_qubits} "
            f"for benchmark compatibility. Adjust max_qubits in device_config.json if needed."
        )
        working_qubits = max_qubits
    else:
        working_qubits = num_qubits
    
    # Check if device has specific coupling map
    if hasattr(device, "profile") and hasattr(device.profile, "coupling_map") and device.profile.coupling_map:
        graph = rx.PyGraph()
        graph.add_nodes_from(range(working_qubits))
        for edge in device.profile.coupling_map:
            if len(edge) == 2 and edge[0] < working_qubits and edge[1] < working_qubits:
                graph.add_edge(edge[0], edge[1], None)
        return graph
    
    # Use default topology from configuration
    default_topology = topology_defaults.get("default_topology", TopologyType.LINE)
    
    return create_topology_graph(default_topology, working_qubits)

