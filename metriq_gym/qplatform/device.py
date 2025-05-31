"""Device introspection and metadata utilities for quantum computing platforms.

This module provides standardized, device- and provider-agnostic access to metadata
and introspection utilities for quantum devices, including both remote devices (via qBraid)
and local simulators with configurable topologies.
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
from metriq_gym.simulators.adapters import LocalDevice
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
    except (FileNotFoundError, json.JSONDecodeError):
        # Return default configuration if file not found or invalid
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
    
    elif topology_type == TopologyType.HEAVY_HEX:
        # Simplified heavy hex pattern (IBM-style)
        # For small systems, create a bipartite approximation
        if num_qubits <= 6:
            # Small heavy hex as line
            for i in range(num_qubits - 1):
                graph.add_edge(i, i + 1, None)
        else:
            # Larger heavy hex approximation
            for i in range(0, num_qubits - 2, 2):
                if i + 1 < num_qubits:
                    graph.add_edge(i, i + 1, None)
                if i + 2 < num_qubits:
                    graph.add_edge(i + 1, i + 2, None)
    
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


# ================================================================================================
# DEVICE CONNECTIVITY FUNCTIONS
# ================================================================================================

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
    """Get connectivity graph for local devices using configuration.
    
    Args:
        device: Local device instance
        
    Returns:
        Bipartite connectivity graph suitable for benchmarks
    """
    # Load configuration
    config = _load_config_file("device_config.json")
    topology_defaults = config.get("topology_defaults", {})
    
    # Get device properties safely
    num_qubits = getattr(device, "num_qubits", 64)
    max_qubits = topology_defaults.get("max_qubits", 10)
    
    # Apply qubit limit with warning for large devices
    if num_qubits > max_qubits:
        import logging
        logger = logging.getLogger(__name__)
        logger.info(
            f"Device reports {num_qubits} qubits, limiting to {max_qubits} "
            f"for benchmark compatibility. Adjust max_qubits in config if needed."
        )
        working_qubits = max_qubits
    else:
        working_qubits = min(num_qubits, max_qubits)
    
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
    
    # Use configured default topology
    default_topology = topology_defaults.get("default_topology", TopologyType.LINE)
    return create_topology_graph(default_topology, working_qubits)


# ================================================================================================
# DYNAMIC REGISTRATION FOR LOCAL DEVICES
# ================================================================================================

def _register_local_device_support():
    """Register local device support dynamically to avoid import issues.
    
    This function registers singledispatch functions for local devices
    when they are available, avoiding circular import problems.
    """

    # Register version function for LocalDevice
    @version.register(LocalDevice)
    def _(device: LocalDevice) -> str:
        return _get_local_device_version(device)
    
    # Register connectivity_graph function for LocalDevice  
    @connectivity_graph.register(LocalDevice)
    def _(device: LocalDevice) -> rx.PyGraph:
        return _get_local_device_connectivity_graph(device)


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
        # Fallback: minimal bipartite graph
        graph = rx.PyGraph()
        graph.add_nodes_from(range(2))
        graph.add_edge(0, 1, None)
        return graph

def get_topology_config() -> dict:
    """Get the current topology configuration.
    
    Returns:
        Dictionary containing topology configuration
    """
    return _load_config_file("device_config.json")


def get_available_topologies() -> list[str]:
    """Get list of available topology types.
    
    Returns:
        List of topology type names
    """
    config = get_topology_config()
    topologies = config.get("topologies", {})
    return list(topologies.keys())


def is_topology_bipartite(topology_type: str, num_qubits: int = None) -> bool:
    """Check if a topology type is bipartite for given number of qubits.
    
    Args:
        topology_type: Type of topology to check
        num_qubits: Number of qubits (needed for some topologies)
        
    Returns:
        True if topology is bipartite, False otherwise
    """
    config = get_topology_config()
    topologies = config.get("topologies", {})
    
    if topology_type not in topologies:
        return True  # Default to bipartite-safe
    
    topology_config = topologies[topology_type]
    
    # Check direct bipartite flag
    if "bipartite" in topology_config:
        return topology_config["bipartite"]
    
    # Check conditional bipartite
    condition = topology_config.get("bipartite_condition")
    if condition == "even_nodes" and num_qubits is not None:
        return num_qubits % 2 == 0
    elif condition == "small_only" and num_qubits is not None:
        return num_qubits <= 2
    
    # Default to True for safety
    return True