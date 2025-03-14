from dataclasses import dataclass, field
from typing import cast
from qbraid import QuantumDevice
from qbraid.runtime import QiskitBackend, BraketDevice

import networkx as nx
import rustworkx as rx
import numpy as np


@dataclass
class GraphColoring:
    """A simple class containing graph coloring data.

    Attributes:
        num_nodes: Number of qubits (nodes) in the graph.
        edge_color_map: Maps each edge index to a color (integer).
        edge_index_map: Maps edge indices to actual qubit pairs.
        num_colors: Total number of colors assigned in the graph.
    """

    num_nodes: int
    edge_color_map: dict
    edge_index_map: dict
    num_colors: int = field(init=False)

    def __post_init__(self):
        self.num_colors = max(self.edge_color_map.values()) + 1

    @classmethod
    def from_dict(cls, data: dict) -> "GraphColoring":
        """Reconstruct GraphColoring from a dictionary, ensuring integer keys."""
        return cls(
            num_nodes=data["num_nodes"],
            edge_color_map={int(k): v for k, v in data["edge_color_map"].items()},
            edge_index_map={int(k): v for k, v in data["edge_index_map"].items()},
        )


def largest_connected_size(good_graph: rx.PyGraph) -> int:
    """Finds the size of the largest connected component in the CHSH subgraph.

    Args:
        good_graph: The graph of qubit pairs that violated CHSH inequality.

    Returns:
        The size of the largest connected component.
    """
    if good_graph.num_nodes() == 0:
        return 0
    cc = rx.connected_components(good_graph)
    largest_cc = cc[np.argmax([len(g) for g in cc])]
    return len(largest_cc)


def device_graph_coloring(topology_graph: rx.PyGraph) -> GraphColoring:
    """Performs graph coloring for a quantum device's topology.

    The goal is to assign colors to edges such that no two adjacent edges have the same color.
    This ensures independent BSEQ experiments can be run in parallel. Identifies qubit pairs (edges)
    that can be executed without interference. These pairs are grouped by "color." The coloring reduces
    the complexity of the benchmarking process by organizing the graph into independent sets of qubit pairs.

    Args:
        topology_graph: The topology graph (coupling map) of the quantum device.

    Returns:
        GraphColoring: An object containing the coloring information.
    """
    num_nodes = topology_graph.num_nodes()

    # Graphs are bipartite, so use that feature to prevent extra colors from greedy search.
    # This graph is colored using a bipartite edge-coloring algorithm.
    edge_color_map = rx.graph_bipartite_edge_color(topology_graph)

    # Get the index of the edges.
    edge_index_map = dict(topology_graph.edge_index_map())
    return GraphColoring(
        num_nodes=num_nodes, edge_color_map=edge_color_map, edge_index_map=edge_index_map
    )


def device_topology(device: QuantumDevice) -> rx.PyGraph:
    """Extracts the device topology graph from a QuantumDevice object.

    Args:
        device: The QuantumDevice object.

    Returns:
        The device topology graph.
    """
    if isinstance(device, QiskitBackend):
        return device._backend.coupling_map.graph.to_undirected(multigraph=False)
    elif isinstance(device, BraketDevice):
        return cast(
            rx.PyGraph,
            rx.networkx_converter(nx.Graph(device._device.topology_graph.to_undirected())),
        )

    raise ValueError("Device type not supported.")
