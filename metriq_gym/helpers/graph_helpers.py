from dataclasses import dataclass, field

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
        if self.edge_color_map:
            self.num_colors = max(self.edge_color_map.values()) + 1
        else:
            self.num_colors = 0

    @classmethod
    def from_dict(cls, data: dict) -> "GraphColoring":
        """Reconstruct GraphColoring from a dictionary, ensuring integer keys."""
        return cls(
            num_nodes=data["num_nodes"],
            edge_color_map={int(k): v for k, v in data["edge_color_map"].items()},
            edge_index_map={int(k): v for k, v in data["edge_index_map"].items()},
        )


def limit_colors(coloring: GraphColoring, max_colors: int) -> "GraphColoring":
    """Adjusts the coloring to use at most max_colors colors.

    Args:
        coloring: The original GraphColoring instance to limit.
        max_colors: Maximum number of colors to use.

    Returns:
        A new GraphColoring instance with limited colors.
    """

    new_edge_color_map = {
        edge: color for edge, color in coloring.edge_color_map.items() if color < max_colors
    }

    return GraphColoring(
        num_nodes=coloring.num_nodes,
        edge_color_map=new_edge_color_map,
        edge_index_map=coloring.edge_index_map,
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

    Chooses between:
      - Bipartite edge-coloring for bipartite graphs (optimal),
      - Misra-Gries edge-coloring for complete graphs (optimal, uses exactly max_degree colors),
      - Greedy edge-coloring otherwise (fast, good for sparse graphs).

    Args:
        topology_graph: The topology graph (coupling map) of the quantum device.

    Returns:
        GraphColoring: An object containing the coloring information.
    """
    num_nodes = topology_graph.num_nodes()

    if rx.is_bipartite(topology_graph):
        edge_color_map = rx.graph_bipartite_edge_color(topology_graph)
    else:
        # Check if graph is complete
        num_edges = len(topology_graph.edge_list())
        expected_edges_if_complete = num_nodes * (num_nodes - 1) // 2
        is_complete = num_edges == expected_edges_if_complete

        if is_complete:
            # Use Misra-Gries for complete graphs (optimal: exactly max_degree colors)
            edge_color_map = rx.graph_misra_gries_edge_color(topology_graph)
        else:
            # Use greedy for sparse graphs (faster, often good results)
            edge_color_map = rx.graph_greedy_edge_color(topology_graph)

    edge_index_map = dict(topology_graph.edge_index_map())
    return GraphColoring(
        num_nodes=num_nodes,
        edge_color_map=edge_color_map,
        edge_index_map=edge_index_map,
    )
