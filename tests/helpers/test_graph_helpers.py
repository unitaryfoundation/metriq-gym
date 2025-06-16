import pytest
import rustworkx as rx
from collections import defaultdict


from metriq_gym.helpers.graph_helpers import (
    device_graph_coloring,
    largest_connected_size,
    GraphColoring,
    _complete_graph_edge_color,
)


# Tests for largest_connected_size:
def test_largest_connected_size_single_component():
    """Test a C5 graph with a single connected component."""
    graph = rx.PyGraph()
    graph.add_nodes_from(range(5))
    graph.add_edges_from([(0, 1, 1), (1, 2, 1), (2, 3, 1), (3, 4, 1)])

    assert largest_connected_size(graph) == 5


def test_largest_connected_size_multiple_components():
    """Test a graph with two connected components."""
    graph = rx.PyGraph()
    graph.add_nodes_from(range(7))
    graph.add_edges_from([(0, 1, 1), (1, 2, 1), (3, 4, 1), (4, 5, 1)])
    print(graph)
    assert largest_connected_size(graph) == 3


def test_largest_connected_size_disconnected_nodes():
    """Test a graph with disconnected nodes."""
    graph = rx.PyGraph()
    graph.add_nodes_from(range(5))
    graph.add_edges_from([(0, 1, 1), (2, 3, 1)])

    assert largest_connected_size(graph) == 2


def test_largest_connected_size_empty_graph():
    """Test an empty graph."""
    graph = rx.PyGraph()
    assert largest_connected_size(graph) == 0


# Tests for device_graph_coloring:
def test_device_graph_coloring_basic():
    """Test graph coloring for a simple connected graph."""
    graph = rx.PyGraph()
    graph.add_nodes_from(range(4))
    graph.add_edges_from([(0, 1, 1), (1, 2, 1), (2, 3, 1)])

    coloring = device_graph_coloring(graph)

    assert isinstance(coloring, GraphColoring)
    assert coloring.num_nodes == 4
    assert set(coloring.edge_index_map.keys()) == set(coloring.edge_color_map.keys())
    assert max(coloring.edge_color_map.values()) + 1 <= 3  # Should use at most 3 colors


def test_device_graph_coloring_disconnected():
    """Test graph coloring for a graph with two disconnected components."""
    graph = rx.PyGraph()
    graph.add_nodes_from(range(6))
    graph.add_edges_from([(0, 1, 1), (2, 3, 1), (4, 5, 1)])

    coloring = device_graph_coloring(graph)

    assert isinstance(coloring, GraphColoring)
    assert coloring.num_nodes == 6
    assert max(coloring.edge_color_map.values()) + 1 <= 2  # Should use at most 2 colors


def test_device_graph_coloring_bipartite():
    """Test graph coloring for a bipartite graph."""
    graph = rx.PyGraph()
    graph.add_nodes_from(range(6))
    graph.add_edges_from([(0, 3, 1), (0, 4, 1), (1, 4, 1), (1, 5, 1), (2, 5, 1)])

    coloring = device_graph_coloring(graph)

    assert isinstance(coloring, GraphColoring)
    assert coloring.num_nodes == 6
    assert max(coloring.edge_color_map.values()) + 1 == 2  # Bipartite graphs need only 2 colors


def test_device_graph_coloring_multigraph():
    """Test graph coloring for a multigraph with parallel edges."""
    graph = rx.PyGraph(multigraph=True)
    graph.add_nodes_from(range(3))
    graph.add_edges_from([(0, 1, 1), (0, 1, 2), (1, 2, 1)])

    coloring = device_graph_coloring(graph)

    assert isinstance(coloring, GraphColoring)
    assert coloring.num_nodes == 3
    assert max(coloring.edge_color_map.values()) + 1 <= 3  # Should use at most 3 colors
    assert len(coloring.edge_index_map) == 3  # Should match the number of edges


@pytest.mark.parametrize(
    "n, expected_colors",
    [
        (0, 0),
        (1, 0),
        (2, 1),
        (3, 3),
        (4, 3),
        (5, 5),
        (8, 7),
        (9, 9),
    ],
)
def test_complete_graph_coloring(n, expected_colors):
    """Test one-factorization coloring of complete graphs K_n for various n."""
    graph = rx.generators.complete_graph(n)
    # Call the function from the source file.
    color_map = _complete_graph_edge_color(graph)

    if n < 2:
        assert not color_map, f"Color map should be empty for n={n}, but was not."
        return

    assert color_map, f"Color map is unexpectedly empty for n={n}."
    assert len(color_map) == graph.num_edges(), "Not all edges were colored."

    num_colors_found = max(color_map.values()) + 1
    assert num_colors_found == expected_colors, (
        f"Expected {expected_colors} colors for K{n}, but found {num_colors_found}."
    )

    # Verify that the coloring is valid.
    # 1. Check that all edges are colored.
    assert len(color_map) == graph.num_edges(), (
        "The number of colored edges does not match the graph."
    )

    # 2. Group edges by color.
    edges_by_color = defaultdict(list)
    for edge_idx, color in color_map.items():
        edges_by_color[color].append(edge_idx)

    # 3. For each color, check that the edges form a valid matching.
    edge_map = graph.edge_index_map()
    for color, edge_indices in edges_by_color.items():
        nodes_in_matching = set()
        for edge_idx in edge_indices:
            u, v, _ = edge_map[edge_idx]  # Unpack the 3-tuple
            assert u not in nodes_in_matching, f"Node {u} is repeated in color {color}"
            assert v not in nodes_in_matching, f"Node {v} is repeated in color {color}"
            nodes_in_matching.add(u)
            nodes_in_matching.add(v)
