import rustworkx as rx
from unittest.mock import patch

from metriq_gym.helpers.graph_helpers import (
    limit_colors,
    device_graph_coloring,
    largest_connected_size,
    GraphColoring,
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


def test_device_graph_coloring_complete_graph_uses_misra_gries():
    """Test that complete graphs use Misra-Gries algorithm."""
    # Create a complete graph K5 (5 nodes, all connected)
    graph = rx.PyGraph()
    graph.add_nodes_from(range(5))
    # Add all possible edges to make it complete
    edges = [(i, j, 1) for i in range(5) for j in range(i + 1, 5)]
    graph.add_edges_from(edges)

    # Verify graph is complete (10 edges for K5)
    assert len(graph.edge_list()) == 10

    # Mock the rustworkx functions to verify which one is called
    with (
        patch("metriq_gym.helpers.graph_helpers.rx.graph_misra_gries_edge_color") as mock_misra,
        patch("metriq_gym.helpers.graph_helpers.rx.graph_greedy_edge_color") as mock_greedy,
    ):
        # Set return values for the mocks
        mock_misra.return_value = {0: 0, 1: 1, 2: 2, 3: 3, 4: 0, 5: 1, 6: 2, 7: 3, 8: 4, 9: 4}

        coloring = device_graph_coloring(graph)

        # Verify Misra-Gries was called and greedy was not
        mock_misra.assert_called_once_with(graph)
        mock_greedy.assert_not_called()

        assert isinstance(coloring, GraphColoring)
        assert coloring.num_nodes == 5


def test_device_graph_coloring_sparse_graph_uses_greedy():
    """Test that sparse (non-complete) graphs use greedy algorithm."""
    # Create a sparse graph (not complete, not bipartite)
    # Triangle (3 nodes) - this is complete K3, so add a 4th node with only one edge
    graph = rx.PyGraph()
    graph.add_nodes_from(range(4))
    # Add edges to form a triangle plus one additional edge
    # This makes it non-bipartite (has odd cycle) and non-complete (only 4 edges instead of 6)
    graph.add_edges_from([(0, 1, 1), (1, 2, 1), (2, 0, 1), (0, 3, 1)])

    # Verify graph is not complete (4 edges instead of 6 for K4)
    assert len(graph.edge_list()) == 4
    assert not rx.is_bipartite(graph)

    # Mock the rustworkx functions to verify which one is called
    with (
        patch("metriq_gym.helpers.graph_helpers.rx.graph_misra_gries_edge_color") as mock_misra,
        patch("metriq_gym.helpers.graph_helpers.rx.graph_greedy_edge_color") as mock_greedy,
    ):
        # Set return value for the greedy mock
        mock_greedy.return_value = {0: 0, 1: 1, 2: 2, 3: 1}

        coloring = device_graph_coloring(graph)

        # Verify greedy was called and Misra-Gries was not
        mock_greedy.assert_called_once_with(graph)
        mock_misra.assert_not_called()

        assert isinstance(coloring, GraphColoring)
        assert coloring.num_nodes == 4


def test_graph_coloring_limit_colors():
    graph = rx.generators.complete_graph(num_nodes=4)

    coloring = device_graph_coloring(graph)

    assert isinstance(coloring, GraphColoring)
    assert coloring.num_nodes == 4
    assert coloring.num_colors == 4

    # copy the first two colors to ensure those are what remain
    max_colors = 2
    retained_colors = {e: c for e, c in coloring.edge_color_map.items() if c < max_colors}
    coloring = limit_colors(coloring, max_colors)
    assert coloring.num_colors == max_colors
    assert coloring.edge_color_map == retained_colors


def test_graph_coloring_limit_colors_no_effect():
    graph = rx.PyGraph()
    graph.add_nodes_from(range(4))
    graph.add_edges_from([(0, 1, 1), (1, 2, 1), (2, 3, 1)])

    coloring = device_graph_coloring(graph)

    assert isinstance(coloring, GraphColoring)
    assert coloring.num_nodes == 4
    assert coloring.num_colors == 2

    original_edge_color_map = coloring.edge_color_map.copy()
    coloring = limit_colors(coloring, 4)
    assert coloring.num_colors == 2
    assert coloring.edge_color_map == original_edge_color_map
