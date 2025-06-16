import rustworkx as rx


from metriq_gym.helpers.graph_helpers import (
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
