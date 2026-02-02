"""Unit tests for EPLG benchmark."""

import rustworkx as rx

from metriq_gym.benchmarks.eplg import (
    random_chain_from_graph,
)


def test_random_chain_from_graph_path():
    """Test random chain from path graph."""
    graph = rx.generators.path_graph(10)
    chain = random_chain_from_graph(graph, 5, seed=42)

    assert len(chain) == 5
    assert len(set(chain)) == 5
    # Verify it's a valid path
    for i in range(len(chain) - 1):
        assert graph.has_edge(chain[i], chain[i + 1])


def test_random_chain_from_graph_complete():
    """Test random chain from complete graph."""
    graph = rx.generators.complete_graph(10)
    chain = random_chain_from_graph(graph, 5, seed=42)

    assert len(chain) == 5
    assert len(set(chain)) == 5
