"""Unit tests for EPLG benchmark."""

import pytest
import rustworkx as rx

from metriq_gym.benchmarks.eplg import (
    eplg_score_at_lengths,
    random_chain_from_graph,
    EPLGResult,
)


def test_eplg_score_at_lengths_exact_match():
    """Test EPLG score when targets match chain lengths exactly."""
    chain_lens = [10, 20, 50, 100]
    chain_eplgs = [0.01, 0.02, 0.03, 0.04]

    score, picked_vals, picks = eplg_score_at_lengths(chain_lens, chain_eplgs)

    assert score == pytest.approx(0.025)  # average of 0.01, 0.02, 0.03, 0.04
    assert picked_vals == [0.01, 0.02, 0.03, 0.04]
    assert picks == [(10, 10), (20, 20), (50, 50), (100, 100)]


def test_eplg_score_at_lengths_nearest_neighbor():
    """Test EPLG score with fallback to nearest chain length."""
    chain_lens = [4, 6, 8]
    chain_eplgs = [0.01, 0.02, 0.03]

    score, picked_vals, picks = eplg_score_at_lengths(chain_lens, chain_eplgs)

    # All targets (10, 20, 50, 100) fall back to nearest (8)
    assert picked_vals == [0.03, 0.03, 0.03, 0.03]
    assert picks == [(10, 8), (20, 8), (50, 8), (100, 8)]


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


def test_eplg_result_compute_score():
    """Test EPLGResult score computation."""
    result = EPLGResult(
        chain_lengths=[4, 6, 8],
        chain_eplgs=[0.01, 0.02, 0.03],
        eplg_10=0.03,
        eplg_20=0.03,
        eplg_50=0.03,
        eplg_100=0.03,
    )

    score = result.compute_score()
    assert score.value == pytest.approx(0.03)


def test_eplg_result_compute_score_partial():
    """Test EPLGResult score with some None values."""
    result = EPLGResult(
        chain_lengths=[4],
        chain_eplgs=[0.01],
        eplg_10=0.01,
        eplg_20=None,
        eplg_50=None,
        eplg_100=None,
    )

    score = result.compute_score()
    assert score.value == pytest.approx(0.01)
