"""Unit tests for EPLG benchmark."""

from unittest.mock import MagicMock, patch

import pytest
import rustworkx as rx
from qiskit_ibm_runtime.fake_provider import FakeMarrakesh

from metriq_gym.benchmarks.eplg import (
    EPLG,
    random_chain_from_graph,
)
from metriq_gym.qplatform.device import coupling_map_to_graph


def assert_valid_chain(graph: rx.PyGraph, chain: list[int], length: int) -> None:
    """Assert that a chain is simple, has the requested size, and follows graph edges."""
    assert len(chain) == length
    assert len(set(chain)) == length
    assert all(graph.has_edge(source, target) for source, target in zip(chain, chain[1:]))


def test_random_chain_from_graph_path():
    """Test random chain from path graph."""
    graph = rx.generators.path_graph(10)
    with patch("metriq_gym.benchmarks.eplg.rx.vf2_mapping") as vf2_mapping:
        chain = random_chain_from_graph(graph, 5, seed=42)

    assert chain == [0, 1, 2, 3, 4]
    vf2_mapping.assert_not_called()
    # Verify it's a valid path
    for i in range(len(chain) - 1):
        assert graph.has_edge(chain[i], chain[i + 1])


def test_random_chain_from_graph_complete():
    """Test random chain from complete graph."""
    graph = rx.generators.complete_graph(10)
    chain = random_chain_from_graph(graph, 5, seed=42)

    assert len(chain) == 5
    assert len(set(chain)) == 5


def test_random_chain_from_graph_falls_back_for_marrakesh():
    """A valid Marrakesh 100-chain is found after the greedy search gets trapped."""
    backend = FakeMarrakesh()
    graph = coupling_map_to_graph(backend.target.build_coupling_map(two_q_gate="cz"))

    chain = random_chain_from_graph(graph, 100, seed=12345)

    assert_valid_chain(graph, chain, 100)


def test_random_chain_vf2_fallback_is_seeded_and_deterministic():
    """The fallback handles a Miami-like grid reproducibly without greedy attempts."""
    graph = rx.generators.grid_graph(10, 12)

    first = random_chain_from_graph(graph, 100, seed=12345, restarts=0)
    second = random_chain_from_graph(graph, 100, seed=12345, restarts=0)

    assert first == second
    assert_valid_chain(graph, first, 100)


def test_random_chain_from_graph_supports_non_contiguous_node_indices():
    """Removed graph nodes do not break adjacency or fallback mapping."""
    graph = rx.PyGraph()
    graph.add_nodes_from([None] * 7)
    graph.add_edges_from_no_data([(0, 2), (2, 4), (4, 6)])
    graph.remove_nodes_from([1, 3, 5])

    chain = random_chain_from_graph(graph, 4, seed=12345, restarts=0)

    assert_valid_chain(graph, chain, 4)
    assert set(chain) == {0, 2, 4, 6}


def test_random_chain_rejects_insufficient_connectivity():
    """Definitively incompatible component sizes produce an actionable error."""
    graph = rx.PyGraph()
    graph.add_nodes_from([None] * 8)
    graph.add_edges_from_no_data([(0, 1), (1, 2), (2, 3), (4, 5), (5, 6), (6, 7)])

    with pytest.raises(ValueError, match="largest connected component has only 4 nodes"):
        random_chain_from_graph(graph, 5, seed=12345)


def test_random_chain_rejects_more_nodes_than_the_graph_contains():
    """A request larger than the entire graph fails before searching."""
    graph = rx.generators.path_graph(4)

    with pytest.raises(ValueError, match="connectivity graph has only 4 nodes"):
        random_chain_from_graph(graph, 5, seed=12345)


def test_random_chain_reports_inconclusive_bounded_search():
    """A large component without a long simple path is not called incompatible."""
    graph = rx.generators.star_graph(10)

    with pytest.raises(RuntimeError, match="search budget may have been exhausted"):
        random_chain_from_graph(graph, 5, seed=12345, restarts=0)


def test_eplg_warns_when_device_gate_support_cannot_be_confirmed():
    """EPLG tells users how to adapt its device-dependent gate parameters."""
    graph = rx.generators.path_graph(2)
    layer_fidelity = MagicMock()
    layer_fidelity.circuits.return_value = []

    params = MagicMock()
    params.num_qubits_in_chain = 2
    params.two_qubit_gate = "cz"
    params.one_qubit_basis_gates = ["rz", "rx", "x"]
    params.lengths = [2]
    params.num_samples = 1
    params.seed = 12345
    params.decompose_clifford_ops = False

    with (
        patch(
            "metriq_gym.benchmarks.eplg.connectivity_graph_for_gate",
            return_value=None,
        ),
        patch("metriq_gym.benchmarks.eplg.connectivity_graph", return_value=graph),
        patch(
            "metriq_gym.benchmarks.eplg.random_chain_from_graph",
            return_value=[0, 1],
        ),
        patch(
            "metriq_gym.benchmarks.eplg.LayerFidelity",
            return_value=layer_fidelity,
        ),
        pytest.warns(
            RuntimeWarning,
            match="provider/device-specific.*change them to a native universal gate set",
        ),
    ):
        EPLG(MagicMock(), params)._build_circuits(MagicMock())
